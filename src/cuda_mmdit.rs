//! The MMDiT denoiser on the device, the twin of [`Dit::forward_with`].
//!
//! Like the T5 encoder in [`crate::cuda_t5`], it runs on the BF16 kernels the
//! language model trains with: the residual streams and every GEMM
//! accumulator are FP32, and only the weights and the operands a GEMM reads
//! are BF16. Attention is the fused BF16 kernel, unmasked.
//!
//! The conditioning vector is built on the host, because it is a handful of
//! small layers over one vector. Every block's modulation is then one GEMV:
//! the modulation projections of all blocks are stacked into one weight and
//! run once per call.
//!
//! NOTE: the norms, the head norms and the bias adds write FP32 and a
//! separate cast narrows the GEMM operands. Fusing the cast into those
//! kernels saves one pass per GEMM if a profile ever shows it.

use crate::conv::Dense;
use crate::cuda_training::{cfg, cuda_err};
use crate::gpu_model::{Act, FlashShape, Gpu};
use crate::gpu_transformer::GpuContext;
use crate::matrix::Matrix;
use crate::mmdit::{Conditioning, Dit, Positions};
use crate::network::NetworkError;
use cudarc::driver::{CudaFunction, CudaSlice, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{Ptx, compile_ptx};
use std::sync::{Arc, OnceLock};

const KERNELS: &str = r#"
#define ROW_THREADS 256
__device__ __forceinline__ float row_sum(float v,float*red){
  int tid=threadIdx.x;red[tid]=v;__syncthreads();
  for(int s=ROW_THREADS/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
  float r=red[0];__syncthreads();return r;
}
// Layer norm with no learned scale, then `x * (1 + scale) + shift`, with the
// shift at `t[c]` and the scale at `t[cols + c]`. One block per row.
extern "C" __global__ void layer_norm_modulate(float*out,const float*x,const float*t,int rows,int cols,float eps){
  __shared__ float red[ROW_THREADS];int tid=threadIdx.x;
  for(int r=blockIdx.x;r<rows;r+=gridDim.x){
    const float*s=x+(size_t)r*cols;float*d=out+(size_t)r*cols;
    float a=0.f;for(int c=tid;c<cols;c+=ROW_THREADS)a+=s[c];
    float mean=row_sum(a,red)/cols;
    float b=0.f;for(int c=tid;c<cols;c+=ROW_THREADS){float v=s[c]-mean;b+=v*v;}
    float inv=rsqrtf(row_sum(b,red)/cols+eps);
    for(int c=tid;c<cols;c+=ROW_THREADS)d[c]=(s[c]-mean)*inv*(1.f+t[cols+c])+t[c];
  }
}
// The per-head RMS norm on a query or key, then the rotary position: pairs
// are adjacent dimensions, FLUX's order, not the language model's halves.
// One thread per (row, head); `first` is the row's index in the tables.
extern "C" __global__ void head_norm_rope(float*x,const float*scale,const float*cs,const float*sn,int rows,int heads,int head_dim,int stride,int off,int first){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*heads)return;int r=i/heads,h=i-r*heads;
  float*v=x+(size_t)r*stride+off+h*head_dim;
  float s=0.f;for(int d=0;d<head_dim;d++)s+=v[d]*v[d];
  float inv=rsqrtf(s/head_dim+1e-6f);int half=head_dim/2;
  const float*c=cs+(size_t)(first+r)*half;const float*n=sn+(size_t)(first+r)*half;
  for(int p=0;p<half;p++){
    float a=v[2*p]*inv*scale[2*p],b=v[2*p+1]*inv*scale[2*p+1];
    v[2*p]=a*c[p]-b*n[p];v[2*p+1]=a*n[p]+b*c[p];
  }
}
// `dst = act(src[:, off..off + cols] + bias[off..])`, where the activation is
// the tanh GELU or none. `dst` is `[rows, cols]`, `src` has row stride `stride`.
extern "C" __global__ void bias_gelu(float*dst,const float*src,const float*bias,int rows,int cols,int stride,int off,int gelu,int has_bias){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*cols)return;int r=i/cols,c=i-r*cols;
  float v=src[(size_t)r*stride+off+c];if(has_bias)v+=bias[off+c];
  if(gelu)v=0.5f*v*(1.f+tanhf(0.7978846f*(v+0.044715f*v*v*v)));
  dst[i]=v;
}
"#;

fn ptx() -> Result<&'static Ptx, NetworkError> {
    static PTX: OnceLock<Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| compile_ptx(KERNELS).map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| NetworkError::Cuda(format!("MMDiT kernel compilation failed: {error}")))
}

/// A layer's weight on the device, and its bias if it has one.
struct Linear {
    weight: Act,
    bias: Option<CudaSlice<f32>>,
    units: usize,
    inner: usize,
}

struct DeviceStream {
    qkv: Linear,
    query_norm: CudaSlice<f32>,
    key_norm: CudaSlice<f32>,
    projection: Linear,
    mlp_in: Linear,
    mlp_out: Linear,
}

struct DeviceSingle {
    fused_in: Linear,
    /// `fused_out`'s columns split at the attention's width: the attention
    /// half and the feed-forward half, two GEMMs into one output.
    attention_out: Linear,
    mlp_out: Linear,
    query_norm: CudaSlice<f32>,
    key_norm: CudaSlice<f32>,
}

/// An MMDiT denoiser resident on the device.
pub struct DeviceDit {
    context: Arc<GpuContext>,
    layer_norm_modulate: CudaFunction,
    head_norm_rope: CudaFunction,
    bias_gelu: CudaFunction,
    image_in: Linear,
    text_in: Linear,
    /// Every block's modulation projection stacked, in the order
    /// [`DeviceDit::forward`] reads them: each double block's image then text
    /// stream, each single block, then the final layer.
    modulation: Linear,
    double: Vec<(DeviceStream, DeviceStream)>,
    single: Vec<DeviceSingle>,
    final_out: Linear,
}

impl std::fmt::Debug for DeviceDit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceDit").finish_non_exhaustive()
    }
}

impl DeviceDit {
    /// Uploads `dit`'s weights as BF16, or as int8 where it holds them at one
    /// byte. The host model stays the source of the conditioning vector, so
    /// it is borrowed again by [`DeviceDit::forward_with`].
    pub fn upload(context: &Arc<GpuContext>, dit: &Dit) -> Result<Self, NetworkError> {
        if context.half_accumulate {
            return Err(NetworkError::InvalidConfig(
                "the denoiser needs a BF16 context, not an FP16 one".into(),
            ));
        }
        let head_dim = dit.config().head_dim();
        if context.flash.is_none() || !crate::cuda_flash::eligible(true, head_dim) {
            return Err(NetworkError::InvalidConfig(format!(
                "the device denoiser needs the fused attention kernel, which takes \
                 heads of 64 or 128 on Ampere or later; this one has {head_dim}"
            )));
        }
        let gpu = Gpu { context };
        let module = context
            .stream
            .context()
            .load_module(ptx()?.clone())
            .map_err(cuda_err("MMDiT module loading"))?;
        let get = |name: &str| {
            module
                .load_function(name)
                .map_err(cuda_err("MMDiT kernel lookup"))
        };
        let linear = |weight: Matrix, bias: Option<&[f32]>, quantized: bool| {
            let (units, inner) = (weight.rows, weight.cols);
            let wide = gpu.upload(&weight.data)?;
            Ok::<_, NetworkError>(Linear {
                weight: match quantized {
                    true => gpu.quantized(&wide.slice(..), units, inner)?,
                    false => gpu.narrowed(&wide.slice(..), units * inner, true)?,
                },
                bias: bias.map(|bias| gpu.upload(bias)).transpose()?,
                units,
                inner,
            })
        };
        let dense = |layer: &Dense| {
            linear(
                layer.weight_f32().into_owned(),
                layer.bias.as_deref(),
                layer.is_quantized(),
            )
        };
        let stream = |stream: &crate::mmdit::Stream| {
            Ok::<_, NetworkError>(DeviceStream {
                qkv: dense(&stream.qkv)?,
                query_norm: gpu.upload(&stream.query_norm.scale)?,
                key_norm: gpu.upload(&stream.key_norm.scale)?,
                projection: dense(&stream.projection)?,
                mlp_in: dense(&stream.mlp_in)?,
                mlp_out: dense(&stream.mlp_out)?,
            })
        };

        let modulations: Vec<&Dense> = dit
            .double
            .iter()
            .flat_map(|block| {
                [
                    &block.image.modulation.projection,
                    &block.text.modulation.projection,
                ]
            })
            .chain(dit.single.iter().map(|block| &block.modulation.projection))
            .chain([&dit.final_modulation])
            .collect();
        let (mut weights, mut biases) = (Vec::new(), Vec::new());
        for layer in &modulations {
            weights.extend_from_slice(&layer.weight_f32().data);
            match &layer.bias {
                Some(bias) => biases.extend_from_slice(bias),
                None => biases.resize(biases.len() + layer.out_dim(), 0.0),
            }
        }
        let inner = dit.config().d_model;
        let modulation = linear(
            Matrix::from_vec(weights.len() / inner, inner, weights),
            Some(&biases),
            modulations.iter().any(|layer| layer.is_quantized()),
        )?;

        let width = dit.config().d_model;
        let single = dit
            .single
            .iter()
            .map(|block| {
                let fused_out = block.fused_out.weight_f32();
                let mlp_dim = fused_out.cols - width;
                let columns = |from: usize, count: usize| {
                    let mut part = Matrix::new(fused_out.rows, count);
                    for row in 0..fused_out.rows {
                        part.row_mut(row)
                            .copy_from_slice(&fused_out.row(row)[from..from + count]);
                    }
                    part
                };
                let quantized = block.fused_out.is_quantized();
                Ok(DeviceSingle {
                    fused_in: dense(&block.fused_in)?,
                    attention_out: linear(
                        columns(0, width),
                        block.fused_out.bias.as_deref(),
                        quantized,
                    )?,
                    mlp_out: linear(columns(width, mlp_dim), None, quantized)?,
                    query_norm: gpu.upload(&block.query_norm.scale)?,
                    key_norm: gpu.upload(&block.key_norm.scale)?,
                })
            })
            .collect::<Result<_, NetworkError>>()?;

        Ok(Self {
            layer_norm_modulate: get("layer_norm_modulate")?,
            head_norm_rope: get("head_norm_rope")?,
            bias_gelu: get("bias_gelu")?,
            image_in: dense(&dit.image_in)?,
            text_in: dense(&dit.text_in)?,
            modulation,
            double: dit
                .double
                .iter()
                .map(|block| Ok((stream(&block.image)?, stream(&block.text)?)))
                .collect::<Result<_, NetworkError>>()?,
            single,
            final_out: dense(&dit.final_out)?,
            context: context.clone(),
        })
    }

    /// The prediction for `patches` at `timestep` under `conditioning`,
    /// `[patches, out_channels]`. `dit` must be the model this was uploaded
    /// from.
    pub fn forward_with(
        &self,
        dit: &Dit,
        patches: &Matrix,
        timestep: f32,
        conditioning: &Conditioning,
    ) -> Result<Matrix, NetworkError> {
        if patches.rows != conditioning.height * conditioning.width {
            return Err(NetworkError::InvalidTarget {
                expected: conditioning.height * conditioning.width,
                actual: patches.rows,
            });
        }
        let gpu = Gpu {
            context: &self.context,
        };
        let config = dit.config();
        let (width, heads, eps) = (config.d_model, config.num_heads, config.eps);
        let head_dim = config.head_dim();
        let (text_rows, image_rows) = (conditioning.text.rows, patches.rows);
        let rows = text_rows + image_rows;

        let vector: Vec<f32> = dit
            .vector(timestep, conditioning)?
            .iter()
            .map(|value| crate::ffn::silu(*value))
            .collect();
        let vector = gpu.narrowed(&gpu.upload(&vector)?.slice(..), width, true)?;
        let modulation = self.apply(&gpu, &self.modulation, &vector.all(), 1, false)?;
        let positions = Positions::new(config, text_rows, conditioning.height, conditioning.width);
        let (cos, sin) = (gpu.upload(&positions.cos)?, gpu.upload(&positions.sin)?);

        let mut text = self.input(&gpu, &self.text_in, &conditioning.text)?;
        let mut image = self.input(&gpu, &self.image_in, patches)?;
        let mut at = 0;
        let mut triple = |count: usize| {
            let view = modulation.slice(at..at + count * width);
            at += count * width;
            view
        };

        for (image_stream, text_stream) in &self.double {
            let image_mod = triple(6);
            let text_mod = triple(6);
            let streams = [
                (text_stream, &text_mod, 0, text_rows),
                (image_stream, &image_mod, text_rows, image_rows),
            ];
            let mut qkv = gpu.uninit(rows * 3 * width)?;
            for &(stream, modulation, first, count) in &streams {
                let hidden = if first == 0 { &text } else { &image };
                let normed = self.norm(&gpu, hidden, &modulation.slice(..), count, width, eps)?;
                let mut part = gpu.uninit(count * 3 * width)?;
                self.project(
                    &gpu,
                    &stream.qkv,
                    &normed,
                    count,
                    &mut part.slice_mut(..),
                    0.0,
                )?;
                let mut target = qkv.slice_mut(first * 3 * width..(first + count) * 3 * width);
                self.bias(&part, &stream.qkv, count, 0, 3 * width, false, &mut target)?;
                for (scale, offset) in [(&stream.query_norm, 0), (&stream.key_norm, width)] {
                    self.head_norm_rope(
                        &mut qkv, scale, &cos, &sin, first, count, heads, head_dim, offset,
                    )?;
                }
            }
            let attended = self.attention(&gpu, &qkv, rows, heads, width, 3 * width)?;
            for &(stream, modulation, first, count) in &streams {
                let hidden = if first == 0 { &mut text } else { &mut image };
                let attended = attended.at(first * width);
                let output = self.apply(&gpu, &stream.projection, &attended, count, false)?;
                *hidden = gpu.gate_residual(
                    &hidden.slice(..),
                    &output.slice(..),
                    &modulation.slice(..),
                    count,
                    width,
                    count,
                )?;
                let modulation = modulation.slice(3 * width..);
                let normed = self.norm(&gpu, hidden, &modulation, count, width, eps)?;
                let inner = self.apply(&gpu, &stream.mlp_in, &normed.all(), count, true)?;
                let inner = gpu.narrowed(&inner.slice(..), inner.len(), true)?;
                let output = self.apply(&gpu, &stream.mlp_out, &inner.all(), count, false)?;
                *hidden = gpu.gate_residual(
                    &hidden.slice(..),
                    &output.slice(..),
                    &modulation,
                    count,
                    width,
                    count,
                )?;
            }
        }

        let mut hidden = gpu.uninit(rows * width)?;
        let stream = &self.context.stream;
        let copy = cuda_err("device to device copy");
        stream
            .memcpy_dtod(&text, &mut hidden.slice_mut(..text_rows * width))
            .map_err(copy)?;
        stream
            .memcpy_dtod(&image, &mut hidden.slice_mut(text_rows * width..))
            .map_err(cuda_err("device to device copy"))?;
        for block in &self.single {
            let modulation = triple(3);
            let normed = self.norm(&gpu, &hidden, &modulation, rows, width, eps)?;
            let fused_width = block.fused_in.units;
            let mlp_dim = fused_width - 3 * width;
            let mut fused = gpu.uninit(rows * fused_width)?;
            self.project(
                &gpu,
                &block.fused_in,
                &normed,
                rows,
                &mut fused.slice_mut(..),
                0.0,
            )?;
            let mut qkv = gpu.uninit(rows * 3 * width)?;
            let mut mlp = gpu.uninit(rows * mlp_dim)?;
            let linear = &block.fused_in;
            self.bias(
                &fused,
                linear,
                rows,
                0,
                3 * width,
                false,
                &mut qkv.slice_mut(..),
            )?;
            self.bias(
                &fused,
                linear,
                rows,
                3 * width,
                mlp_dim,
                true,
                &mut mlp.slice_mut(..),
            )?;
            for (scale, offset) in [(&block.query_norm, 0), (&block.key_norm, width)] {
                self.head_norm_rope(
                    &mut qkv, scale, &cos, &sin, 0, rows, heads, head_dim, offset,
                )?;
            }
            let attended = self.attention(&gpu, &qkv, rows, heads, width, 3 * width)?;
            let mlp = gpu.narrowed(&mlp.slice(..), rows * mlp_dim, true)?;
            let mut output = gpu.uninit(rows * width)?;
            self.project(
                &gpu,
                &block.mlp_out,
                &mlp,
                rows,
                &mut output.slice_mut(..),
                0.0,
            )?;
            self.project(
                &gpu,
                &block.attention_out,
                &attended,
                rows,
                &mut output.slice_mut(..),
                1.0,
            )?;
            let mut biased = gpu.uninit(rows * width)?;
            let attention_out = &block.attention_out;
            self.bias(
                &output,
                attention_out,
                rows,
                0,
                width,
                false,
                &mut biased.slice_mut(..),
            )?;
            hidden = gpu.gate_residual(
                &hidden.slice(..),
                &biased.slice(..),
                &modulation,
                rows,
                width,
                rows,
            )?;
        }

        // The final layer's modulation is a shift then a scale, which is the
        // layout the norm kernel reads.
        let image = hidden.slice(text_rows * width..);
        let mut normed = gpu.uninit(image_rows * width)?;
        self.launch_norm(&image, &triple(2), &mut normed, image_rows, width, eps)?;
        let normed = gpu.narrowed(&normed.slice(..), image_rows * width, true)?;
        let output = self.apply(&gpu, &self.final_out, &normed.all(), image_rows, false)?;
        Ok(Matrix::from_vec(
            image_rows,
            self.final_out.units,
            gpu.download(&output)?,
        ))
    }

    /// A host matrix through an input layer, FP32 out.
    fn input(
        &self,
        gpu: &Gpu<'_>,
        linear: &Linear,
        tokens: &Matrix,
    ) -> Result<CudaSlice<f32>, NetworkError> {
        let tokens_on_device = gpu.upload(&tokens.data)?;
        let narrow = gpu.narrowed(&tokens_on_device.slice(..), tokens.data.len(), true)?;
        self.apply(gpu, linear, &narrow.all(), tokens.rows, false)
    }

    /// `linear` over `rows` narrow rows, its bias added and the GELU applied
    /// when asked, FP32 out.
    fn apply(
        &self,
        gpu: &Gpu<'_>,
        linear: &Linear,
        x: &CudaView<'_, u8>,
        rows: usize,
        gelu: bool,
    ) -> Result<CudaSlice<f32>, NetworkError> {
        let mut raw = gpu.uninit(rows * linear.units)?;
        gpu.project(
            &linear.weight,
            linear.units,
            linear.inner,
            x,
            true,
            &mut raw.slice_mut(..),
            rows,
            0.0,
        )?;
        let mut out = gpu.uninit(rows * linear.units)?;
        self.bias(
            &raw,
            linear,
            rows,
            0,
            linear.units,
            gelu,
            &mut out.slice_mut(..),
        )?;
        Ok(out)
    }

    fn project(
        &self,
        gpu: &Gpu<'_>,
        linear: &Linear,
        x: &Act,
        rows: usize,
        out: &mut CudaViewMut<'_, f32>,
        beta: f32,
    ) -> Result<(), NetworkError> {
        gpu.project(
            &linear.weight,
            linear.units,
            linear.inner,
            &x.all(),
            true,
            out,
            rows,
            beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn bias(
        &self,
        source: &CudaSlice<f32>,
        linear: &Linear,
        rows: usize,
        offset: usize,
        cols: usize,
        gelu: bool,
        out: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), NetworkError> {
        // A layer without a bias still goes through the kernel for the GELU
        // and the column slice; the source stands in as the unread bias.
        let bias = linear.bias.as_ref().unwrap_or(source);
        unsafe {
            self.context
                .stream
                .launch_builder(&self.bias_gelu)
                .arg(out)
                .arg(source)
                .arg(bias)
                .arg(&(rows as i32))
                .arg(&(cols as i32))
                .arg(&(linear.units as i32))
                .arg(&(offset as i32))
                .arg(&i32::from(gelu))
                .arg(&i32::from(linear.bias.is_some()))
                .launch(cfg(rows * cols))
                .map_err(cuda_err("MMDiT bias kernel"))?;
        }
        Ok(())
    }

    /// The modulated layer norm, narrowed for the GEMM that reads it.
    fn norm(
        &self,
        gpu: &Gpu<'_>,
        hidden: &CudaSlice<f32>,
        modulation: &CudaView<'_, f32>,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<Act, NetworkError> {
        let mut normed = gpu.uninit(rows * width)?;
        self.launch_norm(&hidden.slice(..), modulation, &mut normed, rows, width, eps)?;
        gpu.narrowed(&normed.slice(..), rows * width, true)
    }

    fn launch_norm(
        &self,
        hidden: &CudaView<'_, f32>,
        modulation: &CudaView<'_, f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.layer_norm_modulate)
                .arg(out)
                .arg(hidden)
                .arg(modulation)
                .arg(&(rows as i32))
                .arg(&(width as i32))
                .arg(&eps)
                .launch(LaunchConfig {
                    grid_dim: (rows.clamp(1, 1024) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(cuda_err("MMDiT layer norm kernel"))?;
        }
        Ok(())
    }

    /// The head norm and the rotation on `count` rows of `qkv` from row
    /// `first` on, at column `offset`.
    #[allow(clippy::too_many_arguments)]
    fn head_norm_rope(
        &self,
        qkv: &mut CudaSlice<f32>,
        scale: &CudaSlice<f32>,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        first: usize,
        count: usize,
        heads: usize,
        head_dim: usize,
        offset: usize,
    ) -> Result<(), NetworkError> {
        let stride = 3 * heads * head_dim;
        let mut rows = qkv.slice_mut(first * stride..);
        unsafe {
            self.context
                .stream
                .launch_builder(&self.head_norm_rope)
                .arg(&mut rows)
                .arg(scale)
                .arg(cos)
                .arg(sin)
                .arg(&(count as i32))
                .arg(&(heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(stride as i32))
                .arg(&(offset as i32))
                .arg(&(first as i32))
                .launch(cfg(count * heads))
                .map_err(cuda_err("MMDiT head norm kernel"))?;
        }
        Ok(())
    }

    /// Unmasked attention over the whole `[rows, qkv_width]` sequence,
    /// `[rows, width]` narrow out.
    fn attention(
        &self,
        gpu: &Gpu<'_>,
        qkv: &CudaSlice<f32>,
        rows: usize,
        heads: usize,
        width: usize,
        qkv_width: usize,
    ) -> Result<Act, NetworkError> {
        let flash = self.context.flash.as_ref().expect("checked at upload");
        let qkv = gpu.narrowed(&qkv.slice(..), rows * qkv_width, true)?;
        let mut merged = gpu.act(rows * width, true)?;
        let mut log_sum_exp = gpu.uninit(heads * rows)?;
        let shape = FlashShape {
            rows,
            seq_len: rows,
            kv_len: rows,
            sequences: 1,
            heads,
            group: 1,
            qkv_width,
            kv_width: qkv_width,
            query_width: width,
            key_base: width,
            value_base: 2 * width,
            causal: false,
        };
        let scale = ((width / heads) as f32).sqrt().recip();
        gpu.flash_attention(flash, &qkv, &qkv, &mut merged, &mut log_sum_exp, shape, scale)?;
        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_training::cuda_doctor;
    use crate::mmdit::DitConfig;
    use crate::mmdit::tests::{checkpoint, conditioning, tiny};
    use crate::transformer::Precision;

    fn cuda_or_skip() -> Option<Arc<GpuContext>> {
        match cuda_doctor(0, 8192) {
            Ok(_) => Some(GpuContext::new(0).expect("the CUDA doctor passed")),
            Err(NetworkError::Cuda(message))
                if message.contains("NO_DEVICE") || message.contains("no CUDA-capable device") =>
            {
                crate::cuda_training::skip_without_cuda();
                None
            }
            Err(error) => panic!("CUDA is present but the CUDA doctor failed: {error}"),
        }
    }

    /// The tiny model with heads the fused attention takes.
    fn config() -> DitConfig {
        DitConfig {
            d_model: 128,
            num_heads: 2,
            axes_dim: vec![32, 32],
            ..tiny()
        }
    }

    #[test]
    fn a_device_dit_matches_the_host_or_skips_without_device() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let config = config();
        let path = std::env::temp_dir().join("rusting_brain_cuda_mmdit.safetensors");
        checkpoint(&config, &path);
        let (height, width) = (3, 5);
        let patches = Matrix::from_vec(
            height * width,
            config.in_channels,
            (0..height * width * config.in_channels)
                .map(|index| (index as f32 * 0.23).cos())
                .collect(),
        );
        let conditioning = conditioning(&config, height, width);
        for precision in [Precision::F32, Precision::Q8] {
            let dit = Dit::load_at(&path, "", config.clone(), precision).unwrap();
            let device = DeviceDit::upload(&context, &dit).unwrap();
            let expected = dit.forward_with(&patches, 0.4, &conditioning).unwrap();
            let actual = device
                .forward_with(&dit, &patches, 0.4, &conditioning)
                .unwrap();

            assert_eq!((actual.rows, actual.cols), (expected.rows, expected.cols));
            // BF16 weights alone move this random model's output by about 2%
            // (0.8% at Q8, whose weights both sides read exactly), so the
            // whole-model check only catches a misplaced block. The kernels'
            // arithmetic is checked tightly by `the_mmdit_kernels_match_the_host`.
            let error: f32 = actual
                .data
                .iter()
                .zip(&expected.data)
                .map(|(actual, expected)| (actual - expected).powi(2))
                .sum::<f32>()
                .sqrt();
            let norm: f32 = expected.data.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                error < 3e-2 * norm,
                "{precision:?}: error {error} of {norm}"
            );
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn the_mmdit_kernels_match_the_host() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let config = config();
        let path = std::env::temp_dir().join("rusting_brain_cuda_mmdit_kernels.safetensors");
        checkpoint(&config, &path);
        let dit = Dit::load_at(&path, "", config.clone(), Precision::F32).unwrap();
        std::fs::remove_file(path).ok();
        let device = DeviceDit::upload(&context, &dit).unwrap();
        let gpu = Gpu { context: &context };
        let (width, heads) = (config.d_model, config.num_heads);
        let (text_rows, rows) = (2, 7);
        let values = |count: usize, step: f32| -> Vec<f32> {
            (0..count)
                .map(|index| (index as f32 * step).sin() * 1.7)
                .collect()
        };
        let close = |name: &str, actual: &[f32], expected: &[f32]| {
            let worst = actual
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(worst < 1e-4, "{name}: worst difference {worst}");
        };

        // The head norm and rotation on the query columns of a qkv buffer,
        // text rows and image rows launched apart as the forward does.
        let qkv = values(rows * 3 * width, 0.37);
        let positions = Positions::new(&config, text_rows, 1, rows - text_rows);
        let mut on_device = gpu.upload(&qkv).unwrap();
        let (cos, sin) = (
            gpu.upload(&positions.cos).unwrap(),
            gpu.upload(&positions.sin).unwrap(),
        );
        let block = &device.single[0];
        let head_dim = width / heads;
        for (first, count) in [(0, text_rows), (text_rows, rows - text_rows)] {
            device
                .head_norm_rope(
                    &mut on_device,
                    &block.key_norm,
                    &cos,
                    &sin,
                    first,
                    count,
                    heads,
                    head_dim,
                    width,
                )
                .unwrap();
        }
        let actual = gpu.download(&on_device).unwrap();
        let mut keys = Matrix::new(rows, width);
        for row in 0..rows {
            keys.row_mut(row)
                .copy_from_slice(&qkv[row * 3 * width + width..row * 3 * width + 2 * width]);
        }
        dit.single[0].key_norm.apply(&mut keys, heads);
        positions.apply(&mut keys, heads);
        let actual: Vec<f32> = (0..rows)
            .flat_map(|row| actual[row * 3 * width + width..row * 3 * width + 2 * width].to_vec())
            .collect();
        close("head norm and rotation", &actual, &keys.data);

        // The modulated layer norm, shift then scale.
        let hidden = Matrix::from_vec(rows, width, values(rows * width, 0.11));
        let modulation = values(2 * width, 0.53);
        let mut normed = gpu.uninit(rows * width).unwrap();
        device
            .launch_norm(
                &gpu.upload(&hidden.data).unwrap().slice(..),
                &gpu.upload(&modulation).unwrap().slice(..),
                &mut normed,
                rows,
                width,
                config.eps,
            )
            .unwrap();
        let expected = crate::mmdit::modulate(
            &crate::mmdit::layer_norm(&hidden, config.eps),
            &modulation[..width],
            &modulation[width..],
        );
        close(
            "layer norm",
            &gpu.download(&normed).unwrap(),
            &expected.data,
        );

        // A column slice of a biased layer's output, with the GELU.
        let linear = &block.fused_in;
        let source = values(rows * linear.units, 0.07);
        let bias = gpu.download(linear.bias.as_ref().unwrap()).unwrap();
        let (offset, cols) = (3 * width, linear.units - 3 * width);
        let mut out = gpu.uninit(rows * cols).unwrap();
        device
            .bias(
                &gpu.upload(&source).unwrap(),
                linear,
                rows,
                offset,
                cols,
                true,
                &mut out.slice_mut(..),
            )
            .unwrap();
        let expected: Vec<f32> = (0..rows * cols)
            .map(|index| {
                let (row, col) = (index / cols, index % cols);
                crate::ffn::gelu(source[row * linear.units + offset + col] + bias[offset + col])
            })
            .collect();
        close("bias and GELU", &gpu.download(&out).unwrap(), &expected);
    }

    /// A small FLUX-shaped denoiser over a 512x512 image (32x32 patches) and
    /// 256 text tokens, host against device. Run by hand: `cargo test
    /// --release --features cuda --lib time_a_dit -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn time_a_dit_over_a_512_image() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let config = DitConfig {
            in_channels: 64,
            out_channels: 64,
            d_model: 768,
            num_heads: 6,
            double_blocks: 2,
            single_blocks: 4,
            mlp_ratio: 4.0,
            text_dim: 512,
            pooled_dim: 256,
            axes_dim: vec![64, 64],
            ..tiny()
        };
        let path = std::env::temp_dir().join("rusting_brain_cuda_mmdit_time.safetensors");
        checkpoint(&config, &path);
        let models = [Precision::F32, Precision::Q8]
            .map(|precision| Dit::load_at(&path, "", config.clone(), precision).unwrap());
        std::fs::remove_file(path).ok();
        let (height, width) = (32, 32);
        let patches = Matrix::from_vec(
            height * width,
            config.in_channels,
            (0..height * width * config.in_channels)
                .map(|index| (index as f32 * 0.23).cos())
                .collect(),
        );
        let conditioning = Conditioning {
            text: Matrix::from_vec(
                256,
                config.text_dim,
                (0..256 * config.text_dim)
                    .map(|index| (index as f32 * 0.37).sin())
                    .collect(),
            ),
            ..conditioning(&config, height, width)
        };

        // BF16 weights against the byte path, each against its own host.
        for (dit, precision) in models.iter().zip([Precision::F32, Precision::Q8]) {
            let device = DeviceDit::upload(&context, dit).unwrap();
            let started = std::time::Instant::now();
            let expected = dit.forward_with(&patches, 0.4, &conditioning).unwrap();
            let host = started.elapsed();
            device
                .forward_with(dit, &patches, 0.4, &conditioning)
                .unwrap();
            let mut device_times = Vec::new();
            let mut actual = None;
            for _ in 0..5 {
                let started = std::time::Instant::now();
                actual = Some(
                    device
                        .forward_with(dit, &patches, 0.4, &conditioning)
                        .unwrap(),
                );
                device_times.push(started.elapsed());
            }
            let actual = actual.unwrap();
            let error: f32 = actual
                .data
                .iter()
                .zip(&expected.data)
                .map(|(actual, expected)| (actual - expected).powi(2))
                .sum::<f32>()
                .sqrt();
            let norm: f32 = expected.data.iter().map(|v| v * v).sum::<f32>().sqrt();
            println!(
                "{precision:?}: host {host:?}, device {device_times:?}, relative error {}",
                error / norm
            );
        }
    }
}
