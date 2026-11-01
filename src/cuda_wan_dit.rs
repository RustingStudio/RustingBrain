//! The Wan video transformer on the device, the twin of [`WanDit::forward`].
//!
//! It runs like [`crate::cuda_mmdit`]: the residual stream and every GEMM
//! accumulator are FP32, the weights BF16 or int8 and the operands a GEMM
//! reads BF16. Both attentions are the fused BF16 kernel, unmasked. The
//! timestep embedding and every block's modulation are built on the host,
//! a few small layers over one row per frame, and uploaded once per call.
//!
//! Int8 weights read BF16 activations too, unlike [`crate::cuda_t5`]: FP32
//! ones took the real-weight step from 4.4e-2 to 3.7e-2 against diffusers
//! (the host's int8 alone is 3.3e-2) and made the 704x1280x17 step 3.5
//! times slower, 8.1 s against 2.3 s.
//!
//! NOTE: the cross-attention keys and values depend only on the prompt,
//! but are projected again on every call; cache them per prompt if a profile
//! of a whole sampling run shows them (512 rows against 4400 at 704x1280).

use crate::conv::{Dense, FeatureMap};
use crate::cuda_training::{cfg, cuda_err};
use crate::gpu_model::{Act, FlashShape, Gpu};
use crate::gpu_transformer::GpuContext;
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::wan_dit::{WanBlock, WanDit, rope};
use cudarc::driver::{CudaFunction, CudaSlice, CudaView, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{Ptx, compile_ptx};
use std::sync::{Arc, OnceLock};

const KERNELS: &str = r#"
#define ROW_THREADS 256
// The RMS norm over the `cols` values of each row at `x + r * stride + off`,
// times `weight`, in place, then with `rope` the rotation of each adjacent
// pair by the row's angle for the pair's index within its head. Tables are
// `[rows, head_dim / 2]`. One block per row.
extern "C" __global__ void rms_rope(float*x,const float*weight,const float*cs,const float*sn,int rows,int cols,int stride,int off,int head_dim,int rope,float eps){
  __shared__ float red[ROW_THREADS];int tid=threadIdx.x,half=head_dim/2;
  for(int r=blockIdx.x;r<rows;r+=gridDim.x){
    float2*v=(float2*)(x+(size_t)r*stride+off);const float2*w=(const float2*)weight;
    float a=0.f;for(int p=tid;p<cols/2;p+=ROW_THREADS){float2 e=v[p];a+=e.x*e.x+e.y*e.y;}
    red[tid]=a;__syncthreads();
    for(int s=ROW_THREADS/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    float inv=rsqrtf(red[0]/cols+eps);__syncthreads();
    const float*c=cs+(size_t)r*half;const float*n=sn+(size_t)r*half;
    for(int p=tid;p<cols/2;p+=ROW_THREADS){
      float2 e=v[p],g=w[p];float a=e.x*inv*g.x,b=e.y*inv*g.y;
      if(rope){int q=p%half;v[p]=make_float2(a*c[q]-b*n[q],a*n[q]+b*c[q]);}
      else v[p]=make_float2(a,b);
    }
  }
}
"#;

fn ptx() -> Result<&'static Ptx, NetworkError> {
    static PTX: OnceLock<Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| compile_ptx(KERNELS).map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| NetworkError::Cuda(format!("Wan kernel compilation failed: {error}")))
}

/// A layer's weight on the device and its bias.
struct Linear {
    weight: Act,
    bias: CudaSlice<f32>,
    units: usize,
    inner: usize,
}

struct DeviceBlock {
    /// Query, key and value stacked, `[3 * dim, dim]`.
    qkv: Linear,
    query_norm: CudaSlice<f32>,
    key_norm: CudaSlice<f32>,
    output: Linear,
    cross_query: Linear,
    /// The cross-attention key and value stacked, `[2 * dim, dim]`.
    cross_kv: Linear,
    cross_query_norm: CudaSlice<f32>,
    cross_key_norm: CudaSlice<f32>,
    cross_output: Linear,
    /// `norm2`'s bias then its weight less one, the shift and scale layout
    /// `layer_norm_modulate` reads.
    norm2: CudaSlice<f32>,
    ffn_in: Linear,
    ffn_out: Linear,
}

/// A Wan transformer's blocks resident on the device.
pub struct DeviceWanDit {
    context: Arc<GpuContext>,
    layer_norm_modulate: CudaFunction,
    bias_gelu: CudaFunction,
    rms_rope: CudaFunction,
    patch: Linear,
    text_in: Linear,
    text_out: Linear,
    proj_out: Linear,
    blocks: Vec<DeviceBlock>,
    /// Every block's `scale_shift_table`, `[layers, 6 * dim]`, added to the
    /// timestep projection on the host.
    tables: Vec<f32>,
}

impl std::fmt::Debug for DeviceWanDit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceWanDit").finish_non_exhaustive()
    }
}

fn invalid(what: String) -> NetworkError {
    NetworkError::InvalidConfig(format!("Wan transformer upload: {what}"))
}

fn check_dense(name: &str, layer: &Dense, units: usize, inner: usize) -> Result<(), NetworkError> {
    let bias = layer.bias.as_ref().map(Vec::len);
    if layer.out_dim() != units || layer.in_dim() != inner || bias != Some(units) {
        return Err(invalid(format!(
            "{name} is [{}, {}] with a bias of {bias:?}, not [{units}, {inner}] with {units}",
            layer.out_dim(),
            layer.in_dim()
        )));
    }
    Ok(())
}

fn check_len(name: &str, values: &[f32], len: usize) -> Result<(), NetworkError> {
    if values.len() != len {
        return Err(invalid(format!("{name} holds {}, not {len}", values.len())));
    }
    Ok(())
}

/// Every shape the kernels index by, checked before anything is uploaded.
fn validate(model: &WanDit) -> Result<(), NetworkError> {
    let config = &model.config;
    config.validate()?;
    let (dim, ffn) = (config.dim(), config.ffn_dim);
    let [pt, ph, pw] = config.patch;
    check_dense(
        "patch_embedding",
        &model.patch,
        dim,
        config.in_channels * pt * ph * pw,
    )?;
    check_dense(
        "text_embedder.linear_1",
        &model.text_in,
        dim,
        config.text_dim,
    )?;
    check_dense("text_embedder.linear_2", &model.text_out, dim, dim)?;
    check_dense(
        "proj_out",
        &model.proj_out,
        config.out_channels * pt * ph * pw,
        dim,
    )?;
    check_len("scale_shift_table", &model.out_table, 2 * dim)?;
    if model.blocks.len() != config.layers {
        return Err(invalid(format!(
            "{} blocks, not {}",
            model.blocks.len(),
            config.layers
        )));
    }
    for (index, block) in model.blocks.iter().enumerate() {
        let name = |part: &str| format!("blocks.{index}.{part}");
        let square = [
            ("attn1.to_q", &block.query),
            ("attn1.to_k", &block.key),
            ("attn1.to_v", &block.value),
            ("attn1.to_out.0", &block.output),
            ("attn2.to_q", &block.cross_query),
            ("attn2.to_k", &block.cross_key),
            ("attn2.to_v", &block.cross_value),
            ("attn2.to_out.0", &block.cross_output),
        ];
        for (part, layer) in square {
            check_dense(&name(part), layer, dim, dim)?;
        }
        check_dense(&name("ffn.net.0.proj"), &block.ffn_in, ffn, dim)?;
        check_dense(&name("ffn.net.2"), &block.ffn_out, dim, ffn)?;
        let vectors = [
            ("attn1.norm_q", &block.query_norm),
            ("attn1.norm_k", &block.key_norm),
            ("attn2.norm_q", &block.cross_query_norm),
            ("attn2.norm_k", &block.cross_key_norm),
            ("norm2.weight", &block.norm_weight),
            ("norm2.bias", &block.norm_bias),
        ];
        for (part, values) in vectors {
            check_len(&name(part), values, dim)?;
        }
        check_len(&name("scale_shift_table"), &block.table, 6 * dim)?;
    }
    Ok(())
}

impl DeviceWanDit {
    /// Uploads `model`'s blocks and embedders as BF16, or as int8 where it
    /// holds them at one byte, after checking every shape against its
    /// config. The host model stays the source of the timestep embedding,
    /// so it is borrowed again by [`DeviceWanDit::forward`].
    pub fn upload(context: &Arc<GpuContext>, model: &WanDit) -> Result<Self, NetworkError> {
        validate(model)?;
        if context.half_accumulate {
            return Err(NetworkError::InvalidConfig(
                "the Wan transformer needs a BF16 context, not an FP16 one".into(),
            ));
        }
        let head_dim = model.config.head_dim;
        if context.flash.is_none() || !crate::cuda_flash::eligible(true, head_dim) {
            return Err(NetworkError::InvalidConfig(format!(
                "the device Wan transformer needs the fused attention kernel, which takes \
                 heads of 64 or 128 on Ampere or later; this one has {head_dim}"
            )));
        }
        let gpu = Gpu { context };
        let load = |ptx: &Ptx, names: &[&str]| {
            let module = context
                .stream
                .context()
                .load_module(ptx.clone())
                .map_err(cuda_err("Wan module loading"))?;
            names
                .iter()
                .map(|name| {
                    module
                        .load_function(name)
                        .map_err(cuda_err("Wan kernel lookup"))
                })
                .collect::<Result<Vec<_>, _>>()
        };
        let mut mmdit = load(
            crate::cuda_mmdit::ptx()?,
            &["layer_norm_modulate", "bias_gelu"],
        )?;
        let rms_rope = load(ptx()?, &["rms_rope"])?.remove(0);
        let bias_gelu = mmdit.remove(1);
        let layer_norm_modulate = mmdit.remove(0);

        // Several layers stacked into one GEMM, each with its bias.
        let stacked = |layers: &[&Dense]| {
            let (mut weights, mut biases) = (Vec::new(), Vec::new());
            let (mut q, mut scales) = (Vec::new(), Vec::new());
            let int8 = layers.iter().all(|layer| layer.int8_parts().is_some());
            for layer in layers {
                match layer.int8_parts().filter(|_| int8) {
                    Some((bytes, row_scales)) => {
                        q.extend_from_slice(bytes);
                        scales.extend_from_slice(row_scales);
                    }
                    None => weights.extend_from_slice(&layer.weight_f32().data),
                }
                biases.extend_from_slice(layer.bias.as_deref().expect("checked by validate"));
            }
            let (units, inner) = (biases.len(), layers[0].in_dim());
            // int8 weights widen on the device: a quarter of the PCIe bytes
            // and no single-threaded host dequantize of 4x the data.
            let wide = match int8 {
                true => gpu.dequantized(&q, &scales, units, inner)?,
                false => gpu.upload(&weights)?,
            };
            Ok::<_, NetworkError>(Linear {
                weight: match layers[0].is_quantized() {
                    true => gpu.quantized(&wide.slice(..), units, inner)?,
                    false => gpu.narrowed(&wide.slice(..), units * inner, true)?,
                },
                bias: gpu.upload(&biases)?,
                units,
                inner,
            })
        };
        let dense = |layer: &Dense| stacked(&[layer]);
        let blocks = model
            .blocks
            .iter()
            .map(|block: &WanBlock| {
                let norm2: Vec<f32> = block
                    .norm_bias
                    .iter()
                    .copied()
                    .chain(block.norm_weight.iter().map(|weight| weight - 1.0))
                    .collect();
                Ok(DeviceBlock {
                    qkv: stacked(&[&block.query, &block.key, &block.value])?,
                    query_norm: gpu.upload(&block.query_norm)?,
                    key_norm: gpu.upload(&block.key_norm)?,
                    output: dense(&block.output)?,
                    cross_query: dense(&block.cross_query)?,
                    cross_kv: stacked(&[&block.cross_key, &block.cross_value])?,
                    cross_query_norm: gpu.upload(&block.cross_query_norm)?,
                    cross_key_norm: gpu.upload(&block.cross_key_norm)?,
                    cross_output: dense(&block.cross_output)?,
                    norm2: gpu.upload(&norm2)?,
                    ffn_in: dense(&block.ffn_in)?,
                    ffn_out: dense(&block.ffn_out)?,
                })
            })
            .collect::<Result<_, NetworkError>>()?;
        Ok(Self {
            layer_norm_modulate,
            bias_gelu,
            rms_rope,
            patch: dense(&model.patch)?,
            text_in: dense(&model.text_in)?,
            text_out: dense(&model.text_out)?,
            proj_out: dense(&model.proj_out)?,
            blocks,
            tables: model
                .blocks
                .iter()
                .flat_map(|block| block.table.iter().copied())
                .collect(),
            context: context.clone(),
        })
    }

    /// [`WanDit::forward`] on the device. `model` must be the model this was
    /// uploaded from.
    pub fn forward(
        &self,
        model: &WanDit,
        latent: &[FeatureMap],
        timesteps: &[f32],
        context: &Matrix,
    ) -> Result<Vec<FeatureMap>, NetworkError> {
        let config = &model.config;
        if self.blocks.len() != config.layers || self.patch.units != config.dim() {
            return Err(invalid(
                "forward with a model other than the one uploaded".into(),
            ));
        }
        let grid = model.grid(latent, timesteps, context)?;
        let gpu = Gpu {
            context: &self.context,
        };
        let (dim, heads, eps) = (config.dim(), config.num_heads, config.eps);
        let frames = grid[0];
        let per_frame = grid[1] * grid[2];
        let rows = frames * per_frame;
        let (temb, projection) = model.time(timesteps, frames)?;

        // Per block, the attention's then the feed-forward's shift, scale and
        // gate, each `[frames, 3 * dim]`, the layout `gate_residual` reads;
        // then the output's shift and scale, `[frames, 2 * dim]`.
        let mut modulation = Vec::with_capacity((self.blocks.len() * 6 + 2) * frames * dim);
        for table in self.tables.chunks(6 * dim) {
            for half in [0, 3 * dim] {
                for frame in projection.data.chunks(6 * dim) {
                    let range = half..half + 3 * dim;
                    modulation.extend(
                        frame[range.clone()]
                            .iter()
                            .zip(&table[range])
                            .map(|(a, b)| a + b),
                    );
                }
            }
        }
        let output_at = modulation.len();
        for frame in temb.data.chunks(dim) {
            for chunk in model.out_table.chunks(dim) {
                modulation.extend(chunk.iter().zip(frame).map(|(a, b)| a + b));
            }
        }
        let modulation = gpu.upload(&modulation)?;
        let positions = rope(config, grid);
        let (cos, sin) = (gpu.upload(&positions.cos)?, gpu.upload(&positions.sin)?);

        let patches = model.patches(latent, grid);
        let mut hidden = self.input(&gpu, &self.patch, &patches)?;
        let text = self.input(&gpu, &self.text_in, context)?;
        let text = self.gelu(&gpu, &text, context.rows, dim)?;
        let text = self.apply(&gpu, &self.text_out, &text, context.rows)?;
        let text = gpu.narrowed(&text.slice(..), context.rows * dim, true)?;
        let scale = (config.head_dim as f32).sqrt().recip();
        let flash = self.context.flash.as_ref().expect("checked at upload");

        for (index, block) in self.blocks.iter().enumerate() {
            let at = index * 6 * frames * dim;
            let attention = modulation.slice(at..at + 3 * frames * dim);
            let feed_forward = modulation.slice(at + 3 * frames * dim..at + 6 * frames * dim);

            let normed =
                self.frame_norm(&gpu, &hidden, &attention, frames, per_frame, 3, dim, eps)?;
            let mut qkv = self.apply(&gpu, &block.qkv, &normed, rows)?;
            for (weight, offset) in [(&block.query_norm, 0), (&block.key_norm, dim)] {
                self.rms_rope(
                    &mut qkv,
                    weight,
                    Some((&cos, &sin)),
                    rows,
                    3 * dim,
                    offset,
                    config.head_dim,
                    eps,
                )?;
            }
            let qkv = gpu.narrowed(&qkv.slice(..), rows * 3 * dim, true)?;
            let mut merged = gpu.act(rows * dim, true)?;
            let mut log_sum_exp = gpu.uninit(heads * rows)?;
            let shape = FlashShape {
                rows,
                seq_len: rows,
                kv_len: rows,
                sequences: 1,
                heads,
                group: 1,
                qkv_width: 3 * dim,
                kv_width: 3 * dim,
                query_width: dim,
                key_base: dim,
                value_base: 2 * dim,
                causal: false,
            };
            gpu.flash_attention(
                flash,
                &qkv,
                &qkv,
                &mut merged,
                &mut log_sum_exp,
                shape,
                scale,
            )?;
            let out = self.apply(&gpu, &block.output, &merged, rows)?;
            hidden = gpu.gate_residual(
                &hidden.slice(..),
                &out.slice(..),
                &attention,
                rows,
                dim,
                per_frame,
            )?;

            let normed = self.norm(
                &gpu,
                &hidden.slice(..),
                &block.norm2.slice(..),
                rows,
                dim,
                eps,
            )?;
            let mut query = self.apply(&gpu, &block.cross_query, &normed, rows)?;
            self.rms_rope(
                &mut query,
                &block.cross_query_norm,
                None,
                rows,
                dim,
                0,
                config.head_dim,
                eps,
            )?;
            let mut kv = self.apply(&gpu, &block.cross_kv, &text, context.rows)?;
            self.rms_rope(
                &mut kv,
                &block.cross_key_norm,
                None,
                context.rows,
                2 * dim,
                0,
                config.head_dim,
                eps,
            )?;
            let query = gpu.narrowed(&query.slice(..), rows * dim, true)?;
            let kv = gpu.narrowed(&kv.slice(..), context.rows * 2 * dim, true)?;
            let shape = FlashShape {
                kv_len: context.rows,
                qkv_width: dim,
                kv_width: 2 * dim,
                key_base: 0,
                value_base: dim,
                ..shape
            };
            gpu.flash_attention(
                flash,
                &query,
                &kv,
                &mut merged,
                &mut log_sum_exp,
                shape,
                scale,
            )?;
            let out = self.apply(&gpu, &block.cross_output, &merged, rows)?;
            gpu.add(&mut hidden, &out, 0, rows * dim, true)?;

            let normed =
                self.frame_norm(&gpu, &hidden, &feed_forward, frames, per_frame, 3, dim, eps)?;
            let pre = self.apply(&gpu, &block.ffn_in, &normed, rows)?;
            let activated = self.gelu(&gpu, &pre, rows, block.ffn_in.units)?;
            let out = self.apply(&gpu, &block.ffn_out, &activated, rows)?;
            hidden = gpu.gate_residual(
                &hidden.slice(..),
                &out.slice(..),
                &feed_forward,
                rows,
                dim,
                per_frame,
            )?;
        }

        let output = modulation.slice(output_at..);
        let normed = self.frame_norm(&gpu, &hidden, &output, frames, per_frame, 2, dim, eps)?;
        let out = self.apply(&gpu, &self.proj_out, &normed, rows)?;
        let out = Matrix::from_vec(rows, self.proj_out.units, gpu.download(&out)?);
        Ok(model.unpatchify(&out, grid, config.out_channels))
    }

    /// `tokens` uploaded, narrowed and through `linear`.
    fn input(
        &self,
        gpu: &Gpu<'_>,
        linear: &Linear,
        tokens: &Matrix,
    ) -> Result<CudaSlice<f32>, NetworkError> {
        let wide = gpu.upload(&tokens.data)?;
        let narrow = gpu.narrowed(&wide.slice(..), tokens.data.len(), true)?;
        self.apply(gpu, linear, &narrow, tokens.rows)
    }

    /// `linear` over `rows` narrow rows, its bias added, FP32 out.
    fn apply(
        &self,
        gpu: &Gpu<'_>,
        linear: &Linear,
        x: &Act,
        rows: usize,
    ) -> Result<CudaSlice<f32>, NetworkError> {
        let units = linear.units;
        let mut raw = gpu.uninit(rows * units)?;
        gpu.project(
            &linear.weight,
            units,
            linear.inner,
            &x.all(),
            true,
            &mut raw.slice_mut(..),
            rows,
            0.0,
        )?;
        let mut out = gpu.uninit(rows * units)?;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.bias_gelu)
                .arg(&mut out)
                .arg(&raw)
                .arg(&linear.bias)
                .arg(&(rows as i32))
                .arg(&(units as i32))
                .arg(&(units as i32))
                .arg(&0i32)
                .arg(&0i32)
                .arg(&1i32)
                .arg(&0i32)
                .launch(cfg(rows * units))
                .map_err(cuda_err("Wan bias kernel"))?;
        }
        Ok(out)
    }

    /// The tanh GELU of `[rows, cols]`, narrow.
    fn gelu(
        &self,
        gpu: &Gpu<'_>,
        source: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<Act, NetworkError> {
        let mut out = gpu.act(rows * cols, true)?;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.bias_gelu)
                .arg(out.destination())
                .arg(source)
                .arg(source)
                .arg(&(rows as i32))
                .arg(&(cols as i32))
                .arg(&(cols as i32))
                .arg(&0i32)
                .arg(&1i32)
                .arg(&0i32)
                .arg(&1i32)
                .launch(cfg(rows * cols))
                .map_err(cuda_err("Wan GELU kernel"))?;
        }
        Ok(out)
    }

    /// The modulated layer norm of each frame's `per_frame` rows, with that
    /// frame's shift and scale at `chunks * dim` apart in `modulation`,
    /// narrowed for the GEMM that reads it.
    #[allow(clippy::too_many_arguments)]
    fn frame_norm(
        &self,
        gpu: &Gpu<'_>,
        hidden: &CudaSlice<f32>,
        modulation: &CudaView<'_, f32>,
        frames: usize,
        per_frame: usize,
        chunks: usize,
        dim: usize,
        eps: f32,
    ) -> Result<Act, NetworkError> {
        let mut normed = gpu.uninit(frames * per_frame * dim)?;
        for frame in 0..frames {
            let rows = frame * per_frame * dim..(frame + 1) * per_frame * dim;
            let mut out = normed.slice_mut(rows.clone());
            let shift_scale = modulation.slice(frame * chunks * dim..);
            self.launch_norm(
                &mut out,
                &hidden.slice(rows),
                &shift_scale,
                per_frame,
                dim,
                eps,
            )?;
        }
        gpu.narrowed(&normed.slice(..), frames * per_frame * dim, true)
    }

    /// One layer norm over every row with one shift and scale, narrowed.
    fn norm(
        &self,
        gpu: &Gpu<'_>,
        hidden: &CudaView<'_, f32>,
        shift_scale: &CudaView<'_, f32>,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<Act, NetworkError> {
        let mut normed = gpu.uninit(rows * dim)?;
        self.launch_norm(
            &mut normed.slice_mut(..),
            hidden,
            shift_scale,
            rows,
            dim,
            eps,
        )?;
        gpu.narrowed(&normed.slice(..), rows * dim, true)
    }

    fn launch_norm(
        &self,
        out: &mut cudarc::driver::CudaViewMut<'_, f32>,
        hidden: &CudaView<'_, f32>,
        shift_scale: &CudaView<'_, f32>,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.layer_norm_modulate)
                .arg(out)
                .arg(hidden)
                .arg(shift_scale)
                .arg(&(rows as i32))
                .arg(&(dim as i32))
                .arg(&eps)
                .launch(row_blocks(rows))
                .map_err(cuda_err("Wan layer norm kernel"))?;
        }
        Ok(())
    }

    /// The RMS norm of the `cols` values at `offset` in each of `rows` rows
    /// `stride` apart, then the rotation if `tables` are given.
    #[allow(clippy::too_many_arguments)]
    fn rms_rope(
        &self,
        x: &mut CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        tables: Option<(&CudaSlice<f32>, &CudaSlice<f32>)>,
        rows: usize,
        stride: usize,
        offset: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<(), NetworkError> {
        let (cos, sin) = tables.unwrap_or((weight, weight));
        let cols = weight.len();
        unsafe {
            self.context
                .stream
                .launch_builder(&self.rms_rope)
                .arg(x)
                .arg(weight)
                .arg(cos)
                .arg(sin)
                .arg(&(rows as i32))
                .arg(&(cols as i32))
                .arg(&(stride as i32))
                .arg(&(offset as i32))
                .arg(&(head_dim as i32))
                .arg(&i32::from(tables.is_some()))
                .arg(&eps)
                .launch(row_blocks(rows))
                .map_err(cuda_err("Wan RMS norm kernel"))?;
        }
        Ok(())
    }
}

fn row_blocks(rows: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows.clamp(1, 1024) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_training::cuda_doctor;
    use crate::transformer::Precision;
    use crate::wan_dit::tests::{checkpoint, context, latent, tiny};

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

    #[test]
    fn a_device_wan_transformer_matches_the_host_or_skips_without_device() {
        let Some(gpu) = cuda_or_skip() else { return };
        let config = tiny();
        let path = checkpoint(&config, "rusting_brain_cuda_wan_dit.safetensors");
        // Three frames of two timesteps, so the per-frame modulation and the
        // gate's frame index are both exercised.
        let (latent, context) = (latent(&config, 3, 6, 8), context(&config, 7));
        let timesteps = [0.0, 700.0, 700.0];
        let flat = |frames: &[FeatureMap]| {
            frames
                .iter()
                .flat_map(|f| f.data.clone())
                .collect::<Vec<_>>()
        };
        for precision in [Precision::F32, Precision::Q8] {
            let host = WanDit::load_at(&path, config.clone(), precision).unwrap();
            let want = host.forward(&latent, &timesteps, &context).unwrap();
            let mut device = WanDit::load_at(&path, config.clone(), precision).unwrap();
            device.attach_device(&gpu).unwrap();
            let got = device.forward(&latent, &timesteps, &context).unwrap();
            let error = crate::wan_vae::tests::relative(&flat(&got), &flat(&want));
            eprintln!("device wan transformer {precision:?}: relative L2 {error:.3e}");
            assert!(error < 2e-2, "{precision:?}: {error}");
            // Detach and reattach from the kept host blocks.
            let mut kept = WanDit::load_at(&path, config.clone(), precision).unwrap();
            kept.attach_device_keeping_host(&gpu).unwrap();
            kept.detach_device();
            assert!(!kept.on_device());
            let on_host = kept.forward(&latent, &timesteps, &context).unwrap();
            assert_eq!(
                flat(&on_host),
                flat(&want),
                "{precision:?}: host after detach"
            );
            kept.attach_device_keeping_host(&gpu).unwrap();
            assert!(kept.on_device());
            let again = kept.forward(&latent, &timesteps, &context).unwrap();
            assert_eq!(flat(&again), flat(&got), "{precision:?}: reattached");
        }
    }

    #[test]
    fn malformed_shapes_are_refused_before_upload_or_skip_without_device() {
        let Some(gpu) = cuda_or_skip() else { return };
        let config = tiny();
        let path = checkpoint(&config, "rusting_brain_cuda_wan_dit_bad.safetensors");
        let refused = |model: &WanDit| {
            assert!(matches!(
                DeviceWanDit::upload(&gpu, model),
                Err(NetworkError::InvalidConfig(_))
            ))
        };
        let load = || WanDit::load_at(&path, config.clone(), Precision::F32).unwrap();
        let mut model = load();
        model.blocks[1].key_norm.pop();
        refused(&model);
        let mut model = load();
        model.blocks.pop();
        refused(&model);
        let mut model = load();
        model.blocks[0].ffn_out = Dense::new(Matrix::new(128, 255), Some(vec![0.0; 128])).unwrap();
        refused(&model);
        let mut model = load();
        model.proj_out.bias = None;
        refused(&model);
        let mut model = load();
        model.config.head_dim = 32;
        model.config.num_heads = 4;
        refused(&model);
        // A latent the model cannot take is refused before any launch.
        let mut model = load();
        model.attach_device(&gpu).unwrap();
        let context = context(&config, 3);
        assert!(matches!(
            model.forward(&latent(&config, 2, 5, 6), &[1.0], &context),
            Err(NetworkError::InvalidConfig(_))
        ));
    }
}
