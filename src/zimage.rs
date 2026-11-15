//! Z-Image-Turbo's denoiser (Tongyi-MAI, Apache-2.0): diffusers'
//! `ZImageTransformer2DModel`.
//!
//! One stream of 3840-wide blocks. Two modulated "noise refiner" blocks run
//! over the image tokens and two unmodulated "context refiner" blocks over the
//! caption. Then 30 modulated blocks run over `[image, caption]` together.
//! Each block is RMS-normed attention and a SwiGLU feed-forward, both behind
//! a tanh gate, with 3-axis RoPE on adjacent pairs.
//!
//! The blocks run on CUDA only; the 6 B weights are Q8 on the device. The
//! small parts (embedders, modulations, final layer) run on the host in F32.

// Without CUDA the model loads but cannot run, so most of it is unused.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use crate::conv::Dense;
use crate::diffusion::Denoiser;
use crate::matrix::Matrix;
use crate::mmdit::Conditioning;
use crate::network::NetworkError;
use crate::safetensors::ShardedSafeTensors;
use crate::transformer::Precision;
use rayon::prelude::*;

/// Sequences are padded to a multiple of this many rows, as diffusers does.
const SEQ_MULTIPLE: usize = 32;

/// The parts of `transformer/config.json` the model reads.
#[derive(Clone, Debug, PartialEq)]
pub struct ZImageConfig {
    pub dim: usize,
    pub heads: usize,
    pub layers: usize,
    pub refiners: usize,
    pub in_channels: usize,
    pub patch: usize,
    pub caption_dim: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub axes: Vec<usize>,
    pub t_scale: f32,
}

impl ZImageConfig {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, NetworkError> {
        let path = path.as_ref();
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)
            .map_err(|error| NetworkError::InvalidConfig(format!("{}: {error}", path.display())))?;
        let number = |key: &str| {
            json[key].as_f64().ok_or_else(|| {
                NetworkError::InvalidConfig(format!("{}: no number {key}", path.display()))
            })
        };
        let axes: Vec<usize> = json["axes_dims"]
            .as_array()
            .map(|axes| {
                axes.iter()
                    .filter_map(|a| a.as_u64())
                    .map(|a| a as usize)
                    .collect()
            })
            .unwrap_or_default();
        let config = Self {
            dim: number("dim")? as usize,
            heads: number("n_heads")? as usize,
            layers: number("n_layers")? as usize,
            refiners: number("n_refiner_layers")? as usize,
            in_channels: number("in_channels")? as usize,
            patch: json["all_patch_size"][0].as_u64().unwrap_or(2) as usize,
            caption_dim: number("cap_feat_dim")? as usize,
            eps: number("norm_eps")? as f32,
            rope_theta: number("rope_theta")?,
            axes,
            t_scale: number("t_scale")? as f32,
        };
        if config.axes.iter().sum::<usize>() != config.dim / config.heads {
            return Err(NetworkError::InvalidConfig(format!(
                "{}: the RoPE axes {:?} do not fill a {}-wide head",
                path.display(),
                config.axes,
                config.dim / config.heads
            )));
        }
        Ok(config)
    }

    /// Input and output width of a token: the latent channels of one patch.
    pub fn token_width(&self) -> usize {
        self.in_channels * self.patch * self.patch
    }
}

/// One block's big linears and its norms, held on the host until
/// [`ZImage::attach_device`] uploads them.
pub(crate) struct Block {
    /// `to_q`, `to_k` and `to_v` stacked, `[3 * dim, dim]`.
    pub(crate) qkv: Dense,
    pub(crate) out: Dense,
    /// `w1` then `w3`, `[2 * ffn, dim]`: the gate half first.
    pub(crate) w13: Dense,
    pub(crate) w2: Dense,
    /// `norm_q`, `norm_k`, `attention_norm1`, `attention_norm2`,
    /// `ffn_norm1`, `ffn_norm2`.
    pub(crate) norms: [Vec<f32>; 6],
}

/// Z-Image's transformer.
pub struct ZImage {
    config: ZImageConfig,
    time_in: Dense,
    time_out: Dense,
    caption_norm: Vec<f32>,
    caption_in: Dense,
    image_in: Dense,
    image_pad: Vec<f32>,
    caption_pad: Vec<f32>,
    final_modulation: Dense,
    final_out: Dense,
    /// The `adaLN_modulation` of every modulated block: the noise refiner's,
    /// then the main layers'.
    modulations: Vec<Dense>,
    /// Noise refiner, context refiner, then the main layers. Emptied once
    /// they are on the device.
    blocks: Vec<Block>,
    conditioning: Option<Conditioning>,
    #[cfg(feature = "cuda")]
    device: Option<device::DeviceZImage>,
}

impl std::fmt::Debug for ZImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZImage")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

fn linear(file: &mut ShardedSafeTensors, name: &str) -> Result<Dense, NetworkError> {
    let bias = file
        .tensor(&format!("{name}.bias"))
        .ok()
        .map(|(bias, _)| bias);
    Dense::new(file.matrix(&format!("{name}.weight"))?, bias)
}

/// Several bias-free linears over the same input as one, rows stacked.
fn stacked(
    file: &mut ShardedSafeTensors,
    names: &[String],
    precision: Precision,
) -> Result<Dense, NetworkError> {
    let mut data = Vec::new();
    let (mut rows, mut cols) = (0, 0);
    for name in names {
        let part = file.matrix(&format!("{name}.weight"))?;
        rows += part.rows;
        cols = part.cols;
        data.extend_from_slice(&part.data);
    }
    let mut layer = Dense::new(Matrix::from_vec(rows, cols, data), None)?;
    if precision != Precision::F32 {
        // FP8 is not wired for this model; it takes Q8 like everything else.
        match crate::mmdit::q8_group() {
            0 => layer.quantize(),
            group => layer.quantize_grouped(group),
        }
    }
    Ok(layer)
}

impl ZImage {
    /// Reads the transformer. The block linears are stored at `precision`
    /// (Q8 grouped unless F32); the rest stays F32, about 1 GB.
    pub fn read(
        file: &mut ShardedSafeTensors,
        config: ZImageConfig,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        let vector = |file: &mut ShardedSafeTensors, name: &str| file.tensor(name).map(|(v, _)| v);
        let mut blocks = Vec::new();
        let mut modulations = Vec::new();
        let groups = [
            ("noise_refiner", config.refiners, true),
            ("context_refiner", config.refiners, false),
            ("layers", config.layers, true),
        ];
        for (group, count, modulated) in groups {
            for index in 0..count {
                let at = |part: &str| format!("{group}.{index}.{part}");
                let attention = |part: &str| at(&format!("attention.{part}"));
                blocks.push(Block {
                    qkv: stacked(
                        file,
                        &[attention("to_q"), attention("to_k"), attention("to_v")],
                        precision,
                    )?,
                    out: stacked(file, &[attention("to_out.0")], precision)?,
                    w13: stacked(
                        file,
                        &[at("feed_forward.w1"), at("feed_forward.w3")],
                        precision,
                    )?,
                    w2: stacked(file, &[at("feed_forward.w2")], precision)?,
                    norms: [
                        vector(file, &attention("norm_q.weight"))?,
                        vector(file, &attention("norm_k.weight"))?,
                        vector(file, &at("attention_norm1.weight"))?,
                        vector(file, &at("attention_norm2.weight"))?,
                        vector(file, &at("ffn_norm1.weight"))?,
                        vector(file, &at("ffn_norm2.weight"))?,
                    ],
                });
                if modulated {
                    modulations.push(linear(file, &at("adaLN_modulation.0"))?);
                }
            }
        }
        let key = format!("{}-1", config.patch);
        Ok(Self {
            time_in: linear(file, "t_embedder.mlp.0")?,
            time_out: linear(file, "t_embedder.mlp.2")?,
            caption_norm: vector(file, "cap_embedder.0.weight")?,
            caption_in: linear(file, "cap_embedder.1")?,
            image_in: linear(file, &format!("all_x_embedder.{key}"))?,
            image_pad: vector(file, "x_pad_token")?,
            caption_pad: vector(file, "cap_pad_token")?,
            final_modulation: linear(file, &format!("all_final_layer.{key}.adaLN_modulation.1"))?,
            final_out: linear(file, &format!("all_final_layer.{key}.linear"))?,
            modulations,
            blocks,
            config,
            conditioning: None,
            #[cfg(feature = "cuda")]
            device: None,
        })
    }

    pub fn config(&self) -> &ZImageConfig {
        &self.config
    }

    pub fn set_conditioning(&mut self, conditioning: Conditioning) {
        self.conditioning = Some(conditioning);
        #[cfg(feature = "cuda")]
        if let Some(device) = &mut self.device {
            device.prepared = None;
        }
    }

    /// Uploads the blocks and drops their host copy.
    #[cfg(feature = "cuda")]
    pub fn attach_device(
        &mut self,
        context: &std::sync::Arc<crate::gpu_transformer::GpuContext>,
    ) -> Result<(), NetworkError> {
        self.device = Some(device::DeviceZImage::upload(
            context,
            &self.config,
            &self.blocks,
        )?);
        self.blocks = Vec::new();
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn on_device(&self) -> bool {
        self.device.is_some()
    }

    /// The time embedding every modulation reads: `t` is `1 - sigma`.
    fn time(&self, sigma: f32) -> Result<Vec<f32>, NetworkError> {
        let table = crate::mmdit::sinusoid((1.0 - sigma) * self.config.t_scale, 256);
        let hidden: Vec<f32> = self
            .time_in
            .apply(&table)?
            .into_iter()
            .map(crate::ffn::silu)
            .collect();
        self.time_out.apply(&hidden)
    }

    /// Every modulated block's `[1 + scale_msa, tanh gate_msa, 1 + scale_mlp,
    /// tanh gate_mlp]`, block after block.
    fn modulation(&self, time: &[f32]) -> Result<Vec<f32>, NetworkError> {
        let dim = self.config.dim;
        let parts = self
            .modulations
            .par_iter()
            .map(|layer| {
                let mut values = layer.apply(time)?;
                for (index, value) in values.iter_mut().enumerate() {
                    *value = match index / dim {
                        0 | 2 => 1.0 + *value,
                        _ => value.tanh(),
                    };
                }
                Ok(values)
            })
            .collect::<Result<Vec<_>, NetworkError>>()?;
        Ok(parts.concat())
    }

    /// The image tokens embedded, `[padded, dim]`. `latents` is packed the
    /// pipeline's way, a patch's channels as `c * p² + y * p + x`; the model
    /// reads them as `(y * p + x) * channels + c`.
    fn embed_image(&self, latents: &Matrix, padded: usize) -> Result<Matrix, NetworkError> {
        let (channels, cells) = (
            self.config.in_channels,
            self.config.patch * self.config.patch,
        );
        let mut tokens = Matrix::new(latents.rows, latents.cols);
        for (row, source) in tokens
            .data
            .chunks_mut(latents.cols)
            .zip(latents.data.chunks(latents.cols))
        {
            for c in 0..channels {
                for cell in 0..cells {
                    row[cell * channels + c] = source[c * cells + cell];
                }
            }
        }
        let embedded = self.image_in.forward(&tokens)?;
        Ok(pad_rows(embedded, padded, &self.image_pad))
    }

    /// The caption tokens embedded, `[padded, dim]`.
    fn embed_caption(&self, text: &Matrix, padded: usize) -> Result<Matrix, NetworkError> {
        let normed = crate::text_encoder::rms_norm(text, &self.caption_norm, self.config.eps);
        Ok(pad_rows(
            self.caption_in.forward(&normed)?,
            padded,
            &self.caption_pad,
        ))
    }

    /// The final layer over the image rows, back in the pipeline's packing,
    /// and negated: the model predicts noise minus image, the sampler wants
    /// the velocity from noise to image.
    fn finish(&self, hidden: Matrix, time: &[f32]) -> Result<Matrix, NetworkError> {
        let silu: Vec<f32> = time.iter().copied().map(crate::ffn::silu).collect();
        let scale = self.final_modulation.apply(&silu)?;
        let mut normed = hidden;
        normed.data.par_chunks_mut(normed.cols).for_each(|row| {
            let n = row.len() as f32;
            let mean = row.iter().sum::<f32>() / n;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let inv = (var + 1e-6).sqrt().recip();
            for (value, scale) in row.iter_mut().zip(&scale) {
                *value = (*value - mean) * inv * (1.0 + scale);
            }
        });
        let out = self.final_out.forward(&normed)?;
        let (channels, cells) = (
            self.config.in_channels,
            self.config.patch * self.config.patch,
        );
        let mut packed = Matrix::new(out.rows, out.cols);
        for (row, source) in packed
            .data
            .chunks_mut(out.cols)
            .zip(out.data.chunks(out.cols))
        {
            for c in 0..channels {
                for cell in 0..cells {
                    row[c * cells + cell] = -source[cell * channels + c];
                }
            }
        }
        Ok(packed)
    }

    /// The RoPE `cos` and `sin` tables, `[image_rows + caption_rows, head_dim
    /// / 2]`, in the unified order: image tokens at `(caption_rows + 1, y,
    /// x)`, image padding at `(0, 0, 0)`, caption token `i` at `(i + 1, 0,
    /// 0)`.
    fn rope(
        &self,
        height: usize,
        width: usize,
        image_rows: usize,
        caption_rows: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let half = self.config.dim / self.config.heads / 2;
        let frequencies: Vec<(usize, f64)> = self
            .config
            .axes
            .iter()
            .enumerate()
            .flat_map(|(axis, &dim)| (0..dim / 2).map(move |j| (axis, (2 * j) as f64 / dim as f64)))
            .map(|(axis, exponent)| (axis, self.config.rope_theta.powf(exponent).recip()))
            .collect();
        let rows = image_rows + caption_rows;
        let (mut cos, mut sin) = (vec![0.0; rows * half], vec![0.0; rows * half]);
        for row in 0..rows {
            let position = match row {
                r if r < height * width => [caption_rows + 1, r / width, r % width],
                r if r < image_rows => [0, 0, 0],
                r => [r - image_rows + 1, 0, 0],
            };
            for (p, &(axis, frequency)) in frequencies.iter().enumerate() {
                // diffusers takes the angle in F64 and rounds it to F32 first.
                let angle = (position[axis] as f64 * frequency) as f32;
                cos[row * half + p] = angle.cos();
                sin[row * half + p] = angle.sin();
            }
        }
        (cos, sin)
    }
}

/// `matrix` with rows of `pad` added up to `rows`.
fn pad_rows(mut matrix: Matrix, rows: usize, pad: &[f32]) -> Matrix {
    for _ in matrix.rows..rows {
        matrix.data.extend_from_slice(pad);
    }
    matrix.rows = rows.max(matrix.rows);
    matrix
}

impl Denoiser for ZImage {
    fn denoise(&mut self, latents: &Matrix, sigma: f32) -> Result<Matrix, NetworkError> {
        #[cfg(feature = "cuda")]
        if let Some(mut device) = self.device.take() {
            let result = device.run(self, latents, sigma);
            self.device = Some(device);
            return result;
        }
        let _ = (latents, sigma);
        Err(NetworkError::InvalidConfig(
            "Z-Image runs on CUDA only: move the pipeline to a device first".into(),
        ))
    }
}

#[cfg(feature = "cuda")]
mod device {
    use super::{Block, SEQ_MULTIPLE, ZImage, ZImageConfig};
    use crate::conv::Dense;
    use crate::cuda_flash::compile_ptx;
    use crate::cuda_training::{cfg, cuda_err};
    use crate::gpu_model::{Act, FlashShape, Gpu};
    use crate::gpu_transformer::GpuContext;
    use crate::matrix::Matrix;
    use crate::network::NetworkError;
    use cudarc::driver::{CudaFunction, CudaSlice, CudaView, LaunchConfig, PushKernelArg};
    use std::sync::Arc;

    const KERNELS: &str = r#"
#define ROW_THREADS 256
__device__ __forceinline__ float row_sum(float v,float*red){
  int tid=threadIdx.x;red[tid]=v;__syncthreads();
  for(int s=ROW_THREADS/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
  float r=red[0];__syncthreads();return r;
}
__device__ __forceinline__ void st(void*p,size_t i,float v,int narrow){
  if(!narrow){((float*)p)[i]=v;return;}
  unsigned u=__float_as_uint(v);((unsigned short*)p)[i]=(unsigned short)((u+0x7fffu+((u>>16)&1u))>>16);
}
// `out = x / rms(x) * w * scale`, BF16 if `narrow`. One block per row.
extern "C" __global__ void rms_scale(void*out,const float*x,const float*w,const float*scale,int rows,int cols,float eps,int narrow){
  __shared__ float red[ROW_THREADS];int tid=threadIdx.x;
  for(int r=blockIdx.x;r<rows;r+=gridDim.x){
    const float*s=x+(size_t)r*cols;
    float a=0.f;for(int c=tid;c<cols;c+=ROW_THREADS)a+=s[c]*s[c];
    float inv=rsqrtf(row_sum(a,red)/cols+eps);
    for(int c=tid;c<cols;c+=ROW_THREADS)st(out,(size_t)r*cols+c,s[c]*inv*w[c]*scale[c],narrow);
  }
}
// `x += gate * y / rms(y) * w`. One block per row.
extern "C" __global__ void rms_gate_add(float*x,const float*y,const float*w,const float*gate,int rows,int cols,float eps){
  __shared__ float red[ROW_THREADS];int tid=threadIdx.x;
  for(int r=blockIdx.x;r<rows;r+=gridDim.x){
    const float*s=y+(size_t)r*cols;float*d=x+(size_t)r*cols;
    float a=0.f;for(int c=tid;c<cols;c+=ROW_THREADS)a+=s[c]*s[c];
    float inv=rsqrtf(row_sum(a,red)/cols+eps);
    for(int c=tid;c<cols;c+=ROW_THREADS)d[c]+=gate[c]*(s[c]*inv*w[c]);
  }
}
// The per-head RMS norm of a query or key, then RoPE on adjacent pairs. One
// warp per (row, head). Row `r` reads the tables at row `first + r`.
extern "C" __global__ void head_norm_rope(float*x,const float*scale,const float*cs,const float*sn,int rows,int heads,int head_dim,int stride,int off,int first,float eps){
  int w=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31;if(w>=rows*heads)return;
  int r=w/heads,h=w-r*heads,half=head_dim/2;
  float2*v=(float2*)(x+(size_t)r*stride+off+h*head_dim);const float2*k=(const float2*)scale;
  float s=0.f;
  for(int p=lane;p<half;p+=32){float2 a=v[p];s+=a.x*a.x+a.y*a.y;}
  for(int o=16;o>0;o>>=1)s+=__shfl_xor_sync(0xffffffffu,s,o);
  float inv=rsqrtf(s/head_dim+eps);
  const float*c=cs+(size_t)(first+r)*half;const float*n=sn+(size_t)(first+r)*half;
  for(int p=lane;p<half;p+=32){
    float2 e=v[p],g=k[p];float a=e.x*inv*g.x,b=e.y*inv*g.y;
    v[p]=make_float2(a*c[p]-b*n[p],a*n[p]+b*c[p]);
  }
}
"#;

    struct Linear {
        weight: Act,
        units: usize,
        inner: usize,
    }

    struct DeviceBlock {
        qkv: Linear,
        out: Linear,
        w13: Linear,
        w2: Linear,
        norms: [CudaSlice<f32>; 6],
    }

    /// What one conditioning fixes: the refined caption and the RoPE tables.
    pub(super) struct Prepared {
        caption: CudaSlice<f32>,
        caption_rows: usize,
        image_rows: usize,
        cos: CudaSlice<f32>,
        sin: CudaSlice<f32>,
    }

    pub(super) struct DeviceZImage {
        context: Arc<GpuContext>,
        rms_scale: CudaFunction,
        rms_gate_add: CudaFunction,
        head_norm_rope: CudaFunction,
        blocks: Vec<DeviceBlock>,
        /// The scale and gate of an unmodulated block.
        ones: CudaSlice<f32>,
        pub(super) prepared: Option<Prepared>,
        dim: usize,
        heads: usize,
        refiners: usize,
        eps: f32,
    }

    fn rows_config(rows: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (rows.clamp(1, 1024) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    impl DeviceZImage {
        pub(super) fn upload(
            context: &Arc<GpuContext>,
            config: &ZImageConfig,
            blocks: &[Block],
        ) -> Result<Self, NetworkError> {
            let head_dim = config.dim / config.heads;
            if context.half_accumulate
                || context.flash.is_none()
                || !crate::cuda_flash::eligible(true, head_dim)
            {
                return Err(NetworkError::InvalidConfig(
                    "Z-Image needs a BF16 context with the fused attention kernel (Ampere or later)"
                        .into(),
                ));
            }
            if blocks.len() != 2 * config.refiners + config.layers {
                return Err(NetworkError::InvalidConfig(
                    "Z-Image's blocks are no longer on the host".into(),
                ));
            }
            let gpu = Gpu { context };
            let module = context
                .stream
                .context()
                .load_module(compile_ptx(KERNELS).map_err(|error| {
                    NetworkError::Cuda(format!("Z-Image kernel compilation failed: {error}"))
                })?)
                .map_err(cuda_err("Z-Image module loading"))?;
            let get = |name: &str| {
                module
                    .load_function(name)
                    .map_err(cuda_err("Z-Image kernel lookup"))
            };
            // The host weight is already on its int8 grid, so quantizing the
            // widened copy again lands on the same values and scales.
            let linear = |layer: &Dense| -> Result<Linear, NetworkError> {
                let (units, inner) = (layer.out_dim(), layer.in_dim());
                let wide = gpu.upload(&layer.weight_f32().data)?;
                let weight = match (layer.is_quantized(), layer.int8_group()) {
                    (true, Some(group)) => {
                        gpu.quantized_grouped(&wide.slice(..), units, inner, group)?
                    }
                    (true, None) => gpu.quantized(&wide.slice(..), units, inner)?,
                    (false, _) => gpu.narrowed(&wide.slice(..), units * inner, true)?,
                };
                Ok(Linear {
                    weight,
                    units,
                    inner,
                })
            };
            let blocks = blocks
                .iter()
                .map(|block| {
                    let [a, b, c, d, e, f] = &block.norms;
                    Ok(DeviceBlock {
                        qkv: linear(&block.qkv)?,
                        out: linear(&block.out)?,
                        w13: linear(&block.w13)?,
                        w2: linear(&block.w2)?,
                        norms: [
                            gpu.upload(a)?,
                            gpu.upload(b)?,
                            gpu.upload(c)?,
                            gpu.upload(d)?,
                            gpu.upload(e)?,
                            gpu.upload(f)?,
                        ],
                    })
                })
                .collect::<Result<Vec<_>, NetworkError>>()?;
            Ok(Self {
                context: context.clone(),
                rms_scale: get("rms_scale")?,
                rms_gate_add: get("rms_gate_add")?,
                head_norm_rope: get("head_norm_rope")?,
                blocks,
                ones: gpu.upload(&vec![1.0; config.dim])?,
                prepared: None,
                dim: config.dim,
                heads: config.heads,
                refiners: config.refiners,
                eps: config.eps,
            })
        }

        /// One step: the velocity for `latents` at `sigma`.
        pub(super) fn run(
            &mut self,
            model: &ZImage,
            latents: &Matrix,
            sigma: f32,
        ) -> Result<Matrix, NetworkError> {
            let conditioning = model.conditioning.as_ref().ok_or_else(|| {
                NetworkError::InvalidConfig("Z-Image was handed no conditioning".into())
            })?;
            if latents.rows != conditioning.height * conditioning.width
                || latents.cols != model.config.token_width()
            {
                return Err(NetworkError::InvalidTarget {
                    expected: conditioning.height * conditioning.width,
                    actual: latents.rows,
                });
            }
            let image_rows = latents.rows.next_multiple_of(SEQ_MULTIPLE);
            if self
                .prepared
                .as_ref()
                .is_none_or(|p| p.image_rows != image_rows)
            {
                self.prepared = None;
                self.prepared = Some(self.prepare(model, image_rows)?);
            }
            let gpu = Gpu {
                context: &self.context,
            };
            let dim = self.dim;
            let time = model.time(sigma)?;
            let modulation = gpu.upload(&model.modulation(&time)?)?;
            let mut image = gpu.upload(&model.embed_image(latents, image_rows)?.data)?;
            let prepared = self.prepared.as_ref().expect("prepared above");
            let tables = (&prepared.cos, &prepared.sin);
            let at = |index: usize| Some(modulation.slice(index * 4 * dim..(index + 1) * 4 * dim));
            for (index, block) in self.blocks[..self.refiners].iter().enumerate() {
                self.block(&gpu, block, &mut image, image_rows, 0, at(index), tables)?;
            }
            let rows = image_rows + prepared.caption_rows;
            let mut unified = gpu.uninit(rows * dim)?;
            let copy = || cuda_err("device to device copy");
            let stream = &self.context.stream;
            stream
                .memcpy_dtod(&image, &mut unified.slice_mut(..image_rows * dim))
                .map_err(copy())?;
            stream
                .memcpy_dtod(
                    &prepared.caption,
                    &mut unified.slice_mut(image_rows * dim..),
                )
                .map_err(copy())?;
            drop(image);
            for (index, block) in self.blocks[2 * self.refiners..].iter().enumerate() {
                let slot = self.refiners + index;
                self.block(&gpu, block, &mut unified, rows, 0, at(slot), tables)?;
            }
            let mut hidden = gpu.download(&unified)?;
            hidden.truncate(latents.rows * dim);
            model.finish(Matrix::from_vec(latents.rows, dim, hidden), &time)
        }

        /// Embeds the caption, runs the context refiner over it and builds
        /// the RoPE tables, once per conditioning.
        fn prepare(&self, model: &ZImage, image_rows: usize) -> Result<Prepared, NetworkError> {
            let conditioning = model.conditioning.as_ref().expect("checked by the caller");
            let gpu = Gpu {
                context: &self.context,
            };
            let caption_rows = conditioning.text.rows.next_multiple_of(SEQ_MULTIPLE);
            let (cos, sin) = model.rope(
                conditioning.height,
                conditioning.width,
                image_rows,
                caption_rows,
            );
            let (cos, sin) = (gpu.upload(&cos)?, gpu.upload(&sin)?);
            let mut caption =
                gpu.upload(&model.embed_caption(&conditioning.text, caption_rows)?.data)?;
            for block in &self.blocks[self.refiners..2 * self.refiners] {
                self.block(
                    &gpu,
                    block,
                    &mut caption,
                    caption_rows,
                    image_rows,
                    None,
                    (&cos, &sin),
                )?;
            }
            Ok(Prepared {
                caption,
                caption_rows,
                image_rows,
                cos,
                sin,
            })
        }

        /// One block over `rows` rows of `x`, in place. `modulation` is the
        /// block's four `dim`-wide vectors, `None` for an unmodulated block;
        /// row `r` reads the RoPE tables at `first + r`.
        #[allow(clippy::too_many_arguments)]
        fn block(
            &self,
            gpu: &Gpu<'_>,
            block: &DeviceBlock,
            x: &mut CudaSlice<f32>,
            rows: usize,
            first: usize,
            modulation: Option<CudaView<'_, f32>>,
            (cos, sin): (&CudaSlice<f32>, &CudaSlice<f32>),
        ) -> Result<(), NetworkError> {
            let (dim, heads) = (self.dim, self.heads);
            let part = |index: usize| match &modulation {
                Some(modulation) => modulation.slice(index * dim..(index + 1) * dim),
                None => self.ones.slice(..),
            };
            let [
                norm_q,
                norm_k,
                attention_norm1,
                attention_norm2,
                ffn_norm1,
                ffn_norm2,
            ] = &block.norms;

            let normed = self.rms_scale(gpu, x, attention_norm1, &part(0), rows)?;
            let mut qkv = gpu.uninit(rows * 3 * dim)?;
            self.project(gpu, &block.qkv, &normed, rows, &mut qkv)?;
            for (scale, offset) in [(norm_q, 0), (norm_k, dim)] {
                unsafe {
                    self.context
                        .stream
                        .launch_builder(&self.head_norm_rope)
                        .arg(&mut qkv)
                        .arg(scale)
                        .arg(cos)
                        .arg(sin)
                        .arg(&(rows as i32))
                        .arg(&(heads as i32))
                        .arg(&((dim / heads) as i32))
                        .arg(&((3 * dim) as i32))
                        .arg(&(offset as i32))
                        .arg(&(first as i32))
                        .arg(&self.eps)
                        .launch(cfg(rows * heads * 32))
                        .map_err(cuda_err("Z-Image head norm kernel"))?;
                }
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
            let flash = self.context.flash.as_ref().expect("checked at upload");
            let scale = ((dim / heads) as f32).sqrt().recip();
            gpu.flash_attention(
                flash,
                &qkv,
                &qkv,
                &mut merged,
                &mut log_sum_exp,
                shape,
                scale,
            )?;
            drop(qkv);
            let mut attended = gpu.uninit(rows * dim)?;
            self.project(gpu, &block.out, &merged, rows, &mut attended)?;
            self.rms_gate_add(x, &attended, attention_norm2, &part(1), rows)?;

            let normed = self.rms_scale(gpu, x, ffn_norm1, &part(2), rows)?;
            let ffn = block.w13.units / 2;
            let mut gate_up = gpu.uninit(rows * 2 * ffn)?;
            self.project(gpu, &block.w13, &normed, rows, &mut gate_up)?;
            let hidden = gpu.swiglu(&gate_up, rows, ffn, true)?;
            drop(gate_up);
            let mut fed = gpu.uninit(rows * dim)?;
            self.project(gpu, &block.w2, &hidden, rows, &mut fed)?;
            self.rms_gate_add(x, &fed, ffn_norm2, &part(3), rows)
        }

        fn project(
            &self,
            gpu: &Gpu<'_>,
            linear: &Linear,
            x: &Act,
            rows: usize,
            out: &mut CudaSlice<f32>,
        ) -> Result<(), NetworkError> {
            gpu.project(
                &linear.weight,
                linear.units,
                linear.inner,
                &x.all(),
                true,
                &mut out.slice_mut(..),
                rows,
                0.0,
            )
        }

        fn rms_scale(
            &self,
            gpu: &Gpu<'_>,
            x: &CudaSlice<f32>,
            weight: &CudaSlice<f32>,
            scale: &CudaView<'_, f32>,
            rows: usize,
        ) -> Result<Act, NetworkError> {
            let mut out = gpu.act(rows * self.dim, true)?;
            unsafe {
                self.context
                    .stream
                    .launch_builder(&self.rms_scale)
                    .arg(out.destination())
                    .arg(x)
                    .arg(weight)
                    .arg(scale)
                    .arg(&(rows as i32))
                    .arg(&(self.dim as i32))
                    .arg(&self.eps)
                    .arg(&1i32)
                    .launch(rows_config(rows))
                    .map_err(cuda_err("Z-Image norm kernel"))?;
            }
            Ok(out)
        }

        fn rms_gate_add(
            &self,
            x: &mut CudaSlice<f32>,
            branch: &CudaSlice<f32>,
            weight: &CudaSlice<f32>,
            gate: &CudaView<'_, f32>,
            rows: usize,
        ) -> Result<(), NetworkError> {
            unsafe {
                self.context
                    .stream
                    .launch_builder(&self.rms_gate_add)
                    .arg(x)
                    .arg(branch)
                    .arg(weight)
                    .arg(gate)
                    .arg(&(rows as i32))
                    .arg(&(self.dim as i32))
                    .arg(&self.eps)
                    .launch(rows_config(rows))
                    .map_err(cuda_err("Z-Image gated residual kernel"))?;
            }
            Ok(())
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use crate::conv::FeatureMap;
    use crate::diffusion::Scheduler;
    use crate::matrix::Matrix;
    use crate::mmdit::Conditioning;
    use crate::pipeline::{ChatPrompt, ImagePipeline, PromptEncoder};
    use crate::tokenizer::Bpe;
    use crate::transformer::Precision;

    fn relative(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len(), "lengths differ");
        let (gap, norm) = got
            .iter()
            .zip(want)
            .fold((0.0f64, 0.0f64), |(g, n), (a, b)| {
                (g + ((a - b) as f64).powi(2), n + (*b as f64).powi(2))
            });
        (gap / norm).sqrt() as f32
    }

    /// Z-Image-Turbo at 1024², 8 steps, against RustingFrame's
    /// `tools/zimage_refs.py` (diffusers in F32 on the CPU), stage by stage.
    /// `RB_ZIMAGE_REF` is the script's output, `RB_ZIMAGE_MODEL` the model
    /// (default the local copy). `RB_ZIMAGE_ENCODER=f32` also runs the prompt
    /// encoder in F32 on the host (about 14 GB). `RB_ZIMAGE_OUT=<png>` saves
    /// the image. The pipeline is Q8 on device 0.
    ///
    /// The Q8 stages miss the plan's gates (1e-2 step 0, 3e-2 latent and
    /// image), so their limits here are the measured errors with some room:
    /// encoder 1.07e-2, step 0 1.51e-2, latent 9.08e-2, image 8.63e-2. The
    /// int8 weights cause it, not a bug: F32 activations change nothing, and
    /// `RB_Q8_GROUP=16` cuts step 0 to 1.15e-2 and the latent to 6.6e-2.
    #[test]
    #[ignore = "needs Z-Image-Turbo, the output of tools/zimage_refs.py and a CUDA device"]
    fn z_image_turbo_matches_diffusers() {
        let env = |name: &str| std::env::var(name).ok();
        let dir = std::path::PathBuf::from(env("RB_ZIMAGE_REF").expect("RB_ZIMAGE_REF"));
        let model = std::path::PathBuf::from(
            env("RB_ZIMAGE_MODEL")
                .unwrap_or_else(|| "/mnt/ollama/RustingFrameModels/Z-Image-Turbo".into()),
        );
        let settings: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
                .unwrap();
        let prompt = settings["prompt"].as_str().unwrap();
        let mut reference =
            crate::safetensors::SafeTensors::open(dir.join("zimage.safetensors")).unwrap();
        let mut read = |name: &str| reference.tensor(name).unwrap().0;
        let mut report = Vec::new();
        let mut gate = |stage: &str, error: f32, limit: f32| {
            eprintln!("zimage {stage}: relative L2 {error:.3e} (gate {limit:.0e})");
            report.push((stage.to_string(), error, limit));
        };

        // Token ids, exactly; only the real ones are encoded.
        let tokenizer = Bpe::from_file(model.join("tokenizer/tokenizer.json")).unwrap();
        let chat = ChatPrompt::z_image(tokenizer.token_id("<|endoftext|>").unwrap());
        let (ids, real) = chat.tokenize(&tokenizer, prompt).unwrap();
        let want_ids: Vec<u32> = read("input_ids").iter().map(|&id| id as u32).collect();
        let want_real = read("attention_mask").iter().filter(|&&m| m > 0.0).count();
        assert_eq!((ids.as_slice(), real), (&want_ids[..want_real], want_real));
        let embeds = read("prompt_embeds");

        if env("RB_ZIMAGE_ENCODER").as_deref() == Some("f32") {
            let mut file = crate::safetensors::ShardedSafeTensors::open(
                model.join("text_encoder/model.safetensors.index.json"),
            )
            .unwrap();
            let mut config = crate::text_encoder::TextEncoderConfig::from_file(
                model.join("text_encoder/config.json"),
            )
            .unwrap();
            config.layers = 35;
            let encoder = crate::text_encoder::TextEncoder::read_at(
                &mut file,
                "model.",
                config,
                Precision::F32,
            )
            .unwrap();
            drop(file);
            let got = encoder.forward_layers(&ids, real, &chat.layers).unwrap();
            gate(
                "prompt embeds, encoder F32",
                relative(&got.data, &embeds),
                1e-2,
            );
        }

        let started = std::time::Instant::now();
        let mut pipeline = ImagePipeline::load_at(&model, Precision::Q8).unwrap();
        eprintln!("zimage load: {:?}", started.elapsed());
        assert_eq!(pipeline.config.steps, 8);
        assert!(
            matches!(&pipeline.encoder, PromptEncoder::Sequence { chat: Some(c), .. } if *c == chat)
        );

        let started = std::time::Instant::now();
        pipeline.to_cuda(0).unwrap();
        eprintln!("zimage to device: {:?}", started.elapsed());
        let (got, _) = pipeline.encoder.encode(prompt).unwrap();
        gate(
            "prompt embeds, encoder Q8",
            relative(&got.data, &embeds),
            1.2e-2,
        );

        assert!(pipeline.config.dynamic_shift.is_none());
        let levels = Scheduler::flow_match(pipeline.config.shift)
            .sigmas(8)
            .unwrap();
        gate("sigmas", relative(&levels, &read("sigmas")), 1e-5);

        let grid = 128;
        let packed = |pipeline: &ImagePipeline, data: Vec<f32>| {
            pipeline
                .pack(&FeatureMap {
                    channels: 16,
                    height: grid,
                    width: grid,
                    data,
                })
                .unwrap()
        };
        let noise = packed(&pipeline, read("noise"));
        let velocity = packed(&pipeline, read("step0_out").iter().map(|v| -v).collect());
        let conditioning = Conditioning {
            text: Matrix::from_vec(want_real, 2560, embeds),
            pooled: Vec::new(),
            height: grid / 2,
            width: grid / 2,
            guidance: 0.0,
        };
        pipeline.denoiser.set_conditioning(conditioning.clone());
        let started = std::time::Instant::now();
        let prediction =
            crate::diffusion::Denoiser::denoise(&mut pipeline.denoiser, &noise, levels[0]).unwrap();
        eprintln!("zimage step 0 (with the caption): {:?}", started.elapsed());
        gate(
            "step-0 prediction",
            relative(&prediction.data, &velocity.data),
            2e-2,
        );

        let started = std::time::Instant::now();
        let latent = pipeline
            .sample_latent_from_noise(conditioning, None, noise, None, |_, _| true)
            .unwrap();
        eprintln!("zimage 8 steps: {:?}", started.elapsed());
        let want_latent = packed(&pipeline, read("final_latents"));
        gate(
            "final latent",
            relative(&latent.data, &want_latent.data),
            1.2e-1,
        );

        let image = read("image");
        let decoded = pipeline.decode(&want_latent, grid / 2, grid / 2).unwrap();
        gate(
            "image, reference latent",
            relative(&decoded.data, &image),
            1e-2,
        );
        let decoded = pipeline.decode(&latent, grid / 2, grid / 2).unwrap();
        gate(
            "image, whole chain",
            relative(&decoded.data, &image),
            1.2e-1,
        );
        #[cfg(feature = "images")]
        if let Some(out) = env("RB_ZIMAGE_OUT") {
            crate::vae::save_png(&decoded, out).unwrap();
        }

        let failed: Vec<_> = report
            .iter()
            .filter(|(_, error, limit)| error > limit)
            .collect();
        assert!(failed.is_empty(), "over the gate: {failed:?}");
    }
}
