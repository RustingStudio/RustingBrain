//! FLUX 3 Image Klein's denoiser (Black Forest Labs, FLUX Kommunity License):
//! diffusers' `Flux3Transformer2DModel`, the step-distilled flow-map kind.
//!
//! 3072-wide blocks, each a layer norm, a modulation, and attention and a
//! SwiGLU feed-forward run side by side on the same modulated input. Five
//! blocks run over the text alone and five over the image alone, then 28 over
//! `[text, image]` together, with one modulation per kind of token. The
//! modulation reads the step's noise level and the level the step jumps to.
//!
//! Text-to-image only: the reference-image stream (`image_cond`) is not
//! loaded. The blocks run on CUDA only, Q8 on the device, through
//! [`crate::zimage`]'s kernels; the small parts run on the host in F32.

// Without CUDA the model loads but cannot run, so most of it is unused.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use crate::conv::Dense;
use crate::diffusion::Denoiser;
use crate::ffn::silu;
use crate::matrix::Matrix;
use crate::mmdit::Conditioning;
use crate::network::NetworkError;
use crate::safetensors::ShardedSafeTensors;
use crate::transformer::Precision;
use crate::zimage::{linear, stacked};
use rayon::prelude::*;

/// The parts of `transformer/config.json` the model reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Flux3Config {
    pub dim: usize,
    pub heads: usize,
    /// Blocks each kind of token runs through alone.
    pub layers: usize,
    /// Blocks over the joint sequence.
    pub single_layers: usize,
    pub in_channels: usize,
    pub text_dim: usize,
    pub eps: f32,
    pub rope_theta: f64,
    /// RoPE widths of the `(t, h, w, l)` position axes.
    pub axes: Vec<usize>,
}

impl Flux3Config {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, NetworkError> {
        let path = path.as_ref();
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)
            .map_err(|error| NetworkError::InvalidConfig(format!("{}: {error}", path.display())))?;
        let number = |key: &str| {
            json[key].as_f64().ok_or_else(|| {
                NetworkError::InvalidConfig(format!("{}: no number {key}", path.display()))
            })
        };
        let heads = number("num_attention_heads")? as usize;
        let head_dim = number("attention_head_dim")? as usize;
        let config = Self {
            dim: heads * head_dim,
            heads,
            layers: number("num_layers")? as usize,
            single_layers: number("num_single_layers")? as usize,
            in_channels: number("in_channels")? as usize,
            text_dim: number("joint_attention_dim")? as usize,
            eps: number("eps")? as f32,
            rope_theta: number("rope_theta")?,
            axes: json["axes_dims_rope"]
                .as_array()
                .map(|axes| {
                    axes.iter()
                        .filter_map(|a| a.as_u64())
                        .map(|a| a as usize)
                        .collect()
                })
                .unwrap_or_default(),
        };
        if config.axes.len() != 4 || config.axes.iter().sum::<usize>() != head_dim {
            return Err(NetworkError::InvalidConfig(format!(
                "{}: the RoPE axes {:?} are not four that fill a {head_dim}-wide head",
                path.display(),
                config.axes
            )));
        }
        if json["flow_map"].as_bool() != Some(true) {
            return Err(NetworkError::InvalidConfig(format!(
                "{}: only the flow-map (step-distilled) FLUX 3 is supported",
                path.display()
            )));
        }
        Ok(config)
    }
}

/// One block's linears and its query and key norms, held on the host until
/// [`Flux3::attach_device`] uploads them.
pub(crate) struct Block {
    /// `to_q`, `to_k` and `to_v` stacked, `[3 * dim, dim]`.
    qkv: Dense,
    out: Dense,
    /// `[2 * ffn, dim]`: the gate half first.
    ff_in: Dense,
    ff_out: Dense,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
}

/// FLUX 3's transformer.
pub struct Flux3 {
    config: Flux3Config,
    /// `linear_1` and `linear_2` of the timestep embedder, then of the
    /// target timestep embedder.
    time: [Dense; 2],
    target: [Dense; 2],
    /// Text then image, for the per-kind blocks; text then image, for the
    /// joint ones. Each makes `[shift, scale, gate]`.
    modulations: [Dense; 4],
    text_in: Dense,
    image_in: Dense,
    final_modulation: Dense,
    final_out: Dense,
    /// Text blocks, image blocks, then the joint ones. Emptied once they are
    /// on the device.
    blocks: Vec<Block>,
    conditioning: Option<Conditioning>,
    /// The noise levels of the run, which name the level each step jumps to.
    sigmas: Vec<f32>,
    #[cfg(feature = "cuda")]
    device: Option<device::DeviceFlux3>,
}

impl std::fmt::Debug for Flux3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Flux3")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Flux3 {
    /// Reads the transformer. The block linears are stored at `precision`
    /// (Q8 grouped unless F32); the rest stays F32, about 0.8 GB.
    pub fn read(
        file: &mut ShardedSafeTensors,
        config: Flux3Config,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        let vector = |file: &mut ShardedSafeTensors, name: &str| file.tensor(name).map(|(v, _)| v);
        let mut blocks = Vec::new();
        for (group, count) in [
            ("txt_mode_blocks", config.layers),
            ("mode_blocks.image", config.layers),
            ("single_transformer_blocks", config.single_layers),
        ] {
            for index in 0..count {
                let at = |part: &str| format!("{group}.{index}.{part}");
                blocks.push(Block {
                    qkv: stacked(
                        file,
                        &[at("attn.to_q"), at("attn.to_k"), at("attn.to_v")],
                        precision,
                    )?,
                    out: stacked(file, &[at("attn.to_out.0")], precision)?,
                    ff_in: stacked(file, &[at("ff.linear_in")], precision)?,
                    ff_out: stacked(file, &[at("ff.linear_out")], precision)?,
                    norm_q: vector(file, &at("attn.norm_q.weight"))?,
                    norm_k: vector(file, &at("attn.norm_k.weight"))?,
                });
            }
        }
        let pair = |file: &mut ShardedSafeTensors, name: &str| -> Result<[Dense; 2], NetworkError> {
            Ok([
                linear(file, &format!("{name}.linear_1"))?,
                linear(file, &format!("{name}.linear_2"))?,
            ])
        };
        Ok(Self {
            time: pair(file, "timestep_embedder")?,
            target: pair(file, "target_timestep_embedder")?,
            modulations: [
                linear(file, "early_stream_modulations.txt.linear")?,
                linear(file, "early_stream_modulations.image.linear")?,
                linear(file, "single_stream_modulations.txt.linear")?,
                linear(file, "single_stream_modulations.image.linear")?,
            ],
            text_in: linear(file, "context_embedder")?,
            image_in: linear(file, "x_embedders.image")?,
            final_modulation: linear(file, "norm_out.linear")?,
            final_out: linear(file, "proj_out")?,
            blocks,
            config,
            conditioning: None,
            sigmas: Vec::new(),
            #[cfg(feature = "cuda")]
            device: None,
        })
    }

    pub fn config(&self) -> &Flux3Config {
        &self.config
    }

    pub fn set_conditioning(&mut self, conditioning: Conditioning) {
        self.conditioning = Some(conditioning);
        #[cfg(feature = "cuda")]
        if let Some(device) = &mut self.device {
            device.prepared = None;
        }
    }

    /// Hands it the noise levels of the run to come, noisiest first. A step
    /// at one of them tells the model it jumps to the next lower one; with no
    /// schedule every step says it jumps to zero.
    pub fn set_schedule(&mut self, sigmas: Vec<f32>) {
        self.sigmas = sigmas;
    }

    /// Uploads the blocks and drops their host copy.
    #[cfg(feature = "cuda")]
    pub fn attach_device(
        &mut self,
        context: &std::sync::Arc<crate::gpu_transformer::GpuContext>,
    ) -> Result<(), NetworkError> {
        self.device = Some(device::DeviceFlux3::upload(
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

    /// The four modulations for a step at `sigma`, `[shift, scale, gate]`
    /// each, in the order of the `modulations` field; and the image tokens'
    /// time embedding, which the final layer reads.
    fn modulation(&self, sigma: f32) -> Result<(Vec<f32>, Vec<f32>), NetworkError> {
        let embed = |layers: &[Dense; 2], time: f32| -> Result<Vec<f32>, NetworkError> {
            let table = crate::mmdit::sinusoid(time * 1000.0, layers[0].in_dim());
            let hidden: Vec<f32> = layers[0].apply(&table)?.into_iter().map(silu).collect();
            layers[1].apply(&hidden)
        };
        let next = self
            .sigmas
            .iter()
            .copied()
            .filter(|level| *level < sigma)
            .fold(0.0, f32::max);
        let target = embed(&self.target, next)?;
        let with_target = |mut time: Vec<f32>| {
            time.iter_mut().zip(&target).for_each(|(t, add)| *t += add);
            time
        };
        // The image sits at the step's level; the text sits at zero.
        let image = with_target(embed(&self.time, sigma)?);
        let text = with_target(embed(&self.time, 0.0)?);
        let active = |time: &[f32]| time.iter().copied().map(silu).collect::<Vec<f32>>();
        let (image_active, text_active) = (active(&image), active(&text));
        let mut all = Vec::new();
        for (layer, time) in self
            .modulations
            .iter()
            .zip([&text_active, &image_active, &text_active, &image_active])
        {
            all.extend(layer.apply(time)?);
        }
        Ok((all, image))
    }

    /// The final layer over the image rows: the velocity, packed as the
    /// pipeline packs a latent.
    fn finish(&self, hidden: Matrix, image_time: &[f32]) -> Result<Matrix, NetworkError> {
        let active: Vec<f32> = image_time.iter().copied().map(silu).collect();
        // `[scale, shift]`, the other way round from the blocks.
        let modulation = self.final_modulation.apply(&active)?;
        let (scale, shift) = modulation.split_at(hidden.cols);
        let eps = self.config.eps;
        let mut normed = hidden;
        normed.data.par_chunks_mut(normed.cols).for_each(|row| {
            let n = row.len() as f32;
            let mean = row.iter().sum::<f32>() / n;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let inv = (var + eps).sqrt().recip();
            for ((value, scale), shift) in row.iter_mut().zip(scale).zip(shift) {
                *value = (*value - mean) * inv * (1.0 + scale) + shift;
            }
        });
        self.final_out.forward(&normed)
    }

    /// The RoPE `cos` and `sin` tables, `[text_rows + height * width,
    /// head_dim / 2]`: text token `i` at `(0, 0, 0, i)`, then the image token
    /// in row `y`, column `x` at `(0, y / 2, x / 2, 0)`.
    fn rope(&self, height: usize, width: usize, text_rows: usize) -> (Vec<f32>, Vec<f32>) {
        let half = self.config.dim / self.config.heads / 2;
        let frequencies: Vec<(usize, f64)> = self
            .config
            .axes
            .iter()
            .enumerate()
            .flat_map(|(axis, &dim)| (0..dim / 2).map(move |j| (axis, (2 * j) as f64 / dim as f64)))
            .map(|(axis, exponent)| (axis, self.config.rope_theta.powf(exponent).recip()))
            .collect();
        let rows = text_rows + height * width;
        let (mut cos, mut sin) = (vec![0.0; rows * half], vec![0.0; rows * half]);
        for row in 0..rows {
            let position = match row.checked_sub(text_rows) {
                None => [0.0, 0.0, 0.0, row as f64],
                Some(cell) => [
                    0.0,
                    (cell / width) as f64 * 0.5,
                    (cell % width) as f64 * 0.5,
                    0.0,
                ],
            };
            for (p, &(axis, frequency)) in frequencies.iter().enumerate() {
                let angle = position[axis] * frequency;
                cos[row * half + p] = angle.cos() as f32;
                sin[row * half + p] = angle.sin() as f32;
            }
        }
        (cos, sin)
    }
}

impl Denoiser for Flux3 {
    fn denoise(&mut self, latents: &Matrix, sigma: f32) -> Result<Matrix, NetworkError> {
        #[cfg(feature = "cuda")]
        if let Some(mut device) = self.device.take() {
            let result = device.run(self, latents, sigma);
            self.device = Some(device);
            return result;
        }
        let _ = (latents, sigma);
        Err(NetworkError::InvalidConfig(
            "FLUX 3 runs on CUDA only: move the pipeline to a device first".into(),
        ))
    }
}

#[cfg(feature = "cuda")]
mod device {
    use super::{Block, Flux3, Flux3Config};
    use crate::cuda_flash::compile_ptx;
    use crate::cuda_training::{cfg, cuda_err};
    use crate::gpu_model::{Act, FlashShape, Gpu};
    use crate::gpu_transformer::GpuContext;
    use crate::matrix::Matrix;
    use crate::network::NetworkError;
    use crate::zimage::device::{KERNELS as SHARED, Linear, rows_config};
    use cudarc::driver::{CudaFunction, CudaSlice, CudaView, PushKernelArg};
    use std::sync::Arc;

    /// Compiled after Z-Image's kernels, whose helpers and `head_norm_rope`
    /// these use. `m` is a modulation, `[shift, scale, gate]`; rows before
    /// `split` read `ma`, the rest `mb`.
    const KERNELS: &str = r#"
// `out = layer_norm(x) * (1 + scale) + shift`, BF16 if `narrow`. One block per row.
extern "C" __global__ void norm_mod(void*out,const float*x,const float*ma,const float*mb,int split,int rows,int cols,float eps,int narrow){
  __shared__ float red[ROW_THREADS];int tid=threadIdx.x;
  for(int r=blockIdx.x;r<rows;r+=gridDim.x){
    const float*s=x+(size_t)r*cols;const float*m=r<split?ma:mb;
    float a=0.f;for(int c=tid;c<cols;c+=ROW_THREADS)a+=s[c];
    float mean=row_sum(a,red)/cols;
    a=0.f;for(int c=tid;c<cols;c+=ROW_THREADS){float d=s[c]-mean;a+=d*d;}
    float inv=rsqrtf(row_sum(a,red)/cols+eps);
    for(int c=tid;c<cols;c+=ROW_THREADS)st(out,(size_t)r*cols+c,(s[c]-mean)*inv*(1.f+m[cols+c])+m[c],narrow);
  }
}
// `x += gate * (a + b)`.
extern "C" __global__ void gate_add(float*x,const float*a,const float*b,const float*ma,const float*mb,int split,int rows,int cols){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(size_t)rows*cols)return;
  int r=(int)(i/cols),c=(int)(i-(size_t)r*cols);const float*m=r<split?ma:mb;
  x[i]+=m[2*cols+c]*(a[i]+b[i]);
}
"#;

    struct DeviceBlock {
        qkv: Linear,
        out: Linear,
        ff_in: Linear,
        ff_out: Linear,
        norm_q: CudaSlice<f32>,
        norm_k: CudaSlice<f32>,
    }

    /// What one conditioning fixes: the embedded text and the RoPE tables.
    pub(super) struct Prepared {
        text: Vec<f32>,
        text_rows: usize,
        image_rows: usize,
        cos: CudaSlice<f32>,
        sin: CudaSlice<f32>,
    }

    pub(super) struct DeviceFlux3 {
        context: Arc<GpuContext>,
        norm_mod: CudaFunction,
        gate_add: CudaFunction,
        head_norm_rope: CudaFunction,
        blocks: Vec<DeviceBlock>,
        pub(super) prepared: Option<Prepared>,
        dim: usize,
        heads: usize,
        layers: usize,
        eps: f32,
    }

    impl DeviceFlux3 {
        pub(super) fn upload(
            context: &Arc<GpuContext>,
            config: &Flux3Config,
            blocks: &[Block],
        ) -> Result<Self, NetworkError> {
            if context.half_accumulate
                || context.flash.is_none()
                || !crate::cuda_flash::eligible(true, config.dim / config.heads)
            {
                return Err(NetworkError::InvalidConfig(
                    "FLUX 3 needs a BF16 context with the fused attention kernel (Ampere or later)"
                        .into(),
                ));
            }
            if blocks.len() != 2 * config.layers + config.single_layers {
                return Err(NetworkError::InvalidConfig(
                    "FLUX 3's blocks are no longer on the host".into(),
                ));
            }
            let gpu = Gpu { context };
            let module = context
                .stream
                .context()
                .load_module(
                    compile_ptx(&format!("{SHARED}{KERNELS}")).map_err(|error| {
                        NetworkError::Cuda(format!("FLUX 3 kernel compilation failed: {error}"))
                    })?,
                )
                .map_err(cuda_err("FLUX 3 module loading"))?;
            let get = |name: &str| {
                module
                    .load_function(name)
                    .map_err(cuda_err("FLUX 3 kernel lookup"))
            };
            let blocks = blocks
                .iter()
                .map(|block| {
                    Ok(DeviceBlock {
                        qkv: Linear::upload(&gpu, &block.qkv)?,
                        out: Linear::upload(&gpu, &block.out)?,
                        ff_in: Linear::upload(&gpu, &block.ff_in)?,
                        ff_out: Linear::upload(&gpu, &block.ff_out)?,
                        norm_q: gpu.upload(&block.norm_q)?,
                        norm_k: gpu.upload(&block.norm_k)?,
                    })
                })
                .collect::<Result<Vec<_>, NetworkError>>()?;
            Ok(Self {
                context: context.clone(),
                norm_mod: get("norm_mod")?,
                gate_add: get("gate_add")?,
                head_norm_rope: get("head_norm_rope")?,
                blocks,
                prepared: None,
                dim: config.dim,
                heads: config.heads,
                layers: config.layers,
                eps: config.eps,
            })
        }

        /// One step: the velocity for `latents` at `sigma`.
        pub(super) fn run(
            &mut self,
            model: &Flux3,
            latents: &Matrix,
            sigma: f32,
        ) -> Result<Matrix, NetworkError> {
            let conditioning = model.conditioning.as_ref().ok_or_else(|| {
                NetworkError::InvalidConfig("FLUX 3 was handed no conditioning".into())
            })?;
            let image_rows = conditioning.height * conditioning.width;
            if latents.rows != image_rows || latents.cols != model.config.in_channels {
                return Err(NetworkError::InvalidTarget {
                    expected: image_rows,
                    actual: latents.rows,
                });
            }
            let gpu = Gpu {
                context: &self.context,
            };
            if self
                .prepared
                .as_ref()
                .is_none_or(|p| p.image_rows != image_rows)
            {
                self.prepared = None;
                let text = model.text_in.forward(&conditioning.text)?;
                let (cos, sin) = model.rope(conditioning.height, conditioning.width, text.rows);
                self.prepared = Some(Prepared {
                    text_rows: text.rows,
                    text: text.data,
                    image_rows,
                    cos: gpu.upload(&cos)?,
                    sin: gpu.upload(&sin)?,
                });
            }
            let prepared = self.prepared.as_ref().expect("prepared above");
            let (dim, layers, text_rows) = (self.dim, self.layers, prepared.text_rows);
            let tables = (&prepared.cos, &prepared.sin);

            let (modulation, image_time) = model.modulation(sigma)?;
            let modulation = gpu.upload(&modulation)?;
            let m = |index: usize| modulation.slice(index * 3 * dim..(index + 1) * 3 * dim);

            // The text blocks read the level the step jumps to, so they run
            // every step.
            let mut text = gpu.upload(&prepared.text)?;
            for block in &self.blocks[..layers] {
                self.block(&gpu, block, &mut text, text_rows, 0, (&m(0), &m(0), 0), tables)?;
            }
            let mut image = gpu.upload(&model.image_in.forward(latents)?.data)?;
            for block in &self.blocks[layers..2 * layers] {
                let rows = (image_rows, text_rows);
                self.block(&gpu, block, &mut image, rows.0, rows.1, (&m(1), &m(1), 0), tables)?;
            }

            let rows = text_rows + image_rows;
            let mut joint = gpu.uninit(rows * dim)?;
            let copy = || cuda_err("device to device copy");
            let stream = &self.context.stream;
            stream
                .memcpy_dtod(&text, &mut joint.slice_mut(..text_rows * dim))
                .map_err(copy())?;
            stream
                .memcpy_dtod(&image, &mut joint.slice_mut(text_rows * dim..))
                .map_err(copy())?;
            drop((text, image));
            for block in &self.blocks[2 * layers..] {
                self.block(&gpu, block, &mut joint, rows, 0, (&m(2), &m(3), text_rows), tables)?;
            }
            let hidden = gpu.download(&joint)?[text_rows * dim..].to_vec();
            model.finish(Matrix::from_vec(image_rows, dim, hidden), &image_time)
        }

        /// One block over `rows` rows of `x`, in place. Rows before `split`
        /// are modulated by `ma`, the rest by `mb`; row `r` reads the RoPE
        /// tables at `first + r`.
        #[allow(clippy::too_many_arguments)]
        fn block(
            &self,
            gpu: &Gpu<'_>,
            block: &DeviceBlock,
            x: &mut CudaSlice<f32>,
            rows: usize,
            first: usize,
            (ma, mb, split): (&CudaView<'_, f32>, &CudaView<'_, f32>, usize),
            (cos, sin): (&CudaSlice<f32>, &CudaSlice<f32>),
        ) -> Result<(), NetworkError> {
            let (dim, heads) = (self.dim, self.heads);
            let stream = &self.context.stream;

            let mut normed: Act = gpu.act(rows * dim, true)?;
            unsafe {
                stream
                    .launch_builder(&self.norm_mod)
                    .arg(normed.destination())
                    .arg(&*x)
                    .arg(ma)
                    .arg(mb)
                    .arg(&(split as i32))
                    .arg(&(rows as i32))
                    .arg(&(dim as i32))
                    .arg(&self.eps)
                    .arg(&1i32)
                    .launch(rows_config(rows))
                    .map_err(cuda_err("FLUX 3 norm kernel"))?;
            }

            let mut qkv = gpu.uninit(rows * 3 * dim)?;
            block.qkv.project(gpu, &normed, rows, &mut qkv)?;
            for (scale, offset) in [(&block.norm_q, 0), (&block.norm_k, dim)] {
                unsafe {
                    stream
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
                        .map_err(cuda_err("FLUX 3 head norm kernel"))?;
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
            block.out.project(gpu, &merged, rows, &mut attended)?;
            drop(merged);

            // The feed-forward reads the same modulated input as attention.
            let ffn = block.ff_in.units / 2;
            let mut gate_up = gpu.uninit(rows * 2 * ffn)?;
            block.ff_in.project(gpu, &normed, rows, &mut gate_up)?;
            drop(normed);
            let hidden = gpu.swiglu(&gate_up, rows, ffn, true)?;
            drop(gate_up);
            let mut fed = gpu.uninit(rows * dim)?;
            block.ff_out.project(gpu, &hidden, rows, &mut fed)?;

            unsafe {
                stream
                    .launch_builder(&self.gate_add)
                    .arg(x)
                    .arg(&attended)
                    .arg(&fed)
                    .arg(ma)
                    .arg(mb)
                    .arg(&(split as i32))
                    .arg(&(rows as i32))
                    .arg(&(dim as i32))
                    .launch(cfg(rows * dim))
                    .map_err(cuda_err("FLUX 3 gated residual kernel"))?;
            }
            Ok(())
        }
    }
}

#[cfg(all(test, feature = "cuda", feature = "images"))]
mod tests {
    use crate::mmdit::Conditioning;
    use crate::pipeline::ImagePipeline;
    use crate::transformer::Precision;

    /// FLUX 3 Image Klein end to end, 16 steps, Q8 on device 0. There is no
    /// reference to compare with, so this only checks that a picture comes
    /// out; `RB_FLUX3_OUT=<png>` saves it to look at. `RB_FLUX3_MODEL`,
    /// `RB_FLUX3_PROMPT`, `RB_FLUX3_SIZE` (default 1024) and `RB_FLUX3_SEED`
    /// set the rest.
    #[test]
    #[ignore = "needs FLUX 3 Image Klein and a CUDA device"]
    fn flux3_klein_renders() {
        let env = |name: &str| std::env::var(name).ok();
        let model = env("RB_FLUX3_MODEL")
            .unwrap_or_else(|| "/mnt/ollama/RustingFrameModels/flux-3-image-klein".into());
        let prompt = env("RB_FLUX3_PROMPT")
            .unwrap_or_else(|| "a lighthouse on a rocky coast in fog".into());
        let number = |name: &str, default: u64| env(name).map_or(default, |v| v.parse().unwrap());
        let grid = number("RB_FLUX3_SIZE", 1024) as usize / 16;
        let seed = Some(number("RB_FLUX3_SEED", 7));

        let started = std::time::Instant::now();
        let mut pipeline = ImagePipeline::load_at(&model, Precision::Q8).unwrap();
        eprintln!("flux3 load: {:?}", started.elapsed());
        assert_eq!(pipeline.config.steps, 16);
        let started = std::time::Instant::now();
        pipeline.to_cuda(0).unwrap();
        eprintln!("flux3 to device: {:?}", started.elapsed());

        let started = std::time::Instant::now();
        let (text, _) = pipeline.encoder.encode(&prompt).unwrap();
        eprintln!("flux3 encode: {:?}, {} rows", started.elapsed(), text.rows);
        assert_eq!((text.rows % 80, text.cols), (0, 20480));
        let conditioning = Conditioning {
            text,
            pooled: Vec::new(),
            height: grid,
            width: grid,
            guidance: 0.0,
        };
        let noise = crate::diffusion::noise(grid * grid, 128, seed);
        let started = std::time::Instant::now();
        let latent = pipeline
            .sample_latent_from_noise(conditioning, None, noise, seed, |_, _| true)
            .unwrap();
        eprintln!("flux3 16 steps: {:?}", started.elapsed());
        let started = std::time::Instant::now();
        let image = pipeline.decode(&latent, grid, grid).unwrap();
        eprintln!("flux3 decode: {:?}", started.elapsed());

        let n = image.data.len() as f32;
        let mean = image.data.iter().sum::<f32>() / n;
        let spread = (image.data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
        eprintln!("flux3 image: mean {mean:.3}, spread {spread:.3}");
        if let Some(out) = env("RB_FLUX3_OUT") {
            crate::vae::save_png(&image, out).unwrap();
        }
        assert!(spread.is_finite() && spread > 0.05, "a flat or broken picture");
    }
}
