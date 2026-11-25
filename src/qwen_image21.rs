//! Qwen-Image 2.1's denoiser (Alibaba, Qwen Research License): diffusers'
//! `QwenImage21Transformer2DModel`.
//!
//! One stream of 4096-wide blocks over `[text, image]`. Each block is a layer
//! norm, a scale, attention, a gated residual, then the same around a SwiGLU
//! feed-forward. Every block reads the same four modulation vectors, made
//! from the step's noise level; the text rows read the ones for level zero.
//!
//! Attention is block-causal: a text row sees the rows before it, an image
//! row sees those and its own picture. So nothing the text rows compute
//! depends on the image or the step, and they run once per prompt; a step
//! runs the image rows alone against the keys and values kept from that pass.
//!
//! An edit puts each picture it starts from into the text, as latent cells
//! where the prompt names the picture ([`Reference`]). Those rows are read at
//! level zero and kept with the text's.
//!
//! The blocks run on CUDA only, Q8 on the device, through [`crate::zimage`]'s
//! kernels; the small parts run on the host in F32.

// Without CUDA the model loads but cannot run, so most of it is unused.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use crate::conv::Dense;
use crate::diffusion::Denoiser;
use crate::ffn::{gelu, silu};
use crate::matrix::Matrix;
use crate::mmdit::Conditioning;
use crate::network::NetworkError;
use crate::safetensors::ShardedSafeTensors;
use crate::transformer::Precision;
use crate::zimage::{linear, stacked};
use rayon::prelude::*;

/// The parts of `transformer/config.json` the model reads.
#[derive(Clone, Debug, PartialEq)]
pub struct QwenImage21Config {
    pub dim: usize,
    pub heads: usize,
    pub layers: usize,
    pub in_channels: usize,
    pub text_dim: usize,
    pub eps: f32,
    /// RoPE widths of the `(frame, height, width)` position axes.
    pub axes: Vec<usize>,
}

impl QwenImage21Config {
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
            in_channels: number("in_channels")? as usize,
            text_dim: number("context_in_dim")? as usize,
            eps: number("eps")? as f32,
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
        if config.axes.len() != 3 || config.axes.iter().sum::<usize>() != head_dim {
            return Err(NetworkError::InvalidConfig(format!(
                "{}: the RoPE axes {:?} are not three that fill a {head_dim}-wide head",
                path.display(),
                config.axes
            )));
        }
        // What running the text rows once rests on.
        if json["causal_condition"].as_bool() != Some(true) || number("patch_size")? != 1.0 {
            return Err(NetworkError::InvalidConfig(format!(
                "{}: only the causal-condition, unpatched Qwen-Image 2.1 is supported",
                path.display()
            )));
        }
        Ok(config)
    }
}

/// One block's linears and its query and key norms, held on the host until
/// [`QwenImage21::attach_device`] uploads them.
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

/// A picture an edit starts from.
#[derive(Clone, Debug, PartialEq)]
pub struct Reference {
    /// How many rows of the conditioning's text come before it.
    pub at: usize,
    /// Its normalised latent cells in reading order, `[height * width,
    /// in_channels]`.
    pub latents: Matrix,
    pub height: usize,
    pub width: usize,
}

/// Qwen-Image 2.1's transformer.
pub struct QwenImage21 {
    config: QwenImage21Config,
    /// `linear_1` and `linear_2` of the timestep embedder.
    time: [Dense; 2],
    /// Makes `[scale, gate]` for attention, then for the feed-forward.
    modulation: Dense,
    /// Stored as `scale - 1`.
    text_norm: Vec<f32>,
    text_in: [Dense; 2],
    image_in: Dense,
    final_scale: Dense,
    final_out: Dense,
    /// Emptied once they are on the device.
    blocks: Vec<Block>,
    conditioning: Option<Conditioning>,
    references: Vec<Reference>,
    #[cfg(feature = "cuda")]
    device: Option<device::DeviceQwenImage21>,
}

impl std::fmt::Debug for QwenImage21 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenImage21")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl QwenImage21 {
    /// Reads the transformer. The block linears are stored at `precision`
    /// (Q8 grouped unless F32); the rest stays F32, about 0.5 GB.
    pub fn read(
        file: &mut ShardedSafeTensors,
        config: QwenImage21Config,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        let vector = |file: &mut ShardedSafeTensors, name: &str| file.tensor(name).map(|(v, _)| v);
        let mut blocks = Vec::new();
        for index in 0..config.layers {
            let at = |part: &str| format!("transformer_blocks.{index}.{part}");
            blocks.push(Block {
                qkv: stacked(
                    file,
                    &[at("attn.to_q"), at("attn.to_k"), at("attn.to_v")],
                    precision,
                )?,
                out: stacked(file, &[at("attn.to_out.0")], precision)?,
                ff_in: stacked(
                    file,
                    &[at("img_mlp.gate_layer"), at("img_mlp.proj")],
                    precision,
                )?,
                ff_out: stacked(file, &[at("img_mlp.out")], precision)?,
                norm_q: vector(file, &at("attn.norm_q.weight"))?,
                norm_k: vector(file, &at("attn.norm_k.weight"))?,
            });
        }
        Ok(Self {
            time: [
                linear(file, "time_text_embed.timestep_embedder.linear_1")?,
                linear(file, "time_text_embed.timestep_embedder.linear_2")?,
            ],
            modulation: linear(file, "modulation.1")?,
            text_norm: vector(file, "txt_in.text_norm.weight")?,
            text_in: [
                linear(file, "txt_in.in_layer")?,
                linear(file, "txt_in.out_layer")?,
            ],
            image_in: linear(file, "img_in")?,
            final_scale: linear(file, "norm_out.linear")?,
            final_out: linear(file, "proj_out")?,
            blocks,
            config,
            conditioning: None,
            references: Vec::new(),
            #[cfg(feature = "cuda")]
            device: None,
        })
    }

    pub fn config(&self) -> &QwenImage21Config {
        &self.config
    }

    pub fn set_conditioning(&mut self, conditioning: Conditioning) {
        self.conditioning = Some(conditioning);
        self.release();
    }

    /// The pictures the next conditioning edits, in the order its text names
    /// them; none for text-to-image. The text has to be the one encoded with
    /// these pictures in it.
    pub fn set_references(&mut self, references: Vec<Reference>) {
        self.references = references;
        self.release();
    }

    /// Frees what the last conditioning left on the device: about 2 GB after
    /// an edit of a 1024x1024 picture. The next step builds it again.
    pub fn release(&mut self) {
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
        self.device = Some(device::DeviceQwenImage21::upload(
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

    /// The time embedding of a noise level.
    fn time(&self, sigma: f32) -> Result<Vec<f32>, NetworkError> {
        let table = crate::mmdit::sinusoid(sigma * 1000.0, self.time[0].in_dim());
        let hidden: Vec<f32> = self.time[0].apply(&table)?.into_iter().map(silu).collect();
        self.time[1].apply(&hidden)
    }

    /// `[scale, gate, scale, gate]`, attention's pair then the
    /// feed-forward's, which every block reads.
    fn modulation(&self, time: &[f32]) -> Result<Vec<f32>, NetworkError> {
        let active: Vec<f32> = time.iter().copied().map(silu).collect();
        self.modulation.apply(&active)
    }

    /// The prompt states as the blocks read them: an RMS norm, then a
    /// two-layer projection.
    fn text(&self, states: &Matrix) -> Result<Matrix, NetworkError> {
        let eps = self.config.eps;
        let mut normed = states.clone();
        normed.data.par_chunks_mut(normed.cols).for_each(|row| {
            let mean = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
            let inv = (mean + eps).sqrt().recip();
            for (value, weight) in row.iter_mut().zip(&self.text_norm) {
                *value *= inv * (weight + 1.0);
            }
        });
        let mut hidden = self.text_in[0].forward(&normed)?;
        hidden.data.par_iter_mut().for_each(|v| *v = gelu(*v));
        self.text_in[1].forward(&hidden)
    }

    /// The final layer over the image rows: the velocity, one row a latent
    /// cell.
    fn finish(&self, hidden: Matrix, time: &[f32]) -> Result<Matrix, NetworkError> {
        let active: Vec<f32> = time.iter().copied().map(silu).collect();
        let scale = self.final_scale.apply(&active)?;
        let eps = self.config.eps;
        let mut normed = hidden;
        normed.data.par_chunks_mut(normed.cols).for_each(|row| {
            let n = row.len() as f32;
            let mean = row.iter().sum::<f32>() / n;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let inv = (var + eps).sqrt().recip();
            for (value, scale) in row.iter_mut().zip(&scale) {
                *value = (*value - mean) * inv * (1.0 + scale);
            }
        });
        self.final_out.forward(&normed)
    }

    /// The RoPE `cos` and `sin` tables, `[rows, head_dim / 2]`. `pictures` is
    /// each picture's `(first row, height, width)` in row order; the other
    /// rows are text. A text row sits one past the row before it on all three
    /// axes. A picture's rows share the frame the text reached, on a grid
    /// centred on zero, and the text after it resumes `max(height, width)`
    /// later.
    fn rope(&self, rows: usize, pictures: &[(usize, usize, usize)]) -> (Vec<f32>, Vec<f32>) {
        let half = self.config.dim / self.config.heads / 2;
        let frequencies: Vec<(usize, f64)> = self
            .config
            .axes
            .iter()
            .enumerate()
            .flat_map(|(axis, &dim)| (0..dim / 2).map(move |j| (axis, (2 * j) as f64 / dim as f64)))
            .map(|(axis, exponent)| (axis, 10_000f64.powf(-exponent)))
            .collect();
        let centred = |index: usize, side: usize| index as f64 - (side - side / 2) as f64;
        let mut positions = Vec::with_capacity(rows);
        let mut next = 0;
        let text = |positions: &mut Vec<[f64; 3]>, until: usize, next: &mut usize| {
            while positions.len() < until {
                positions.push([*next as f64; 3]);
                *next += 1;
            }
        };
        for &(first, height, width) in pictures {
            text(&mut positions, first, &mut next);
            positions.extend((0..height * width).map(|cell| {
                [
                    next as f64,
                    centred(cell / width, height),
                    centred(cell % width, width),
                ]
            }));
            next += height.max(width);
        }
        text(&mut positions, rows, &mut next);
        let (mut cos, mut sin) = (vec![0.0; rows * half], vec![0.0; rows * half]);
        for (row, position) in positions.iter().enumerate() {
            for (p, &(axis, frequency)) in frequencies.iter().enumerate() {
                let angle = position[axis] * frequency;
                cos[row * half + p] = angle.cos() as f32;
                sin[row * half + p] = angle.sin() as f32;
            }
        }
        (cos, sin)
    }
}

impl Denoiser for QwenImage21 {
    fn denoise(&mut self, latents: &Matrix, sigma: f32) -> Result<Matrix, NetworkError> {
        #[cfg(feature = "cuda")]
        if let Some(mut device) = self.device.take() {
            let result = device.run(self, latents, sigma);
            self.device = Some(device);
            return result;
        }
        let _ = (latents, sigma);
        Err(NetworkError::InvalidConfig(
            "Qwen-Image 2.1 runs on CUDA only: move the pipeline to a device first".into(),
        ))
    }
}

#[cfg(feature = "cuda")]
mod device {
    use super::{Block, QwenImage21, QwenImage21Config};
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
    /// these use.
    const KERNELS: &str = r#"
// `out = layer_norm(x) * (1 + scale)`, BF16 if `narrow`. One block per row.
extern "C" __global__ void layer_norm_scale(void*out,const float*x,const float*scale,int rows,int cols,float eps,int narrow){
  __shared__ float red[ROW_THREADS];int tid=threadIdx.x;
  for(int r=blockIdx.x;r<rows;r+=gridDim.x){
    const float*s=x+(size_t)r*cols;
    float a=0.f;for(int c=tid;c<cols;c+=ROW_THREADS)a+=s[c];
    float mean=row_sum(a,red)/cols;
    a=0.f;for(int c=tid;c<cols;c+=ROW_THREADS){float d=s[c]-mean;a+=d*d;}
    float inv=rsqrtf(row_sum(a,red)/cols+eps);
    for(int c=tid;c<cols;c+=ROW_THREADS)st(out,(size_t)r*cols+c,(s[c]-mean)*inv*(1.f+scale[c]),narrow);
  }
}
// `x += tanh(gate) * y`.
extern "C" __global__ void tanh_gate_add(float*x,const float*y,const float*gate,int rows,int cols){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(size_t)rows*cols)return;
  x[i]+=tanhf(gate[i%cols])*y[i];
}
// Columns `[off, off + n)` of each `stride`-wide row of `x`, as BF16 rows of `n`.
extern "C" __global__ void narrow_cols(void*out,const float*x,int rows,int stride,int off,int n){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(size_t)rows*n)return;
  st(out,i,x[(i/n)*stride+off+i%n],1);
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

    /// What one conditioning fixes: every block's keys and values for the
    /// rows before the image (the text, and the pictures an edit starts
    /// from), and the RoPE tables.
    pub(super) struct Prepared {
        /// Per block, those rows' `[k, v]` after the head norm and RoPE,
        /// BF16.
        kept: Vec<Act>,
        rows: usize,
        grid: (usize, usize),
        cos: CudaSlice<f32>,
        sin: CudaSlice<f32>,
    }

    pub(super) struct DeviceQwenImage21 {
        context: Arc<GpuContext>,
        layer_norm_scale: CudaFunction,
        tanh_gate_add: CudaFunction,
        narrow_cols: CudaFunction,
        head_norm_rope: CudaFunction,
        blocks: Vec<DeviceBlock>,
        pub(super) prepared: Option<Prepared>,
        dim: usize,
        heads: usize,
        eps: f32,
    }

    impl DeviceQwenImage21 {
        pub(super) fn upload(
            context: &Arc<GpuContext>,
            config: &QwenImage21Config,
            blocks: &[Block],
        ) -> Result<Self, NetworkError> {
            if context.half_accumulate
                || context.flash.is_none()
                || !crate::cuda_flash::eligible(true, config.dim / config.heads)
            {
                return Err(NetworkError::InvalidConfig(
                    "Qwen-Image 2.1 needs a BF16 context with the fused attention kernel \
                     (Ampere or later)"
                        .into(),
                ));
            }
            if blocks.len() != config.layers {
                return Err(NetworkError::InvalidConfig(
                    "Qwen-Image 2.1's blocks are no longer on the host".into(),
                ));
            }
            let gpu = Gpu { context };
            let module = context
                .stream
                .context()
                .load_module(
                    compile_ptx(&format!("{SHARED}{KERNELS}")).map_err(|error| {
                        NetworkError::Cuda(format!(
                            "Qwen-Image 2.1 kernel compilation failed: {error}"
                        ))
                    })?,
                )
                .map_err(cuda_err("Qwen-Image 2.1 module loading"))?;
            let get = |name: &str| {
                module
                    .load_function(name)
                    .map_err(cuda_err("Qwen-Image 2.1 kernel lookup"))
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
                layer_norm_scale: get("layer_norm_scale")?,
                tanh_gate_add: get("tanh_gate_add")?,
                narrow_cols: get("narrow_cols")?,
                head_norm_rope: get("head_norm_rope")?,
                blocks,
                prepared: None,
                dim: config.dim,
                heads: config.heads,
                eps: config.eps,
            })
        }

        /// One step: the velocity for `latents` at `sigma`.
        pub(super) fn run(
            &mut self,
            model: &QwenImage21,
            latents: &Matrix,
            sigma: f32,
        ) -> Result<Matrix, NetworkError> {
            let conditioning = model.conditioning.as_ref().ok_or_else(|| {
                NetworkError::InvalidConfig("Qwen-Image 2.1 was handed no conditioning".into())
            })?;
            let grid = (conditioning.height, conditioning.width);
            let image_rows = grid.0 * grid.1;
            if latents.rows != image_rows || latents.cols != model.config.in_channels {
                return Err(NetworkError::InvalidTarget {
                    expected: image_rows,
                    actual: latents.rows,
                });
            }
            let gpu = Gpu {
                context: &self.context,
            };
            if self.prepared.as_ref().is_none_or(|p| p.grid != grid) {
                self.prepared = None;
                let text = model.text(&conditioning.text)?;
                // The rows before the image: the text, with each picture an
                // edit starts from where the prompt names it.
                let dim = self.dim;
                let mut prefix = Vec::new();
                let mut pictures = Vec::new();
                let mut taken = 0;
                for reference in &model.references {
                    let cells = reference.height * reference.width;
                    if reference.at < taken
                        || reference.at > text.rows
                        || reference.latents.rows != cells
                    {
                        return Err(NetworkError::InvalidConfig(
                            "a Qwen-Image 2.1 reference does not fit its conditioning".into(),
                        ));
                    }
                    prefix.extend_from_slice(&text.data[taken * dim..reference.at * dim]);
                    pictures.push((prefix.len() / dim, reference.height, reference.width));
                    prefix.extend_from_slice(&model.image_in.forward(&reference.latents)?.data);
                    taken = reference.at;
                }
                prefix.extend_from_slice(&text.data[taken * dim..]);
                let rows = prefix.len() / dim;
                let spans: Vec<(usize, usize)> = pictures
                    .iter()
                    .map(|&(first, height, width)| (first, first + height * width))
                    .collect();
                pictures.push((rows, grid.0, grid.1));
                let (cos, sin) = model.rope(rows + image_rows, &pictures);
                let (cos, sin) = (gpu.upload(&cos)?, gpu.upload(&sin)?);
                // These rows sit at level zero whatever the step.
                let modulation = gpu.upload(&model.modulation(&model.time(0.0)?)?)?;
                let mut x = gpu.upload(&prefix)?;
                let kept = self
                    .blocks
                    .iter()
                    .map(|block| {
                        let tables = (&cos, &sin);
                        self.block(&gpu, block, &mut x, rows, &modulation, tables, None, &spans)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.prepared = Some(Prepared {
                    kept,
                    rows,
                    grid,
                    cos,
                    sin,
                });
            }
            let prepared = self.prepared.as_ref().expect("prepared above");
            let tables = (&prepared.cos, &prepared.sin);

            let time = model.time(sigma)?;
            let modulation = gpu.upload(&model.modulation(&time)?)?;
            let mut x = gpu.upload(&model.image_in.forward(latents)?.data)?;
            for (block, kept) in self.blocks.iter().zip(&prepared.kept) {
                let kept = Some((kept, prepared.rows));
                self.block(&gpu, block, &mut x, image_rows, &modulation, tables, kept, &[])?;
            }
            let hidden = gpu.download(&x)?;
            model.finish(Matrix::from_vec(image_rows, self.dim, hidden), &time)
        }

        /// One block over `rows` rows of `x`, in place; the rows' `[k, v]` as
        /// attention read them come back.
        ///
        /// With nothing `kept` these are the rows before the image: causal
        /// over themselves, but for the `pictures` among them, `(first row,
        /// end)` each, whose rows also see the rest of their picture. With
        /// `(the kept [k, v], its rows)`, they are the image rows, which
        /// follow the kept ones and see all of them and each other.
        #[allow(clippy::too_many_arguments)]
        fn block(
            &self,
            gpu: &Gpu<'_>,
            block: &DeviceBlock,
            x: &mut CudaSlice<f32>,
            rows: usize,
            modulation: &CudaSlice<f32>,
            (cos, sin): (&CudaSlice<f32>, &CudaSlice<f32>),
            kept: Option<(&Act, usize)>,
            pictures: &[(usize, usize)],
        ) -> Result<Act, NetworkError> {
            let (dim, heads) = (self.dim, self.heads);
            let stream = &self.context.stream;
            let part = |index: usize| modulation.slice(index * dim..(index + 1) * dim);
            let before = kept.map_or(0, |(_, rows)| rows);

            let normed = self.norm(gpu, x, &part(0), rows)?;
            let mut wide = gpu.uninit(rows * 3 * dim)?;
            block.qkv.project(gpu, &normed, rows, &mut wide)?;
            drop(normed);
            for (scale, offset) in [(&block.norm_q, 0), (&block.norm_k, dim)] {
                unsafe {
                    stream
                        .launch_builder(&self.head_norm_rope)
                        .arg(&mut wide)
                        .arg(scale)
                        .arg(cos)
                        .arg(sin)
                        .arg(&(rows as i32))
                        .arg(&(heads as i32))
                        .arg(&((dim / heads) as i32))
                        .arg(&((3 * dim) as i32))
                        .arg(&(offset as i32))
                        .arg(&(before as i32))
                        .arg(&self.eps)
                        .launch(cfg(rows * heads * 32))
                        .map_err(cuda_err("Qwen-Image 2.1 head norm kernel"))?;
                }
            }
            // The keys and values: these rows', then the kept ones. Their
            // order does not matter once RoPE is in them.
            let narrow = Act::element(true);
            let queries = self.narrow(gpu, &wide.slice(..), rows, 0, dim)?;
            let mut keys = gpu.act((rows + before) * 2 * dim, true)?;
            self.narrow_into(&mut keys, &wide.slice(..), rows, dim, 2 * dim)?;
            if let Some((kept, _)) = kept {
                let bytes = rows * 2 * dim * narrow;
                stream
                    .memcpy_dtod(&kept.all(), &mut keys.destination().slice_mut(bytes..))
                    .map_err(cuda_err("device to device copy"))?;
            }
            let mut merged = gpu.act(rows * dim, true)?;
            let shape = FlashShape {
                rows,
                seq_len: rows,
                kv_len: rows + before,
                sequences: 1,
                heads,
                group: 1,
                qkv_width: dim,
                kv_width: 2 * dim,
                query_width: dim,
                key_base: 0,
                value_base: dim,
                causal: kept.is_none(),
            };
            self.attend(gpu, &queries, &keys, &mut merged, shape)?;
            drop(queries);
            // ponytail: a picture's rows are attended twice, causally above
            // and here without the mask over the keys up to the picture's
            // end, once per conditioning; hand the kernel a query offset if
            // that pass ever shows.
            for &(first, end) in pictures {
                let count = end - first;
                let queries = self.narrow(gpu, &wide.slice(first * 3 * dim..), count, 0, dim)?;
                let mut seen = gpu.act(count * dim, true)?;
                let shape = FlashShape {
                    rows: count,
                    seq_len: count,
                    kv_len: end,
                    causal: false,
                    ..shape
                };
                self.attend(gpu, &queries, &keys, &mut seen, shape)?;
                let bytes = first * dim * narrow..end * dim * narrow;
                stream
                    .memcpy_dtod(&seen.all(), &mut merged.destination().slice_mut(bytes))
                    .map_err(cuda_err("device to device copy"))?;
            }
            drop(wide);
            let mut attended = gpu.uninit(rows * dim)?;
            block.out.project(gpu, &merged, rows, &mut attended)?;
            drop(merged);
            self.gate_add(x, &attended, &part(1), rows)?;
            drop(attended);

            // The feed-forward reads what attention left.
            let normed = self.norm(gpu, x, &part(2), rows)?;
            let ffn = block.ff_in.units / 2;
            let mut gate_up = gpu.uninit(rows * 2 * ffn)?;
            block.ff_in.project(gpu, &normed, rows, &mut gate_up)?;
            drop(normed);
            let hidden = gpu.swiglu(&gate_up, rows, ffn, true)?;
            drop(gate_up);
            let mut fed = gpu.uninit(rows * dim)?;
            block.ff_out.project(gpu, &hidden, rows, &mut fed)?;
            self.gate_add(x, &fed, &part(3), rows)?;
            Ok(keys)
        }

        /// Attention of `queries` over `keys`, `[k, v]` a row, into `merged`.
        fn attend(
            &self,
            gpu: &Gpu<'_>,
            queries: &Act,
            keys: &Act,
            merged: &mut Act,
            shape: FlashShape,
        ) -> Result<(), NetworkError> {
            let mut log_sum_exp = gpu.uninit(shape.heads * shape.rows)?;
            let flash = self.context.flash.as_ref().expect("checked at upload");
            let scale = ((self.dim / self.heads) as f32).sqrt().recip();
            gpu.flash_attention(flash, queries, keys, merged, &mut log_sum_exp, shape, scale)
        }

        /// Columns `[offset, offset + width)` of `rows` rows of a `[q, k, v]`
        /// buffer, BF16.
        fn narrow(
            &self,
            gpu: &Gpu<'_>,
            wide: &CudaView<'_, f32>,
            rows: usize,
            offset: usize,
            width: usize,
        ) -> Result<Act, NetworkError> {
            let mut narrowed = gpu.act(rows * width, true)?;
            self.narrow_into(&mut narrowed, wide, rows, offset, width)?;
            Ok(narrowed)
        }

        /// [`Self::narrow`] into the start of `out`.
        fn narrow_into(
            &self,
            out: &mut Act,
            wide: &CudaView<'_, f32>,
            rows: usize,
            offset: usize,
            width: usize,
        ) -> Result<(), NetworkError> {
            unsafe {
                self.context
                    .stream
                    .launch_builder(&self.narrow_cols)
                    .arg(out.destination())
                    .arg(wide)
                    .arg(&(rows as i32))
                    .arg(&((3 * self.dim) as i32))
                    .arg(&(offset as i32))
                    .arg(&(width as i32))
                    .launch(cfg(rows * width))
                    .map_err(cuda_err("Qwen-Image 2.1 narrowing kernel"))?;
            }
            Ok(())
        }

        /// `layer_norm(x) * (1 + scale)`, BF16.
        fn norm(
            &self,
            gpu: &Gpu<'_>,
            x: &CudaSlice<f32>,
            scale: &CudaView<'_, f32>,
            rows: usize,
        ) -> Result<Act, NetworkError> {
            let mut normed = gpu.act(rows * self.dim, true)?;
            unsafe {
                self.context
                    .stream
                    .launch_builder(&self.layer_norm_scale)
                    .arg(normed.destination())
                    .arg(x)
                    .arg(scale)
                    .arg(&(rows as i32))
                    .arg(&(self.dim as i32))
                    .arg(&self.eps)
                    .arg(&1i32)
                    .launch(rows_config(rows))
                    .map_err(cuda_err("Qwen-Image 2.1 norm kernel"))?;
            }
            Ok(normed)
        }

        /// `x += tanh(gate) * y`.
        fn gate_add(
            &self,
            x: &mut CudaSlice<f32>,
            y: &CudaSlice<f32>,
            gate: &CudaView<'_, f32>,
            rows: usize,
        ) -> Result<(), NetworkError> {
            unsafe {
                self.context
                    .stream
                    .launch_builder(&self.tanh_gate_add)
                    .arg(x)
                    .arg(y)
                    .arg(gate)
                    .arg(&(rows as i32))
                    .arg(&(self.dim as i32))
                    .launch(cfg(rows * self.dim))
                    .map_err(cuda_err("Qwen-Image 2.1 gated residual kernel"))?;
            }
            Ok(())
        }
    }
}

#[cfg(all(test, feature = "cuda", feature = "images"))]
mod tests {
    use crate::pipeline::ImagePipeline;
    use crate::transformer::Precision;

    /// Qwen-Image 2.1 end to end, Q8 on device 0. There is no reference to
    /// compare with, so this only checks that a picture comes out;
    /// `RB_QWEN_IMAGE_OUT=<png>` saves it to look at. `RB_QWEN_IMAGE_MODEL`,
    /// `RB_QWEN_IMAGE_PROMPT`, `RB_QWEN_IMAGE_SIZE` (default 1024),
    /// `RB_QWEN_IMAGE_STEPS` (default the model's own) and `RB_QWEN_IMAGE_SEED`
    /// set the rest, and `RB_QWEN_IMAGE_LORA` merges a LoRA file.
    /// `RB_QWEN_IMAGE_REFERENCE=<picture>` makes it an edit of that picture,
    /// resized to the output's size, or to `RB_QWEN_IMAGE_REFERENCE_SIZE`.
    #[test]
    #[ignore = "needs Qwen-Image 2.1 and a CUDA device"]
    fn qwen_image21_renders() {
        let env = |name: &str| std::env::var(name).ok();
        let model = env("RB_QWEN_IMAGE_MODEL")
            .unwrap_or_else(|| "/home/vasylt/models/Qwen-Image-2.1".into());
        let prompt = env("RB_QWEN_IMAGE_PROMPT")
            .unwrap_or_else(|| "a lighthouse on a rocky coast in fog".into());
        let number = |name: &str, default: u64| env(name).map_or(default, |v| v.parse().unwrap());
        let size = number("RB_QWEN_IMAGE_SIZE", 1024) as usize;
        let grid = size / 16;
        let seed = Some(number("RB_QWEN_IMAGE_SEED", 7));

        let started = std::time::Instant::now();
        let loras: Vec<(std::path::PathBuf, f32)> = env("RB_QWEN_IMAGE_LORA")
            .map(|path| (path.into(), 1.0))
            .into_iter()
            .collect();
        let mut pipeline = ImagePipeline::load_with_loras(&model, Precision::Q8, &loras).unwrap();
        eprintln!("qwen-image load: {:?}", started.elapsed());
        let steps = pipeline.config.steps as u64;
        pipeline.config.steps = number("RB_QWEN_IMAGE_STEPS", steps) as usize;
        let started = std::time::Instant::now();
        pipeline.to_cuda(0).unwrap();
        eprintln!("qwen-image to device: {:?}", started.elapsed());

        if let Some(reference) = env("RB_QWEN_IMAGE_REFERENCE") {
            let started = std::time::Instant::now();
            let bytes = std::fs::read(reference).unwrap();
            let side = number("RB_QWEN_IMAGE_REFERENCE_SIZE", size as u64) as usize;
            let picture = crate::vae::from_image_bytes(&bytes, side, side).unwrap();
            pipeline.set_references(&[picture]).unwrap();
            eprintln!("qwen-image reference: {:?}", started.elapsed());
        }
        let started = std::time::Instant::now();
        let conditioning = pipeline.condition(&prompt, size, size).unwrap();
        let rows = conditioning.text.rows;
        eprintln!("qwen-image encode: {:?}, {rows} rows", started.elapsed());
        assert_eq!(conditioning.text.cols, 4096);
        let noise = crate::diffusion::noise(grid * grid, 64, seed);
        let started = std::time::Instant::now();
        let latent = pipeline
            .sample_latent_from_noise(conditioning, None, noise, seed, |_, _| true)
            .unwrap();
        eprintln!(
            "qwen-image {} steps: {:?}",
            pipeline.config.steps,
            started.elapsed()
        );
        let started = std::time::Instant::now();
        let image = pipeline.decode(&latent, grid, grid).unwrap();
        eprintln!("qwen-image decode: {:?}", started.elapsed());
        assert_eq!(image.channels, 3);

        let n = image.data.len() as f32;
        let mean = image.data.iter().sum::<f32>() / n;
        let spread = (image.data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
        eprintln!("qwen-image image: mean {mean:.3}, spread {spread:.3}");
        if let Some(out) = env("RB_QWEN_IMAGE_OUT") {
            crate::vae::save_png(&image, out).unwrap();
        }
        assert!(spread.is_finite() && spread > 0.05, "a flat or broken picture");
    }
}
