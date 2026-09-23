//! A deep-learning library in Rust, from a two-layer XOR network up to a
//! 300M-parameter transformer language model.
//!
//! There are two model types, and they share the matrix, optimizer, and
//! serialization code underneath:
//!
//! - [`Network`] — a dense feed-forward network for tabular regression and
//!   classification. Built with [`Network::builder`], trained with
//!   [`Network::fit`].
//! - [`TransformerLm`] — a decoder-only language model with grouped-query
//!   attention, rotary positions, RMSNorm, SwiGLU feed-forwards, and optional
//!   sparse mixture-of-experts layers. Built with [`TransformerLm::builder`],
//!   trained one batch at a time with [`TransformerLm::train_step`].
//!
//! # A language model in twenty lines
//!
//! ```no_run
//! use rusting_brain::{Optimizer, TransformerLm};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut model = TransformerLm::builder()
//!     .vocab_size(32_000)
//!     .d_model(512)
//!     .n_layers(8)
//!     .heads(8, 2, 64)          // 8 query heads, 2 key/value heads (GQA)
//!     .d_ff(1408)
//!     .experts(8, 2)            // 8 experts, 2 active per token
//!     .moe_layers(2..8)         // layers 0-1 stay dense
//!     .max_seq_len(1024)
//!     .optimizer(Optimizer::adam(3e-4))
//!     .seed(42)
//!     .build()?;
//!
//! println!("{}", model.parameter_counts());   // total vs. active
//!
//! let batch: Vec<Vec<u32>> = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
//! let loss = model.train_step(&batch)?;
//! println!("loss {:.4}", loss.lm_loss);
//! # Ok(())
//! # }
//! ```
//!
//! # Generating text
//!
//! [`TransformerLm::generate`] prefills a per-layer [`KvCache`] with the prompt
//! and then decodes one token at a time, so the *n*-th token costs one row of
//! attention instead of *n*. [`Sampler`] decides which token that is.
//!
//! ```no_run
//! # use rusting_brain::{Sampler, TransformerLm};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let model = TransformerLm::builder().build()?;
//! # let prompt = [1u32, 2, 3];
//! let mut sampler = Sampler::temperature(0.8, Some(42)).top_k(40).top_p(0.95);
//! let generated = model.generate(&prompt, 50, &mut sampler)?;
//! # Ok(())
//! # }
//! ```
//!
//! [`TransformerLm::forward_cached`] is the cache underneath, for driving
//! decoding by hand.
//!
//! # Running on a GPU
//!
//! CUDA is an optional feature and is off by default; the crate builds and the
//! tests pass without a driver, a toolkit, or a device.
//!
//! ```bash
//! cargo add rusting_brain --features cuda
//! ```
//!
//! [`TransformerLm::to_cuda`] moves the parameters onto a device and every
//! later `train_step`, `forward_batch`, and `backward` runs there. The device
//! path is fail-closed: if the driver, cuBLAS, a kernel, an allocation, or a
//! numerical check fails, the call returns [`NetworkError`] rather than
//! silently falling back to the CPU.
//!
//! ```no_run
//! # use rusting_brain::TransformerLm;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let mut model = TransformerLm::builder().build()?;
//! # #[cfg(feature = "cuda")] {
//! model.set_mixed_precision(true);      // BF16 GEMMs, FP32 master weights
//! model.to_cuda(0, 9_000)?;             // device 0, 9000 MiB budget
//! # }
//! # Ok(())
//! # }
//! ```
//!
//! Dense [`Network`]s use a separate device path,
//! [`TrainingBackend::Cuda`], or [`metal_training`] on Apple Silicon.
//!
//! # Where to read next
//!
//! - `tutorials/` — a sixteen-chapter course, from what a neural network is to
//!   training a language model end to end.
//! - `docs/baseline.md` — measured throughput per architecture on an RTX 3060.
//! - `IMPORT_MODELS.md` — running ONNX models exported from TensorFlow or
//!   PyTorch.

pub mod accelerator;
pub mod activations;
pub mod adaln;
pub mod attention;
pub mod batch;
pub mod causal_lm_loss;
pub mod checkpoint;
pub mod clip;
pub mod conv;
pub mod dataset;
pub mod diffusion;
pub mod ema;
pub mod embedding;
pub mod ffn;
pub mod flow_transformer;
pub mod gltf;
pub mod losses;
pub mod masked_lm;
pub mod matrix;
pub mod mesh;
pub mod mmdit;
pub mod moe;
pub mod network;
pub mod norm;
pub mod optimizers;
pub mod param;
pub mod pipeline;
pub mod quantized;
pub mod rng;
pub mod rope;
pub mod safetensors;
pub mod sampling;
pub mod serialization;
pub mod shape_vae;
pub mod shards;
pub mod t5;
pub mod text_encoder;
pub mod token_file;
pub mod tokenizer;
pub mod transformer;
pub mod transformer_block;
pub mod unet;
pub mod vae;
pub mod vision;
pub mod vit_encoder;

#[cfg(feature = "cuda")]
pub(crate) mod cuda_flash;
#[cfg(feature = "cuda")]
pub mod cuda_image;
#[cfg(feature = "cuda")]
pub mod cuda_training;
#[cfg(feature = "cuda")]
pub(crate) mod gpu_cross;
#[cfg(feature = "cuda")]
pub(crate) mod gpu_flow;
#[cfg(feature = "cuda")]
pub(crate) mod gpu_matrix;
#[cfg(feature = "cuda")]
pub(crate) mod gpu_model;
#[cfg(feature = "cuda")]
pub(crate) mod gpu_shape;
#[cfg(feature = "cuda")]
pub mod gpu_test;
#[cfg(feature = "cuda")]
pub mod gpu_transformer;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal_training;

pub mod onnx;
pub(crate) mod onnx_export;

pub use accelerator::{
    AcceleratorDoctorReport, AcceleratorStats, TrainingSession, accelerator_doctor,
    estimate_tensor_memory_mib,
};
pub use activations::Activation;
pub use adaln::{
    AdaLayerNorm, AdaLnCache, Modulation, ModulationCache, TimestepCache, TimestepEmbedding,
    gate_residual, gate_residual_backward,
};
pub use attention::{CrossAttentionCache, KvCache, MultiHeadAttention};
pub use batch::{Layout, TokenBatch};
pub use causal_lm_loss::{CausalLmLoss, TotalLoss, causal_lm_loss, causal_lm_loss_batch};
pub use checkpoint::{load as load_checkpoint, save as save_checkpoint};
pub use clip::{ClipTextConfig, ClipTextEncoder};
pub use conv::{
    Conv2d, ConvCache, ConvGeometry, FeatureMap, GroupNorm, ImageBatch, TrainableConv2d,
    pixel_shuffle, pixel_unshuffle, upsample_nearest,
};
#[cfg(feature = "cuda")]
pub use cuda_training::{CudaDoctorReport, CudaTrainingSession, CudaTrainingStats, cuda_doctor};
pub use dataset::{
    Augment, BatchCursor, BatchSource, Dataset, DatasetBatch, DatasetStream, JsonlStream,
    Standardizer,
};
pub use diffusion::{Denoiser, SamplingConfig, Scheduler, Solver, noise, sample, sample_from};
pub use ema::Ema;
pub use embedding::Embedding;
pub use ffn::{GeluMlp, SwiGlu};
pub use flow_transformer::{
    FlowConfig, FlowDenoiser, FlowTransformer, flow_match_target, sample_timesteps,
};
pub use gltf::Glb;
#[cfg(feature = "cuda")]
pub use gpu_transformer::GpuContext;
pub use losses::Loss;
pub use masked_lm::{MaskedBatch, masked_lm_loss};
pub use matrix::Matrix;
pub use mesh::{
    Bvh, Mesh, QuerySampling, Transform, marching_tetrahedra, marching_tetrahedra_sparse,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal_training::{MetalTrainingSession, metal_doctor};
pub use mmdit::{Conditioning, Dit, DitConfig};
pub use moe::{Expert, MoeConfig, MoeLayer, Router};
pub use network::{
    CudaTrainingCheckpoint, Dense, DenseLayer, Network, NetworkBuilder, NetworkError, TrainConfig,
    TrainingBackend, TrainingHistory,
};
pub use norm::RmsNorm;
pub use optimizers::{Optimizer, Schedule};
pub use param::{CudaDevice, Linear, Lora, Param};
pub use pipeline::{
    DynamicShift, ImageDenoiser, ImagePipeline, PipelineConfig, PooledEncoder, PromptEncoder,
};
pub use rng::RunRng;
pub use rope::Rope;
pub use safetensors::{Dtype, SafeTensors, ShardedSafeTensors, TensorInfo};
pub use sampling::Sampler;
pub use shape_vae::{ShapeVae, ShapeVaeConfig, fourier_features};
pub use shards::{Batch, BatchConfig, BatchStream, Corpus, Example, Shards};
pub use t5::{T5Config, T5Encoder};
pub use text_encoder::{TextEncoder, TextEncoderConfig};
pub use token_file::{TokenFile, TokenStream};
pub use tokenizer::{Bpe, Unigram};
pub use transformer::{
    Decoder, LoraConfig, ParameterCounts, Precision, TransformerBuilder, TransformerConfig,
    TransformerLm,
};
pub use transformer_block::{FeedForward, TransformerBlock};
pub use unet::{Unet, UnetConfig};
pub use vae::{VaeConfig, VaeDecoder, VaeEncoder, to_rgb8};
pub use vision::{VisionTransformer, VitConfig};
#[cfg(feature = "images")]
pub use vit_encoder::read_image;
pub use vit_encoder::{CLIP_MEAN, CLIP_STD, VitEncoder, VitEncoderConfig, composite_rgba};
