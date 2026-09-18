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
//! [`TransformerLm::forward_cached`] appends to a per-layer [`KvCache`], so
//! decoding the *n*-th token costs one row of attention instead of *n*:
//!
//! ```no_run
//! # use rusting_brain::TransformerLm;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let model = TransformerLm::builder().build()?;
//! # let prompt = [1u32, 2, 3];
//! let mut caches = model.new_kv_caches();
//! let mut logits = model.forward_cached(&prompt, &mut caches)?;   // prefill
//!
//! for _ in 0..50 {
//!     let next = argmax(logits.row(logits.rows - 1));
//!     logits = model.forward_cached(&[next], &mut caches)?;       // decode
//! }
//! # fn argmax(row: &[f32]) -> u32 { 0 }
//! # Ok(())
//! # }
//! ```
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
pub mod attention;
pub mod batch;
pub mod causal_lm_loss;
pub mod dataset;
pub mod embedding;
pub mod ffn;
pub mod losses;
pub mod matrix;
pub mod moe;
pub mod network;
pub mod norm;
pub mod optimizers;
pub mod param;
pub mod rope;
pub mod serialization;
pub mod transformer;
pub mod transformer_block;

#[cfg(feature = "cuda")]
pub(crate) mod cuda_flash;
#[cfg(feature = "cuda")]
pub mod cuda_training;
#[cfg(feature = "cuda")]
pub(crate) mod gpu_matrix;
#[cfg(feature = "cuda")]
pub(crate) mod gpu_model;
#[cfg(feature = "cuda")]
pub mod gpu_test;
#[cfg(feature = "cuda")]
pub mod gpu_transformer;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal_training;

pub mod onnx;

pub use accelerator::{
    AcceleratorDoctorReport, AcceleratorStats, TrainingSession, accelerator_doctor,
    estimate_tensor_memory_mib,
};
pub use activations::Activation;
pub use attention::{KvCache, MultiHeadAttention};
pub use batch::{Layout, TokenBatch};
pub use causal_lm_loss::{CausalLmLoss, TotalLoss, causal_lm_loss, causal_lm_loss_batch};
#[cfg(feature = "cuda")]
pub use cuda_training::{CudaDoctorReport, CudaTrainingSession, CudaTrainingStats, cuda_doctor};
pub use dataset::{Dataset, DatasetBatch};
pub use embedding::Embedding;
pub use ffn::{GeluMlp, SwiGlu};
#[cfg(feature = "cuda")]
pub use gpu_transformer::GpuContext;
pub use losses::Loss;
pub use matrix::Matrix;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal_training::{MetalTrainingSession, metal_doctor};
pub use moe::{Expert, MoeConfig, MoeLayer, Router};
pub use network::{
    CudaTrainingCheckpoint, Dense, DenseLayer, Network, NetworkBuilder, NetworkError, TrainConfig,
    TrainingBackend, TrainingHistory,
};
pub use norm::RmsNorm;
pub use optimizers::Optimizer;
pub use param::{Linear, Param};
pub use rope::Rope;
pub use transformer::{
    ParameterCounts, Precision, TransformerBuilder, TransformerConfig, TransformerLm,
};
pub use transformer_block::{FeedForward, TransformerBlock};
