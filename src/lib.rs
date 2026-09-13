pub mod accelerator;
pub mod activations;
pub mod attention;
pub mod batch;
pub mod causal_lm_loss;
pub mod dataset;
pub mod embedding;
pub mod ffn;
pub mod layers;
pub mod losses;
pub mod matrix;
pub mod moe;
pub mod network;
pub mod norm;
pub mod optimizers;
pub mod param;
pub mod rope;
pub mod serialization;
pub mod tensor;
pub mod transformer;
pub mod transformer_block;

#[cfg(feature = "cuda")]
pub mod cuda_training;
#[cfg(feature = "cuda")]
pub mod gpu_matrix;
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
pub use transformer::{ParameterCounts, Precision, TransformerBuilder, TransformerConfig, TransformerLm};
pub use transformer_block::{FeedForward, TransformerBlock};
