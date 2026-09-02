pub mod activations;
pub mod dataset;
pub mod layers;
pub mod losses;
pub mod matrix;
pub mod network;
pub mod optimizers;
pub mod serialization;
pub mod tensor;

#[cfg(feature = "cuda")]
pub mod cuda_training;
#[cfg(feature = "cuda")]
pub mod gpu_matrix;
#[cfg(feature = "cuda")]
pub mod gpu_test;

pub mod onnx;

pub use activations::Activation;
#[cfg(feature = "cuda")]
pub use cuda_training::{CudaDoctorReport, cuda_doctor};
pub use dataset::{Dataset, DatasetBatch};
pub use losses::Loss;
pub use matrix::Matrix;
pub use network::{
    CudaTrainingCheckpoint, Dense, DenseLayer, Network, NetworkBuilder, NetworkError, TrainConfig,
    TrainingBackend, TrainingHistory,
};
pub use optimizers::Optimizer;
