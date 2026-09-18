//! Dense feed-forward networks: layers, the builder, and the training loop.
//!
//! This is the [`Network`] side of the crate, for tabular regression and
//! classification. The transformer side starts at
//! [`TransformerLm`](crate::TransformerLm).

use crate::activations::Activation;
use crate::dataset::Dataset;
use crate::losses::Loss;
use crate::matrix::Matrix;
use crate::optimizers::Optimizer;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NetworkError {
    #[error("network needs an input size and at least one dense layer")]
    EmptyArchitecture,
    #[error("input length {actual} does not match expected length {expected}")]
    InvalidInput { expected: usize, actual: usize },
    #[error("target length {actual} does not match expected length {expected}")]
    InvalidTarget { expected: usize, actual: usize },
    #[error("dataset is empty")]
    EmptyDataset,
    #[error("invalid network snapshot: {0}")]
    InvalidSnapshot(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("CUDA backend is unavailable: this crate was built without the `cuda` feature")]
    CudaFeatureDisabled,
    #[error("CUDA backend error: {0}")]
    Cuda(String),
    #[error("CUDA backend does not support {0}")]
    UnsupportedCuda(String),
    #[error(
        "CUDA memory budget exceeded: estimated {estimated_mib} MiB exceeds budget {budget_mib} MiB"
    )]
    CudaMemoryBudget {
        estimated_mib: usize,
        budget_mib: usize,
    },
    #[error("invalid CUDA checkpoint: {0}")]
    InvalidCudaCheckpoint(String),
    #[error(
        "Metal backend is unavailable: this crate was built without the `metal` feature, \
         or for a platform that has no Metal"
    )]
    MetalFeatureDisabled,
    #[error("Metal backend error: {0}")]
    Metal(String),
    #[error("Metal backend does not support {0}")]
    UnsupportedMetal(String),
    #[error(
        "Metal memory budget exceeded: estimated {estimated_mib} MiB exceeds budget {budget_mib} MiB"
    )]
    MetalMemoryBudget {
        estimated_mib: usize,
        budget_mib: usize,
    },
    #[error("accelerator backend error: {0}")]
    Accelerator(String),
    #[error("invalid transformer configuration: {0}")]
    InvalidConfig(String),
    #[error("token id {id} is outside the vocabulary of {vocab_size}")]
    TokenOutOfRange { id: u32, vocab_size: usize },
    #[error("sequence length {length} exceeds the configured maximum {max_seq_len}")]
    SequenceTooLong { length: usize, max_seq_len: usize },
}

/// Selects where fitting is performed. The accelerator backends are
/// deliberately fail-closed: a CUDA or Metal request never falls back to CPU
/// training, because silently turning a GPU study into a CPU one changes how
/// long it runs by more than an order of magnitude.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingBackend {
    Cpu,
    Cuda {
        device: usize,
        memory_budget_mib: usize,
    },
    /// Apple Silicon (and any other Metal device). `device` indexes
    /// `metal::Device::all()`; 0 is the system default.
    Metal {
        device: usize,
        memory_budget_mib: usize,
    },
}

impl TrainingBackend {
    /// Lowercase backend name, for messages and report files.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda { .. } => "cuda",
            Self::Metal { .. } => "metal",
        }
    }

    pub fn is_accelerated(self) -> bool {
        !matches!(self, Self::Cpu)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Dense {
    pub units: usize,
    pub activation: Activation,
}

impl Dense {
    pub fn new(units: usize, activation: Activation) -> Self {
        assert!(units > 0);
        Self { units, activation }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DenseLayer {
    pub weights: Matrix,
    pub biases: Matrix,
    pub activation: Activation,
}

#[derive(Clone, Debug)]
pub struct NetworkBuilder {
    input_size: Option<usize>,
    layers: Vec<Dense>,
    loss: Loss,
    optimizer: Optimizer,
    seed: Option<u64>,
}

impl NetworkBuilder {
    pub fn new() -> Self {
        Self {
            input_size: None,
            layers: Vec::new(),
            loss: Loss::Mse,
            optimizer: Optimizer::sgd(0.01),
            seed: None,
        }
    }

    pub fn input_size(mut self, input_size: usize) -> Self {
        assert!(input_size > 0);
        self.input_size = Some(input_size);
        self
    }

    pub fn dense(mut self, units: usize, activation: Activation) -> Self {
        self.layers.push(Dense::new(units, activation));
        self
    }

    pub fn loss(mut self, loss: Loss) -> Self {
        self.loss = loss;
        self
    }

    pub fn optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Uses a deterministic random stream for weight initialization.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn build(self) -> Network {
        Network::from_builder(self).expect("invalid network architecture")
    }

    pub fn try_build(self) -> Result<Network, NetworkError> {
        Network::from_builder(self)
    }
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub shuffle: bool,
    pub seed: Option<u64>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            shuffle: true,
            seed: Some(42),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingHistory {
    pub losses: Vec<f32>,
}

/// Device-independent optimizer state for resuming CUDA fitting. It contains no
/// CUDA handles or pointers and is safe to serialize as JSON.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CudaTrainingCheckpoint {
    pub version: u32,
    pub epoch: usize,
    pub optimizer_step: usize,
    pub shuffle_seed: Option<u64>,
    pub input_size: usize,
    pub layers: Vec<DenseLayer>,
    pub loss: Loss,
    pub optimizer: Optimizer,
    pub adam_m_weights: Vec<Matrix>,
    pub adam_v_weights: Vec<Matrix>,
    pub adam_m_biases: Vec<Matrix>,
    pub adam_v_biases: Vec<Matrix>,
}

impl CudaTrainingCheckpoint {
    /// Writes the checkpoint as compact JSON straight into a buffered file.
    /// `to_string_pretty` on a multi-megabyte optimizer state spends most of
    /// its time emitting indentation that nothing reads back, and it has to
    /// materialize the whole document in memory first; streaming the compact
    /// form is the same data in roughly a third of the bytes.
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), NetworkError> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        serde_json::to_writer(&mut writer, self)?;
        std::io::Write::flush(&mut writer)?;
        Ok(())
    }

    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, NetworkError> {
        let checkpoint: Self = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        if checkpoint.version != 1 {
            return Err(NetworkError::InvalidCudaCheckpoint(
                "unsupported version".into(),
            ));
        }
        let snapshot = NetworkSnapshot {
            version: 1,
            input_size: checkpoint.input_size,
            layers: checkpoint.layers.clone(),
            loss: checkpoint.loss,
        };
        validate_snapshot(&snapshot)
            .map_err(|e| NetworkError::InvalidCudaCheckpoint(e.to_string()))?;
        Ok(checkpoint)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct NetworkSnapshot {
    version: u32,
    input_size: usize,
    layers: Vec<DenseLayer>,
    loss: Loss,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Network {
    pub(crate) input_size: usize,
    pub(crate) layers: Vec<DenseLayer>,
    pub(crate) loss: Loss,
    pub(crate) optimizer: Optimizer,
    pub(crate) adam_step: usize,
    pub(crate) adam_m_weights: Vec<Matrix>,
    pub(crate) adam_v_weights: Vec<Matrix>,
    pub(crate) adam_m_biases: Vec<Matrix>,
    pub(crate) adam_v_biases: Vec<Matrix>,
}

#[derive(Clone, Debug)]
struct LayerCache {
    input: Vec<f32>,
    output: Vec<f32>,
}

/// Two buffers wide enough for any layer, ping-ponged by `forward_inference`.
struct InferenceScratch {
    current: Vec<f32>,
    next: Vec<f32>,
}

impl InferenceScratch {
    fn for_network(network: &Network) -> Self {
        let widest = widest_layer(network);
        Self {
            current: vec![0.0; widest],
            next: vec![0.0; widest],
        }
    }
}

fn widest_layer(network: &Network) -> usize {
    network
        .layers
        .iter()
        .map(|layer| layer.weights.rows)
        .chain(std::iter::once(network.input_size))
        .max()
        .unwrap_or(0)
}

/// How many samples `forward_inference_tile` evaluates side by side.
///
/// A single-sample forward pass is a chain of `value += weight * input`, and
/// because f32 addition is not associative the compiler may not reorder it into
/// independent partial sums. The chain therefore runs at the latency of one
/// dependent add per weight -- measured at 0.9 GFLOP/s per core on a 908-input
/// network, a small fraction of what the core can issue.
///
/// Evaluating a tile of samples together fixes that without touching the
/// arithmetic: each sample still accumulates over its inputs in exactly the
/// same order, so every result is bit-for-bit what the scalar path produced,
/// but the tile's accumulators are independent of each other and the inner loop
/// vectorises across them. Eight is one AVX2 f32 vector.
const INFERENCE_TILE: usize = 8;

/// Samples handed to one rayon task. A multiple of [`INFERENCE_TILE`], and big
/// enough that scheduling one task costs nothing next to running it.
const INFERENCE_CHUNK: usize = 512;

/// Samples per rayon task in [`Network::train_batch_parallel`]. Backward is
/// roughly three times the work of forward per sample, so a smaller chunk than
/// [`INFERENCE_CHUNK`] still hides the task overhead while giving a typical
/// 1024-sample batch enough pieces to fill every core of an M-series or
/// desktop CPU.
pub const GRADIENT_CHUNK: usize = 64;

/// Ping-ponged activations for a tile of samples, laid out unit-major with the
/// tile contiguous (`value[unit * INFERENCE_TILE + sample]`) so the inner loop
/// reads one vector per unit.
struct TiledInferenceScratch {
    current: Vec<f32>,
    next: Vec<f32>,
    single: InferenceScratch,
}

impl TiledInferenceScratch {
    fn for_network(network: &Network) -> Self {
        let widest = widest_layer(network);
        Self {
            current: vec![0.0; widest * INFERENCE_TILE],
            next: vec![0.0; widest * INFERENCE_TILE],
            single: InferenceScratch::for_network(network),
        }
    }
}

#[derive(Clone, Debug)]
struct Gradients {
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
}

impl Network {
    pub fn builder() -> NetworkBuilder {
        NetworkBuilder::new()
    }

    pub fn new(layers: Vec<usize>, learning_rate: f32) -> Self {
        assert!(layers.len() >= 2);

        let last = layers.len() - 2;
        let mut builder = NetworkBuilder::new()
            .input_size(layers[0])
            .loss(Loss::Mse)
            .optimizer(Optimizer::sgd(learning_rate));

        for (i, &units) in layers.iter().enumerate().skip(1) {
            let activation = if i - 1 == last {
                Activation::Linear
            } else {
                Activation::Relu
            };
            builder = builder.dense(units, activation);
        }

        builder.build()
    }

    fn from_builder(builder: NetworkBuilder) -> Result<Self, NetworkError> {
        let input_size = builder.input_size.ok_or(NetworkError::EmptyArchitecture)?;
        if builder.layers.is_empty() {
            return Err(NetworkError::EmptyArchitecture);
        }

        let mut rng = builder
            .seed
            .map_or_else(StdRng::from_entropy, StdRng::seed_from_u64);
        let mut previous = input_size;
        let mut layers = Vec::with_capacity(builder.layers.len());

        for dense in builder.layers {
            let scale = (2.0 / previous as f32).sqrt();
            let weights = (0..dense.units * previous)
                .map(|_| rng.gen_range(-scale..scale))
                .collect();

            layers.push(DenseLayer {
                weights: Matrix::from_vec(dense.units, previous, weights),
                biases: Matrix::new(dense.units, 1),
                activation: dense.activation,
            });
            previous = dense.units;
        }

        Ok(Self::with_layers(
            input_size,
            layers,
            builder.loss,
            builder.optimizer,
        ))
    }

    fn with_layers(
        input_size: usize,
        layers: Vec<DenseLayer>,
        loss: Loss,
        optimizer: Optimizer,
    ) -> Self {
        let adam_m_weights = layers
            .iter()
            .map(|layer| Matrix::new(layer.weights.rows, layer.weights.cols))
            .collect();
        let adam_v_weights = layers
            .iter()
            .map(|layer| Matrix::new(layer.weights.rows, layer.weights.cols))
            .collect();
        let adam_m_biases = layers
            .iter()
            .map(|layer| Matrix::new(layer.biases.rows, layer.biases.cols))
            .collect();
        let adam_v_biases = layers
            .iter()
            .map(|layer| Matrix::new(layer.biases.rows, layer.biases.cols))
            .collect();

        Self {
            input_size,
            layers,
            loss,
            optimizer,
            adam_step: 0,
            adam_m_weights,
            adam_v_weights,
            adam_m_biases,
            adam_v_biases,
        }
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn output_size(&self) -> usize {
        self.layers.last().map_or(0, |layer| layer.biases.rows)
    }

    pub fn layers(&self) -> &[DenseLayer] {
        &self.layers
    }

    pub fn loss(&self) -> Loss {
        self.loss
    }

    pub fn predict(&self, input: &[f32]) -> Result<Vec<f32>, NetworkError> {
        self.validate_input(input)?;
        Ok(self.forward_internal(input).0)
    }

    /// [`Network::predict`] for callers that have already checked the shape.
    ///
    /// # Panics
    ///
    /// Panics if `input` is not exactly [`Network::input_size`] long. Use
    /// `predict` for anything reading input the program did not construct.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.predict(input).expect("invalid input shape")
    }

    pub fn predict_batch(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, NetworkError> {
        for input in inputs {
            self.validate_input(input)?;
        }
        // Chunks are large enough that rayon's per-task overhead disappears
        // next to the work, and a multiple of the tile so only the dataset's
        // own remainder takes the single-sample path.
        Ok(inputs
            .par_chunks(INFERENCE_CHUNK)
            .map_init(
                || TiledInferenceScratch::for_network(self),
                |scratch, chunk| self.forward_inference_chunk(chunk, scratch),
            )
            .flatten()
            .collect())
    }

    pub fn train(&mut self, input: &[f32], target: &[f32]) -> Result<f32, NetworkError> {
        self.train_batch(&[input.to_vec()], &[target.to_vec()])
    }

    pub fn train_batch(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
    ) -> Result<f32, NetworkError> {
        if inputs.is_empty() {
            return Err(NetworkError::EmptyDataset);
        }
        assert_eq!(inputs.len(), targets.len());

        let mut gradients = Gradients::zeros(&self.layers);
        let mut loss = 0.0;

        for (input, target) in inputs.iter().zip(targets) {
            self.validate_input(input)?;
            self.validate_target(target)?;

            let (prediction, caches) = self.forward_internal(input);
            loss += self.loss.value(&prediction, target);
            let sample_grads = self.backward(&prediction, target, &caches);
            gradients.add_assign(&sample_grads);
        }

        let scale = 1.0 / inputs.len() as f32;
        gradients.scale(scale);
        self.apply_gradients(&gradients);

        Ok(loss * scale)
    }

    /// Same gradient step as [`Network::train_batch`], spread over rayon.
    ///
    /// Samples are reduced in fixed-size chunks taken in index order and the
    /// chunk gradients are then summed in that same order, so the result
    /// depends only on the batch contents -- not on the thread count, the pool
    /// size, or how rayon happened to split the work. It is *not* bit-for-bit
    /// equal to `train_batch`, because a chunked sum associates the additions
    /// differently; it is equal to `train_batch` for batches of at most
    /// [`GRADIENT_CHUNK`] samples, which take that path directly.
    pub fn train_batch_parallel(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        _num_threads: usize,
    ) -> Result<f32, NetworkError> {
        if inputs.is_empty() {
            return Err(NetworkError::EmptyDataset);
        }
        assert_eq!(inputs.len(), targets.len());
        if inputs.len() <= GRADIENT_CHUNK {
            return self.train_batch(inputs, targets);
        }
        for (input, target) in inputs.iter().zip(targets) {
            self.validate_input(input)?;
            self.validate_target(target)?;
        }
        // `map` + an ordered `collect` rather than `reduce`: rayon's reduction
        // order is a function of the thread count, and two machines must not
        // disagree about a trained model.
        let partials: Vec<(Gradients, f32)> = inputs
            .par_chunks(GRADIENT_CHUNK)
            .zip(targets.par_chunks(GRADIENT_CHUNK))
            .map(|(inputs, targets)| {
                let mut gradients = Gradients::zeros(&self.layers);
                let mut loss = 0.0;
                for (input, target) in inputs.iter().zip(targets) {
                    let (prediction, caches) = self.forward_internal(input);
                    loss += self.loss.value(&prediction, target);
                    let sample_grads = self.backward(&prediction, target, &caches);
                    gradients.add_assign(&sample_grads);
                }
                (gradients, loss)
            })
            .collect();

        let mut gradients = Gradients::zeros(&self.layers);
        let mut loss = 0.0;
        for (chunk_gradients, chunk_loss) in &partials {
            gradients.add_assign(chunk_gradients);
            loss += chunk_loss;
        }
        let scale = 1.0 / inputs.len() as f32;
        gradients.scale(scale);
        self.apply_gradients(&gradients);

        Ok(loss * scale)
    }

    pub fn fit(
        &mut self,
        dataset: &Dataset,
        config: TrainConfig,
    ) -> Result<TrainingHistory, NetworkError> {
        if dataset.is_empty() {
            return Err(NetworkError::EmptyDataset);
        }

        let mut working = dataset.clone();
        let mut losses = Vec::with_capacity(config.epochs);

        for epoch in 0..config.epochs {
            if config.shuffle {
                let seed = config.seed.map(|seed| seed + epoch as u64);
                working.shuffle(seed);
            }

            let mut epoch_loss = 0.0;
            let mut batches = 0;
            for batch in working.batches(config.batch_size.max(1)) {
                epoch_loss += self.train_batch_parallel(batch.inputs, batch.targets, 0)?;
                batches += 1;
            }

            losses.push(epoch_loss / batches as f32);
        }

        Ok(TrainingHistory { losses })
    }

    /// Fits using the requested backend. CPU behavior is unchanged; CUDA never
    /// silently delegates to CPU on an initialization or execution failure.
    pub fn fit_with_backend(
        &mut self,
        dataset: &Dataset,
        config: TrainConfig,
        backend: TrainingBackend,
    ) -> Result<TrainingHistory, NetworkError> {
        match backend {
            TrainingBackend::Cpu => self.fit(dataset, config),
            TrainingBackend::Cuda {
                device,
                memory_budget_mib,
            } => {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda_training::fit_cuda(self, dataset, config, device, memory_budget_mib)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (dataset, config, device, memory_budget_mib);
                    Err(NetworkError::CudaFeatureDisabled)
                }
            }
            TrainingBackend::Metal {
                device,
                memory_budget_mib,
            } => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    crate::metal_training::fit_metal(
                        self,
                        dataset,
                        config,
                        device,
                        memory_budget_mib,
                    )
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    let _ = (dataset, config, device, memory_budget_mib);
                    Err(NetworkError::MetalFeatureDisabled)
                }
            }
        }
    }

    pub fn cuda_checkpoint(
        &self,
        epoch: usize,
        shuffle_seed: Option<u64>,
    ) -> CudaTrainingCheckpoint {
        CudaTrainingCheckpoint {
            version: 1,
            epoch,
            optimizer_step: self.adam_step,
            shuffle_seed,
            input_size: self.input_size,
            layers: self.layers.clone(),
            loss: self.loss,
            optimizer: self.optimizer.clone(),
            adam_m_weights: self.adam_m_weights.clone(),
            adam_v_weights: self.adam_v_weights.clone(),
            adam_m_biases: self.adam_m_biases.clone(),
            adam_v_biases: self.adam_v_biases.clone(),
        }
    }

    pub fn restore_cuda_checkpoint(
        &mut self,
        checkpoint: CudaTrainingCheckpoint,
    ) -> Result<(), NetworkError> {
        if checkpoint.version != 1 {
            return Err(NetworkError::InvalidCudaCheckpoint(
                "unsupported version".into(),
            ));
        }
        let snapshot = NetworkSnapshot {
            version: 1,
            input_size: checkpoint.input_size,
            layers: checkpoint.layers.clone(),
            loss: checkpoint.loss,
        };
        validate_snapshot(&snapshot)?;
        let n = checkpoint.layers.len();
        if checkpoint.adam_m_weights.len() != n
            || checkpoint.adam_v_weights.len() != n
            || checkpoint.adam_m_biases.len() != n
            || checkpoint.adam_v_biases.len() != n
        {
            return Err(NetworkError::InvalidCudaCheckpoint(
                "moment tensor count does not match layers".into(),
            ));
        }
        for i in 0..n {
            if checkpoint.adam_m_weights[i].data.len() != checkpoint.layers[i].weights.data.len()
                || checkpoint.adam_v_weights[i].data.len()
                    != checkpoint.layers[i].weights.data.len()
                || checkpoint.adam_m_biases[i].data.len() != checkpoint.layers[i].biases.data.len()
                || checkpoint.adam_v_biases[i].data.len() != checkpoint.layers[i].biases.data.len()
                || checkpoint.adam_m_weights[i]
                    .data
                    .iter()
                    .chain(&checkpoint.adam_v_weights[i].data)
                    .chain(&checkpoint.adam_m_biases[i].data)
                    .chain(&checkpoint.adam_v_biases[i].data)
                    .any(|v| !v.is_finite())
            {
                return Err(NetworkError::InvalidCudaCheckpoint(format!(
                    "invalid moments for layer {i}"
                )));
            }
        }
        self.input_size = checkpoint.input_size;
        self.layers = checkpoint.layers;
        self.loss = checkpoint.loss;
        self.optimizer = checkpoint.optimizer;
        self.adam_step = checkpoint.optimizer_step;
        self.adam_m_weights = checkpoint.adam_m_weights;
        self.adam_v_weights = checkpoint.adam_v_weights;
        self.adam_m_biases = checkpoint.adam_m_biases;
        self.adam_v_biases = checkpoint.adam_v_biases;
        Ok(())
    }

    pub fn evaluate_loss(&self, dataset: &Dataset) -> Result<f32, NetworkError> {
        if dataset.is_empty() {
            return Err(NetworkError::EmptyDataset);
        }

        for (input, target) in dataset.inputs.iter().zip(&dataset.targets) {
            self.validate_input(input)?;
            self.validate_target(target)?;
        }

        // Per-sample losses are collected in dataset order and only then summed,
        // so the total does not depend on how rayon splits the work.
        let losses: Vec<f32> = dataset
            .inputs
            .par_chunks(INFERENCE_CHUNK)
            .zip(dataset.targets.par_chunks(INFERENCE_CHUNK))
            .map_init(
                || TiledInferenceScratch::for_network(self),
                |scratch, (inputs, targets)| {
                    self.forward_inference_chunk(inputs, scratch)
                        .into_iter()
                        .zip(targets)
                        .map(|(prediction, target)| self.loss.value(&prediction, target))
                        .collect::<Vec<f32>>()
                },
            )
            .flatten()
            .collect();

        Ok(losses.iter().sum::<f32>() / dataset.len() as f32)
    }

    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), NetworkError> {
        let snapshot = NetworkSnapshot {
            version: 1,
            input_size: self.input_size,
            layers: self.layers.clone(),
            loss: self.loss,
        };
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        serde_json::to_writer(&mut writer, &snapshot)?;
        std::io::Write::flush(&mut writer)?;
        Ok(())
    }

    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, NetworkError> {
        let json = std::fs::read_to_string(path)?;
        let snapshot: NetworkSnapshot = serde_json::from_str(&json)?;
        validate_snapshot(&snapshot)?;
        Ok(Self::with_layers(
            snapshot.input_size,
            snapshot.layers,
            snapshot.loss,
            Optimizer::sgd(0.01),
        ))
    }

    /// Inference-only forward pass that reuses two scratch buffers instead of
    /// allocating a `Vec` and a discarded `LayerCache` per layer per sample.
    /// The arithmetic and its order are identical to `forward_internal`, so
    /// predictions stay bit-for-bit the same.
    fn forward_inference<'s>(&self, input: &[f32], scratch: &'s mut InferenceScratch) -> &'s [f32] {
        let InferenceScratch { current, next } = scratch;
        let mut width = input.len();
        current[..width].copy_from_slice(input);

        for layer in &self.layers {
            let units = layer.weights.rows;
            let source = &current[..width];
            let destination = &mut next[..units];

            for (row, out) in destination.iter_mut().enumerate() {
                let mut value = layer.biases.data[row];
                let row_offset = row * layer.weights.cols;
                for (col, input_value) in source.iter().enumerate() {
                    value += layer.weights.data[row_offset + col] * input_value;
                }
                *out = value;
            }

            layer.activation.apply_to_slice(destination);
            std::mem::swap(current, next);
            width = units;
        }

        &current[..width]
    }

    /// Forward exactly `INFERENCE_TILE` samples, writing each one's output row
    /// into `outputs`.
    ///
    /// The column loop, and therefore each sample's accumulation order, is the
    /// same one `forward_inference` walks; only the number of samples in flight
    /// changes. See [`INFERENCE_TILE`].
    fn forward_inference_tile(
        &self,
        inputs: &[Vec<f32>],
        scratch: &mut TiledInferenceScratch,
        outputs: &mut Vec<Vec<f32>>,
    ) {
        debug_assert_eq!(inputs.len(), INFERENCE_TILE);
        let mut width = self.input_size;
        for (column, value) in scratch.current[..width * INFERENCE_TILE]
            .chunks_exact_mut(INFERENCE_TILE)
            .enumerate()
        {
            for (sample, slot) in value.iter_mut().enumerate() {
                *slot = inputs[sample][column];
            }
        }

        for layer in &self.layers {
            let units = layer.weights.rows;
            let source = &scratch.current[..width * INFERENCE_TILE];
            let destination = &mut scratch.next[..units * INFERENCE_TILE];

            for (row, out) in destination.chunks_exact_mut(INFERENCE_TILE).enumerate() {
                let mut accumulator = [layer.biases.data[row]; INFERENCE_TILE];
                let weights = &layer.weights.data[row * layer.weights.cols..][..width];
                for (weight, values) in weights.iter().zip(source.chunks_exact(INFERENCE_TILE)) {
                    for (slot, value) in accumulator.iter_mut().zip(values) {
                        *slot += weight * value;
                    }
                }
                out.copy_from_slice(&accumulator);
            }

            if layer.activation == Activation::Softmax {
                // Softmax normalises across a sample's units, not elementwise,
                // so it cannot see the interleaved tile.
                let mut column = vec![0.0; units];
                for sample in 0..INFERENCE_TILE {
                    for unit in 0..units {
                        column[unit] = destination[unit * INFERENCE_TILE + sample];
                    }
                    layer.activation.apply_to_slice(&mut column);
                    for unit in 0..units {
                        destination[unit * INFERENCE_TILE + sample] = column[unit];
                    }
                }
            } else {
                layer.activation.apply_to_slice(destination);
            }
            std::mem::swap(&mut scratch.current, &mut scratch.next);
            width = units;
        }

        for sample in 0..INFERENCE_TILE {
            outputs.push(
                (0..width)
                    .map(|unit| scratch.current[unit * INFERENCE_TILE + sample])
                    .collect(),
            );
        }
    }

    /// Forward a slice of samples, tile by tile, with any remainder falling
    /// back to the single-sample path.
    fn forward_inference_chunk(
        &self,
        inputs: &[Vec<f32>],
        scratch: &mut TiledInferenceScratch,
    ) -> Vec<Vec<f32>> {
        let mut outputs = Vec::with_capacity(inputs.len());
        let mut tiles = inputs.chunks_exact(INFERENCE_TILE);
        for tile in &mut tiles {
            self.forward_inference_tile(tile, scratch, &mut outputs);
        }
        for input in tiles.remainder() {
            outputs.push(self.forward_inference(input, &mut scratch.single).to_vec());
        }
        outputs
    }

    fn forward_internal(&self, input: &[f32]) -> (Vec<f32>, Vec<LayerCache>) {
        let mut current = input.to_vec();
        let mut caches = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let layer_input = current;
            let mut output = vec![0.0; layer.biases.rows];

            for (row, out) in output.iter_mut().enumerate() {
                let mut value = layer.biases.data[row];
                let row_offset = row * layer.weights.cols;
                for (col, input_value) in layer_input.iter().enumerate() {
                    value += layer.weights.data[row_offset + col] * input_value;
                }
                *out = value;
            }

            layer.activation.apply_to_slice(&mut output);
            caches.push(LayerCache {
                input: layer_input,
                output: output.clone(),
            });
            current = output;
        }

        (current, caches)
    }

    fn backward(&self, prediction: &[f32], target: &[f32], caches: &[LayerCache]) -> Gradients {
        let mut gradients = Gradients::zeros(&self.layers);
        let last_idx = self.layers.len() - 1;
        let mut delta =
            self.loss
                .output_delta(prediction, target, self.layers[last_idx].activation);

        for layer_idx in (0..self.layers.len()).rev() {
            let layer = &self.layers[layer_idx];
            let cache = &caches[layer_idx];

            for (row, delta_value) in delta.iter().enumerate() {
                gradients.biases[layer_idx].data[row] += *delta_value;
                let row_offset = row * layer.weights.cols;
                for (col, input_value) in cache.input.iter().enumerate() {
                    gradients.weights[layer_idx].data[row_offset + col] +=
                        delta_value * input_value;
                }
            }

            if layer_idx > 0 {
                let previous_output = &caches[layer_idx - 1].output;
                let mut previous_delta = vec![0.0; layer.weights.cols];

                for (col, previous_delta_value) in previous_delta.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for (row, delta_value) in delta.iter().enumerate() {
                        sum += layer.weights.data[row * layer.weights.cols + col] * delta_value;
                    }
                    *previous_delta_value = sum
                        * self.layers[layer_idx - 1]
                            .activation
                            .derivative(previous_output[col]);
                }

                delta = previous_delta;
            }
        }

        gradients
    }

    fn apply_gradients(&mut self, gradients: &Gradients) {
        match self.optimizer.clone() {
            Optimizer::Sgd { learning_rate } => {
                for (layer, (weight_grad, bias_grad)) in self
                    .layers
                    .iter_mut()
                    .zip(gradients.weights.iter().zip(&gradients.biases))
                {
                    for (weight, grad) in layer.weights.data.iter_mut().zip(&weight_grad.data) {
                        *weight += learning_rate * grad;
                    }
                    for (bias, grad) in layer.biases.data.iter_mut().zip(&bias_grad.data) {
                        *bias += learning_rate * grad;
                    }
                }
            }
            Optimizer::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
            } => {
                self.adam_step += 1;
                let bias_correction1 = 1.0 - beta1.powi(self.adam_step as i32);
                let bias_correction2 = 1.0 - beta2.powi(self.adam_step as i32);

                for layer_idx in 0..self.layers.len() {
                    apply_adam(
                        &mut self.layers[layer_idx].weights.data,
                        &gradients.weights[layer_idx].data,
                        &mut self.adam_m_weights[layer_idx].data,
                        &mut self.adam_v_weights[layer_idx].data,
                        AdamHyperparams {
                            learning_rate,
                            beta1,
                            beta2,
                            epsilon,
                            bias_correction1,
                            bias_correction2,
                            weight_decay,
                        },
                    );
                    apply_adam(
                        &mut self.layers[layer_idx].biases.data,
                        &gradients.biases[layer_idx].data,
                        &mut self.adam_m_biases[layer_idx].data,
                        &mut self.adam_v_biases[layer_idx].data,
                        AdamHyperparams {
                            learning_rate,
                            beta1,
                            beta2,
                            epsilon,
                            bias_correction1,
                            bias_correction2,
                            weight_decay: 0.0,
                        },
                    );
                }
            }
        }
    }

    pub(crate) fn validate_input(&self, input: &[f32]) -> Result<(), NetworkError> {
        if input.len() != self.input_size {
            return Err(NetworkError::InvalidInput {
                expected: self.input_size,
                actual: input.len(),
            });
        }
        Ok(())
    }

    pub(crate) fn validate_target(&self, target: &[f32]) -> Result<(), NetworkError> {
        let output_size = self.output_size();
        if target.len() != output_size {
            return Err(NetworkError::InvalidTarget {
                expected: output_size,
                actual: target.len(),
            });
        }
        Ok(())
    }
}

fn validate_snapshot(snapshot: &NetworkSnapshot) -> Result<(), NetworkError> {
    if snapshot.version != 1 {
        return Err(NetworkError::InvalidSnapshot(format!(
            "unsupported version {}",
            snapshot.version
        )));
    }
    if snapshot.input_size == 0 || snapshot.layers.is_empty() {
        return Err(NetworkError::InvalidSnapshot(
            "input size and layers must be non-empty".to_owned(),
        ));
    }
    let mut expected_inputs = snapshot.input_size;
    for (index, layer) in snapshot.layers.iter().enumerate() {
        if layer.weights.rows == 0
            || layer.weights.cols != expected_inputs
            || layer.weights.data.len() != layer.weights.rows * layer.weights.cols
            || layer.biases.rows != layer.weights.rows
            || layer.biases.cols != 1
            || layer.biases.data.len() != layer.biases.rows
            || layer.weights.data.iter().any(|value| !value.is_finite())
            || layer.biases.data.iter().any(|value| !value.is_finite())
        {
            return Err(NetworkError::InvalidSnapshot(format!(
                "layer {index} has inconsistent dimensions or non-finite values"
            )));
        }
        expected_inputs = layer.weights.rows;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct AdamHyperparams {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    weight_decay: f32,
}

fn apply_adam(
    values: &mut [f32],
    gradients: &[f32],
    moment1: &mut [f32],
    moment2: &mut [f32],
    params: AdamHyperparams,
) {
    for (((value, gradient), m), v) in values.iter_mut().zip(gradients).zip(moment1).zip(moment2) {
        *m = params.beta1 * *m + (1.0 - params.beta1) * *gradient;
        *v = params.beta2 * *v + (1.0 - params.beta2) * gradient * gradient;

        let m_hat = *m / params.bias_correction1;
        let v_hat = *v / params.bias_correction2;
        // Decoupled (AdamW-style) weight decay: shrink the parameter toward
        // zero independently of the adaptive gradient step.
        *value -= params.learning_rate * params.weight_decay * *value;
        *value += params.learning_rate * m_hat / (v_hat.sqrt() + params.epsilon);
    }
}

impl Gradients {
    fn zeros(layers: &[DenseLayer]) -> Self {
        Self {
            weights: layers
                .iter()
                .map(|layer| Matrix::new(layer.weights.rows, layer.weights.cols))
                .collect(),
            biases: layers
                .iter()
                .map(|layer| Matrix::new(layer.biases.rows, layer.biases.cols))
                .collect(),
        }
    }

    fn add_assign(&mut self, other: &Self) {
        for (left, right) in self.weights.iter_mut().zip(&other.weights) {
            for (l, r) in left.data.iter_mut().zip(&right.data) {
                *l += r;
            }
        }

        for (left, right) in self.biases.iter_mut().zip(&other.biases) {
            for (l, r) in left.data.iter_mut().zip(&right.data) {
                *l += r;
            }
        }
    }

    fn scale(&mut self, scale: f32) {
        for matrix in self.weights.iter_mut().chain(&mut self.biases) {
            for value in &mut matrix.data {
                *value *= scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_learns_with_adam() {
        let dataset = Dataset::new(
            vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]],
        );
        let mut model = Network::builder()
            .input_size(2)
            .dense(8, Activation::Tanh)
            .dense(1, Activation::Sigmoid)
            .loss(Loss::BinaryCrossEntropy)
            .optimizer(Optimizer::adam(0.05))
            .build();

        let initial = model.evaluate_loss(&dataset).unwrap();
        model
            .fit(
                &dataset,
                TrainConfig {
                    epochs: 2_000,
                    batch_size: 4,
                    shuffle: true,
                    seed: Some(7),
                },
            )
            .unwrap();
        let final_loss = model.evaluate_loss(&dataset).unwrap();

        assert!(final_loss < initial);
        assert!(final_loss < 0.2, "final loss was {final_loss}");
    }

    #[test]
    fn save_load_preserves_predictions() {
        let model = Network::builder()
            .input_size(2)
            .dense(3, Activation::Relu)
            .dense(1, Activation::Linear)
            .build();
        let expected = model.predict(&[0.2, 0.4]).unwrap();
        let path = std::env::temp_dir().join("rusting_brain_model_test.json");

        model.save_json(&path).unwrap();
        let loaded = Network::load_json(&path).unwrap();
        let actual = loaded.predict(&[0.2, 0.4]).unwrap();
        let _ = std::fs::remove_file(path);

        assert_eq!(expected, actual);
    }

    #[test]
    fn predict_rejects_wrong_input_size() {
        let model = Network::builder()
            .input_size(2)
            .dense(1, Activation::Linear)
            .build();

        let error = model.predict(&[1.0]).unwrap_err();

        assert!(matches!(
            error,
            NetworkError::InvalidInput {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn identical_builder_seeds_produce_identical_predictions() {
        let build = || {
            Network::builder()
                .input_size(2)
                .dense(3, Activation::Relu)
                .dense(1, Activation::Linear)
                .seed(17)
                .build()
        };
        assert_eq!(
            build().predict(&[0.25, -0.5]).unwrap(),
            build().predict(&[0.25, -0.5]).unwrap()
        );
    }

    /// `predict_batch` and `evaluate_loss` run a parallel, scratch-buffer
    /// forward pass. Research artifacts are compared across runs, so the fast
    /// path must stay bit-for-bit identical to the per-sample scalar path
    /// rather than merely close.
    #[test]
    fn batched_inference_is_bit_identical_to_scalar_inference() {
        let model = Network::builder()
            .input_size(9)
            .dense(13, Activation::Relu)
            .dense(7, Activation::Tanh)
            .dense(3, Activation::Sigmoid)
            .loss(Loss::Mse)
            .seed(0x5eed)
            .build();
        let inputs: Vec<Vec<f32>> = (0..257)
            .map(|row| {
                (0..9)
                    .map(|col| ((row * 31 + col * 17) as f32 - 400.0) / 91.0)
                    .collect()
            })
            .collect();
        let targets: Vec<Vec<f32>> = inputs
            .iter()
            .map(|x| vec![x[0].abs().min(1.0), 0.5, x[2].abs().min(1.0)])
            .collect();

        let batched = model.predict_batch(&inputs).unwrap();
        for (input, actual) in inputs.iter().zip(&batched) {
            assert_eq!(&model.predict(input).unwrap(), actual);
        }

        let mut expected = 0.0f32;
        for (input, target) in inputs.iter().zip(&targets) {
            expected += model.loss.value(&model.predict(input).unwrap(), target);
        }
        let dataset = Dataset::new(inputs, targets);
        assert_eq!(
            model.evaluate_loss(&dataset).unwrap(),
            expected / dataset.len() as f32
        );
    }
}
