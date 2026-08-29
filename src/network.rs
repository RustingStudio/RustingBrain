use crate::activations::Activation;
use crate::dataset::Dataset;
use crate::losses::Loss;
use crate::matrix::Matrix;
use crate::optimizers::Optimizer;
use rand::Rng;
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
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
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
}

impl NetworkBuilder {
    pub fn new() -> Self {
        Self {
            input_size: None,
            layers: Vec::new(),
            loss: Loss::Mse,
            optimizer: Optimizer::sgd(0.01),
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct NetworkSnapshot {
    version: u32,
    input_size: usize,
    layers: Vec<DenseLayer>,
    loss: Loss,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Network {
    input_size: usize,
    layers: Vec<DenseLayer>,
    loss: Loss,
    optimizer: Optimizer,
    adam_step: usize,
    adam_m_weights: Vec<Matrix>,
    adam_v_weights: Vec<Matrix>,
    adam_m_biases: Vec<Matrix>,
    adam_v_biases: Vec<Matrix>,
}

#[derive(Clone, Debug)]
struct LayerCache {
    input: Vec<f32>,
    output: Vec<f32>,
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

        let mut rng = rand::thread_rng();
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
        self.layers.last().unwrap().biases.rows
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

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.predict(input).expect("invalid input shape")
    }

    pub fn predict_batch(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, NetworkError> {
        inputs.iter().map(|input| self.predict(input)).collect()
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

    pub fn train_batch_parallel(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        _num_threads: usize,
    ) -> Result<f32, NetworkError> {
        self.train_batch(inputs, targets)
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
                epoch_loss += self.train_batch(batch.inputs, batch.targets)?;
                batches += 1;
            }

            losses.push(epoch_loss / batches as f32);
        }

        Ok(TrainingHistory { losses })
    }

    pub fn evaluate_loss(&self, dataset: &Dataset) -> Result<f32, NetworkError> {
        if dataset.is_empty() {
            return Err(NetworkError::EmptyDataset);
        }

        let mut loss = 0.0;
        for (input, target) in dataset.inputs.iter().zip(&dataset.targets) {
            self.validate_input(input)?;
            self.validate_target(target)?;
            loss += self.loss.value(&self.predict(input)?, target);
        }

        Ok(loss / dataset.len() as f32)
    }

    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), NetworkError> {
        let snapshot = NetworkSnapshot {
            version: 1,
            input_size: self.input_size,
            layers: self.layers.clone(),
            loss: self.loss,
        };
        let json = serde_json::to_string_pretty(&snapshot)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, NetworkError> {
        let json = std::fs::read_to_string(path)?;
        let snapshot: NetworkSnapshot = serde_json::from_str(&json)?;
        Ok(Self::with_layers(
            snapshot.input_size,
            snapshot.layers,
            snapshot.loss,
            Optimizer::sgd(0.01),
        ))
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
                        },
                    );
                }
            }
        }
    }

    fn validate_input(&self, input: &[f32]) -> Result<(), NetworkError> {
        if input.len() != self.input_size {
            return Err(NetworkError::InvalidInput {
                expected: self.input_size,
                actual: input.len(),
            });
        }
        Ok(())
    }

    fn validate_target(&self, target: &[f32]) -> Result<(), NetworkError> {
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

#[derive(Clone, Copy)]
struct AdamHyperparams {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    bias_correction1: f32,
    bias_correction2: f32,
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
}
