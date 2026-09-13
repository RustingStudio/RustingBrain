//! Position-wise feed-forward networks.
//!
//! [`SwiGlu`] is the default and is also the shape of one MoE expert.
//! [`GeluMlp`] is the plain two-matrix alternative, kept for comparison.

use crate::matrix::Matrix;
use crate::param::{Linear, Param};
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Elements per rayon task in the point-wise activation passes. Big enough
/// that the scheduling overhead disappears against the work, small enough that
/// a `[2048, 308]` hidden layer still splits across every core.
const CHUNK: usize = 8192;

/// `SiLU(x * Wg^T) * (x * Wu^T) * Wd^T`, all three projections bias-free.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SwiGlu {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

/// Pre-activation gate and up projections, which the backward pass needs and
/// cannot recover from the output.
#[derive(Clone, Debug)]
pub struct SwiGluCache {
    input: Matrix,
    gate: Matrix,
    up: Matrix,
    hidden: Matrix,
}

impl SwiGlu {
    pub fn new(d_model: usize, d_ff: usize, rng: &mut StdRng) -> Self {
        Self {
            gate: Linear::new(d_model, d_ff, rng),
            up: Linear::new(d_model, d_ff, rng),
            down: Linear::new(d_ff, d_model, rng),
        }
    }

    pub fn d_model(&self) -> usize {
        self.gate.in_features()
    }

    pub fn d_ff(&self) -> usize {
        self.gate.out_features()
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        self.forward_train(input).0
    }

    pub fn forward_train(&self, input: &Matrix) -> (Matrix, SwiGluCache) {
        let gate = self.gate.forward(input);
        let up = self.up.forward(input);

        // `silu` is an `exp` per element over a `[rows, d_ff]` matrix, which
        // is the largest point-wise pass in a step. It is embarrassingly
        // parallel, so it runs on every core rather than one.
        let mut hidden = Matrix::new(gate.rows, gate.cols);
        hidden
            .data
            .par_chunks_mut(CHUNK)
            .zip(gate.data.par_chunks(CHUNK))
            .zip(up.data.par_chunks(CHUNK))
            .for_each(|((destination, gate), up)| {
                for ((slot, &g), &u) in destination.iter_mut().zip(gate).zip(up) {
                    *slot = silu(g) * u;
                }
            });

        let output = self.down.forward(&hidden);
        (
            output,
            SwiGluCache {
                input: input.clone(),
                gate,
                up,
                hidden,
            },
        )
    }

    pub fn backward(&mut self, cache: &SwiGluCache, grad_output: &Matrix) -> Matrix {
        let grad_hidden = self.down.backward(&cache.hidden, grad_output);

        let mut grad_gate = Matrix::new(cache.gate.rows, cache.gate.cols);
        let mut grad_up = Matrix::new(cache.up.rows, cache.up.cols);
        grad_gate
            .data
            .par_chunks_mut(CHUNK)
            .zip(grad_up.data.par_chunks_mut(CHUNK))
            .zip(grad_hidden.data.par_chunks(CHUNK))
            .zip(cache.gate.data.par_chunks(CHUNK))
            .zip(cache.up.data.par_chunks(CHUNK))
            .for_each(|((((grad_gate, grad_up), upstream), gate), up)| {
                for index in 0..upstream.len() {
                    let g = gate[index];
                    grad_gate[index] = upstream[index] * up[index] * silu_derivative(g);
                    grad_up[index] = upstream[index] * silu(g);
                }
            });

        let mut grad_input = self.gate.backward(&cache.input, &grad_gate);
        let from_up = self.up.backward(&cache.input, &grad_up);
        grad_input
            .data
            .par_chunks_mut(CHUNK)
            .zip(from_up.data.par_chunks(CHUNK))
            .for_each(|(destination, source)| {
                for (slot, value) in destination.iter_mut().zip(source) {
                    *slot += value;
                }
            });

        grad_input
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn linears_mut(&mut self) -> Vec<&mut Linear> {
        vec![&mut self.gate, &mut self.up, &mut self.down]
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = self.gate.params_mut();
        params.extend(self.up.params_mut());
        params.extend(self.down.params_mut());
        params
    }

    /// Weight count, used by the parameter-count report.
    pub fn num_parameters(&self) -> usize {
        self.gate.weight.len() + self.up.weight.len() + self.down.weight.len()
    }
}

/// `GELU(x * W1^T) * W2^T`, the pre-SwiGLU feed-forward shape.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GeluMlp {
    pub up: Linear,
    pub down: Linear,
}

#[derive(Clone, Debug)]
pub struct GeluMlpCache {
    input: Matrix,
    pre_activation: Matrix,
    hidden: Matrix,
}

impl GeluMlp {
    pub fn new(d_model: usize, d_ff: usize, rng: &mut StdRng) -> Self {
        Self {
            up: Linear::new(d_model, d_ff, rng),
            down: Linear::new(d_ff, d_model, rng),
        }
    }

    pub fn d_model(&self) -> usize {
        self.up.in_features()
    }

    pub fn d_ff(&self) -> usize {
        self.up.out_features()
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        self.forward_train(input).0
    }

    pub fn forward_train(&self, input: &Matrix) -> (Matrix, GeluMlpCache) {
        let pre_activation = self.up.forward(input);

        let mut hidden = Matrix::new(pre_activation.rows, pre_activation.cols);
        hidden
            .data
            .par_chunks_mut(CHUNK)
            .zip(pre_activation.data.par_chunks(CHUNK))
            .for_each(|(destination, source)| {
                for (slot, &value) in destination.iter_mut().zip(source) {
                    *slot = gelu(value);
                }
            });

        let output = self.down.forward(&hidden);
        (
            output,
            GeluMlpCache {
                input: input.clone(),
                pre_activation,
                hidden,
            },
        )
    }

    pub fn backward(&mut self, cache: &GeluMlpCache, grad_output: &Matrix) -> Matrix {
        let mut grad_hidden = self.down.backward(&cache.hidden, grad_output);
        grad_hidden
            .data
            .par_chunks_mut(CHUNK)
            .zip(cache.pre_activation.data.par_chunks(CHUNK))
            .for_each(|(destination, source)| {
                for (slot, &value) in destination.iter_mut().zip(source) {
                    *slot *= gelu_derivative(value);
                }
            });
        self.up.backward(&cache.input, &grad_hidden)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn linears_mut(&mut self) -> Vec<&mut Linear> {
        vec![&mut self.up, &mut self.down]
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = self.up.params_mut();
        params.extend(self.down.params_mut());
        params
    }

    pub fn num_parameters(&self) -> usize {
        self.up.weight.len() + self.down.weight.len()
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn silu_derivative(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

/// The tanh approximation, which is what the GPT-2 and BERT reference
/// implementations use and is a good deal cheaper than `erf`.
pub fn gelu(x: f32) -> f32 {
    const COEFFICIENT: f32 = 0.797_884_6; // sqrt(2 / pi)
    0.5 * x * (1.0 + (COEFFICIENT * (x + 0.044_715 * x * x * x)).tanh())
}

fn gelu_derivative(x: f32) -> f32 {
    const COEFFICIENT: f32 = 0.797_884_6;
    let inner = COEFFICIENT * (x + 0.044_715 * x * x * x);
    let tanh = inner.tanh();
    let inner_derivative = COEFFICIENT * (1.0 + 3.0 * 0.044_715 * x * x);
    0.5 * (1.0 + tanh) + 0.5 * x * (1.0 - tanh * tanh) * inner_derivative
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn sum_of_outputs(output: &Matrix) -> f32 {
        output.data.iter().sum()
    }

    #[test]
    fn swiglu_preserves_the_model_dimension() {
        let mut rng = StdRng::seed_from_u64(1);
        let ffn = SwiGlu::new(8, 32, &mut rng);
        let output = ffn.forward(&Matrix::random(5, 8));

        assert_eq!(output.rows, 5);
        assert_eq!(output.cols, 8);
    }

    #[test]
    fn swiglu_backward_matches_finite_differences() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut ffn = SwiGlu::new(4, 6, &mut rng);
        let input = Matrix::from_vec(2, 4, vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.1, -0.3]);

        let (output, cache) = ffn.forward_train(&input);
        let grad_output = Matrix::from_vec(output.rows, output.cols, vec![1.0; output.data.len()]);
        let grad_input = ffn.backward(&cache, &grad_output);

        let epsilon = 1e-3;
        for index in 0..input.data.len() {
            let mut bumped = input.clone();
            bumped.data[index] += epsilon;
            let high = sum_of_outputs(&ffn.forward(&bumped));
            bumped.data[index] -= 2.0 * epsilon;
            let low = sum_of_outputs(&ffn.forward(&bumped));
            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (grad_input.data[index] - numeric).abs() < 1e-2,
                "index {index}: {} vs {numeric}",
                grad_input.data[index]
            );
        }
    }

    #[test]
    fn swiglu_gate_weight_gradient_matches_finite_differences() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut ffn = SwiGlu::new(3, 4, &mut rng);
        let input = Matrix::from_vec(2, 3, vec![0.4, -0.8, 1.2, 0.9, 0.3, -1.1]);

        let (output, cache) = ffn.forward_train(&input);
        let grad_output = Matrix::from_vec(output.rows, output.cols, vec![1.0; output.data.len()]);
        ffn.backward(&cache, &grad_output);
        let analytic = ffn.gate.weight.grad.data.clone();

        let epsilon = 1e-3;
        for (index, &expected) in analytic.iter().enumerate() {
            let mut probe = ffn.clone();
            probe.gate.weight.value.data[index] += epsilon;
            let high = sum_of_outputs(&probe.forward(&input));
            probe.gate.weight.value.data[index] -= 2.0 * epsilon;
            let low = sum_of_outputs(&probe.forward(&input));
            let numeric = (high - low) / (2.0 * epsilon);
            assert!((expected - numeric).abs() < 1e-2);
        }
    }

    #[test]
    fn gelu_mlp_backward_matches_finite_differences() {
        let mut rng = StdRng::seed_from_u64(4);
        let mut mlp = GeluMlp::new(4, 5, &mut rng);
        let input = Matrix::from_vec(2, 4, vec![0.2, -1.3, 0.8, 1.9, -0.4, 0.6, -2.0, 0.05]);

        let (output, cache) = mlp.forward_train(&input);
        let grad_output = Matrix::from_vec(output.rows, output.cols, vec![1.0; output.data.len()]);
        let grad_input = mlp.backward(&cache, &grad_output);

        let epsilon = 1e-3;
        for index in 0..input.data.len() {
            let mut bumped = input.clone();
            bumped.data[index] += epsilon;
            let high = sum_of_outputs(&mlp.forward(&bumped));
            bumped.data[index] -= 2.0 * epsilon;
            let low = sum_of_outputs(&mlp.forward(&bumped));
            let numeric = (high - low) / (2.0 * epsilon);
            assert!((grad_input.data[index] - numeric).abs() < 1e-2);
        }
    }

    #[test]
    fn activations_match_their_definitions_at_known_points() {
        assert!(silu(0.0).abs() < 1e-6);
        assert!((gelu(0.0)).abs() < 1e-6);
        // Both are asymptotically the identity for large positive inputs.
        assert!((silu(10.0) - 10.0).abs() < 1e-3);
        assert!((gelu(10.0) - 10.0).abs() < 1e-3);
    }
}
