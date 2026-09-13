//! Normalization layers. RMSNorm is what the decoder blocks use.

use crate::matrix::Matrix;
use crate::param::Param;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Rows per parallel tile in [`RmsNorm::backward`].
///
/// The scale gradient is a sum over every row, so the rows are grouped into
/// fixed tiles and the per-tile partials are added up in order afterwards. The
/// result is then independent of how rayon split the work, which is the same
/// convention `network.rs` and `causal_lm_loss.rs` follow.
const ROWS_PER_TILE: usize = 32;

/// Root-mean-square normalization: `x / sqrt(mean(x^2) + eps) * weight`.
///
/// No mean subtraction and no bias, which is what Qwen, Llama and the rest of
/// the current decoder family use. A `LayerNorm` would be this plus a mean term
/// and a bias `Param`; nothing here assumes RMSNorm specifically, so the block
/// can hold either once one exists.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RmsNorm {
    pub weight: Param,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(d_model: usize, eps: f32) -> Self {
        Self {
            weight: Param::filled(1, d_model, 1.0),
            eps,
        }
    }

    pub fn d_model(&self) -> usize {
        self.weight.value.cols
    }

    /// `[tokens, d_model] -> [tokens, d_model]`, row by row.
    pub fn forward(&self, input: &Matrix) -> Matrix {
        debug_assert_eq!(input.cols, self.d_model());
        let scale = &self.weight.value.data;
        let mut output = Matrix::new(input.rows, input.cols);

        // Rows are independent. Normalization is a large share of the
        // non-GEMM work in a step, and left serial it is the one place a
        // twelve-core host runs on one core.
        output
            .data
            .par_chunks_mut(input.cols)
            .zip(input.data.par_chunks(input.cols))
            .for_each(|(destination, source)| {
                let inverse_rms = self.inverse_rms(source);
                for ((slot, &value), &weight) in destination.iter_mut().zip(source).zip(scale) {
                    *slot = value * inverse_rms * weight;
                }
            });

        output
    }

    /// Accumulates the scale gradient and returns `dL/dinput`.
    ///
    /// The inverse RMS is recomputed rather than cached: it is one pass over a
    /// row the backward pass is already reading, and caching it would mean
    /// every caller carrying a per-layer buffer around.
    pub fn backward(&mut self, input: &Matrix, grad_output: &Matrix) -> Matrix {
        debug_assert_eq!(input.rows, grad_output.rows);
        debug_assert_eq!(input.cols, self.d_model());

        let width = self.d_model();
        let scale = &self.weight.value.data;
        let mut grad_input = Matrix::new(input.rows, width);

        let partials: Vec<Vec<f32>> = grad_input
            .data
            .par_chunks_mut(ROWS_PER_TILE * width)
            .enumerate()
            .map(|(tile, destination)| {
                let mut partial = vec![0.0f32; width];

                for (offset, target) in destination.chunks_mut(width).enumerate() {
                    let row = tile * ROWS_PER_TILE + offset;
                    let source = input.row(row);
                    let upstream = grad_output.row(row);
                    let inverse_rms = self.inverse_rms(source);

                    // d(inverse_rms)/dx_j pulls in every element of the row, so
                    // the shared term is summed first and then spread back over
                    // the row.
                    let mut projection = 0.0;
                    for ((&value, &upstream), &weight) in source.iter().zip(upstream).zip(scale) {
                        projection += upstream * weight * value;
                    }
                    let shared = projection * inverse_rms.powi(3) / width as f32;

                    for (index, slot) in target.iter_mut().enumerate() {
                        *slot =
                            upstream[index] * scale[index] * inverse_rms - source[index] * shared;
                    }

                    for ((slot, &value), &upstream) in partial.iter_mut().zip(source).zip(upstream)
                    {
                        *slot += upstream * value * inverse_rms;
                    }
                }

                partial
            })
            .collect();

        for partial in &partials {
            for (slot, value) in self.weight.grad.data.iter_mut().zip(partial) {
                *slot += value;
            }
        }

        grad_input
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        vec![&mut self.weight]
    }

    fn inverse_rms(&self, row: &[f32]) -> f32 {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
        (mean_square + self.eps).sqrt().recip()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_normalizes_to_unit_rms() {
        let norm = RmsNorm::new(4, 1e-6);
        let input = Matrix::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]);

        let output = norm.forward(&input);

        let mean_square = output.data.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((mean_square - 1.0).abs() < 1e-4);
    }

    #[test]
    fn forward_does_not_subtract_the_mean() {
        // A constant row has zero variance; LayerNorm would map it to zeros,
        // RMSNorm must leave its sign and magnitude ratio intact.
        let norm = RmsNorm::new(3, 1e-6);
        let output = norm.forward(&Matrix::from_vec(1, 3, vec![5.0, 5.0, 5.0]));

        for value in output.data {
            assert!((value - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn weight_scales_each_channel() {
        let mut norm = RmsNorm::new(2, 1e-6);
        norm.weight.value.data = vec![2.0, 0.5];

        let output = norm.forward(&Matrix::from_vec(1, 2, vec![1.0, 1.0]));

        assert!((output.data[0] - 2.0).abs() < 1e-4);
        assert!((output.data[1] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn backward_matches_finite_differences() {
        let mut norm = RmsNorm::new(4, 1e-6);
        norm.weight.value.data = vec![1.5, 0.5, -2.0, 1.0];
        let input = Matrix::from_vec(2, 4, vec![0.3, -1.2, 2.5, 0.1, -0.7, 0.4, 1.1, -1.9]);
        let grad_output = Matrix::from_vec(2, 4, vec![1.0; 8]);

        let grad_input = norm.backward(&input, &grad_output);

        let epsilon = 1e-3;
        for index in 0..input.data.len() {
            let mut bumped = input.clone();
            bumped.data[index] += epsilon;
            let high: f32 = norm.forward(&bumped).data.iter().sum();
            bumped.data[index] -= 2.0 * epsilon;
            let low: f32 = norm.forward(&bumped).data.iter().sum();
            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (grad_input.data[index] - numeric).abs() < 1e-2,
                "index {index}: {} vs {numeric}",
                grad_input.data[index]
            );
        }
    }

    #[test]
    fn weight_gradient_matches_finite_differences() {
        let mut norm = RmsNorm::new(3, 1e-6);
        let input = Matrix::from_vec(2, 3, vec![0.5, -1.0, 2.0, 1.5, 0.25, -0.5]);
        let grad_output = Matrix::from_vec(2, 3, vec![1.0; 6]);

        norm.backward(&input, &grad_output);
        let analytic = norm.weight.grad.data.clone();

        let epsilon = 1e-3;
        for (index, &expected) in analytic.iter().enumerate() {
            let mut probe = norm.clone();
            probe.weight.value.data[index] += epsilon;
            let high: f32 = probe.forward(&input).data.iter().sum();
            probe.weight.value.data[index] -= 2.0 * epsilon;
            let low: f32 = probe.forward(&input).data.iter().sum();
            let numeric = (high - low) / (2.0 * epsilon);
            assert!((expected - numeric).abs() < 1e-2);
        }
    }
}
