//! Next-token cross-entropy, plus the auxiliary losses MoE layers report.

use crate::activations::softmax;
use crate::batch::TokenBatch;
use crate::matrix::Matrix;
use crate::network::NetworkError;
use rayon::prelude::*;

/// Cross-entropy over shifted targets, and its gradient with respect to the
/// logits.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalLmLoss {
    /// Mean negative log-likelihood over the predicted positions.
    pub loss: f32,
    /// `[tokens, vocab_size]`. The final row is zero: nothing follows the last
    /// token, so it predicts nothing.
    pub grad_logits: Matrix,
    /// How many positions the mean was taken over.
    pub predicted: usize,
}

impl CausalLmLoss {
    /// `exp(loss)`, the usual way to report language-model quality.
    pub fn perplexity(&self) -> f32 {
        self.loss.exp()
    }
}

/// Language-modelling loss plus every MoE layer's auxiliary losses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TotalLoss {
    pub lm_loss: f32,
    /// Sum of the load-balancing and z-losses over all MoE layers, already
    /// weighted by their configured coefficients.
    pub auxiliary_loss: f32,
}

impl TotalLoss {
    /// What the optimizer is minimizing.
    pub fn total(&self) -> f32 {
        self.lm_loss + self.auxiliary_loss
    }
}

/// Cross-entropy of `logits[t]` against `ids[t + 1]`.
///
/// The shift happens here rather than in the caller, because getting it wrong
/// produces a model that trains happily and predicts the token it was just
/// given.
pub fn causal_lm_loss(logits: &Matrix, ids: &[u32]) -> Result<CausalLmLoss, NetworkError> {
    causal_lm_loss_batch(logits, &TokenBatch::new(&[ids])?)
}

/// The same loss over a packed batch.
///
/// Padding rows and the last position of every sequence predict nothing, so
/// their gradient rows stay zero; the mean is taken over the positions that do
/// predict, which is what makes a ragged batch weight every token equally
/// rather than every sequence equally.
pub fn causal_lm_loss_batch(
    logits: &Matrix,
    batch: &TokenBatch,
) -> Result<CausalLmLoss, NetworkError> {
    if logits.rows != batch.rows() {
        return Err(NetworkError::InvalidTarget {
            expected: logits.rows,
            actual: batch.rows(),
        });
    }

    let vocab_size = logits.cols;
    let seq_len = batch.seq_len();
    let ids = batch.ids();
    let predicted = batch.predicted();
    if predicted == 0 {
        return Err(NetworkError::InvalidConfig(
            "a causal language-modelling loss needs at least two tokens".into(),
        ));
    }

    let mut grad_logits = Matrix::new(logits.rows, vocab_size);
    let lengths = batch.lengths();
    let inverse_predicted = 1.0 / predicted as f32;

    // One softmax per row over a 32k-wide vocabulary is the single most
    // expensive host phase of a step, and the rows are independent. `map` into
    // an ordered `collect` rather than a parallel reduction, so the summed
    // loss does not depend on how rayon split the work.
    let per_row: Vec<f32> = grad_logits
        .data
        .par_chunks_mut(vocab_size)
        .enumerate()
        .map(|(row_index, row)| {
            let position = row_index % seq_len;
            // Padding rows and the last position of a sequence predict nothing.
            if position + 1 >= lengths[row_index / seq_len] {
                return Ok(0.0);
            }
            let target = ids[row_index + 1] as usize;
            if target >= vocab_size {
                return Err(NetworkError::TokenOutOfRange {
                    id: ids[row_index + 1],
                    vocab_size,
                });
            }

            row.copy_from_slice(logits.row(row_index));
            softmax(row);

            // `softmax` already subtracts the row maximum, so the probability
            // is safe to take a logarithm of once clamped away from zero.
            let contribution = -row[target].max(f32::MIN_POSITIVE).ln();

            for value in row.iter_mut() {
                *value *= inverse_predicted;
            }
            row[target] -= inverse_predicted;
            Ok(contribution)
        })
        .collect::<Result<Vec<f32>, NetworkError>>()?;
    let loss: f32 = per_row.iter().sum();

    Ok(CausalLmLoss {
        loss: loss / predicted as f32,
        grad_logits,
        predicted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_logits_give_the_log_of_the_vocabulary() {
        let logits = Matrix::new(3, 4);
        let result = causal_lm_loss(&logits, &[0, 1, 2]).unwrap();

        assert_eq!(result.predicted, 2);
        assert!((result.loss - 4.0f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn the_last_position_has_no_target() {
        let logits = Matrix::from_vec(2, 2, vec![0.5, -0.5, 2.0, -2.0]);
        let result = causal_lm_loss(&logits, &[0, 1]).unwrap();

        assert_eq!(result.grad_logits.row(1), &[0.0, 0.0]);
        assert!(result.grad_logits.row(0).iter().any(|&g| g != 0.0));
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let logits = Matrix::from_vec(3, 3, vec![0.2, -1.0, 0.5, 1.5, 0.3, -0.7, 0.1, 0.9, -0.2]);
        let ids = [2u32, 0, 1];
        let analytic = causal_lm_loss(&logits, &ids).unwrap().grad_logits;

        let epsilon = 1e-3;
        for index in 0..logits.data.len() {
            let mut bumped = logits.clone();
            bumped.data[index] += epsilon;
            let high = causal_lm_loss(&bumped, &ids).unwrap().loss;
            bumped.data[index] -= 2.0 * epsilon;
            let low = causal_lm_loss(&bumped, &ids).unwrap().loss;
            let numeric = (high - low) / (2.0 * epsilon);
            assert!((analytic.data[index] - numeric).abs() < 1e-3);
        }
    }

    #[test]
    fn confident_and_correct_beats_confident_and_wrong() {
        let right = Matrix::from_vec(2, 2, vec![-5.0, 5.0, 0.0, 0.0]);
        let wrong = Matrix::from_vec(2, 2, vec![5.0, -5.0, 0.0, 0.0]);

        let right = causal_lm_loss(&right, &[0, 1]).unwrap().loss;
        let wrong = causal_lm_loss(&wrong, &[0, 1]).unwrap().loss;

        assert!(right < wrong);
    }

    #[test]
    fn a_single_token_has_nothing_to_predict() {
        assert!(causal_lm_loss(&Matrix::new(1, 4), &[0]).is_err());
    }

    #[test]
    fn total_loss_adds_the_auxiliary_terms() {
        let total = TotalLoss {
            lm_loss: 2.0,
            auxiliary_loss: 0.25,
        };
        assert!((total.total() - 2.25).abs() < 1e-6);
    }
}
