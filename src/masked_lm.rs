//! Masked language modelling: the encoder-only training objective.
//!
//! A causal model learns from every position at once because each one predicts
//! the token after it, and a bidirectional model cannot: every position already
//! reads the token it would be asked to predict. The objective instead corrupts
//! part of the input and scores the model on reconstructing what was there, so
//! a prediction has to come from both sides of the hole.
//!
//! The batch carries the corrupted ids, so the forward pass, the blocks and the
//! head are exactly the ones a causal model uses. Only the loss differs: it
//! reads a row's own token rather than the next one, and only at the positions
//! that were corrupted.

use crate::batch::TokenBatch;
use crate::causal_lm_loss::{CausalLmLoss, cross_entropy_rows};
use crate::matrix::Matrix;
use crate::network::NetworkError;
use rand::{Rng, SeedableRng, rngs::StdRng};

/// A batch whose inputs were corrupted and whose original tokens are the
/// targets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskedBatch {
    inputs: TokenBatch,
    targets: Vec<u32>,
    masked: Vec<bool>,
}

impl MaskedBatch {
    /// Corrupts `probability` of the tokens in each sequence and returns the
    /// batch to train on.
    ///
    /// The corruption is BERT's: of the chosen positions, 80% become `mask_id`,
    /// 10% become a uniformly random token and 10% are left alone. The last two
    /// exist because a position that is only ever the mask token teaches the
    /// model nothing about the tokens it will actually see at inference, where
    /// nothing is masked at all.
    ///
    /// Every sequence gets at least one corrupted position, so a short sequence
    /// or a low probability cannot produce a batch with nothing to learn from.
    ///
    /// ```
    /// # use rusting_brain::MaskedBatch;
    /// let batch = MaskedBatch::corrupt(&[[3u32, 4, 5, 6]], 32, 1, 0.15, Some(7))?;
    /// assert_eq!(batch.targets(), &[3, 4, 5, 6]);
    /// assert!(batch.masked() >= 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn corrupt<S: AsRef<[u32]>>(
        sequences: &[S],
        vocab_size: usize,
        mask_id: u32,
        probability: f32,
        seed: Option<u64>,
    ) -> Result<Self, NetworkError> {
        if mask_id as usize >= vocab_size {
            return Err(NetworkError::TokenOutOfRange {
                id: mask_id,
                vocab_size,
            });
        }
        if !(0.0..=1.0).contains(&probability) {
            return Err(NetworkError::InvalidConfig(format!(
                "a masking probability is a fraction, got {probability}"
            )));
        }

        let inputs = TokenBatch::new(sequences)?;
        let targets = inputs.ids().to_vec();
        let seq_len = inputs.seq_len();
        let mut ids = targets.clone();
        let mut masked = vec![false; ids.len()];
        let mut rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        for (sequence, &length) in inputs.lengths().iter().enumerate() {
            let base = sequence * seq_len;
            for position in 0..length {
                if rng.gen_range(0.0..1.0f32) < probability {
                    masked[base + position] = true;
                }
            }
            if !masked[base..base + length].iter().any(|&flag| flag) {
                masked[base + rng.gen_range(0..length)] = true;
            }
            for position in 0..length {
                if !masked[base + position] {
                    continue;
                }
                let roll = rng.gen_range(0.0..1.0f32);
                if roll < 0.8 {
                    ids[base + position] = mask_id;
                } else if roll < 0.9 {
                    ids[base + position] = rng.gen_range(0..vocab_size as u32);
                }
            }
        }

        Self::new(inputs.replacing_ids(&ids)?, &targets, &masked)
    }

    /// The same batch from a caller's own corruption: `inputs` holds what the
    /// model reads, `targets` what it should reconstruct, and `masked` one flag
    /// per row saying which positions count toward the loss.
    ///
    /// Use this for a whole-word or span corruption, which needs the tokenizer
    /// that produced the ids and so cannot live here.
    pub fn new(inputs: TokenBatch, targets: &[u32], masked: &[bool]) -> Result<Self, NetworkError> {
        if targets.len() != inputs.rows() || masked.len() != inputs.rows() {
            return Err(NetworkError::InvalidConfig(format!(
                "a masked batch of {} rows needs {} targets and {} flags, got {} and {}",
                inputs.rows(),
                inputs.rows(),
                inputs.rows(),
                targets.len(),
                masked.len()
            )));
        }
        let seq_len = inputs.seq_len();
        let padding = (0..inputs.rows())
            .find(|&row| masked[row] && row % seq_len >= inputs.lengths()[row / seq_len]);
        if let Some(row) = padding {
            return Err(NetworkError::InvalidConfig(format!(
                "row {row} is padding, so it cannot be a masked target"
            )));
        }
        if !masked.iter().any(|&flag| flag) {
            return Err(NetworkError::InvalidConfig(
                "a masked batch with no masked position has nothing to learn from".into(),
            ));
        }

        Ok(Self {
            inputs,
            targets: targets.to_vec(),
            masked: masked.to_vec(),
        })
    }

    /// What the model reads: the corrupted ids.
    pub fn inputs(&self) -> &TokenBatch {
        &self.inputs
    }

    /// What it should reconstruct: the ids before corruption.
    pub fn targets(&self) -> &[u32] {
        &self.targets
    }

    /// Whether row `row` counts toward the loss.
    pub fn is_masked(&self, row: usize) -> bool {
        self.masked[row]
    }

    /// How many positions the loss covers.
    pub fn masked(&self) -> usize {
        self.masked.iter().filter(|&&flag| flag).count()
    }
}

/// Cross-entropy of each masked row's logits against the token that was there.
///
/// Unmasked and padding rows keep a zero gradient: they are context, and
/// scoring them would be scoring the model on copying its own input.
pub fn masked_lm_loss(logits: &Matrix, batch: &MaskedBatch) -> Result<CausalLmLoss, NetworkError> {
    if logits.rows != batch.inputs.rows() {
        return Err(NetworkError::InvalidTarget {
            expected: logits.rows,
            actual: batch.inputs.rows(),
        });
    }

    let targets = batch.targets();
    cross_entropy_rows(logits, batch.masked(), |row| {
        batch.is_masked(row).then(|| targets[row])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corruption_keeps_the_targets_and_marks_what_it_changed() {
        let sequences = [[3u32, 4, 5, 6, 7, 8, 9, 10]];
        let batch = MaskedBatch::corrupt(&sequences, 32, 1, 0.5, Some(4)).unwrap();

        assert_eq!(batch.targets(), &sequences[0]);
        assert!(batch.masked() > 1);
        for row in 0..batch.inputs().rows() {
            if !batch.is_masked(row) {
                // An untouched position reads exactly what it was.
                assert_eq!(batch.inputs().ids()[row], batch.targets()[row]);
            }
        }
        // Most masked positions became the mask id, and none of the ids left
        // the vocabulary.
        let masked_positions = (0..batch.inputs().rows()).filter(|&row| batch.is_masked(row));
        assert!(
            masked_positions
                .clone()
                .any(|row| batch.inputs().ids()[row] == 1)
        );
        assert!(
            masked_positions
                .map(|row| batch.inputs().ids()[row])
                .all(|id| id < 32)
        );
    }

    #[test]
    fn every_sequence_gets_a_masked_position_even_at_probability_zero() {
        let batch =
            MaskedBatch::corrupt(&[[3u32, 4], [5, 6], [7, 8]], 16, 0, 0.0, Some(1)).unwrap();

        assert_eq!(batch.masked(), 3);
        for sequence in 0..3 {
            let base = sequence * batch.inputs().seq_len();
            assert!((base..base + 2).any(|row| batch.is_masked(row)));
        }
    }

    #[test]
    fn padding_cannot_be_a_target_and_a_short_batch_still_masks_real_tokens() {
        let batch =
            MaskedBatch::corrupt(&[vec![3u32, 4, 5, 6], vec![7, 8]], 16, 1, 1.0, Some(2)).unwrap();
        // Four rows per sequence, but the second sequence has two real tokens.
        assert_eq!(batch.masked(), 6);

        let inputs = TokenBatch::new(&[vec![3u32, 4, 5, 6], vec![7, 8]]).unwrap();
        let mut masked = vec![false; inputs.rows()];
        masked[7] = true; // padding
        let error = MaskedBatch::new(inputs, &[0; 8], &masked)
            .unwrap_err()
            .to_string();
        assert!(error.contains("padding"), "{error}");
    }

    #[test]
    fn the_loss_scores_masked_positions_only() {
        let inputs = TokenBatch::new(&[[1u32, 1, 1]]).unwrap();
        let batch = MaskedBatch::new(inputs, &[2, 0, 1], &[false, true, false]).unwrap();
        let logits = Matrix::new(3, 4);

        let loss = masked_lm_loss(&logits, &batch).unwrap();
        assert_eq!(loss.predicted, 1);
        assert!((loss.loss - 4.0f32.ln()).abs() < 1e-5);
        assert_eq!(loss.grad_logits.row(0), &[0.0; 4]);
        assert_eq!(loss.grad_logits.row(2), &[0.0; 4]);
        assert!(loss.grad_logits.row(1).iter().any(|&value| value != 0.0));
        // The last row has a target here, which a causal loss never gives it.
        let last = MaskedBatch::new(
            TokenBatch::new(&[[1u32, 1, 1]]).unwrap(),
            &[2, 0, 1],
            &[false, false, true],
        )
        .unwrap();
        assert!(masked_lm_loss(&logits, &last).unwrap().grad_logits.row(2)[1] != 0.0);
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let inputs = TokenBatch::new(&[[1u32, 1, 1]]).unwrap();
        let batch = MaskedBatch::new(inputs, &[2, 0, 1], &[true, false, true]).unwrap();
        let logits = Matrix::from_vec(3, 3, vec![0.2, -1.0, 0.5, 1.5, 0.3, -0.7, 0.1, 0.9, -0.2]);
        let analytic = masked_lm_loss(&logits, &batch).unwrap().grad_logits;

        let epsilon = 1e-3;
        for index in 0..logits.data.len() {
            let mut bumped = logits.clone();
            bumped.data[index] += epsilon;
            let high = masked_lm_loss(&bumped, &batch).unwrap().loss;
            bumped.data[index] -= 2.0 * epsilon;
            let low = masked_lm_loss(&bumped, &batch).unwrap().loss;
            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (analytic.data[index] - numeric).abs() < 1e-3,
                "index {index}"
            );
        }
    }
}
