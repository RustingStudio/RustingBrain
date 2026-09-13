//! Token embedding, optionally shared with the output projection.

use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::param::Param;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

/// A `[vocab_size, d_model]` lookup table.
///
/// The same matrix serves as the output unembedding when weights are tied, so
/// [`Embedding::unembed`] lives here rather than in a separate head: tying two
/// owners of one matrix in Rust means either shared interior mutability or one
/// owner with two methods, and the second is far less machinery.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Embedding {
    pub weight: Param,
}

impl Embedding {
    /// Normally-ish distributed small values. Embeddings have no fan-in, so the
    /// He rule used for projections does not apply; 0.02 is the initialization
    /// scale GPT-2 and the Llama family use.
    pub fn new(vocab_size: usize, d_model: usize, rng: &mut StdRng) -> Self {
        let data = (0..vocab_size * d_model)
            .map(|_| rng.gen_range(-0.02..0.02))
            .collect();
        Self {
            weight: Param::new(Matrix::from_vec(vocab_size, d_model, data)),
        }
    }

    pub fn with_seed(vocab_size: usize, d_model: usize, seed: u64) -> Self {
        Self::new(vocab_size, d_model, &mut StdRng::seed_from_u64(seed))
    }

    pub fn vocab_size(&self) -> usize {
        self.weight.value.rows
    }

    pub fn d_model(&self) -> usize {
        self.weight.value.cols
    }

    /// Gathers one row per token id: `[tokens] -> [tokens, d_model]`.
    pub fn forward(&self, ids: &[u32]) -> Result<Matrix, NetworkError> {
        let d_model = self.d_model();
        #[cfg(feature = "cuda")]
        if let Some(device) = &self.weight.device {
            for &id in ids {
                self.row_index(id)?;
            }
            return Ok(device.gather(ids));
        }
        let mut output = Matrix::new(ids.len(), d_model);

        for (position, &id) in ids.iter().enumerate() {
            let row = self.row_index(id)?;
            output
                .row_mut(position)
                .copy_from_slice(self.weight.value.row(row));
        }

        Ok(output)
    }

    /// Scatters `dL/doutput` back into the rows that were read.
    ///
    /// Only the rows actually gathered are touched, so the cost is proportional
    /// to the number of tokens rather than to the vocabulary.
    pub fn backward(&mut self, ids: &[u32], grad_output: &Matrix) -> Result<(), NetworkError> {
        debug_assert_eq!(grad_output.rows, ids.len());
        let d_model = self.d_model();
        #[cfg(feature = "cuda")]
        if self.weight.device.is_some() {
            for &id in ids {
                self.row_index(id)?;
            }
            if let Some(device) = &mut self.weight.device {
                device.scatter_grad(ids, grad_output);
            }
            return Ok(());
        }

        for (position, &id) in ids.iter().enumerate() {
            let row = self.row_index(id)?;
            let grad_row = &grad_output.data[position * d_model..(position + 1) * d_model];
            for (slot, value) in self.weight.grad.row_mut(row).iter_mut().zip(grad_row) {
                *slot += value;
            }
        }

        Ok(())
    }

    /// Tied output projection: `logits = hidden * weight^T`.
    pub fn unembed(&self, hidden: &Matrix) -> Matrix {
        debug_assert_eq!(hidden.cols, self.d_model());
        #[cfg(feature = "cuda")]
        if let Some(device) = &self.weight.device {
            return device.matmul_rhs_transposed(hidden);
        }
        let mut logits = Matrix::new(hidden.rows, self.vocab_size());
        hidden.dot_rhs_transposed(&self.weight.value, &mut logits);
        logits
    }

    /// Backward of [`Embedding::unembed`]. Accumulates into the same gradient
    /// buffer as [`Embedding::backward`], which is exactly what tying means.
    pub fn unembed_backward(&mut self, hidden: &Matrix, grad_logits: &Matrix) -> Matrix {
        debug_assert_eq!(grad_logits.cols, self.vocab_size());
        #[cfg(feature = "cuda")]
        if let Some(device) = &mut self.weight.device {
            device.accumulate_grad(grad_logits, hidden);
            return device.matmul(grad_logits);
        }
        grad_logits.dot_self_transposed_accumulate(hidden, &mut self.weight.grad);

        let mut grad_hidden = Matrix::new(hidden.rows, self.d_model());
        grad_logits.dot(&self.weight.value, &mut grad_hidden);
        grad_hidden
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        vec![&mut self.weight]
    }

    fn row_index(&self, id: u32) -> Result<usize, NetworkError> {
        let row = id as usize;
        if row >= self.vocab_size() {
            return Err(NetworkError::TokenOutOfRange {
                id,
                vocab_size: self.vocab_size(),
            });
        }
        Ok(row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Embedding {
        // Rows are 10, 20, 30 so a gather is obvious by inspection.
        Embedding {
            weight: Param::new(Matrix::from_vec(
                3,
                2,
                vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
            )),
        }
    }

    #[test]
    fn forward_gathers_rows_by_id() {
        let embedding = fixture();
        let output = embedding.forward(&[2, 0, 2]).unwrap();

        assert_eq!(output.rows, 3);
        assert_eq!(output.cols, 2);
        assert_eq!(output.data, vec![30.0, 31.0, 10.0, 11.0, 30.0, 31.0]);
    }

    #[test]
    fn forward_rejects_ids_past_the_vocabulary() {
        assert!(matches!(
            fixture().forward(&[3]),
            Err(NetworkError::TokenOutOfRange { id: 3, .. })
        ));
    }

    #[test]
    fn backward_accumulates_into_repeated_rows_only() {
        let mut embedding = fixture();
        let grad = Matrix::from_vec(3, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);

        embedding.backward(&[2, 0, 2], &grad).unwrap();

        // Row 1 was never gathered, so it must stay untouched.
        assert_eq!(embedding.weight.grad.row(0), &[2.0, 2.0]);
        assert_eq!(embedding.weight.grad.row(1), &[0.0, 0.0]);
        assert_eq!(embedding.weight.grad.row(2), &[4.0, 4.0]);
    }

    #[test]
    fn unembed_projects_onto_the_vocabulary() {
        let embedding = fixture();
        let hidden = Matrix::from_vec(1, 2, vec![1.0, 0.0]);

        let logits = embedding.unembed(&hidden);

        assert_eq!(logits.cols, 3);
        assert_eq!(logits.data, vec![10.0, 20.0, 30.0]);
    }
}
