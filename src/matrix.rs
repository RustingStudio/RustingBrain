//! Row-major `f32` matrices and the matmuls the CPU path is built on.
//!
//! Every product writes into a caller-owned `target` rather than returning a
//! new matrix, so a training step allocates nothing after the first one. The
//! five shapes here are the five that backpropagation actually needs; they
//! exist separately because transposing an operand costs more than reading it
//! in a different order.

use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

/// Dot product with eight accumulators.
///
/// Floating-point addition is not associative, so a single running sum is a
/// dependency chain the compiler is not allowed to break: it stalls on the
/// four-cycle latency of each add where it could be issuing a vector's worth
/// every cycle. Eight independent lanes give it something to pipeline.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut lanes = [0.0f32; 8];
    let tail = a.len() - a.len() % 8;
    for (x, y) in a[..tail].chunks_exact(8).zip(b[..tail].chunks_exact(8)) {
        for lane in 0..8 {
            lanes[lane] += x[lane] * y[lane];
        }
    }
    let mut rest = 0.0f32;
    for (x, y) in a[tail..].iter().zip(&b[tail..]) {
        rest += x * y;
    }
    lanes.iter().sum::<f32>() + rest
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(0.0..1.0)).collect();
        Self { rows, cols, data }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }

    pub fn zeros(&mut self) {
        self.data.fill(0.0);
    }

    pub fn dot(&self, other: &Matrix, target: &mut Matrix) {
        debug_assert_eq!(self.cols, other.rows);
        debug_assert_eq!(target.rows, self.rows);
        debug_assert_eq!(target.cols, other.cols);

        unsafe {
            matrixmultiply::sgemm(
                self.rows,
                self.cols,
                other.cols,
                1.0,
                self.data.as_ptr(),
                self.cols as isize,
                1,
                other.data.as_ptr(),
                other.cols as isize,
                1,
                0.0,
                target.data.as_mut_ptr(),
                target.cols as isize,
                1,
            );
        }
    }

    pub fn dot_rhs_transposed(&self, other: &Matrix, target: &mut Matrix) {
        debug_assert_eq!(self.cols, other.cols);
        debug_assert_eq!(target.rows, self.rows);
        debug_assert_eq!(target.cols, other.rows);

        // A cached decode step feeds one token at a time, and `sgemm` with
        // `m == 1` spends all its time packing panels for a blocked kernel that
        // never gets a second row. A plain row-times-row dot product over
        // contiguous memory beats it by a wide margin, and the compiler
        // vectorizes it.
        if self.rows == 1 {
            let k = self.cols;
            let input = &self.data;
            let row = |weight: &[f32]| dot(input, weight);
            // One decoded token against a 32000-row unembedding streams 16 MB of
            // weights, which is several times what one core can pull from memory
            // in the time the rest of the token takes. The projections inside a
            // block are small enough that the fork costs more than the work, so
            // they stay on this thread.
            const PARALLEL_FLOATS: usize = 1 << 18;
            if other.data.len() >= PARALLEL_FLOATS {
                target
                    .data
                    .par_iter_mut()
                    .zip(other.data.par_chunks_exact(k))
                    .for_each(|(slot, weight)| *slot = row(weight));
            } else {
                for (slot, weight) in target.data.iter_mut().zip(other.data.chunks_exact(k)) {
                    *slot = row(weight);
                }
            }
            return;
        }

        unsafe {
            matrixmultiply::sgemm(
                self.rows,
                self.cols,
                other.rows,
                1.0,
                self.data.as_ptr(),
                self.cols as isize,
                1,
                other.data.as_ptr(),
                1,
                other.cols as isize,
                0.0,
                target.data.as_mut_ptr(),
                target.cols as isize,
                1,
            );
        }
    }

    pub fn dot_self_transposed(&self, other: &Matrix, target: &mut Matrix) {
        debug_assert_eq!(self.rows, other.rows);
        debug_assert_eq!(target.rows, self.cols);
        debug_assert_eq!(target.cols, other.cols);

        unsafe {
            matrixmultiply::sgemm(
                self.cols,
                self.rows,
                other.cols,
                1.0,
                self.data.as_ptr(),
                1,
                self.cols as isize,
                other.data.as_ptr(),
                other.cols as isize,
                1,
                0.0,
                target.data.as_mut_ptr(),
                target.cols as isize,
                1,
            );
        }
    }

    pub fn outer_product(&self, input: &Matrix, target: &mut Matrix) {
        debug_assert_eq!(input.cols, 1);
        debug_assert_eq!(target.rows, self.rows);
        debug_assert_eq!(target.cols, input.rows);

        unsafe {
            matrixmultiply::sgemm(
                self.rows,
                1,
                input.rows,
                1.0,
                self.data.as_ptr(),
                self.cols as isize,
                1,
                input.data.as_ptr(),
                1,
                input.cols as isize,
                0.0,
                target.data.as_mut_ptr(),
                target.cols as isize,
                1,
            );
        }
    }

    pub fn dot_transpose_self(&self, error: &Matrix, target: &mut Matrix) {
        debug_assert_eq!(self.rows, error.rows);
        debug_assert_eq!(target.rows, self.cols);
        debug_assert_eq!(target.cols, 1);

        unsafe {
            matrixmultiply::sgemm(
                self.cols,
                self.rows,
                1,
                1.0,
                self.data.as_ptr(),
                1,
                self.cols as isize,
                error.data.as_ptr(),
                error.cols as isize,
                1,
                0.0,
                target.data.as_mut_ptr(),
                target.cols as isize,
                1,
            );
        }
    }

    pub fn copy_from_slice(&mut self, source: &[f32]) {
        self.data.copy_from_slice(source);
    }

    /// Row `index` as a contiguous slice. The transformer layers keep one token
    /// per row, so this is the natural unit of work for them.
    pub fn row(&self, index: usize) -> &[f32] {
        debug_assert!(index < self.rows);
        &self.data[index * self.cols..(index + 1) * self.cols]
    }

    pub fn row_mut(&mut self, index: usize) -> &mut [f32] {
        debug_assert!(index < self.rows);
        &mut self.data[index * self.cols..(index + 1) * self.cols]
    }

    /// `target += self^T * other`, the accumulating form of
    /// [`Matrix::dot_self_transposed`].
    ///
    /// Weight gradients are summed over a whole batch and often over several
    /// call sites (a shared embedding matrix is written by both the gather and
    /// the unembedding), so the accumulating form avoids a full-size scratch
    /// matrix and an extra pass per call.
    pub fn dot_self_transposed_accumulate(&self, other: &Matrix, target: &mut Matrix) {
        debug_assert_eq!(self.rows, other.rows);
        debug_assert_eq!(target.rows, self.cols);
        debug_assert_eq!(target.cols, other.cols);

        unsafe {
            matrixmultiply::sgemm(
                self.cols,
                self.rows,
                other.cols,
                1.0,
                self.data.as_ptr(),
                1,
                self.cols as isize,
                other.data.as_ptr(),
                other.cols as isize,
                1,
                1.0,
                target.data.as_mut_ptr(),
                target.cols as isize,
                1,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_multiplies_row_major_matrices() {
        let left = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let right = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let mut output = Matrix::new(2, 2);

        left.dot(&right, &mut output);

        assert_eq!(output.data, vec![58.0, 64.0, 139.0, 154.0]);
    }
}
