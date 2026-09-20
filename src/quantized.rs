//! Weights stored as one byte per value, for inference.
//!
//! Decoding one token reads every active weight once and does two flops with
//! it, so a CPU decode step is bound by memory bandwidth and nothing else: this
//! machine reads 60 GB/s and a 116M-active-parameter model needs 464 MB per
//! token as `f32`. One byte per weight is a quarter of that.
//!
//! The scheme is the one [`Precision::Q8`](crate::transformer::Precision)
//! already writes to disk: per row, `scale = absmax / 127` and `round(w /
//! scale)` as `i8`. Rounding to 255 levels costs about 0.4% relative error per
//! weight, which inference absorbs and training does not — so quantizing is
//! one-way, and the model refuses to train or save afterwards.

use crate::matrix::Matrix;
use rayon::prelude::*;

/// A `[rows, cols]` weight matrix as `i8` values with one `f32` scale per row.
#[derive(Clone, Debug, PartialEq)]
pub struct Quantized {
    rows: usize,
    cols: usize,
    data: Vec<i8>,
    scales: Vec<f32>,
}

impl Quantized {
    /// Quantizes row by row. A row of zeros gets a scale of zero, which
    /// dequantizes back to zeros rather than to `NaN`.
    pub(crate) fn from_matrix(value: &Matrix) -> Self {
        let cols = value.cols.max(1);
        let mut data = Vec::with_capacity(value.data.len());
        let mut scales = Vec::with_capacity(value.rows);

        for row in value.data.chunks(cols) {
            let absmax = row.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
            let scale = absmax / 127.0;
            scales.push(scale);
            let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            data.extend(
                row.iter()
                    .map(|v| (v * inverse).round().clamp(-127.0, 127.0) as i8),
            );
        }

        Self {
            rows: value.rows,
            cols: value.cols,
            data,
            scales,
        }
    }

    /// `target[t, r] = input[t, ..] . row(r)`, the shape
    /// [`Matrix::dot_rhs_transposed`] produces.
    pub(crate) fn matmul_rhs_transposed(&self, input: &Matrix) -> Matrix {
        debug_assert_eq!(input.cols, self.cols);
        let mut output = Matrix::new(input.rows, self.rows);
        let cols = self.cols.max(1);

        fn row_of<'a>(activations: &'a [f32]) -> impl Fn((&mut f32, (&[i8], &f32))) + Sync + 'a {
            move |(slot, (weights, scale))| *slot = dot(activations, weights) * scale
        }

        // A sequence of tokens is the cheapest thing to fork on: the rows are
        // independent and each one keeps a whole core busy. One token — a
        // decode step — has no such axis, so the fork goes across the weight's
        // rows instead, and a small weight is not worth forking at all.
        const PARALLEL_VALUES: usize = 1 << 18;
        if input.rows > 1 {
            output
                .data
                .par_chunks_mut(self.rows)
                .enumerate()
                .for_each(|(token, target)| {
                    let activations = input.row(token);
                    target
                        .iter_mut()
                        .zip(self.data.chunks_exact(cols).zip(self.scales.iter()))
                        .for_each(row_of(activations));
                });
        } else if self.data.len() >= PARALLEL_VALUES {
            for (token, target) in output.data.chunks_mut(self.rows).enumerate() {
                let activations = input.row(token);
                target
                    .par_iter_mut()
                    .zip(self.data.par_chunks_exact(cols).zip(self.scales.par_iter()))
                    .for_each(row_of(activations));
            }
        } else {
            for (token, target) in output.data.chunks_mut(self.rows).enumerate() {
                let activations = input.row(token);
                target
                    .iter_mut()
                    .zip(self.data.chunks_exact(cols).zip(self.scales.iter()))
                    .for_each(row_of(activations));
            }
        }
        output
    }

    /// Writes row `index` back as `f32`, for the embedding lookup.
    pub(crate) fn dequantize_row(&self, index: usize, target: &mut [f32]) {
        let cols = self.cols.max(1);
        let scale = self.scales[index];
        let row = &self.data[index * cols..index * cols + cols];
        for (slot, &value) in target.iter_mut().zip(row) {
            *slot = value as f32 * scale;
        }
    }

    /// The whole table back as `f32`.
    ///
    /// The device path uploads BF16, so the one byte per weight this holds has
    /// to be widened somewhere; doing it here, once, keeps every device matmul
    /// on cuBLAS instead of a hand-written dequantizing kernel.
    #[cfg(feature = "cuda")]
    pub(crate) fn dequantize(&self) -> Matrix {
        let mut output = Matrix::new(self.rows, self.cols);
        for index in 0..self.rows {
            self.dequantize_row(index, output.row_mut(index));
        }
        output
    }

    /// Rounds `row` in place through the grid
    /// [`Quantized::from_matrix`] would put it on.
    ///
    /// A scale covers one row and is read off that row alone, so a single
    /// gathered embedding row can be rounded without quantizing the table it
    /// came from.
    pub(crate) fn round_row(row: &mut [f32]) {
        let absmax = row.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let scale = absmax / 127.0;
        if scale <= 0.0 {
            return;
        }
        for value in row {
            *value = (*value / scale).round().clamp(-127.0, 127.0) * scale;
        }
    }

    pub(crate) fn bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }
}

/// `f32` activations against `i8` weights.
///
/// `LANES` independent accumulators, and the widening done into its own array
/// first, because that is the shape LLVM turns into vector instructions. Eight
/// lanes is what a baseline `x86-64` target can hold in registers; with AVX2 it
/// is worth going to thirty-two.
#[inline(always)]
fn dot_lanes<const LANES: usize>(activations: &[f32], weights: &[i8]) -> f32 {
    debug_assert_eq!(activations.len(), weights.len());
    let mut lanes = [0.0f32; LANES];
    let tail = activations.len() - activations.len() % LANES;

    for (x, y) in activations[..tail]
        .chunks_exact(LANES)
        .zip(weights[..tail].chunks_exact(LANES))
    {
        let mut widened = [0.0f32; LANES];
        for lane in 0..LANES {
            widened[lane] = f32::from(y[lane]);
        }
        for lane in 0..LANES {
            lanes[lane] += x[lane] * widened[lane];
        }
    }

    let mut rest = 0.0f32;
    for (x, &y) in activations[tail..].iter().zip(&weights[tail..]) {
        rest += x * f32::from(y);
    }
    lanes.iter().sum::<f32>() + rest
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(activations: &[f32], weights: &[i8]) -> f32 {
    dot_lanes::<32>(activations, weights)
}

/// Three times the throughput of the eight-lane loop on a machine with AVX2,
/// which is every `x86-64` CPU since 2013 — but not what a default `x86-64`
/// build is allowed to emit, so the choice is made here at runtime.
fn dot(activations: &[f32], weights: &[i8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // Safe: the branch is the feature check.
        return unsafe { dot_avx2(activations, weights) };
    }
    dot_lanes::<8>(activations, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantizing_and_multiplying_agrees_with_the_float_matrix() {
        let weights = Matrix::from_vec(3, 4, (0..12).map(|v| v as f32 * 0.37 - 2.0).collect());
        let input = Matrix::from_vec(2, 4, vec![0.5, -1.5, 2.0, 0.25, 1.0, 1.0, -1.0, 0.0]);

        let mut expected = Matrix::new(2, 3);
        input.dot_rhs_transposed(&weights, &mut expected);
        let actual = Quantized::from_matrix(&weights).matmul_rhs_transposed(&input);

        assert_eq!(actual.rows, 2);
        assert_eq!(actual.cols, 3);
        for (got, want) in actual.data.iter().zip(&expected.data) {
            // 255 levels over the row's range, so the error is bounded by the
            // row's absmax and the number of terms, not by the result.
            assert!((got - want).abs() < 0.05, "{got} vs {want}");
        }
    }

    #[test]
    fn a_row_of_zeros_stays_zero() {
        let quantized = Quantized::from_matrix(&Matrix::new(2, 3));
        let mut row = [1.0f32; 3];
        quantized.dequantize_row(0, &mut row);
        assert_eq!(row, [0.0; 3]);
    }

    #[test]
    fn dequantizing_a_row_recovers_it_within_the_step_size() {
        let weights = Matrix::from_vec(2, 3, vec![0.5, -1.0, 0.25, 4.0, -2.0, 0.0]);
        let quantized = Quantized::from_matrix(&weights);

        let mut row = [0.0f32; 3];
        quantized.dequantize_row(1, &mut row);
        for (got, want) in row.iter().zip(&weights.data[3..]) {
            assert!((got - want).abs() < 4.0 / 127.0, "{got} vs {want}");
        }
    }
}
