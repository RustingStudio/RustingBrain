//! Weights stored as one byte per value, or half a byte, for inference.
//!
//! Decoding one token reads every active weight once and does two flops with
//! it, so a CPU decode step is bound by memory bandwidth and nothing else: this
//! machine reads 60 GB/s and a 116M-active-parameter model needs 464 MB per
//! token as `f32`. One byte per weight is a quarter of that.
//!
//! The int8 scheme is the one [`Precision::Q8`](crate::transformer::Precision)
//! already writes to disk: per row, `scale = absmax / 127` and `round(w /
//! scale)` as `i8`. Rounding to 255 levels costs about 0.4% relative error per
//! weight, which inference absorbs and training does not — so quantizing is
//! one-way, and the model refuses to train or save afterwards.
//!
//! The 4-bit scheme stores, for every weight, the index of the nearest of 16
//! levels in a [`Codebook`], two indices to a byte, and one `f32` absmax per
//! block of [`BLOCK`] weights along a row. A weight is `level * absmax`. The
//! block keeps one outlier from coarsening a whole row, which 15 levels could
//! not absorb the way 255 do.

use crate::matrix::Matrix;
use rayon::prelude::*;

/// Weights per 4-bit scale, along a row: the QLoRA block size.
pub const BLOCK: usize = 64;

/// The 16 levels a 4-bit weight can take, as fractions of its block's absmax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codebook {
    /// Evenly spaced, `-7/7` to `7/7` in sevenths (index 0 is unused, so
    /// zero is exact and the grid is symmetric).
    Int4,
    /// NormalFloat4 from the QLoRA paper: the quantiles of a normal
    /// distribution, so a trained weight matrix, whose values are close to
    /// normal, loses less than on the even grid.
    Nf4,
}

impl Codebook {
    /// The levels, ascending.
    pub fn levels(self) -> [f32; 16] {
        match self {
            Codebook::Int4 => std::array::from_fn(|i| (i as f32 - 8.0).max(-7.0) / 7.0),
            Codebook::Nf4 => [
                -1.0,
                -0.696_192_8,
                -0.525_073_05,
                -0.394_917_5,
                -0.284_441_38,
                -0.184_773_43,
                -0.091_050_036,
                0.0,
                0.079_580_3,
                0.160_930_2,
                0.246_112_3,
                0.337_915_24,
                0.440_709_83,
                0.562_617,
                0.722_956_84,
                1.0,
            ],
        }
    }

    /// The index of the level nearest `x`, a value in `[-1, 1]`.
    fn nearest(levels: &[f32; 16], x: f32) -> u8 {
        // The midpoints split the line into the 16 cells; `x` lands in the
        // cell of however many midpoints lie below it.
        (0..15)
            .filter(|&i| x > (levels[i] + levels[i + 1]) / 2.0)
            .count() as u8
    }
}

/// A `[rows, cols]` weight matrix as `i8` values with one `f32` scale per row
/// (or per `group` columns of a row, see [`Quantized::from_matrix_grouped`]),
/// or as 4-bit codebook indices with one `f32` scale per [`BLOCK`] of a row.
#[derive(Clone, Debug, PartialEq)]
pub struct Quantized {
    rows: usize,
    cols: usize,
    values: Values,
    scales: Vec<f32>,
    /// Columns one int8 scale covers; `cols` for a scale per row. Unused by
    /// 4-bit weights, whose block is [`BLOCK`].
    group: usize,
}

#[derive(Clone, Debug, PartialEq)]
enum Values {
    Int8(Vec<i8>),
    /// Two indices a byte, `BLOCK / 2` bytes a block, rows padded to whole
    /// blocks: byte `j` of a block holds weight `j` in its low nibble and
    /// weight `j + BLOCK / 2` in its high one, so either half of a block
    /// widens as one contiguous run.
    Four {
        codebook: Codebook,
        packed: Vec<u8>,
    },
    /// FP8 E4M3 (`e4m3fn`) bytes, one scale per row: `scale = absmax / 448`
    /// and the nearest E4M3 value of `w / scale`. A storage format here: the
    /// host and an sm_86 device widen it before they multiply.
    Fp8(Vec<u8>),
}

/// The largest finite E4M3 (`e4m3fn`) value.
pub(crate) const E4M3_MAX: f32 = 448.0;

/// `x` as E4M3 (`e4m3fn`) bits, rounded to nearest even and saturated at
/// [`E4M3_MAX`]; `NaN` is `0x7f`.
///
/// The positive codes count up in value order, and the step inside an octave
/// `[2^e, 2^(e+1))` is `2^(e-3)` (`2^-9` for the subnormals, `e = -6`), so the
/// code is the octave's first code plus the value in steps. Rounding up out of
/// an octave lands on the next octave's first code by the same sum.
pub(crate) fn f32_to_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = x.abs();
    if a >= E4M3_MAX {
        return sign | 0x7e;
    }
    let octave = ((a.to_bits() >> 23) as i32 - 127).max(-6);
    // A power of two, so the product is exact and only the rounding rounds.
    let steps = (a * f32::from_bits(((130 - octave) as u32) << 23)).round_ties_even() as i32;
    sign | (((octave + 7) << 3) + steps - 8).min(0x7e) as u8
}

/// The 256 E4M3 values, from the decoder the safetensors reader uses.
fn e4m3_table() -> &'static [f32; 256] {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|b| crate::safetensors::f8_to_f32(b as u8, 4, 3)))
}

impl Quantized {
    /// Quantizes row by row. A row of zeros gets a scale of zero, which
    /// dequantizes back to zeros rather than to `NaN`.
    pub(crate) fn from_matrix(value: &Matrix) -> Self {
        Self::from_matrix_grouped(value, value.cols)
    }

    /// [`Quantized::from_matrix`] with one scale per `group` columns of a row
    /// rather than per row: an outlier then costs the precision of `group`
    /// weights instead of the whole row. `group >= cols` is the per-row
    /// scheme; the last group of a row may be shorter.
    pub(crate) fn from_matrix_grouped(value: &Matrix, group: usize) -> Self {
        let cols = value.cols.max(1);
        let group = group.clamp(1, cols);
        let mut data = Vec::with_capacity(value.data.len());
        let mut scales = Vec::with_capacity(value.rows * cols.div_ceil(group));

        for chunk in value.data.chunks(cols).flat_map(|row| row.chunks(group)) {
            let absmax = chunk.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
            let scale = absmax / 127.0;
            scales.push(scale);
            let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            data.extend(
                chunk
                    .iter()
                    .map(|v| (v * inverse).round().clamp(-127.0, 127.0) as i8),
            );
        }

        Self {
            rows: value.rows,
            cols: value.cols,
            values: Values::Int8(data),
            scales,
            group,
        }
    }

    /// Stores `value` as FP8 E4M3 with one scale per row (see
    /// [`Values::Fp8`]). A row of zeros gets a scale of zero.
    pub(crate) fn from_matrix_fp8(value: &Matrix) -> Self {
        let cols = value.cols.max(1);
        let mut data = Vec::with_capacity(value.data.len());
        let mut scales = Vec::with_capacity(value.rows);
        for row in value.data.chunks(cols) {
            let absmax = row.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
            let scale = absmax / E4M3_MAX;
            scales.push(scale);
            data.extend(row.iter().map(|v| match scale > 0.0 {
                true => f32_to_e4m3(v / scale),
                false => 0,
            }));
        }
        Self {
            rows: value.rows,
            cols: value.cols,
            values: Values::Fp8(data),
            scales,
            group: cols,
        }
    }

    /// Quantizes to 4 bits on `codebook`, one scale per [`BLOCK`] of a row. A
    /// block of zeros gets a scale of zero, as in
    /// [`Quantized::from_matrix`].
    pub(crate) fn from_matrix_4bit(value: &Matrix, codebook: Codebook) -> Self {
        let cols = value.cols.max(1);
        let levels = codebook.levels();
        let row_bytes = cols.div_ceil(BLOCK) * BLOCK / 2;
        let mut packed = vec![0u8; value.rows * row_bytes];
        let mut scales = Vec::with_capacity(value.rows * cols.div_ceil(BLOCK));
        for (row, target) in value.data.chunks(cols).zip(packed.chunks_mut(row_bytes)) {
            for (block, values) in row.chunks(BLOCK).enumerate() {
                let absmax = values.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
                scales.push(absmax);
                let inverse = if absmax > 0.0 { 1.0 / absmax } else { 0.0 };
                for (offset, v) in values.iter().enumerate() {
                    let k = block * BLOCK + offset;
                    let index = Codebook::nearest(&levels, v * inverse);
                    target[nibble_byte(k)] |= index << nibble_shift(k);
                }
            }
        }
        Self {
            rows: value.rows,
            cols: value.cols,
            values: Values::Four { codebook, packed },
            scales,
            group: BLOCK,
        }
    }

    /// `target[t, r] = input[t, ..] . row(r)`, the shape
    /// [`Matrix::dot_rhs_transposed`] produces.
    pub(crate) fn matmul_rhs_transposed(&self, input: &Matrix) -> Matrix {
        debug_assert_eq!(input.cols, self.cols);
        let cols = self.cols.max(1);
        let mut output = Matrix::new(input.rows, self.rows);
        let levels = match &self.values {
            Values::Four { codebook, .. } => codebook.levels(),
            _ => [0.0; 16],
        };
        let blocks = cols.div_ceil(BLOCK);
        let row_bytes = blocks * BLOCK / 2;
        let int8_groups = cols.div_ceil(self.group);
        let unit_dot = |activations: &[f32], unit: usize| match &self.values {
            Values::Int8(data) => {
                let row = &data[unit * cols..(unit + 1) * cols];
                let scales = &self.scales[unit * int8_groups..];
                row.chunks(self.group)
                    .zip(activations.chunks(self.group))
                    .zip(scales)
                    .map(|((w, x), scale)| dot(x, w) * scale)
                    .sum()
            }
            Values::Four { packed, .. } => dot4(
                activations,
                &packed[unit * row_bytes..(unit + 1) * row_bytes],
                &self.scales[unit * blocks..(unit + 1) * blocks],
                &levels,
            ),
            Values::Fp8(data) => {
                dot_fp8(activations, &data[unit * cols..(unit + 1) * cols]) * self.scales[unit]
            }
        };

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
                    for (unit, slot) in target.iter_mut().enumerate() {
                        *slot = unit_dot(activations, unit);
                    }
                });
        } else if self.rows * cols >= PARALLEL_VALUES {
            for (token, target) in output.data.chunks_mut(self.rows).enumerate() {
                let activations = input.row(token);
                target
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(unit, slot)| *slot = unit_dot(activations, unit));
            }
        } else {
            for (token, target) in output.data.chunks_mut(self.rows).enumerate() {
                let activations = input.row(token);
                for (unit, slot) in target.iter_mut().enumerate() {
                    *slot = unit_dot(activations, unit);
                }
            }
        }
        output
    }

    /// Writes row `index` back as `f32`, for the embedding lookup.
    pub(crate) fn dequantize_row(&self, index: usize, target: &mut [f32]) {
        let cols = self.cols.max(1);
        match &self.values {
            Values::Int8(data) => {
                let groups = cols.div_ceil(self.group);
                let scales = &self.scales[index * groups..(index + 1) * groups];
                let row = &data[index * cols..index * cols + cols];
                for (k, (slot, &value)) in target.iter_mut().zip(row).enumerate() {
                    *slot = value as f32 * scales[k / self.group];
                }
            }
            Values::Four { codebook, packed } => {
                let levels = codebook.levels();
                let row_bytes = cols.div_ceil(BLOCK) * BLOCK / 2;
                let row = &packed[index * row_bytes..(index + 1) * row_bytes];
                let scales = &self.scales[index * cols.div_ceil(BLOCK)..];
                for (k, slot) in target.iter_mut().enumerate().take(cols) {
                    let nibble = (row[nibble_byte(k)] >> nibble_shift(k)) & 0xf;
                    *slot = levels[usize::from(nibble)] * scales[k / BLOCK];
                }
            }
            Values::Fp8(data) => {
                let table = e4m3_table();
                let scale = self.scales[index];
                for (slot, &byte) in target
                    .iter_mut()
                    .zip(&data[index * cols..(index + 1) * cols])
                {
                    *slot = table[usize::from(byte)] * scale;
                }
            }
        }
    }

    /// The whole table back as `f32`.
    ///
    /// The device path uploads BF16, so the one byte per weight this holds has
    /// to be widened somewhere; doing it here, once, keeps every device matmul
    /// on cuBLAS instead of a hand-written dequantizing kernel. The host
    /// backward pass through an int8 LoRA base widens it here too.
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
        let values = match &self.values {
            Values::Int8(data) => data.len(),
            Values::Four { packed, .. } => packed.len(),
            Values::Fp8(data) => data.len(),
        };
        values + self.scales.len() * 4
    }

    /// The int8 values, row-major, and one scale per row, for an upload, or
    /// `None` for 4-bit weights, which have no device path yet.
    #[cfg(feature = "cuda")]
    pub(crate) fn parts(&self) -> Option<(&[i8], &[f32])> {
        match &self.values {
            Values::Int8(data) if self.group >= self.cols => Some((data, &self.scales)),
            _ => None,
        }
    }

    /// Columns one int8 scale covers, `None` for a scale per row or for
    /// 4-bit weights.
    #[cfg(feature = "cuda")]
    pub(crate) fn int8_group(&self) -> Option<usize> {
        match &self.values {
            Values::Int8(_) if self.group < self.cols => Some(self.group),
            _ => None,
        }
    }

    /// The packed indices, the block scales and the codebook of a 4-bit
    /// weight; `None` for int8.
    #[cfg(feature = "cuda")]
    pub(crate) fn parts_4bit(&self) -> Option<(&[u8], &[f32], Codebook)> {
        match &self.values {
            Values::Four { codebook, packed } => Some((packed, &self.scales, *codebook)),
            _ => None,
        }
    }

    /// The E4M3 bytes, row-major, and one scale per row; `None` unless the
    /// weight is FP8.
    #[cfg(feature = "cuda")]
    pub(crate) fn parts_fp8(&self) -> Option<(&[u8], &[f32])> {
        match &self.values {
            Values::Fp8(data) => Some((data, &self.scales)),
            _ => None,
        }
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

/// `f32` activations against one E4M3 row, widened through the table eight
/// at a time into the lane-wise product of [`dot_lanes`].
fn dot_fp8(activations: &[f32], weights: &[u8]) -> f32 {
    const LANES: usize = 8;
    let table = e4m3_table();
    let mut lanes = [0.0f32; LANES];
    let tail = activations.len() - activations.len() % LANES;
    for (x, w) in activations[..tail]
        .chunks_exact(LANES)
        .zip(weights[..tail].chunks_exact(LANES))
    {
        for lane in 0..LANES {
            lanes[lane] += x[lane] * table[usize::from(w[lane])];
        }
    }
    let rest: f32 = activations[tail..]
        .iter()
        .zip(&weights[tail..])
        .map(|(x, &w)| x * table[usize::from(w)])
        .sum();
    lanes.iter().sum::<f32>() + rest
}

/// Where weight `k` of a 4-bit row sits: see [`Values::Four`].
fn nibble_byte(k: usize) -> usize {
    k / BLOCK * (BLOCK / 2) + k % (BLOCK / 2)
}
fn nibble_shift(k: usize) -> u32 {
    if k % BLOCK < BLOCK / 2 { 0 } else { 4 }
}

/// `f32` activations against one 4-bit row: each block widened through the
/// codebook into an array, both halves as contiguous runs, then the same
/// lane-wise product as [`dot_lanes`], scaled once per block.
#[inline(always)]
fn dot4_lanes<const LANES: usize>(
    activations: &[f32],
    packed: &[u8],
    scales: &[f32],
    levels: &[f32; 16],
) -> f32 {
    let mut total = 0.0f32;
    for ((x, bytes), scale) in activations
        .chunks(BLOCK)
        .zip(packed.chunks_exact(BLOCK / 2))
        .zip(scales)
    {
        let mut widened = [0.0f32; BLOCK];
        let (low, high) = widened.split_at_mut(BLOCK / 2);
        for ((low, high), &byte) in low.iter_mut().zip(high.iter_mut()).zip(bytes) {
            *low = levels[usize::from(byte & 0xf)];
            *high = levels[usize::from(byte >> 4)];
        }
        let mut lanes = [0.0f32; LANES];
        let tail = x.len() - x.len() % LANES;
        for (x, w) in x[..tail]
            .chunks_exact(LANES)
            .zip(widened[..tail].chunks_exact(LANES))
        {
            for lane in 0..LANES {
                lanes[lane] += x[lane] * w[lane];
            }
        }
        let rest: f32 = x[tail..]
            .iter()
            .zip(&widened[tail..])
            .map(|(x, w)| x * w)
            .sum();
        total += (lanes.iter().sum::<f32>() + rest) * scale;
    }
    total
}

/// The codebook held in two registers and looked up eight weights at a time
/// with `vpermps`, bit 3 of each nibble choosing the register: the table walk
/// [`dot4_lanes`] does one weight at a time was what made 4-bit decode twice
/// as slow as int8 on half the bytes. A ragged last block goes through
/// [`dot4_lanes`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot4_avx2(activations: &[f32], packed: &[u8], scales: &[f32], levels: &[f32; 16]) -> f32 {
    use std::arch::x86_64::*;
    // Safe: every pointer stays inside the block counted from the slice lengths.
    unsafe {
        let whole = activations.len() / BLOCK;
        let low_table = _mm256_loadu_ps(levels.as_ptr());
        let high_table = _mm256_loadu_ps(levels.as_ptr().add(8));
        let nibble = _mm256_set1_epi32(0xf);
        let lookup = |codes: __m256i| {
            let low = _mm256_permutevar8x32_ps(low_table, codes);
            let high = _mm256_permutevar8x32_ps(high_table, codes);
            // Bit 3 moved to the sign bit is the blend mask.
            _mm256_blendv_ps(low, high, _mm256_castsi256_ps(_mm256_slli_epi32(codes, 28)))
        };
        let mut total = _mm256_setzero_ps();
        for (block, &scale) in scales[..whole].iter().enumerate() {
            let x = activations.as_ptr().add(block * BLOCK);
            let bytes = packed.as_ptr().add(block * BLOCK / 2);
            let mut sum = _mm256_setzero_ps();
            for step in 0..BLOCK / 16 {
                let codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64(bytes.add(step * 8).cast()));
                let low = lookup(_mm256_and_si256(codes, nibble));
                let high = lookup(_mm256_srli_epi32(codes, 4));
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(x.add(step * 8)), low, sum);
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(x.add(BLOCK / 2 + step * 8)), high, sum);
            }
            total = _mm256_fmadd_ps(sum, _mm256_set1_ps(scale), total);
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), total);
        lanes.iter().sum::<f32>()
            + dot4_lanes::<8>(
                &activations[whole * BLOCK..],
                &packed[whole * BLOCK / 2..],
                &scales[whole..],
                levels,
            )
    }
}

fn dot4(activations: &[f32], packed: &[u8], scales: &[f32], levels: &[f32; 16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // Safe: the branch is the feature check.
        return unsafe { dot4_avx2(activations, packed, scales, levels) };
    }
    dot4_lanes::<8>(activations, packed, scales, levels)
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

    /// The E4M3 encoder against the safetensors decoder: every finite code
    /// round-trips, rounding is to nearest with ties to the even code (the
    /// even mantissa) on a brute-force search over all 254 finite codes,
    /// magnitudes from 448 up saturate, `NaN` stays `NaN`, and the subnormals
    /// round on their 2^-9 grid.
    #[test]
    fn e4m3_encoding_round_trips_rounds_to_even_and_saturates() {
        use crate::safetensors::f8_to_f32;
        let decode = |b: u8| f8_to_f32(b, 4, 3);
        for b in 0..=255u8 {
            if b & 0x7f == 0x7f {
                assert!(decode(b).is_nan(), "{b:#04x} is the NaN");
                continue;
            }
            assert_eq!(f32_to_e4m3(decode(b)), b, "{b:#04x} -> {}", decode(b));
        }
        // The nearest of the 127 non-negative finite codes to `|x|`; of two
        // equally near, which are neighbours, the even one.
        let nearest = |x: f32| -> u8 {
            let a = x.abs();
            let mut best = 0u8;
            for b in 1..=0x7eu8 {
                let (gap, held) = ((decode(b) - a).abs(), (decode(best) - a).abs());
                if gap < held || (gap == held && b % 2 == 0) {
                    best = b;
                }
            }
            best | if x.is_sign_negative() { 0x80 } else { 0 }
        };
        let mut values: Vec<f32> = normal(4000).iter().map(|v| v * 40.0).collect();
        values.extend(normal(2000).iter().map(|v| v * 0.02));
        // Every midpoint between neighbouring codes, which is where the ties
        // are, and a hair either side of it.
        for b in 0..0x7eu8 {
            let mid = (decode(b) + decode(b + 1)) / 2.0;
            for x in [mid, mid.next_up(), mid.next_down()] {
                values.extend([x, -x]);
            }
        }
        for x in values.into_iter().filter(|x| x.abs() < E4M3_MAX) {
            assert_eq!(f32_to_e4m3(x), nearest(x), "{x:e}");
        }
        for (x, want) in [
            (448.0, 0x7e),
            (464.0, 0x7e),
            (1e9, 0x7e),
            (f32::INFINITY, 0x7e),
            (-500.0, 0xfe),
            (f32::NEG_INFINITY, 0xfe),
            (2.0f32.powi(-9), 0x01),
            (2.0f32.powi(-10), 0x00),
            (3.0 * 2.0f32.powi(-10), 0x02),
            (-(2.0f32.powi(-11)), 0x80),
            (2.0f32.powi(-6) - 2.0f32.powi(-10), 0x08),
            (f32::MIN_POSITIVE / 4.0, 0x00),
        ] {
            assert_eq!(f32_to_e4m3(x), want, "{x:e}");
        }
        assert_eq!(f32_to_e4m3(f32::NAN), 0x7f);
        assert!(decode(f32_to_e4m3(f32::NAN)).is_nan());
    }

    /// An FP8 weight multiplies like the float matrix it dequantizes to,
    /// and lands within E4M3's half step (2^-4 relative) of the original.
    #[test]
    fn an_fp8_weight_multiplies_like_its_own_dequantized_matrix() {
        let (rows, cols) = (7, 37);
        let mut weights = Matrix::from_vec(rows, cols, normal(rows * cols));
        weights.row_mut(3).fill(0.0);
        let quantized = Quantized::from_matrix_fp8(&weights);
        let widened = quantized.dequantize();
        assert!(widened.row(3).iter().all(|&v| v == 0.0));
        for (got, want) in widened.data.iter().zip(&weights.data) {
            assert!(
                (got - want).abs() <= want.abs() / 16.0 + 2e-3,
                "{got} vs {want}"
            );
        }
        let input = Matrix::from_vec(3, cols, normal(3 * cols));
        let mut expected = Matrix::new(3, rows);
        input.dot_rhs_transposed(&widened, &mut expected);
        let actual = quantized.matmul_rhs_transposed(&input);
        for (got, want) in actual.data.iter().zip(&expected.data) {
            assert!((got - want).abs() < 1e-4, "{got} vs {want}");
        }
        assert_eq!(quantized.bytes(), rows * cols + rows * 4);
    }

    #[test]
    fn a_scale_per_group_beats_a_scale_per_row_on_an_outlier() {
        // One large weight in the first group; the second group is small and
        // keeps its own scale. A ragged last group (cols = 10, group 4).
        let mut data: Vec<f32> = (0..20).map(|v| (v as f32 * 0.37).sin() * 0.01).collect();
        data[0] = 5.0;
        let weights = Matrix::from_vec(2, 10, data);
        let input = Matrix::from_vec(1, 10, (0..10).map(|v| 1.0 - v as f32 * 0.2).collect());
        let mut expected = Matrix::new(1, 2);
        input.dot_rhs_transposed(&weights, &mut expected);

        let row = Quantized::from_matrix(&weights);
        let grouped = Quantized::from_matrix_grouped(&weights, 4);
        assert_eq!(grouped.scales.len(), 2 * 3);
        let error = |q: &Quantized| {
            let got = q.matmul_rhs_transposed(&input);
            let wide = q.dequantize();
            // The product must be the dequantized matrix's product.
            let mut again = Matrix::new(1, 2);
            input.dot_rhs_transposed(&wide, &mut again);
            for (a, b) in got.data.iter().zip(&again.data) {
                assert!((a - b).abs() < 1e-5, "{a} vs {b}");
            }
            wide.data
                .iter()
                .zip(&weights.data)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
        };
        assert!(error(&grouped) < error(&row) / 2.0);
        assert!((grouped.matmul_rhs_transposed(&input).data[1] - expected.data[1]).abs() < 1e-3);
        // `group >= cols` is the per-row scheme, bit for bit.
        assert_eq!(Quantized::from_matrix_grouped(&weights, 10), row);
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

    /// Deterministic standard-normal values, Box-Muller over an LCG.
    fn normal(len: usize) -> Vec<f32> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut uniform = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 40) as f32 + 0.5) / (1u64 << 24) as f32
        };
        (0..len)
            .map(|_| {
                let (a, b) = (uniform(), uniform());
                (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
            })
            .collect()
    }

    /// 67 columns: a full block, a partial one and an odd nibble at the end.
    #[test]
    fn a_4bit_product_matches_the_float_product_of_its_own_weights() {
        let weights = Matrix::from_vec(3, 67, normal(3 * 67));
        for codebook in [Codebook::Int4, Codebook::Nf4] {
            let quantized = Quantized::from_matrix_4bit(&weights, codebook);
            assert_eq!(quantized.bytes(), 3 * 64 + 3 * 2 * 4);
            let widened = quantized.dequantize();
            for tokens in [1, 2] {
                let input = Matrix::from_vec(tokens, 67, normal(tokens * 67 + 7)[7..].to_vec());
                let mut expected = Matrix::new(tokens, 3);
                input.dot_rhs_transposed(&widened, &mut expected);
                let actual = quantized.matmul_rhs_transposed(&input);
                for (got, want) in actual.data.iter().zip(&expected.data) {
                    assert!((got - want).abs() < 1e-4, "{codebook:?}: {got} vs {want}");
                }
            }
        }
    }

    /// Half a byte a weight and four bytes a block of 64: 0.5625 bytes a
    /// weight at a width blocks divide, against 1 plus a row scale for int8.
    #[test]
    fn a_4bit_weight_takes_under_six_tenths_of_a_byte() {
        let weights = Matrix::from_vec(8, 512, normal(8 * 512));
        let four = Quantized::from_matrix_4bit(&weights, Codebook::Nf4).bytes();
        assert_eq!(four, 8 * 256 + 8 * 8 * 4);
        assert!(four * 10 < Quantized::from_matrix(&weights).bytes() * 6);
    }

    #[test]
    fn nf4_loses_less_than_the_even_grid_on_normal_weights() {
        let weights = Matrix::from_vec(16, 256, normal(16 * 256));
        let error = |codebook| {
            let widened = Quantized::from_matrix_4bit(&weights, codebook).dequantize();
            let squares: f32 = widened
                .data
                .iter()
                .zip(&weights.data)
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            (squares / weights.data.len() as f32).sqrt()
        };
        let (int4, nf4) = (error(Codebook::Int4), error(Codebook::Nf4));
        // Each block's absmax is about 2.5 standard deviations, so the even
        // grid's step is about 0.36 and its RMS error about 0.1.
        assert!(int4 < 0.12, "int4 RMS error {int4}");
        assert!(nf4 < int4, "nf4 {nf4} against int4 {int4}");
    }

    #[test]
    fn a_4bit_block_of_zeros_stays_zero_and_its_neighbour_keeps_its_extremes() {
        let mut values = vec![0.0f32; 2 * BLOCK];
        values[BLOCK] = -3.0;
        values[BLOCK + 1] = 3.0;
        let quantized = Quantized::from_matrix_4bit(
            &Matrix::from_vec(1, 2 * BLOCK, values.clone()),
            Codebook::Nf4,
        );
        let mut row = vec![1.0f32; 2 * BLOCK];
        quantized.dequantize_row(0, &mut row);
        assert_eq!(row, values);
    }
}
