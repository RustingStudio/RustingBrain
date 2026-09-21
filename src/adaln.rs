//! Adaptive layer normalization: a block whose normalization is steered by a
//! conditioning vector rather than by weights alone.
//!
//! A diffusion or flow model has to tell every block what timestep it is
//! denoising. The usual answer is to project the timestep into a `(shift,
//! scale, gate)` triple per sub-layer and fold it into the normalization:
//!
//! ```text
//! h = x + gate * Branch(norm(x) * (1 + scale) + shift)
//! ```
//!
//! Done naively that costs one `d_cond -> 3 * d_model` projection per
//! sub-layer, which at `d_model` 640 over sixteen blocks is more parameters
//! than the blocks themselves. This module implements **AdaLN-single**
//! instead: one [`Modulation`] shared by the whole stack produces the base
//! triple, and each [`AdaLayerNorm`] adds its own learned `[3, d_model]`
//! offset. Per-sub-layer cost drops from `3 * d_model * d_cond` to
//! `3 * d_model`.
//!
//! Everything here is zero-initialized, which is the standard trick: at step
//! zero `scale` and `shift` are zero and `gate` is zero, so every block is the
//! identity and the residual stream starts clean. The gradients are not zero,
//! so the first optimizer step moves it.
//!
//! # Shapes
//!
//! A conditioning vector is *per sequence*, not per token. Rows of `x` are
//! grouped into sequences of `seq_len`, and sequence `i` reads row `i` of the
//! conditioning. Every function here takes `seq_len` for that reason.

use crate::ffn::{silu, silu_derivative};
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::norm::RmsNorm;
use crate::param::{Linear, Param};
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A projection whose weight starts at zero, so the layer it feeds starts as
/// the identity.
fn zero_linear(in_features: usize, out_features: usize) -> Linear {
    Linear {
        weight: Param::zeros(out_features, in_features),
        lora: None,
    }
}

/// Turns a scalar timestep into a conditioning vector.
///
/// A fixed sinusoidal basis followed by a two-layer SiLU MLP, which is what
/// every model in this family uses. The basis has no parameters, so the
/// gradient stops at the MLP's input and the timestep itself is never
/// differentiated.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TimestepEmbedding {
    pub input: Linear,
    pub output: Linear,
    /// Width of the sinusoidal basis. Even, so cosines and sines pair up.
    frequencies: usize,
}

/// What [`TimestepEmbedding::backward`] needs from its forward pass.
#[derive(Clone, Debug)]
pub struct TimestepCache {
    basis: Matrix,
    hidden: Matrix,
    activated: Matrix,
}

impl TimestepEmbedding {
    /// `frequencies` is the sinusoidal basis width, `d_hidden` the MLP's
    /// middle, `d_cond` what the blocks downstream read.
    pub fn new(
        frequencies: usize,
        d_hidden: usize,
        d_cond: usize,
        rng: &mut StdRng,
    ) -> Result<Self, NetworkError> {
        if frequencies == 0 || frequencies % 2 != 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "a sinusoidal basis needs an even, non-zero width, not {frequencies}"
            )));
        }
        Ok(Self {
            input: Linear::new(frequencies, d_hidden, rng),
            output: Linear::new(d_hidden, d_cond, rng),
            frequencies,
        })
    }

    /// Width of the conditioning vector this produces.
    pub fn d_cond(&self) -> usize {
        self.output.out_features()
    }

    /// One row of conditioning per timestep.
    pub fn forward_train(&self, times: &[f32]) -> (Matrix, TimestepCache) {
        let mut basis = Matrix::new(times.len(), self.frequencies);
        for (row, &time) in times.iter().enumerate() {
            basis
                .row_mut(row)
                .copy_from_slice(&crate::mmdit::sinusoid(time, self.frequencies));
        }

        let hidden = self.input.forward(&basis);
        let mut activated = hidden.clone();
        activated.data.par_iter_mut().for_each(|v| *v = silu(*v));
        let output = self.output.forward(&activated);

        (
            output,
            TimestepCache {
                basis,
                hidden,
                activated,
            },
        )
    }

    pub fn forward(&self, times: &[f32]) -> Matrix {
        self.forward_train(times).0
    }

    /// Accumulates both projections' gradients. Nothing is returned: the
    /// sinusoidal basis has no parameters and a timestep is not learned.
    pub fn backward(&mut self, cache: &TimestepCache, grad_output: &Matrix) {
        let mut grad_hidden = self.output.backward(&cache.activated, grad_output);
        grad_hidden
            .data
            .par_iter_mut()
            .zip(cache.hidden.data.par_iter())
            .for_each(|(slot, &pre)| *slot *= silu_derivative(pre));
        self.input.backward(&cache.basis, &grad_hidden);
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = self.input.params_mut();
        params.extend(self.output.params_mut());
        params
    }

    pub fn num_parameters(&self) -> usize {
        self.input.weight.len() + self.output.weight.len()
    }
}

/// The one projection the whole stack shares: conditioning to a base
/// `(shift, scale, gate)` triple.
///
/// Zero-initialized, so the triple starts at zero and every
/// [`AdaLayerNorm`] reading it starts as a plain normalization with a closed
/// gate.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Modulation {
    /// `d_cond -> 3 * d_model`, applied after a SiLU.
    pub project: Linear,
    d_model: usize,
}

/// What [`Modulation::backward`] needs from its forward pass.
#[derive(Clone, Debug)]
pub struct ModulationCache {
    conditioning: Matrix,
    activated: Matrix,
}

impl Modulation {
    pub fn new(d_cond: usize, d_model: usize) -> Self {
        Self {
            project: zero_linear(d_cond, 3 * d_model),
            d_model,
        }
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// `[sequences, d_cond] -> [sequences, 3 * d_model]`.
    pub fn forward_train(&self, conditioning: &Matrix) -> (Matrix, ModulationCache) {
        let mut activated = conditioning.clone();
        activated.data.par_iter_mut().for_each(|v| *v = silu(*v));
        let triple = self.project.forward(&activated);
        (
            triple,
            ModulationCache {
                conditioning: conditioning.clone(),
                activated,
            },
        )
    }

    pub fn forward(&self, conditioning: &Matrix) -> Matrix {
        self.forward_train(conditioning).0
    }

    /// Accumulates the projection's gradient and returns `dL/dconditioning`.
    pub fn backward(&mut self, cache: &ModulationCache, grad_triple: &Matrix) -> Matrix {
        let mut grad = self.project.backward(&cache.activated, grad_triple);
        grad.data
            .par_iter_mut()
            .zip(cache.conditioning.data.par_iter())
            .for_each(|(slot, &pre)| *slot *= silu_derivative(pre));
        grad
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        self.project.params_mut()
    }

    pub fn num_parameters(&self) -> usize {
        self.project.weight.len()
    }
}

/// One sub-layer's normalization, steered by the shared triple plus a learned
/// per-sub-layer offset.
///
/// The offset is the whole of AdaLN-single's per-block cost: `3 * d_model`
/// values against the `3 * d_model * d_cond` a private projection would want.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AdaLayerNorm {
    pub norm: RmsNorm,
    /// `[1, 3 * d_model]`: this sub-layer's shift, scale and gate offsets,
    /// laid out in that order and added to the shared triple.
    pub offset: Param,
}

/// What [`AdaLayerNorm::backward`] needs from its forward pass.
#[derive(Clone, Debug)]
pub struct AdaLnCache {
    input: Matrix,
    normalized: Matrix,
    /// The shared triple plus this layer's offset, `[sequences, 3 * d_model]`.
    triple: Matrix,
    seq_len: usize,
}

impl AdaLayerNorm {
    pub fn new(d_model: usize, eps: f32) -> Self {
        Self {
            norm: RmsNorm::new(d_model, eps),
            offset: Param::zeros(1, 3 * d_model),
        }
    }

    pub fn d_model(&self) -> usize {
        self.norm.d_model()
    }

    /// Normalizes and modulates, and hands back the gate for the caller to
    /// apply to whatever the branch produces.
    ///
    /// `input` is `[sequences * seq_len, d_model]` and `triple` is
    /// `[sequences, 3 * d_model]` — one conditioning row per sequence, shared
    /// by all of its tokens. Returns the modulated activations and the
    /// `[sequences, d_model]` gate.
    pub fn forward_train(
        &self,
        input: &Matrix,
        triple: &Matrix,
        seq_len: usize,
    ) -> Result<(Matrix, Matrix, AdaLnCache), NetworkError> {
        let width = self.d_model();
        self.check(input, triple, seq_len)?;

        // The offset is per sub-layer and the triple is per sequence, so the
        // sum is materialized once here rather than re-added per token.
        let mut combined = triple.clone();
        for row in 0..combined.rows {
            for (slot, &offset) in combined
                .row_mut(row)
                .iter_mut()
                .zip(&self.offset.value.data)
            {
                *slot += offset;
            }
        }

        let normalized = self.norm.forward(input);
        let mut modulated = Matrix::new(input.rows, width);

        modulated
            .data
            .par_chunks_mut(width)
            .zip(normalized.data.par_chunks(width))
            .enumerate()
            .for_each(|(row, (destination, source))| {
                let sequence = &combined.data[(row / seq_len) * combined.cols..];
                let (shift, scale) = (&sequence[..width], &sequence[width..2 * width]);
                for column in 0..width {
                    destination[column] = source[column] * (1.0 + scale[column]) + shift[column];
                }
            });

        let mut gate = Matrix::new(combined.rows, width);
        for sequence in 0..combined.rows {
            gate.row_mut(sequence)
                .copy_from_slice(&combined.row(sequence)[2 * width..]);
        }

        Ok((
            modulated,
            gate,
            AdaLnCache {
                input: input.clone(),
                normalized,
                triple: combined,
                seq_len,
            },
        ))
    }

    /// Accumulates the norm's and the offset's gradients and returns
    /// `(dL/dinput, dL/dtriple)`.
    ///
    /// `grad_gate` is `[sequences, d_model]` and comes from
    /// [`gate_residual_backward`]; it is what closes the loop between the gate
    /// handed out by the forward pass and the triple that produced it.
    pub fn backward(
        &mut self,
        cache: &AdaLnCache,
        grad_modulated: &Matrix,
        grad_gate: &Matrix,
    ) -> Result<(Matrix, Matrix), NetworkError> {
        let width = self.d_model();
        let sequences = cache.triple.rows;
        if grad_gate.rows != sequences || grad_gate.cols != width {
            return Err(NetworkError::InvalidConfig(format!(
                "gate gradient is [{}, {}], expected [{sequences}, {width}]",
                grad_gate.rows, grad_gate.cols
            )));
        }

        // dL/dnormalized = upstream * (1 + scale); the shift and scale
        // gradients are sums over the tokens of each sequence.
        let mut grad_normalized = Matrix::new(cache.input.rows, width);
        let mut grad_triple = Matrix::new(sequences, 3 * width);

        for row in 0..cache.input.rows {
            let sequence = row / cache.seq_len;
            let scale = &cache.triple.row(sequence)[width..2 * width];
            let upstream = grad_modulated.row(row);
            let normalized = cache.normalized.row(row);
            let target = grad_normalized.row_mut(row);
            for column in 0..width {
                target[column] = upstream[column] * (1.0 + scale[column]);
            }

            let triple = grad_triple.row_mut(sequence);
            for column in 0..width {
                triple[column] += upstream[column];
                triple[width + column] += upstream[column] * normalized[column];
            }
        }

        for sequence in 0..sequences {
            let upstream = grad_gate.row(sequence);
            let triple = grad_triple.row_mut(sequence);
            for column in 0..width {
                triple[2 * width + column] += upstream[column];
            }
        }

        // The offset is added to every sequence's triple, so its gradient is
        // the sum down the columns of exactly that gradient.
        if !self.offset.is_frozen() {
            for sequence in 0..sequences {
                for (slot, &value) in self
                    .offset
                    .grad
                    .data
                    .iter_mut()
                    .zip(grad_triple.row(sequence))
                {
                    *slot += value;
                }
            }
        }

        let grad_input = self.norm.backward(&cache.input, &grad_normalized);
        Ok((grad_input, grad_triple))
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = self.norm.params_mut();
        params.push(&mut self.offset);
        params
    }

    pub fn num_parameters(&self) -> usize {
        self.norm.weight.len() + self.offset.len()
    }

    fn check(&self, input: &Matrix, triple: &Matrix, seq_len: usize) -> Result<(), NetworkError> {
        let width = self.d_model();
        if input.cols != width {
            return Err(NetworkError::InvalidInput {
                expected: width,
                actual: input.cols,
            });
        }
        if triple.cols != 3 * width {
            return Err(NetworkError::InvalidInput {
                expected: 3 * width,
                actual: triple.cols,
            });
        }
        if seq_len == 0 || input.rows % seq_len != 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "{} rows do not split into sequences of {seq_len}",
                input.rows
            )));
        }
        if input.rows / seq_len != triple.rows {
            return Err(NetworkError::InvalidConfig(format!(
                "{} sequences against {} conditioning rows",
                input.rows / seq_len,
                triple.rows
            )));
        }
        Ok(())
    }
}

/// `residual + gate * branch`, with one gate row per sequence.
///
/// The counterpart of the gate [`AdaLayerNorm::forward_train`] hands out.
/// `branch` and `residual` are `[sequences * seq_len, d_model]`, `gate` is
/// `[sequences, d_model]`.
pub fn gate_residual(
    residual: &Matrix,
    branch: &Matrix,
    gate: &Matrix,
    seq_len: usize,
) -> Result<Matrix, NetworkError> {
    check_gate(residual, branch, gate, seq_len)?;
    let width = residual.cols;
    let mut output = residual.clone();

    output
        .data
        .par_chunks_mut(width)
        .zip(branch.data.par_chunks(width))
        .enumerate()
        .for_each(|(row, (destination, source))| {
            let scale = &gate.data[(row / seq_len) * width..][..width];
            for column in 0..width {
                destination[column] += scale[column] * source[column];
            }
        });

    Ok(output)
}

/// Returns `(dL/dbranch, dL/dgate)` for [`gate_residual`].
///
/// `dL/dresidual` is `grad_output` unchanged, so it is not returned: the
/// caller already holds it.
pub fn gate_residual_backward(
    branch: &Matrix,
    gate: &Matrix,
    grad_output: &Matrix,
    seq_len: usize,
) -> Result<(Matrix, Matrix), NetworkError> {
    check_gate(grad_output, branch, gate, seq_len)?;
    let width = branch.cols;
    let mut grad_branch = Matrix::new(branch.rows, width);
    let mut grad_gate = Matrix::new(gate.rows, width);

    for row in 0..branch.rows {
        let sequence = row / seq_len;
        let scale = gate.row(sequence);
        let upstream = grad_output.row(row);
        let source = branch.row(row);
        let target = grad_branch.row_mut(row);
        for column in 0..width {
            target[column] = upstream[column] * scale[column];
        }

        let accumulator = grad_gate.row_mut(sequence);
        for column in 0..width {
            accumulator[column] += upstream[column] * source[column];
        }
    }

    Ok((grad_branch, grad_gate))
}

fn check_gate(
    residual: &Matrix,
    branch: &Matrix,
    gate: &Matrix,
    seq_len: usize,
) -> Result<(), NetworkError> {
    if residual.rows != branch.rows || residual.cols != branch.cols {
        return Err(NetworkError::InvalidConfig(format!(
            "residual is [{}, {}] but the branch is [{}, {}]",
            residual.rows, residual.cols, branch.rows, branch.cols
        )));
    }
    if seq_len == 0 || branch.rows % seq_len != 0 {
        return Err(NetworkError::InvalidConfig(format!(
            "{} rows do not split into sequences of {seq_len}",
            branch.rows
        )));
    }
    if gate.rows != branch.rows / seq_len || gate.cols != branch.cols {
        return Err(NetworkError::InvalidConfig(format!(
            "gate is [{}, {}], expected [{}, {}]",
            gate.rows,
            gate.cols,
            branch.rows / seq_len,
            branch.cols
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rows(count: usize, width: usize, salt: usize) -> Matrix {
        Matrix::from_vec(
            count,
            width,
            (0..count * width)
                .map(|i| ((i * 37 + salt * 11) % 23) as f32 / 11.0 - 1.0)
                .collect(),
        )
    }

    fn objective(output: &Matrix, weight: &Matrix) -> f32 {
        output
            .data
            .iter()
            .zip(&weight.data)
            .map(|(value, weight)| value * weight)
            .sum()
    }

    #[test]
    fn a_fresh_block_is_the_identity() {
        // Zero-initialized: scale and shift are zero, so the modulation is a
        // plain normalization, and the gate is closed, so the branch cannot
        // move the residual at all.
        let mut rng = StdRng::seed_from_u64(3);
        let embedding = TimestepEmbedding::new(16, 32, 24, &mut rng).unwrap();
        let modulation = Modulation::new(24, 4);
        let layer = AdaLayerNorm::new(4, 1e-5);

        let conditioning = embedding.forward(&[0.3, 0.7]);
        let triple = modulation.forward(&conditioning);
        let input = rows(6, 4, 1);
        let (modulated, gate, _) = layer.forward_train(&input, &triple, 3).unwrap();

        let normalized = layer.norm.forward(&input);
        for (actual, expected) in modulated.data.iter().zip(&normalized.data) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert!(gate.data.iter().all(|&value| value == 0.0));

        let branch = rows(6, 4, 2);
        let output = gate_residual(&input, &branch, &gate, 3).unwrap();
        assert_eq!(output.data, input.data);
    }

    #[test]
    fn the_shared_triple_is_broadcast_over_a_sequence() {
        let mut modulation = Modulation::new(3, 2);
        // Break the zero initialization so the triple actually varies.
        modulation.project.weight.value =
            Matrix::from_vec(6, 3, (0..18).map(|i| (i % 5) as f32 / 4.0 - 0.5).collect());
        let conditioning = rows(2, 3, 4);
        let triple = modulation.forward(&conditioning);

        let layer = AdaLayerNorm::new(2, 1e-5);
        let input = rows(4, 2, 5);
        let (_, gate, _) = layer.forward_train(&input, &triple, 2).unwrap();

        // One gate row per sequence, not per token.
        assert_eq!(gate.rows, 2);
        assert_eq!(gate.cols, 2);
        assert_ne!(gate.row(0), gate.row(1));
    }

    #[test]
    fn mismatched_sequence_counts_are_rejected() {
        let layer = AdaLayerNorm::new(4, 1e-5);
        // Six rows in sequences of three is two sequences, against one
        // conditioning row.
        assert!(
            layer
                .forward_train(&rows(6, 4, 1), &rows(1, 12, 2), 3)
                .is_err()
        );
        // And a sequence length that does not divide the rows.
        assert!(
            layer
                .forward_train(&rows(6, 4, 1), &rows(2, 12, 2), 4)
                .is_err()
        );
    }

    /// The whole sub-layer, end to end, as a block would run it:
    /// `residual + gate * (modulated * weight)`.
    fn sublayer(
        layer: &AdaLayerNorm,
        input: &Matrix,
        triple: &Matrix,
        seq_len: usize,
        weight: &Matrix,
    ) -> Matrix {
        let (modulated, gate, _) = layer.forward_train(input, triple, seq_len).unwrap();
        let mut branch = modulated;
        for (slot, &scale) in branch.data.iter_mut().zip(weight.data.iter().cycle()) {
            *slot *= scale;
        }
        gate_residual(input, &branch, &gate, seq_len).unwrap()
    }

    #[test]
    fn input_gradient_matches_finite_differences() {
        let mut layer = AdaLayerNorm::new(4, 1e-5);
        layer.offset.value = rows(1, 12, 6);
        let seq_len = 3;
        let mut input = rows(6, 4, 1);
        let triple = rows(2, 12, 7);
        let branch_weight = rows(1, 4, 8);
        let upstream = rows(6, 4, 9);

        let (modulated, gate, cache) = layer.forward_train(&input, &triple, seq_len).unwrap();
        let mut branch = modulated.clone();
        for (slot, &scale) in branch
            .data
            .iter_mut()
            .zip(branch_weight.data.iter().cycle())
        {
            *slot *= scale;
        }
        let (grad_branch, grad_gate) =
            gate_residual_backward(&branch, &gate, &upstream, seq_len).unwrap();
        let mut grad_modulated = grad_branch;
        for (slot, &scale) in grad_modulated
            .data
            .iter_mut()
            .zip(branch_weight.data.iter().cycle())
        {
            *slot *= scale;
        }
        let (mut grad_input, _) = layer.backward(&cache, &grad_modulated, &grad_gate).unwrap();
        // The residual path contributes `grad_output` directly.
        for (slot, &value) in grad_input.data.iter_mut().zip(&upstream.data) {
            *slot += value;
        }

        let epsilon = 1e-3;
        for index in [0, 5, 11, 19, 23] {
            let original = input.data[index];
            input.data[index] = original + epsilon;
            let high = sublayer(&layer, &input, &triple, seq_len, &branch_weight);
            input.data[index] = original - epsilon;
            let low = sublayer(&layer, &input, &triple, seq_len, &branch_weight);
            input.data[index] = original;

            let numeric =
                (objective(&high, &upstream) - objective(&low, &upstream)) / (2.0 * epsilon);
            assert!(
                (grad_input.data[index] - numeric).abs() < 1e-2,
                "index {index}: {} vs {numeric}",
                grad_input.data[index]
            );
        }
    }

    #[test]
    fn triple_gradient_matches_finite_differences() {
        let mut layer = AdaLayerNorm::new(4, 1e-5);
        layer.offset.value = rows(1, 12, 6);
        let seq_len = 3;
        let input = rows(6, 4, 1);
        let mut triple = rows(2, 12, 7);
        let branch_weight = rows(1, 4, 8);
        let upstream = rows(6, 4, 9);

        let (modulated, gate, cache) = layer.forward_train(&input, &triple, seq_len).unwrap();
        let mut branch = modulated;
        for (slot, &scale) in branch
            .data
            .iter_mut()
            .zip(branch_weight.data.iter().cycle())
        {
            *slot *= scale;
        }
        let (grad_branch, grad_gate) =
            gate_residual_backward(&branch, &gate, &upstream, seq_len).unwrap();
        let mut grad_modulated = grad_branch;
        for (slot, &scale) in grad_modulated
            .data
            .iter_mut()
            .zip(branch_weight.data.iter().cycle())
        {
            *slot *= scale;
        }
        let (_, grad_triple) = layer.backward(&cache, &grad_modulated, &grad_gate).unwrap();

        // Every column of the triple: shift, scale and gate in turn.
        let epsilon = 1e-3;
        for index in [0, 3, 5, 9, 13, 18, 22] {
            let original = triple.data[index];
            triple.data[index] = original + epsilon;
            let high = sublayer(&layer, &input, &triple, seq_len, &branch_weight);
            triple.data[index] = original - epsilon;
            let low = sublayer(&layer, &input, &triple, seq_len, &branch_weight);
            triple.data[index] = original;

            let numeric =
                (objective(&high, &upstream) - objective(&low, &upstream)) / (2.0 * epsilon);
            assert!(
                (grad_triple.data[index] - numeric).abs() < 1e-2,
                "index {index}: {} vs {numeric}",
                grad_triple.data[index]
            );
        }
    }

    #[test]
    fn offset_gradient_matches_finite_differences() {
        let mut layer = AdaLayerNorm::new(4, 1e-5);
        layer.offset.value = rows(1, 12, 6);
        let seq_len = 3;
        let input = rows(6, 4, 1);
        let triple = rows(2, 12, 7);
        let branch_weight = rows(1, 4, 8);
        let upstream = rows(6, 4, 9);

        let (modulated, gate, cache) = layer.forward_train(&input, &triple, seq_len).unwrap();
        let mut branch = modulated;
        for (slot, &scale) in branch
            .data
            .iter_mut()
            .zip(branch_weight.data.iter().cycle())
        {
            *slot *= scale;
        }
        let (grad_branch, grad_gate) =
            gate_residual_backward(&branch, &gate, &upstream, seq_len).unwrap();
        let mut grad_modulated = grad_branch;
        for (slot, &scale) in grad_modulated
            .data
            .iter_mut()
            .zip(branch_weight.data.iter().cycle())
        {
            *slot *= scale;
        }
        layer.backward(&cache, &grad_modulated, &grad_gate).unwrap();

        let epsilon = 1e-3;
        for index in [1, 4, 7, 10] {
            let expected = layer.offset.grad.data[index];
            let original = layer.offset.value.data[index];

            layer.offset.value.data[index] = original + epsilon;
            let high = sublayer(&layer, &input, &triple, seq_len, &branch_weight);
            layer.offset.value.data[index] = original - epsilon;
            let low = sublayer(&layer, &input, &triple, seq_len, &branch_weight);
            layer.offset.value.data[index] = original;

            let numeric =
                (objective(&high, &upstream) - objective(&low, &upstream)) / (2.0 * epsilon);
            assert!(
                (expected - numeric).abs() < 1e-2,
                "index {index}: {expected} vs {numeric}"
            );
        }
    }

    #[test]
    fn conditioning_gradient_reaches_the_timestep_mlp() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut embedding = TimestepEmbedding::new(8, 12, 6, &mut rng).unwrap();
        let mut modulation = Modulation::new(6, 4);
        // Zero weights would make the projection's own gradient zero here, so
        // start it somewhere else for the check.
        modulation.project.weight.value = rows(12, 6, 2);

        let times = [0.25f32, 0.9];
        let upstream = rows(2, 12, 3);

        let (conditioning, timestep_cache) = embedding.forward_train(&times);
        let (_, modulation_cache) = modulation.forward_train(&conditioning);
        let grad_conditioning = modulation.backward(&modulation_cache, &upstream);
        embedding.backward(&timestep_cache, &grad_conditioning);

        let epsilon = 1e-3;
        for index in [0, 5, 17, 40] {
            let expected = embedding.output.weight.grad.data[index];
            let original = embedding.output.weight.value.data[index];

            embedding.output.weight.value.data[index] = original + epsilon;
            let high = modulation.forward(&embedding.forward(&times));
            embedding.output.weight.value.data[index] = original - epsilon;
            let low = modulation.forward(&embedding.forward(&times));
            embedding.output.weight.value.data[index] = original;

            let numeric =
                (objective(&high, &upstream) - objective(&low, &upstream)) / (2.0 * epsilon);
            assert!(
                (expected - numeric).abs() < 1e-2,
                "index {index}: {expected} vs {numeric}"
            );
        }
    }

    #[test]
    fn a_closed_gate_blocks_the_branch_gradient() {
        let branch = rows(4, 3, 1);
        let gate = Matrix::new(2, 3);
        let upstream = rows(4, 3, 2);
        let (grad_branch, grad_gate) =
            gate_residual_backward(&branch, &gate, &upstream, 2).unwrap();

        assert!(grad_branch.data.iter().all(|&value| value == 0.0));
        // The gate itself still learns, which is what lets a zero-initialized
        // block ever open.
        assert!(grad_gate.data.iter().any(|&value| value != 0.0));
    }
}
