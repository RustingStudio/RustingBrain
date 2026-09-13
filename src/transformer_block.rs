//! One decoder block: pre-norm attention, then a pre-norm feed-forward.
//!
//! Both sub-layers are residual, so a block is
//! `x + attn(norm(x))` followed by `y + ffn(norm(y))`.

use crate::attention::{AttentionCache, KvCache, MultiHeadAttention};
use crate::batch::Layout;
use crate::ffn::{GeluMlp, GeluMlpCache, SwiGlu, SwiGluCache};
use crate::matrix::Matrix;
use crate::moe::{MoeCache, MoeConfig, MoeLayer};
use crate::network::NetworkError;
#[cfg(feature = "cuda")]
use crate::param::Linear;
use crate::param::Param;
use crate::rope::Rope;
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::norm::RmsNorm;

/// The feed-forward half of a block.
///
/// Real MoE models keep the first few layers dense, so this is selected per
/// layer rather than per model.
// The dense variants are the hot path, so they stay inline; only the much
// larger MoE layer is boxed.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum FeedForward {
    SwiGlu(SwiGlu),
    Gelu(GeluMlp),
    Moe(Box<MoeLayer>),
}

#[derive(Clone, Debug)]
pub enum FeedForwardCache {
    SwiGlu(SwiGluCache),
    Gelu(GeluMlpCache),
    Moe(Box<MoeCache>),
}

impl FeedForward {
    pub fn swiglu(d_model: usize, d_ff: usize, rng: &mut StdRng) -> Self {
        Self::SwiGlu(SwiGlu::new(d_model, d_ff, rng))
    }

    pub fn gelu(d_model: usize, d_ff: usize, rng: &mut StdRng) -> Self {
        Self::Gelu(GeluMlp::new(d_model, d_ff, rng))
    }

    pub fn moe(d_model: usize, config: MoeConfig, rng: &mut StdRng) -> Result<Self, NetworkError> {
        Ok(Self::Moe(Box::new(MoeLayer::new(d_model, config, rng)?)))
    }

    pub fn is_moe(&self) -> bool {
        matches!(self, Self::Moe(_))
    }

    pub fn forward(&self, input: &Matrix) -> Result<Matrix, NetworkError> {
        Ok(match self {
            Self::SwiGlu(ffn) => ffn.forward(input),
            Self::Gelu(ffn) => ffn.forward(input),
            Self::Moe(ffn) => ffn.forward(input)?,
        })
    }

    pub fn forward_train(
        &self,
        input: &Matrix,
        layout: Layout<'_>,
    ) -> Result<(Matrix, FeedForwardCache), NetworkError> {
        Ok(match self {
            Self::SwiGlu(ffn) => {
                let (output, cache) = ffn.forward_train(input);
                (output, FeedForwardCache::SwiGlu(cache))
            }
            Self::Gelu(ffn) => {
                let (output, cache) = ffn.forward_train(input);
                (output, FeedForwardCache::Gelu(cache))
            }
            Self::Moe(ffn) => {
                let (output, cache) = ffn.forward_train(input, layout)?;
                (output, FeedForwardCache::Moe(Box::new(cache)))
            }
        })
    }

    pub fn backward(&mut self, cache: &FeedForwardCache, grad_output: &Matrix) -> Matrix {
        match (self, cache) {
            (Self::SwiGlu(ffn), FeedForwardCache::SwiGlu(cache)) => {
                ffn.backward(cache, grad_output)
            }
            (Self::Gelu(ffn), FeedForwardCache::Gelu(cache)) => ffn.backward(cache, grad_output),
            (Self::Moe(ffn), FeedForwardCache::Moe(cache)) => ffn.backward(cache, grad_output),
            _ => panic!("feed-forward cache does not match the layer that produced it"),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn linears_mut(&mut self) -> Vec<&mut Linear> {
        match self {
            Self::SwiGlu(ffn) => ffn.linears_mut(),
            Self::Gelu(ffn) => ffn.linears_mut(),
            Self::Moe(ffn) => ffn.linears_mut(),
        }
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        match self {
            Self::SwiGlu(ffn) => ffn.params_mut(),
            Self::Gelu(ffn) => ffn.params_mut(),
            Self::Moe(ffn) => ffn.params_mut(),
        }
    }

    pub fn num_parameters(&self) -> usize {
        match self {
            Self::SwiGlu(ffn) => ffn.num_parameters(),
            Self::Gelu(ffn) => ffn.num_parameters(),
            Self::Moe(ffn) => ffn.num_parameters(),
        }
    }

    /// Weights one token multiplies against. Equal to
    /// [`FeedForward::num_parameters`] for the dense variants.
    pub fn active_parameters(&self) -> usize {
        match self {
            Self::Moe(ffn) => ffn.active_parameters(),
            other => other.num_parameters(),
        }
    }
}

impl FeedForwardCache {
    /// Auxiliary losses to add to the language-modelling loss. Zero for a dense
    /// feed-forward.
    pub fn auxiliary_loss(&self) -> f32 {
        match self {
            Self::Moe(cache) => cache.auxiliary_loss(),
            _ => 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TransformerBlockCache {
    input: Matrix,
    attention: AttentionCache,
    residual: Matrix,
    feed_forward: FeedForwardCache,
}

impl TransformerBlockCache {
    pub fn auxiliary_loss(&self) -> f32 {
        self.feed_forward.auxiliary_loss()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TransformerBlock {
    pub attention_norm: RmsNorm,
    pub attention: MultiHeadAttention,
    pub feed_forward_norm: RmsNorm,
    pub feed_forward: FeedForward,
}

impl TransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: Rope,
        feed_forward: FeedForward,
        rmsnorm_eps: f32,
        rng: &mut StdRng,
    ) -> Result<Self, NetworkError> {
        Ok(Self {
            attention_norm: RmsNorm::new(d_model, rmsnorm_eps),
            attention: MultiHeadAttention::new(
                d_model,
                num_heads,
                num_kv_heads,
                head_dim,
                rope,
                rng,
            )?,
            feed_forward_norm: RmsNorm::new(d_model, rmsnorm_eps),
            feed_forward,
        })
    }

    pub fn d_model(&self) -> usize {
        self.attention.d_model()
    }

    pub fn forward_train(
        &self,
        input: &Matrix,
        layout: Layout<'_>,
    ) -> Result<(Matrix, TransformerBlockCache), NetworkError> {
        let attention_normed = self.attention_norm.forward(input);
        let (attended, attention) = self.attention.forward_train(&attention_normed, layout)?;

        let mut residual = attended;
        add_in_place(&mut residual, input);

        let feed_forward_normed = self.feed_forward_norm.forward(&residual);
        let (projected, feed_forward) = self
            .feed_forward
            .forward_train(&feed_forward_normed, layout)?;

        let mut output = projected;
        add_in_place(&mut output, &residual);

        Ok((
            output,
            TransformerBlockCache {
                input: input.clone(),
                attention,
                residual,
                feed_forward,
            },
        ))
    }

    /// Incremental forward against a key/value cache, for generation.
    pub fn forward_cached(
        &self,
        input: &Matrix,
        cache: &mut KvCache,
    ) -> Result<Matrix, NetworkError> {
        let attention_normed = self.attention_norm.forward(input);
        let mut residual = self.attention.forward_cached(&attention_normed, cache)?;
        add_in_place(&mut residual, input);

        let feed_forward_normed = self.feed_forward_norm.forward(&residual);
        let mut output = self.feed_forward.forward(&feed_forward_normed)?;
        add_in_place(&mut output, &residual);

        Ok(output)
    }

    pub fn backward(
        &mut self,
        cache: &TransformerBlockCache,
        grad_output: &Matrix,
    ) -> Result<Matrix, NetworkError> {
        let grad_projected = self.feed_forward.backward(&cache.feed_forward, grad_output);
        let mut grad_residual = self
            .feed_forward_norm
            .backward(&cache.residual, &grad_projected);
        // The residual connection passes the upstream gradient through
        // untouched alongside the branch gradient.
        add_in_place(&mut grad_residual, grad_output);

        let grad_attended = self.attention.backward(&cache.attention, &grad_residual)?;
        let mut grad_input = self.attention_norm.backward(&cache.input, &grad_attended);
        add_in_place(&mut grad_input, &grad_residual);

        Ok(grad_input)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn linears_mut(&mut self) -> Vec<&mut Linear> {
        let mut linears = self.attention.linears_mut();
        linears.extend(self.feed_forward.linears_mut());
        linears
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = self.attention_norm.params_mut();
        params.extend(self.attention.params_mut());
        params.extend(self.feed_forward_norm.params_mut());
        params.extend(self.feed_forward.params_mut());
        params
    }

    pub fn num_parameters(&self) -> usize {
        self.attention_norm.weight.len()
            + self.attention.query.weight.len()
            + self.attention.key.weight.len()
            + self.attention.value.weight.len()
            + self.attention.output.weight.len()
            + self.feed_forward_norm.weight.len()
            + self.feed_forward.num_parameters()
    }

    pub fn active_parameters(&self) -> usize {
        self.num_parameters() - self.feed_forward.num_parameters()
            + self.feed_forward.active_parameters()
    }
}

fn add_in_place(target: &mut Matrix, source: &Matrix) {
    debug_assert_eq!(target.data.len(), source.data.len());
    // Two residual adds per block per pass over `[rows, d_model]`. Pure
    // streaming work, so it is worth spreading over the cores' memory ports.
    target
        .data
        .par_chunks_mut(8192)
        .zip(source.data.par_chunks(8192))
        .for_each(|(destination, source)| {
            for (slot, value) in destination.iter_mut().zip(source) {
                *slot += value;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn block(feed_forward: impl FnOnce(&mut StdRng) -> FeedForward) -> TransformerBlock {
        let mut rng = StdRng::seed_from_u64(21);
        let rope = Rope::new(4, 32, 10000.0).unwrap();
        let ffn = feed_forward(&mut rng);
        TransformerBlock::new(8, 2, 1, 4, rope, ffn, 1e-6, &mut rng).unwrap()
    }

    fn inputs(tokens: usize) -> Matrix {
        Matrix::from_vec(
            tokens,
            8,
            (0..tokens * 8)
                .map(|i| ((i * 31) % 19) as f32 / 9.0 - 1.0)
                .collect(),
        )
    }

    #[test]
    fn a_dense_block_keeps_its_shape() {
        let layer = block(|rng| FeedForward::swiglu(8, 16, rng));
        let (output, _) = layer.forward_train(&inputs(4), Layout::default()).unwrap();

        assert_eq!((output.rows, output.cols), (4, 8));
    }

    #[test]
    fn a_moe_block_keeps_its_shape_and_reports_an_auxiliary_loss() {
        let layer = block(|rng| FeedForward::moe(8, MoeConfig::new(4, 2, 6), rng).unwrap());
        let (output, cache) = layer.forward_train(&inputs(5), Layout::default()).unwrap();

        assert_eq!((output.rows, output.cols), (5, 8));
        assert!(layer.feed_forward.is_moe());
        assert!(cache.auxiliary_loss() > 0.0);
    }

    #[test]
    fn a_dense_block_reports_no_auxiliary_loss() {
        let layer = block(|rng| FeedForward::swiglu(8, 16, rng));
        let (_, cache) = layer.forward_train(&inputs(3), Layout::default()).unwrap();

        assert_eq!(cache.auxiliary_loss(), 0.0);
    }

    #[test]
    fn a_cached_decode_reproduces_the_full_sequence_forward() {
        let layer = block(|rng| FeedForward::moe(8, MoeConfig::new(4, 2, 6), rng).unwrap());
        let sequence = inputs(5);
        let (expected, _) = layer.forward_train(&sequence, Layout::default()).unwrap();

        let mut cache = KvCache::new(1, 4);
        for token in 0..sequence.rows {
            let step = Matrix::from_vec(1, 8, sequence.row(token).to_vec());
            let output = layer.forward_cached(&step, &mut cache).unwrap();
            for (actual, expected) in output.data.iter().zip(expected.row(token)) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn dense_block_backward_matches_finite_differences() {
        let mut layer = block(|rng| FeedForward::swiglu(8, 12, rng));
        let input = inputs(3);

        let (output, cache) = layer.forward_train(&input, Layout::default()).unwrap();
        let grad_output = Matrix::from_vec(output.rows, output.cols, vec![1.0; output.data.len()]);
        let grad_input = layer.backward(&cache, &grad_output).unwrap();

        let epsilon = 1e-3;
        for index in 0..input.data.len() {
            let mut bumped = input.clone();
            bumped.data[index] += epsilon;
            let high: f32 = layer
                .forward_train(&bumped, Layout::default())
                .unwrap()
                .0
                .data
                .iter()
                .sum();
            bumped.data[index] -= 2.0 * epsilon;
            let low: f32 = layer
                .forward_train(&bumped, Layout::default())
                .unwrap()
                .0
                .data
                .iter()
                .sum();
            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (grad_input.data[index] - numeric).abs() < 3e-2,
                "index {index}: {} vs {numeric}",
                grad_input.data[index]
            );
        }
    }

    #[test]
    fn moe_block_backward_matches_finite_differences() {
        let mut layer = block(|rng| {
            FeedForward::moe(
                8,
                MoeConfig::new(4, 2, 6)
                    .with_aux_loss_weight(0.0)
                    .with_router_z_loss_weight(0.0),
                rng,
            )
            .unwrap()
        });
        let input = inputs(3);

        let (output, cache) = layer.forward_train(&input, Layout::default()).unwrap();
        let grad_output = Matrix::from_vec(output.rows, output.cols, vec![1.0; output.data.len()]);
        let grad_input = layer.backward(&cache, &grad_output).unwrap();

        let epsilon = 1e-3;
        for index in 0..input.data.len() {
            let mut bumped = input.clone();
            bumped.data[index] += epsilon;
            let high: f32 = layer
                .forward_train(&bumped, Layout::default())
                .unwrap()
                .0
                .data
                .iter()
                .sum();
            bumped.data[index] -= 2.0 * epsilon;
            let low: f32 = layer
                .forward_train(&bumped, Layout::default())
                .unwrap()
                .0
                .data
                .iter()
                .sum();
            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (grad_input.data[index] - numeric).abs() < 3e-2,
                "index {index}: {} vs {numeric}",
                grad_input.data[index]
            );
        }
    }

    #[test]
    fn a_moe_block_has_more_total_than_active_parameters() {
        let layer = block(|rng| FeedForward::moe(8, MoeConfig::new(8, 2, 6), rng).unwrap());

        assert!(layer.num_parameters() > layer.active_parameters());
    }

    #[test]
    fn a_dense_block_has_no_inactive_parameters() {
        let layer = block(|rng| FeedForward::swiglu(8, 16, rng));

        assert_eq!(layer.num_parameters(), layer.active_parameters());
    }
}
