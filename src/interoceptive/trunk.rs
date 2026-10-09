//! The exteroceptive encoder: market window in, hidden state out.

use super::market_sim::TICK_FEATURES;
use super::modulation::add_column_sums;
use crate::batch::Layout;
use crate::matrix::Matrix;
use crate::norm::RmsNorm;
use crate::param::{Linear, Param};
use crate::rope::Rope;
use crate::transformer_block::{FeedForward, TransformerBlock, TransformerBlockCache};
use rand::rngs::StdRng;

/// The trunk [`QNetwork`](super::QNetwork) encodes the market with.
#[derive(Clone, Debug)]
pub enum Trunk {
    Mlp(Box<MarketTrunk>),
    Causal(Box<CausalTrunk>),
}

/// What [`Trunk::forward`] keeps for the backward pass.
#[derive(Clone, Debug)]
pub enum TrunkState {
    Mlp(TrunkCache),
    Causal(CausalCache),
}

impl TrunkState {
    /// The hidden state, `[n, hidden]`.
    pub fn h(&self) -> &Matrix {
        match self {
            TrunkState::Mlp(cache) => &cache.h,
            TrunkState::Causal(cache) => &cache.h,
        }
    }
}

impl Trunk {
    pub fn forward(&self, obs: &Matrix) -> TrunkState {
        match self {
            Trunk::Mlp(trunk) => TrunkState::Mlp(trunk.forward(obs)),
            Trunk::Causal(trunk) => TrunkState::Causal(trunk.forward(obs)),
        }
    }

    pub fn backward(&mut self, obs: &Matrix, state: &TrunkState, grad_h: &Matrix) {
        match (self, state) {
            (Trunk::Mlp(trunk), TrunkState::Mlp(cache)) => trunk.backward(obs, cache, grad_h),
            (Trunk::Causal(trunk), TrunkState::Causal(cache)) => trunk.backward(cache, grad_h),
            _ => panic!("the trunk state is from the other kind of trunk"),
        }
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        match self {
            Trunk::Mlp(trunk) => trunk.params_mut(),
            Trunk::Causal(trunk) => trunk.params_mut(),
        }
    }
}

/// The shape of a [`CausalTrunk`].
#[derive(Clone, Debug, PartialEq)]
pub struct CausalConfig {
    /// Candles in the window: `SimConfig::lookback`.
    pub lookback: usize,
    pub d_model: usize,
    pub layers: usize,
    pub heads: usize,
}

/// A causal transformer over the window, one token per candle, read at the
/// last candle. The features after the window (position and trends) join the
/// last token's state before one ReLU layer to `hidden`.
#[derive(Clone, Debug)]
pub struct CausalTrunk {
    pub lookback: usize,
    pub embed: Linear,
    pub embed_bias: Param,
    pub blocks: Vec<TransformerBlock>,
    pub norm: RmsNorm,
    pub out: Linear,
    pub out_bias: Param,
}

/// Activations [`CausalTrunk::backward`] reads.
#[derive(Clone, Debug)]
pub struct CausalCache {
    tokens: Matrix,
    blocks: Vec<TransformerBlockCache>,
    last: Matrix,
    joined: Matrix,
    pub h: Matrix,
}

impl CausalTrunk {
    /// Panics if the window does not fit in `input` or `d_model` does not
    /// split into `heads`.
    pub fn new(input: usize, hidden: usize, config: &CausalConfig, rng: &mut StdRng) -> Self {
        let (d, heads) = (config.d_model, config.heads);
        let window = config.lookback * TICK_FEATURES;
        assert!(
            config.lookback > 0 && window <= input,
            "the window is wider than the input"
        );
        assert!(
            heads > 0 && d % heads == 0,
            "d_model does not split into heads"
        );
        let head_dim = d / heads;
        let blocks = (0..config.layers)
            .map(|_| {
                let rope = Rope::new(head_dim, config.lookback, 10000.0).unwrap();
                let ffn = FeedForward::swiglu(d, 4 * d, rng);
                TransformerBlock::new(d, heads, heads, head_dim, rope, ffn, 1e-6, rng).unwrap()
            })
            .collect();
        Self {
            lookback: config.lookback,
            embed: Linear::new(TICK_FEATURES, d, rng),
            embed_bias: Param::zeros(1, d),
            blocks,
            norm: RmsNorm::new(d, 1e-6),
            out: Linear::new(d + input - window, hidden, rng),
            out_bias: Param::zeros(1, hidden),
        }
    }

    /// `[n, input] -> [n, hidden]`.
    pub fn forward(&self, obs: &Matrix) -> CausalCache {
        let (n, length) = (obs.rows, self.lookback);
        let window = length * TICK_FEATURES;
        let mut tokens = Matrix::new(n * length, TICK_FEATURES);
        for r in 0..n {
            tokens.data[r * window..(r + 1) * window].copy_from_slice(&obs.row(r)[..window]);
        }
        let mut x = self.embed.forward(&tokens);
        add_bias(&mut x, &self.embed_bias);
        let layout = Layout {
            seq_len: Some(length),
            valid: None,
        };
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (output, cache) = block
                .forward_train(&x, layout)
                .expect("the window's shape is fixed when the trunk is built");
            blocks.push(cache);
            x = output;
        }
        let d = x.cols;
        let mut last = Matrix::new(n, d);
        for r in 0..n {
            last.row_mut(r)
                .copy_from_slice(x.row(r * length + length - 1));
        }
        let normed = self.norm.forward(&last);
        let mut joined = Matrix::new(n, d + obs.cols - window);
        for r in 0..n {
            let row = joined.row_mut(r);
            row[..d].copy_from_slice(normed.row(r));
            row[d..].copy_from_slice(&obs.row(r)[window..]);
        }
        let h = dense_relu(&self.out, &self.out_bias, &joined);
        CausalCache {
            tokens,
            blocks,
            last,
            joined,
            h,
        }
    }

    /// Accumulates every trunk gradient.
    pub fn backward(&mut self, cache: &CausalCache, grad_h: &Matrix) {
        let (n, length, d) = (cache.last.rows, self.lookback, cache.last.cols);
        let grad_z = relu_backward(&cache.h, grad_h);
        add_column_sums(&grad_z, &mut self.out_bias);
        let grad_joined = self.out.backward(&cache.joined, &grad_z);
        let mut grad_normed = Matrix::new(n, d);
        for r in 0..n {
            grad_normed
                .row_mut(r)
                .copy_from_slice(&grad_joined.row(r)[..d]);
        }
        let grad_last = self.norm.backward(&cache.last, &grad_normed);
        let mut grad = Matrix::new(n * length, d);
        for r in 0..n {
            grad.row_mut(r * length + length - 1)
                .copy_from_slice(grad_last.row(r));
        }
        for (block, block_cache) in self.blocks.iter_mut().zip(&cache.blocks).rev() {
            grad = block
                .backward(block_cache, &grad)
                .expect("the window's shape is fixed when the trunk is built");
        }
        add_column_sums(&grad, &mut self.embed_bias);
        self.embed.backward(&cache.tokens, &grad);
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = self.embed.params_mut();
        params.push(&mut self.embed_bias);
        for block in &mut self.blocks {
            params.extend(block.params_mut());
        }
        params.extend(self.norm.params_mut());
        params.extend(self.out.params_mut());
        params.push(&mut self.out_bias);
        params
    }
}

/// Two ReLU layers from the flattened lookback window to `h`.
#[derive(Clone, Debug)]
pub struct MarketTrunk {
    pub l1: Linear,
    pub b1: Param,
    pub l2: Linear,
    pub b2: Param,
}

/// Activations kept for the backward pass.
#[derive(Clone, Debug)]
pub struct TrunkCache {
    pub a1: Matrix,
    pub h: Matrix,
}

impl MarketTrunk {
    pub fn new(input: usize, hidden: usize, rng: &mut StdRng) -> Self {
        Self {
            l1: Linear::new(input, hidden, rng),
            b1: Param::zeros(1, hidden),
            l2: Linear::new(hidden, hidden, rng),
            b2: Param::zeros(1, hidden),
        }
    }

    /// `[n, input] -> [n, hidden]`.
    pub fn forward(&self, obs: &Matrix) -> TrunkCache {
        let a1 = dense_relu(&self.l1, &self.b1, obs);
        let h = dense_relu(&self.l2, &self.b2, &a1);
        TrunkCache { a1, h }
    }

    /// Accumulates every trunk gradient. The observation needs no gradient.
    pub fn backward(&mut self, obs: &Matrix, cache: &TrunkCache, grad_h: &Matrix) {
        let grad_z2 = relu_backward(&cache.h, grad_h);
        add_column_sums(&grad_z2, &mut self.b2);
        let grad_a1 = self.l2.backward(&cache.a1, &grad_z2);
        let grad_z1 = relu_backward(&cache.a1, &grad_a1);
        add_column_sums(&grad_z1, &mut self.b1);
        self.l1.backward(obs, &grad_z1);
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = self.l1.params_mut();
        params.push(&mut self.b1);
        params.extend(self.l2.params_mut());
        params.push(&mut self.b2);
        params
    }
}

fn add_bias(out: &mut Matrix, bias: &Param) {
    for r in 0..out.rows {
        for (v, b) in out.row_mut(r).iter_mut().zip(&bias.value.data) {
            *v += b;
        }
    }
}

fn dense_relu(layer: &Linear, bias: &Param, input: &Matrix) -> Matrix {
    let mut out = layer.forward(input);
    add_bias(&mut out, bias);
    out.data.iter_mut().for_each(|v| *v = v.max(0.0));
    out
}

fn relu_backward(activation: &Matrix, grad: &Matrix) -> Matrix {
    let mut out = grad.clone();
    for (g, a) in out.data.iter_mut().zip(&activation.data) {
        if *a <= 0.0 {
            *g = 0.0;
        }
    }
    out
}
