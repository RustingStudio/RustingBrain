//! The exteroceptive encoder: market window in, hidden state out.

use super::modulation::add_column_sums;
use crate::matrix::Matrix;
use crate::param::{Linear, Param};
use rand::rngs::StdRng;

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

fn dense_relu(layer: &Linear, bias: &Param, input: &Matrix) -> Matrix {
    let mut out = layer.forward(input);
    for r in 0..out.rows {
        for (v, b) in out.row_mut(r).iter_mut().zip(&bias.value.data) {
            *v = (*v + b).max(0.0);
        }
    }
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
