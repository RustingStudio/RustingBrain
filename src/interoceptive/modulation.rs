//! How the body acts on the market trunk: gain on the hidden units, a bias on
//! the action values, and a temperature on exploration.

use crate::matrix::Matrix;
use crate::param::Param;
use rand::Rng;
use rand::rngs::StdRng;

/// Column of `w_bias` per body signal, row per action
/// (`Hold, BuySmall, SellSmall, DeRiskHalf, EmergencyFlatten`).
///
/// The prior is a reflex, not a policy: margin stress pushes hard toward
/// flattening and away from adding, drawdown velocity pushes the same way more
/// gently, slippage pain favours doing nothing, fatigue favours trimming.
/// Training is free to move all of it.
const BIAS_PRIOR: [[f32; 4]; 5] = [
    [0.0, 0.0, 1.0, 0.0],
    [-4.0, -2.0, -1.0, -1.0],
    [-4.0, -2.0, -1.0, -1.0],
    [4.0, 2.0, 0.0, 2.0],
    [8.0, 4.0, 0.0, 1.0],
];

/// Learned coupling from the four body signals into the network.
#[derive(Clone, Debug)]
pub struct InteroceptiveCore {
    /// `[hidden, 4]`.
    pub w_gain: Param,
    /// `[1, hidden]`.
    pub c_gain: Param,
    /// `[actions, 4]`.
    pub w_bias: Param,
}

/// What the backward pass needs from [`InteroceptiveCore::modulate`].
#[derive(Clone, Debug)]
pub struct GainCache {
    /// `tanh(W_gain b + c_gain)`, `[n, hidden]`.
    pub gate: Matrix,
}

impl InteroceptiveCore {
    /// Gain weights start small and the gain bias at zero, so a calm body
    /// passes the trunk's hidden state through almost untouched.
    pub fn new(hidden: usize, actions: usize, rng: &mut StdRng) -> Self {
        assert_eq!(actions, BIAS_PRIOR.len());
        let w_gain = (0..hidden * 4).map(|_| rng.gen_range(-0.1..0.1)).collect();
        Self {
            w_gain: Param::new(Matrix::from_vec(hidden, 4, w_gain)),
            c_gain: Param::zeros(1, hidden),
            w_bias: Param::new(Matrix::from_vec(actions, 4, BIAS_PRIOR.concat())),
        }
    }

    /// `h' = h * (1 + tanh(W_gain b + c_gain))`, row by row.
    pub fn modulate(&self, h: &Matrix, body: &Matrix) -> (Matrix, GainCache) {
        let mut gate = Matrix::new(h.rows, h.cols);
        body.dot_rhs_transposed(&self.w_gain.value, &mut gate);
        let mut out = h.clone();
        for r in 0..h.rows {
            for ((g, o), c) in gate
                .row_mut(r)
                .iter_mut()
                .zip(out.row_mut(r))
                .zip(&self.c_gain.value.data)
            {
                *g = (*g + c).tanh();
                *o *= 1.0 + *g;
            }
        }
        (out, GainCache { gate })
    }

    /// Accumulates the gain gradients and returns `dL/dh`.
    pub fn modulate_backward(
        &mut self,
        h: &Matrix,
        body: &Matrix,
        cache: &GainCache,
        grad_out: &Matrix,
    ) -> Matrix {
        let mut grad_h = grad_out.clone();
        let mut grad_pre = grad_out.clone();
        for i in 0..grad_out.data.len() {
            let g = cache.gate.data[i];
            grad_h.data[i] *= 1.0 + g;
            grad_pre.data[i] *= h.data[i] * (1.0 - g * g);
        }
        grad_pre.dot_self_transposed_accumulate(body, &mut self.w_gain.grad);
        add_column_sums(&grad_pre, &mut self.c_gain);
        grad_h
    }

    /// `logits += W_bias b`.
    pub fn add_bias(&self, body: &Matrix, logits: &mut Matrix) {
        let mut bias = Matrix::new(logits.rows, logits.cols);
        body.dot_rhs_transposed(&self.w_bias.value, &mut bias);
        for (l, b) in logits.data.iter_mut().zip(&bias.data) {
            *l += b;
        }
    }

    pub fn add_bias_backward(&mut self, body: &Matrix, grad_logits: &Matrix) {
        grad_logits.dot_self_transposed_accumulate(body, &mut self.w_bias.grad);
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        vec![&mut self.w_gain, &mut self.c_gain, &mut self.w_bias]
    }
}

/// `tau = tau0 * exp(-beta * drawdown_velocity)`: panic cools exploration.
pub fn temperature(tau0: f32, beta: f32, drawdown_velocity: f32) -> f32 {
    tau0 * (-beta * drawdown_velocity).exp()
}

/// `target[0, j] += sum_i m[i, j]`, a bias gradient.
pub(crate) fn add_column_sums(m: &Matrix, target: &mut Param) {
    for r in 0..m.rows {
        for (t, v) in target.grad.data.iter_mut().zip(m.row(r)) {
            *t += v;
        }
    }
}
