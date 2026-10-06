//! Trunk, core and Q head together, and the DQN agent around them.

use super::Tensor;
use super::buffer::BatchData;
use super::loss::{bellman_targets, td_huber_loss};
use super::market_sim::Action;
use super::modulation::{GainCache, InteroceptiveCore, add_column_sums, temperature};
use super::state::AccountBody;
use super::trunk::{MarketTrunk, TrunkCache};
use crate::matrix::Matrix;
use crate::optimizers::{Optimizer, step_clipped};
use crate::param::{Linear, Param};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Debug, PartialEq)]
pub struct AgentConfig {
    pub obs_dim: usize,
    pub hidden: usize,
    pub gamma: f32,
    pub optimizer: Optimizer,
    pub max_grad_norm: f32,
    /// Exploration temperature for a calm body.
    pub tau0: f32,
    /// How sharply drawdown velocity cools exploration.
    pub beta: f32,
    /// Updates between copies of the online network into the target network.
    pub target_sync: usize,
    pub seed: u64,
}

impl AgentConfig {
    pub fn new(obs_dim: usize) -> Self {
        Self {
            obs_dim,
            hidden: 128,
            gamma: 0.99,
            optimizer: Optimizer::adam(3e-4),
            max_grad_norm: 10.0,
            tau0: 1.0,
            beta: 5.0,
            target_sync: 250,
            seed: 0,
        }
    }
}

/// `Q(s, b) = W_head (h * (1 + tanh(W_gain b + c))) + c_head + W_bias b`,
/// with `h` the trunk's encoding of `s`.
#[derive(Clone, Debug)]
pub struct QNetwork {
    pub trunk: MarketTrunk,
    pub core: InteroceptiveCore,
    pub head: Linear,
    pub head_bias: Param,
}

/// Everything [`QNetwork::backward`] needs from the forward pass.
#[derive(Clone, Debug)]
pub struct ForwardCache {
    pub trunk: TrunkCache,
    pub gain: GainCache,
    /// Modulated hidden state `h'`, `[n, hidden]`.
    pub h_mod: Matrix,
    /// `[n, actions]`.
    pub q: Matrix,
}

impl QNetwork {
    pub fn new(obs_dim: usize, hidden: usize, rng: &mut StdRng) -> Self {
        Self {
            trunk: MarketTrunk::new(obs_dim, hidden, rng),
            core: InteroceptiveCore::new(hidden, Action::COUNT, rng),
            head: Linear::new(hidden, Action::COUNT, rng),
            head_bias: Param::zeros(1, Action::COUNT),
        }
    }

    /// `obs` is `[n, obs_dim]`, `body` is `[n, 4]`.
    pub fn forward(&self, obs: &Matrix, body: &Matrix) -> ForwardCache {
        let trunk = self.trunk.forward(obs);
        let (h_mod, gain) = self.core.modulate(&trunk.h, body);
        let mut q = self.head.forward(&h_mod);
        for r in 0..q.rows {
            for (v, b) in q.row_mut(r).iter_mut().zip(&self.head_bias.value.data) {
                *v += b;
            }
        }
        self.core.add_bias(body, &mut q);
        ForwardCache {
            trunk,
            gain,
            h_mod,
            q,
        }
    }

    /// Accumulates `dL/dparam` for every parameter from `dL/dQ`.
    pub fn backward(&mut self, obs: &Matrix, body: &Matrix, cache: &ForwardCache, grad_q: &Matrix) {
        add_column_sums(grad_q, &mut self.head_bias);
        self.core.add_bias_backward(body, grad_q);
        let grad_h_mod = self.head.backward(&cache.h_mod, grad_q);
        let grad_h = self
            .core
            .modulate_backward(&cache.trunk.h, body, &cache.gain, &grad_h_mod);
        self.trunk.backward(obs, &cache.trunk, &grad_h);
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = self.trunk.params_mut();
        params.extend(self.core.params_mut());
        params.extend(self.head.params_mut());
        params.push(&mut self.head_bias);
        params
    }
}

/// A DQN agent with a target network.
pub struct InteroceptiveAgent {
    pub config: AgentConfig,
    pub online: QNetwork,
    pub target: QNetwork,
    updates: usize,
}

impl InteroceptiveAgent {
    pub fn new(config: AgentConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let online = QNetwork::new(config.obs_dim, config.hidden, &mut rng);
        Self {
            target: online.clone(),
            online,
            config,
            updates: 0,
        }
    }

    /// Q-values and modulated hidden state for every row of `obs`, all under
    /// the same body.
    pub fn forward(&self, obs: &Tensor, body: &AccountBody) -> (Tensor, Tensor) {
        let cache = self.online.forward(obs, &body_rows(body, obs.rows));
        (cache.q, cache.h_mod)
    }

    /// Q-values with one body per row, `bodies` being `[n, 4]`.
    pub fn q_values(&self, obs: &Matrix, bodies: &Matrix) -> Matrix {
        self.online.forward(obs, bodies).q
    }

    /// Epsilon-greedy over the first row of `obs`, where the exploratory
    /// draw is Boltzmann at the body's temperature.
    pub fn select_action(&self, obs: &Tensor, body: &AccountBody, epsilon: f32) -> Action {
        let (q, _) = self.forward(obs, body);
        self.choose(q.row(0), body, epsilon, &mut rand::thread_rng())
    }

    /// The policy on one row of Q-values. With probability `epsilon`, sample
    /// `softmax(q / tau)` with `tau` from [`temperature`]; otherwise argmax.
    /// A panicking body has `tau` near zero, so even exploration turns greedy.
    pub fn choose(
        &self,
        q: &[f32],
        body: &AccountBody,
        epsilon: f32,
        rng: &mut impl Rng,
    ) -> Action {
        let best = argmax(q);
        if rng.gen_range(0.0..1.0f32) >= epsilon {
            return Action::from_index(best);
        }
        let tau = temperature(self.config.tau0, self.config.beta, body.drawdown_velocity);
        if tau < 1e-3 {
            return Action::from_index(best);
        }
        let mut weights = [0.0f32; Action::COUNT];
        for (w, v) in weights.iter_mut().zip(q) {
            *w = ((v - q[best]) / tau).exp();
        }
        let mut draw = rng.gen_range(0.0..weights.iter().sum::<f32>());
        for (i, w) in weights.iter().enumerate() {
            if draw < *w {
                return Action::from_index(i);
            }
            draw -= w;
        }
        Action::from_index(best)
    }

    /// One DQN update against the target network. Returns the Huber loss.
    pub fn train_step(&mut self, batch: &BatchData) -> f32 {
        let next_q = self.target.forward(&batch.next_obs, &batch.next_body).q;
        let mut targets = vec![0.0; batch.len()];
        bellman_targets(
            &batch.rewards,
            &batch.dones,
            &next_q,
            self.config.gamma,
            &mut targets,
        );

        let cache = self.online.forward(&batch.obs, &batch.body);
        let mut grad = Matrix::new(batch.len(), Action::COUNT);
        let loss = td_huber_loss(&cache.q, &batch.actions, &targets, &mut grad);
        self.online.backward(&batch.obs, &batch.body, &cache, &grad);

        self.updates += 1;
        step_clipped(
            &mut self.online.params_mut(),
            &self.config.optimizer,
            self.updates,
            1.0,
            self.config.max_grad_norm,
        )
        .expect("host parameters cannot fail a step");
        if self.updates % self.config.target_sync == 0 {
            self.target = self.online.clone();
        }
        loss
    }

    pub fn updates(&self) -> usize {
        self.updates
    }
}

/// `n` copies of `body` as a `[n, 4]` matrix.
pub fn body_rows(body: &AccountBody, n: usize) -> Matrix {
    Matrix::from_vec(n, 4, body.to_array().repeat(n))
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn margin_stress_raises_the_flatten_logit_reflexively() {
        let agent = InteroceptiveAgent::new(AgentConfig::new(12));
        let mut rng = StdRng::seed_from_u64(7);
        let obs = Matrix::from_vec(1, 12, (0..12).map(|_| rng.gen_range(-1.0..1.0)).collect());
        let calm = AccountBody {
            margin_stress: 0.05,
            ..AccountBody::default()
        };
        let stressed = AccountBody {
            margin_stress: 0.95,
            ..calm
        };

        let flatten = Action::EmergencyFlatten as usize;
        let (q_calm, _) = agent.forward(&obs, &calm);
        let (q_stress, _) = agent.forward(&obs, &stressed);
        let shift = q_stress.row(0)[flatten] - q_calm.row(0)[flatten];
        assert!(shift > 3.0, "flatten logit moved by only {shift}");
        assert_eq!(argmax(q_stress.row(0)), flatten);
    }

    #[test]
    fn panic_makes_exploration_greedy() {
        let agent = InteroceptiveAgent::new(AgentConfig {
            beta: 20.0,
            ..AgentConfig::new(4)
        });
        let panic = AccountBody::from_array([0.1, 1.0, 0.05, 0.0]);
        let q = [0.0, 0.1, 0.0, 0.0, 0.0];
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..100 {
            assert_eq!(agent.choose(&q, &panic, 1.0, &mut rng), Action::BuySmall);
        }
    }

    #[test]
    fn backward_matches_finite_differences() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut net = QNetwork::new(3, 6, &mut rng);
        let mut random = |rows, cols| {
            Matrix::from_vec(
                rows,
                cols,
                (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            )
        };
        let obs = random(4, 3);
        let body = Matrix::from_vec(4, 4, random(4, 4).data.iter().map(|v| v.abs()).collect());
        let weights = random(4, Action::COUNT);
        let loss = |net: &QNetwork| -> f32 {
            let q = net.forward(&obs, &body).q;
            q.data.iter().zip(&weights.data).map(|(a, b)| a * b).sum()
        };

        let cache = net.forward(&obs, &body);
        net.backward(&obs, &body, &cache, &weights);
        let analytic: Vec<Vec<f32>> = net
            .params_mut()
            .iter()
            .map(|p| p.grad.data.clone())
            .collect();

        let eps = 1e-3;
        for (p, grads) in analytic.iter().enumerate() {
            for (i, &g) in grads.iter().enumerate() {
                let mut plus = net.clone();
                plus.params_mut()[p].value.data[i] += eps;
                let mut minus = net.clone();
                minus.params_mut()[p].value.data[i] -= eps;
                let numeric = (loss(&plus) - loss(&minus)) / (2.0 * eps);
                assert!(
                    (numeric - g).abs() < 1e-2 * (1.0 + g.abs()),
                    "param {p}[{i}]: analytic {g}, numeric {numeric}"
                );
            }
        }
    }
}
