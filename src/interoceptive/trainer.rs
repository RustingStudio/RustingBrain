//! The training loop: a pool of environments stepped in parallel, batched
//! action selection, and DQN updates from replay.

use super::agent::{AgentConfig, InteroceptiveAgent};
use super::buffer::{BatchData, ReplayBuffer};
use super::market_sim::{MarketEnvironment, MarketTick, SimConfig, StepInfo};
use super::state::AccountBody;
use crate::matrix::Matrix;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct TrainerConfig {
    pub n_envs: usize,
    /// Transitions to collect, summed over every environment.
    pub total_steps: usize,
    pub batch_size: usize,
    pub buffer_capacity: usize,
    /// Transitions in the buffer before the first update.
    pub warmup: usize,
    /// Updates after each parallel step of the pool.
    pub updates_per_step: usize,
    pub epsilon_start: f32,
    pub epsilon_end: f32,
    /// Transitions over which epsilon falls linearly to its end value.
    pub epsilon_decay_steps: usize,
    pub seed: u64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            n_envs: 8,
            total_steps: 50_000,
            batch_size: 64,
            buffer_capacity: 100_000,
            warmup: 1_000,
            updates_per_step: 1,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay_steps: 25_000,
            seed: 0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrainReport {
    pub transitions: usize,
    pub updates: usize,
    pub episodes: usize,
    pub liquidations: usize,
    pub mean_loss: f32,
    pub mean_episode_reward: f32,
}

pub struct Trainer {
    pub agent: InteroceptiveAgent,
    pub envs: Vec<MarketEnvironment>,
    pub buffer: ReplayBuffer,
    pub config: TrainerConfig,
    /// Transitions collected over every [`Trainer::run`], which is what the
    /// epsilon schedule counts, so a run split into several calls explores
    /// like one long run.
    collected: usize,
    rng: StdRng,
}

impl Trainer {
    /// Environment `i` is seeded `sim.seed + i`, so the pool sees different
    /// episodes of the same data.
    pub fn new(
        data: Arc<Vec<MarketTick>>,
        sim: SimConfig,
        agent: AgentConfig,
        config: TrainerConfig,
    ) -> Self {
        assert_eq!(
            agent.obs_dim,
            sim.obs_dim(),
            "agent and simulator disagree on obs_dim"
        );
        let envs = (0..config.n_envs as u64)
            .map(|i| {
                let sim = SimConfig {
                    seed: sim.seed + i,
                    ..sim.clone()
                };
                MarketEnvironment::new(data.clone(), sim)
            })
            .collect();
        Self {
            agent: InteroceptiveAgent::new(agent),
            envs,
            buffer: ReplayBuffer::new(config.buffer_capacity, sim.obs_dim()),
            rng: StdRng::seed_from_u64(config.seed),
            collected: 0,
            config,
        }
    }

    /// Collects `config.total_steps` more transitions, training as it goes.
    pub fn run(&mut self) -> TrainReport {
        let cfg = self.config.clone();
        let n = self.envs.len();
        let dim = self.agent.config.obs_dim;
        let mut obs = Matrix::new(n, dim);
        let mut next_obs = Matrix::new(n, dim);
        let mut body_matrix = Matrix::new(n, 4);
        let mut bodies: Vec<AccountBody> = self
            .envs
            .iter_mut()
            .zip(obs.data.chunks_exact_mut(dim))
            .map(|(env, row)| env.reset_into(row))
            .collect();
        let mut actions = vec![super::Action::Hold; n];
        let mut infos = vec![StepInfo::default(); n];
        let mut episode_reward = vec![0.0f32; n];
        let mut batch = BatchData::new(cfg.batch_size, dim);
        let (mut report, mut loss_sum, mut reward_sum) = (TrainReport::default(), 0.0, 0.0);

        while report.transitions < cfg.total_steps {
            let progress = ((self.collected + report.transitions) as f32
                / cfg.epsilon_decay_steps.max(1) as f32)
                .min(1.0);
            let epsilon = cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * progress;
            for (row, body) in body_matrix.data.chunks_exact_mut(4).zip(&bodies) {
                row.copy_from_slice(&body.to_array());
            }
            let q = self.agent.q_values(&obs, &body_matrix);
            for (i, action) in actions.iter_mut().enumerate() {
                *action = self
                    .agent
                    .choose(q.row(i), &bodies[i], epsilon, &mut self.rng);
            }

            next_obs
                .data
                .par_chunks_exact_mut(dim)
                .zip(self.envs.par_iter_mut())
                .zip(actions.par_iter())
                .zip(infos.par_iter_mut())
                .for_each(|(((row, env), action), info)| *info = env.step_into(*action, row));

            for i in 0..n {
                let info = infos[i];
                self.buffer.push(
                    obs.row(i),
                    &bodies[i],
                    actions[i] as usize,
                    info.reward,
                    next_obs.row(i),
                    &info.body,
                    info.done,
                );
                episode_reward[i] += info.reward;
                bodies[i] = info.body;
                if info.done {
                    report.episodes += 1;
                    report.liquidations += info.liquidated as usize;
                    reward_sum += episode_reward[i];
                    episode_reward[i] = 0.0;
                    bodies[i] = self.envs[i].reset_into(next_obs.row_mut(i));
                }
            }
            std::mem::swap(&mut obs, &mut next_obs);
            report.transitions += n;

            if self.buffer.len() >= cfg.warmup.max(1) {
                for _ in 0..cfg.updates_per_step {
                    self.buffer.sample_into(&mut batch, &mut self.rng);
                    loss_sum += self.agent.train_step(&batch);
                    report.updates += 1;
                }
            }
        }
        self.collected += report.transitions;
        report.mean_loss = loss_sum / report.updates.max(1) as f32;
        report.mean_episode_reward = reward_sum / report.episodes.max(1) as f32;
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interoceptive::synthetic_ticks;

    #[test]
    fn a_short_run_collects_trains_and_finishes_episodes() {
        let sim = SimConfig {
            lookback: 8,
            trend_horizons: vec![4, 16],
            episode_len: 32,
            ..SimConfig::default()
        };
        let agent = AgentConfig {
            hidden: 16,
            ..AgentConfig::new(sim.obs_dim())
        };
        let config = TrainerConfig {
            n_envs: 4,
            total_steps: 800,
            batch_size: 16,
            warmup: 64,
            epsilon_decay_steps: 400,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(Arc::new(synthetic_ticks(500, 1)), sim, agent, config);
        let report = trainer.run();
        assert_eq!(report.transitions, 800);
        assert!(report.episodes >= 4 * (800 / 4 / 32));
        assert!(report.updates > 100);
        assert!(report.mean_loss.is_finite());
    }
}
