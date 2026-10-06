//! A circular replay buffer over flat, pre-allocated storage.

use super::state::AccountBody;
use crate::matrix::Matrix;
use rand::Rng;
use rand::rngs::StdRng;

/// A sampled minibatch, one row per transition. Reused across samples.
#[derive(Clone, Debug)]
pub struct BatchData {
    pub obs: Matrix,
    pub body: Matrix,
    pub actions: Vec<usize>,
    pub rewards: Vec<f32>,
    pub next_obs: Matrix,
    pub next_body: Matrix,
    /// One where the episode ended, zero otherwise.
    pub dones: Vec<f32>,
}

impl BatchData {
    pub fn new(batch_size: usize, obs_dim: usize) -> Self {
        Self {
            obs: Matrix::new(batch_size, obs_dim),
            body: Matrix::new(batch_size, 4),
            actions: vec![0; batch_size],
            rewards: vec![0.0; batch_size],
            next_obs: Matrix::new(batch_size, obs_dim),
            next_body: Matrix::new(batch_size, 4),
            dones: vec![0.0; batch_size],
        }
    }

    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}

/// `(s, b, a, r, s', b', done)` in structure-of-arrays form. Pushing past
/// capacity overwrites the oldest transition.
pub struct ReplayBuffer {
    obs_dim: usize,
    capacity: usize,
    len: usize,
    next: usize,
    obs: Vec<f32>,
    body: Vec<f32>,
    actions: Vec<u8>,
    rewards: Vec<f32>,
    next_obs: Vec<f32>,
    next_body: Vec<f32>,
    dones: Vec<bool>,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, obs_dim: usize) -> Self {
        assert!(capacity > 0);
        Self {
            obs_dim,
            capacity,
            len: 0,
            next: 0,
            obs: vec![0.0; capacity * obs_dim],
            body: vec![0.0; capacity * 4],
            actions: vec![0; capacity],
            rewards: vec![0.0; capacity],
            next_obs: vec![0.0; capacity * obs_dim],
            next_body: vec![0.0; capacity * 4],
            dones: vec![false; capacity],
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        obs: &[f32],
        body: &AccountBody,
        action: usize,
        reward: f32,
        next_obs: &[f32],
        next_body: &AccountBody,
        done: bool,
    ) {
        let (i, d) = (self.next, self.obs_dim);
        self.obs[i * d..(i + 1) * d].copy_from_slice(obs);
        self.next_obs[i * d..(i + 1) * d].copy_from_slice(next_obs);
        self.body[i * 4..(i + 1) * 4].copy_from_slice(&body.to_array());
        self.next_body[i * 4..(i + 1) * 4].copy_from_slice(&next_body.to_array());
        self.actions[i] = action as u8;
        self.rewards[i] = reward;
        self.dones[i] = done;
        self.next = (i + 1) % self.capacity;
        self.len = (self.len + 1).min(self.capacity);
    }

    /// Fills `batch` with transitions drawn uniformly with replacement.
    pub fn sample_into(&self, batch: &mut BatchData, rng: &mut StdRng) {
        assert!(!self.is_empty(), "sampling an empty replay buffer");
        let d = self.obs_dim;
        for row in 0..batch.len() {
            let i = rng.gen_range(0..self.len);
            batch
                .obs
                .row_mut(row)
                .copy_from_slice(&self.obs[i * d..(i + 1) * d]);
            batch
                .next_obs
                .row_mut(row)
                .copy_from_slice(&self.next_obs[i * d..(i + 1) * d]);
            batch
                .body
                .row_mut(row)
                .copy_from_slice(&self.body[i * 4..(i + 1) * 4]);
            batch
                .next_body
                .row_mut(row)
                .copy_from_slice(&self.next_body[i * 4..(i + 1) * 4]);
            batch.actions[row] = self.actions[i] as usize;
            batch.rewards[row] = self.rewards[i];
            batch.dones[row] = if self.dones[i] { 1.0 } else { 0.0 };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn pushing_past_capacity_overwrites_the_oldest() {
        let mut buffer = ReplayBuffer::new(3, 1);
        let body = AccountBody::default();
        for step in 0..5 {
            let x = step as f32;
            buffer.push(&[x], &body, step, x, &[x + 1.0], &body, false);
        }
        assert_eq!(buffer.len(), 3);
        let mut batch = BatchData::new(64, 1);
        buffer.sample_into(&mut batch, &mut StdRng::seed_from_u64(0));
        assert!(batch.rewards.iter().all(|&r| r >= 2.0));
        for row in 0..64 {
            assert_eq!(batch.next_obs.data[row], batch.obs.data[row] + 1.0);
            assert_eq!(batch.actions[row] as f32, batch.rewards[row]);
        }
    }
}
