//! A circular replay buffer over flat, pre-allocated storage, sampled
//! uniformly or, from [`ReplayBuffer::prioritized`], in proportion to each
//! transition's last TD error (Schaul et al. 2016, the proportional variant).

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
    priority: Option<Priority>,
}

/// The state prioritized sampling adds: a sum tree over `priority^alpha`, so
/// drawing a transition and changing its priority are both `O(log n)`.
struct Priority {
    alpha: f32,
    beta: f32,
    /// Leaves start at `leaves`; node `i` holds the sum of `2i` and `2i + 1`.
    tree: Vec<f64>,
    leaves: usize,
    /// The largest priority seen, given to every new transition so it is
    /// replayed at least once.
    max: f64,
    /// What the last [`ReplayBuffer::sample_into`] drew, for the weights and
    /// for [`ReplayBuffer::update_priorities`].
    sampled: Vec<usize>,
    weights: Vec<f32>,
}

impl Priority {
    fn set(&mut self, index: usize, value: f64) {
        let mut node = self.leaves + index;
        self.tree[node] = value;
        while node > 1 {
            node /= 2;
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1];
        }
    }

    /// The leaf whose cumulative range holds `mass`.
    fn find(&self, mut mass: f64) -> usize {
        let mut node = 1;
        while node < self.leaves {
            if mass < self.tree[2 * node] {
                node *= 2;
            } else {
                mass -= self.tree[2 * node];
                node = 2 * node + 1;
            }
        }
        node - self.leaves
    }
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
            priority: None,
        }
    }

    /// A buffer that samples transition `i` with probability
    /// `p_i^alpha / sum p^alpha`, where `p_i` is its last absolute TD error,
    /// and reports importance weights `(N P(i))^-beta`, scaled so the largest
    /// in a batch is one. `alpha` zero is uniform; the paper uses 0.6 and
    /// anneals `beta` from 0.4 to 1 with [`ReplayBuffer::set_beta`].
    pub fn prioritized(capacity: usize, obs_dim: usize, alpha: f32, beta: f32) -> Self {
        let leaves = capacity.next_power_of_two();
        Self {
            priority: Some(Priority {
                alpha,
                beta,
                tree: vec![0.0; 2 * leaves],
                leaves,
                max: 1.0,
                sampled: Vec::new(),
                weights: Vec::new(),
            }),
            ..Self::new(capacity, obs_dim)
        }
    }

    pub fn is_prioritized(&self) -> bool {
        self.priority.is_some()
    }

    pub fn set_beta(&mut self, beta: f32) {
        if let Some(priority) = &mut self.priority {
            priority.beta = beta;
        }
    }

    /// Importance weights for the rows of the last sample, or `None` for a
    /// uniform buffer, whose weights are all one.
    pub fn weights(&self) -> Option<&[f32]> {
        self.priority.as_ref().map(|p| p.weights.as_slice())
    }

    /// Sets the priority of each transition in the last sample to its
    /// absolute TD error, row for row. Does nothing on a uniform buffer.
    pub fn update_priorities(&mut self, td_errors: &[f32]) {
        let Some(priority) = &mut self.priority else {
            return;
        };
        assert_eq!(td_errors.len(), priority.sampled.len());
        for (row, error) in td_errors.iter().enumerate() {
            // A small floor keeps a transition with zero error reachable.
            let p = (error.abs() as f64 + 1e-6).powf(priority.alpha as f64);
            priority.max = priority.max.max(p);
            let index = priority.sampled[row];
            priority.set(index, p);
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
        if let Some(priority) = &mut self.priority {
            let max = priority.max;
            priority.set(i, max);
        }
        self.next = (i + 1) % self.capacity;
        self.len = (self.len + 1).min(self.capacity);
    }

    /// Fills `batch` from a prioritized buffer: one transition from each of
    /// `batch.len()` equal slices of the total priority, recorded for
    /// [`ReplayBuffer::weights`] and [`ReplayBuffer::update_priorities`]. On a
    /// uniform buffer this is [`ReplayBuffer::sample_into`].
    pub fn sample_prioritized_into(&mut self, batch: &mut BatchData, rng: &mut StdRng) {
        let Some(priority) = &mut self.priority else {
            return self.sample_into(batch, rng);
        };
        assert!(self.len > 0, "sampling an empty replay buffer");
        let total = priority.tree[1];
        let slice = total / batch.len() as f64;
        priority.sampled.clear();
        priority.weights.clear();
        for row in 0..batch.len() {
            let mass = (row as f64 + rng.r#gen::<f64>()) * slice;
            // Rounding can land past the last filled leaf.
            let index = priority.find(mass).min(self.len - 1);
            let probability = priority.tree[priority.leaves + index] / total;
            priority.sampled.push(index);
            priority
                .weights
                .push((self.len as f64 * probability).powf(-priority.beta as f64) as f32);
        }
        let largest = priority.weights.iter().fold(0.0f32, |a, &w| a.max(w));
        priority.weights.iter_mut().for_each(|w| *w /= largest);
        let priority = self.priority.as_ref().unwrap();
        for (row, &index) in priority.sampled.iter().enumerate() {
            self.copy_row(index, batch, row);
        }
    }

    /// Fills `batch` with transitions drawn uniformly with replacement,
    /// ignoring any priorities.
    pub fn sample_into(&self, batch: &mut BatchData, rng: &mut StdRng) {
        assert!(!self.is_empty(), "sampling an empty replay buffer");
        for row in 0..batch.len() {
            self.copy_row(rng.gen_range(0..self.len), batch, row);
        }
    }

    fn copy_row(&self, i: usize, batch: &mut BatchData, row: usize) {
        let d = self.obs_dim;
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

    #[test]
    fn prioritized_sampling_follows_the_td_errors() {
        // Five transitions in a capacity of eight, so the tree has empty leaves.
        let mut buffer = ReplayBuffer::prioritized(8, 1, 1.0, 1.0);
        let body = AccountBody::default();
        for step in 0..5 {
            buffer.push(
                &[step as f32],
                &body,
                step,
                step as f32,
                &[0.0],
                &body,
                false,
            );
        }
        let mut rng = StdRng::seed_from_u64(0);
        // Every transition starts at the same maximum priority.
        let mut batch = BatchData::new(5, 1);
        buffer.sample_prioritized_into(&mut batch, &mut rng);
        assert!(
            buffer
                .weights()
                .unwrap()
                .iter()
                .all(|&w| (w - 1.0).abs() < 1e-6)
        );

        // Give transition k the error k + 1 by sampling and answering until
        // every transition has been answered.
        let mut answered = [false; 5];
        while answered.iter().any(|a| !a) {
            buffer.sample_prioritized_into(&mut batch, &mut rng);
            let errors: Vec<f32> = batch.rewards.iter().map(|r| r + 1.0).collect();
            batch
                .rewards
                .iter()
                .for_each(|&r| answered[r as usize] = true);
            buffer.update_priorities(&errors);
        }

        let mut counts = [0usize; 5];
        let mut batch = BatchData::new(50, 1);
        for _ in 0..2_000 {
            buffer.sample_prioritized_into(&mut batch, &mut rng);
            for (row, &reward) in batch.rewards.iter().enumerate() {
                let k = reward as usize;
                counts[k] += 1;
                // (N P(k))^-1 over the largest in the batch.
                let largest = batch
                    .rewards
                    .iter()
                    .fold(0.0f32, |a, &r| a.max(1.0 / (r + 1.0)));
                let expected = (1.0 / (k as f32 + 1.0)) / largest;
                assert!((buffer.weights().unwrap()[row] - expected).abs() < 1e-3);
            }
        }
        // Priorities 1..=5 sum to 15.
        for (k, &count) in counts.iter().enumerate() {
            let frequency = count as f32 / 100_000.0;
            let expected = (k + 1) as f32 / 15.0;
            assert!(
                (frequency - expected).abs() < 0.01,
                "{k}: {frequency} against {expected}"
            );
        }

        // A new transition gets the largest priority seen, 5.
        buffer.push(&[5.0], &body, 5, 5.0, &[0.0], &body, false);
        let mut seen = 0;
        for _ in 0..1_000 {
            buffer.sample_prioritized_into(&mut batch, &mut rng);
            seen += batch.rewards.iter().filter(|&&r| r == 5.0).count();
        }
        let frequency = seen as f32 / 50_000.0;
        assert!((frequency - 5.0 / 20.0).abs() < 0.01, "{frequency}");
    }
}
