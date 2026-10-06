//! Drive-reduction reward and the DQN temporal-difference loss.

use super::state::AccountBody;
use crate::matrix::Matrix;

/// `(D(b_t) - D(b_t+1)) + lambda * pnl - penalty * liquidated`.
pub fn homeostatic_reward(
    prev: &AccountBody,
    next: &AccountBody,
    pnl: f32,
    liquidated: bool,
    drive_weights: [f32; 4],
    pnl_lambda: f32,
    terminal_penalty: f32,
) -> f32 {
    let relief = prev.drive(drive_weights) - next.drive(drive_weights);
    relief + pnl_lambda * pnl - if liquidated { terminal_penalty } else { 0.0 }
}

/// `y = r + gamma * (1 - done) * max_a Q_target(s', b', a)`, into `out`.
pub fn bellman_targets(
    rewards: &[f32],
    dones: &[f32],
    next_q: &Matrix,
    gamma: f32,
    out: &mut [f32],
) {
    for (i, y) in out.iter_mut().enumerate() {
        let best = next_q
            .row(i)
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        *y = rewards[i] + gamma * (1.0 - dones[i]) * best;
    }
}

/// Mean Huber loss (delta 1) between `Q(s, a_i)` and `targets`, with
/// `dL/dQ` written into `grad`: non-zero only on the action taken.
pub fn td_huber_loss(q: &Matrix, actions: &[usize], targets: &[f32], grad: &mut Matrix) -> f32 {
    grad.zeros();
    let n = q.rows as f32;
    let mut loss = 0.0;
    for (i, (&a, &y)) in actions.iter().zip(targets).enumerate() {
        let error = q.row(i)[a] - y;
        loss += if error.abs() <= 1.0 {
            0.5 * error * error
        } else {
            error.abs() - 0.5
        };
        grad.row_mut(i)[a] = error.clamp(-1.0, 1.0) / n;
    }
    loss / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relief_pays_and_liquidation_costs() {
        let stressed = AccountBody::from_array([0.9, 0.5, 0.05, 0.0]);
        let calm = AccountBody::default();
        let w = [1.0; 4];
        assert!(homeostatic_reward(&stressed, &calm, 0.0, false, w, 1.0, 1.0) > 0.5);
        assert!(homeostatic_reward(&calm, &calm, 0.0, true, w, 1.0, 1.0) == -1.0);
    }

    #[test]
    fn td_loss_touches_only_the_chosen_action_and_done_cuts_the_bootstrap() {
        let next_q = Matrix::from_vec(2, 2, vec![1.0, 3.0, 5.0, 2.0]);
        let mut y = [0.0; 2];
        bellman_targets(&[1.0, 1.0], &[0.0, 1.0], &next_q, 0.5, &mut y);
        assert_eq!(y, [2.5, 1.0]);

        let q = Matrix::from_vec(2, 2, vec![0.0, 2.0, 1.0, 9.0]);
        let mut grad = Matrix::new(2, 2);
        let loss = td_huber_loss(&q, &[1, 0], &y, &mut grad);
        assert!((loss - 0.125 / 2.0).abs() < 1e-6);
        assert_eq!(grad.data, vec![0.0, -0.25, 0.0, 0.0]);
    }
}
