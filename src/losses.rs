//! Loss functions for dense networks.
//!
//! [`Loss::output_delta`] returns `dL/dz` for the output layer directly, not
//! `dL/dy`: for the softmax/cross-entropy and sigmoid/binary-cross-entropy
//! pairs the activation derivative cancels, and computing the product would
//! reintroduce the catastrophic cancellation the pairing exists to avoid.

use crate::activations::Activation;
use serde::{Deserialize, Serialize};

const EPSILON: f32 = 1e-7;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Loss {
    Mse,
    BinaryCrossEntropy,
    CrossEntropy,
}

impl Loss {
    pub fn value(self, prediction: &[f32], target: &[f32]) -> f32 {
        assert_eq!(prediction.len(), target.len());

        match self {
            Loss::Mse => {
                prediction
                    .iter()
                    .zip(target)
                    .map(|(p, t)| {
                        let diff = p - t;
                        diff * diff
                    })
                    .sum::<f32>()
                    / prediction.len() as f32
            }
            Loss::BinaryCrossEntropy => {
                prediction
                    .iter()
                    .zip(target)
                    .map(|(p, t)| {
                        let p = p.clamp(EPSILON, 1.0 - EPSILON);
                        -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
                    })
                    .sum::<f32>()
                    / prediction.len() as f32
            }
            Loss::CrossEntropy => prediction
                .iter()
                .zip(target)
                .map(|(p, t)| -t * p.clamp(EPSILON, 1.0).ln())
                .sum(),
        }
    }

    pub fn output_delta(
        self,
        prediction: &[f32],
        target: &[f32],
        activation: Activation,
    ) -> Vec<f32> {
        assert_eq!(prediction.len(), target.len());

        prediction
            .iter()
            .zip(target)
            .map(|(&p, &t)| {
                let base = t - p;
                match (self, activation) {
                    (Loss::BinaryCrossEntropy, Activation::Sigmoid)
                    | (Loss::CrossEntropy, Activation::Softmax) => base,
                    _ => base * activation.derivative(p),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_matches_expected_value() {
        let value = Loss::Mse.value(&[2.0, 4.0], &[1.0, 1.0]);
        assert!((value - 5.0).abs() < 1e-6);
    }

    #[test]
    fn cross_entropy_with_softmax_delta_is_target_minus_prediction() {
        let delta = Loss::CrossEntropy.output_delta(
            &[0.2, 0.7, 0.1],
            &[0.0, 1.0, 0.0],
            Activation::Softmax,
        );
        assert_eq!(delta, vec![-0.2, 0.3, -0.1]);
    }
}

/// The KL divergence of a diagonal Gaussian from the unit normal, and its
/// gradients with respect to the mean and the log-variance.
///
/// `0.5 * sum(mean^2 + exp(log_variance) - 1 - log_variance)`, averaged over
/// the elements so the scale does not move when the latent grid does. This is
/// the term that stops a variational autoencoder from ignoring its own noise
/// and collapsing into a plain autoencoder with an unusable latent space.
///
/// ```
/// # use rusting_brain::losses::kl_divergence;
/// // The unit normal itself is zero divergence from the unit normal.
/// let (value, grad_mean, grad_log_variance) = kl_divergence(&[0.0; 4], &[0.0; 4]);
/// assert!(value.abs() < 1e-6);
/// assert_eq!(grad_mean, vec![0.0; 4]);
/// assert_eq!(grad_log_variance, vec![0.0; 4]);
/// ```
pub fn kl_divergence(mean: &[f32], log_variance: &[f32]) -> (f32, Vec<f32>, Vec<f32>) {
    assert_eq!(mean.len(), log_variance.len());
    let count = mean.len().max(1) as f32;

    let mut value = 0.0;
    let mut grad_mean = Vec::with_capacity(mean.len());
    let mut grad_log_variance = Vec::with_capacity(mean.len());
    for (mean, log_variance) in mean.iter().zip(log_variance) {
        let variance = log_variance.exp();
        value += 0.5 * (mean * mean + variance - 1.0 - log_variance);
        grad_mean.push(mean / count);
        grad_log_variance.push(0.5 * (variance - 1.0) / count);
    }
    (value / count, grad_mean, grad_log_variance)
}

/// L1 against a target clamped to `±limit`, and its gradient.
///
/// The signed-distance target of a query point far from the surface carries no
/// information a reconstruction needs: what matters is the zero crossing and
/// its neighbourhood. Clamping collapses every far target onto the band edge,
/// so the model stops spending capacity on distance nobody reads.
///
/// Only the target is clamped. Clamping the prediction as well — which is how
/// DeepSDF states it — makes the gradient vanish for any prediction already
/// outside the band, so a model that starts saturated can never come back. The
/// far field is still uninformative either way, and this version has no dead
/// zone.
///
/// ```
/// # use rusting_brain::losses::clamped_l1;
/// // A target far outside the band is pulled in to the edge, so a prediction
/// // sitting at the edge is already correct.
/// let (value, grad) = clamped_l1(&[0.1], &[0.5], 0.1);
/// assert_eq!(value, 0.0);
/// assert_eq!(grad, vec![0.0]);
///
/// // A prediction outside the band still gets pulled back.
/// let (_, grad) = clamped_l1(&[0.9], &[0.5], 0.1);
/// assert_eq!(grad, vec![1.0]);
/// ```
pub fn clamped_l1(prediction: &[f32], target: &[f32], limit: f32) -> (f32, Vec<f32>) {
    assert_eq!(prediction.len(), target.len());
    let count = prediction.len().max(1) as f32;

    let mut value = 0.0;
    let mut grad = Vec::with_capacity(prediction.len());
    for (prediction, target) in prediction.iter().zip(target) {
        let difference = prediction - target.clamp(-limit, limit);
        value += difference.abs();
        grad.push(match difference == 0.0 {
            true => 0.0,
            false => difference.signum() / count,
        });
    }
    (value / count, grad)
}

#[cfg(test)]
mod divergence_tests {
    use super::*;

    #[test]
    fn the_kl_gradients_match_finite_differences() {
        let mut mean = vec![0.4, -1.2, 0.0, 2.0];
        let mut log_variance = vec![-0.5, 0.3, 1.1, -2.0];
        let (_, grad_mean, grad_log_variance) = kl_divergence(&mean, &log_variance);

        let epsilon = 1e-3;
        for index in 0..mean.len() {
            let original = mean[index];
            mean[index] = original + epsilon;
            let high = kl_divergence(&mean, &log_variance).0;
            mean[index] = original - epsilon;
            let low = kl_divergence(&mean, &log_variance).0;
            mean[index] = original;
            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (grad_mean[index] - numeric).abs() < 1e-4,
                "mean {index}: {} vs {numeric}",
                grad_mean[index]
            );

            let original = log_variance[index];
            log_variance[index] = original + epsilon;
            let high = kl_divergence(&mean, &log_variance).0;
            log_variance[index] = original - epsilon;
            let low = kl_divergence(&mean, &log_variance).0;
            log_variance[index] = original;
            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (grad_log_variance[index] - numeric).abs() < 1e-4,
                "log variance {index}: {} vs {numeric}",
                grad_log_variance[index]
            );
        }
    }

    #[test]
    fn the_divergence_grows_as_the_distribution_drifts() {
        let unit = kl_divergence(&[0.0; 3], &[0.0; 3]).0;
        let shifted = kl_divergence(&[1.0; 3], &[0.0; 3]).0;
        let widened = kl_divergence(&[0.0; 3], &[1.0; 3]).0;
        let narrowed = kl_divergence(&[0.0; 3], &[-1.0; 3]).0;

        assert!(unit.abs() < 1e-6);
        // Half the squared shift, which is the mean's whole contribution.
        assert!((shifted - 0.5).abs() < 1e-6);
        assert!(widened > 0.0 && narrowed > 0.0);
    }

    #[test]
    fn clamping_flattens_the_far_field() {
        // Two query points a long way outside the surface, at very different
        // distances: after clamping they ask the model for the same answer.
        let (near, _) = clamped_l1(&[0.1], &[0.4], 0.1);
        let (far, _) = clamped_l1(&[0.1], &[9.0], 0.1);
        assert_eq!(near, far);

        // Inside the band nothing is clamped and it is plain L1.
        let (value, grad) = clamped_l1(&[0.05, -0.02], &[0.01, 0.03], 0.1);
        assert!((value - (0.04 + 0.05) / 2.0).abs() < 1e-6);
        assert_eq!(grad, vec![0.5, -0.5]);
    }
}
