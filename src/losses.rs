//! Loss functions for dense networks.
//!
//! [`Loss::output_delta`] returns `dL/dz` for the output layer directly, not
//! `dL/dy`: for the softmax/cross-entropy and sigmoid/binary-cross-entropy
//! pairs the activation derivative cancels, and computing the product would
//! reintroduce the catastrophic cancellation the pairing exists to avoid.

use crate::activations::Activation;
use crate::matrix::Matrix;
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
    kl_divergence_free_bits(mean, log_variance, 0.0)
}

/// [`kl_divergence`] with a per-unit floor, below which a latent unit is not
/// charged for its divergence and receives no gradient from it.
///
/// A unit whose KL is under `floor_nats` contributes the floor to the reported
/// value and nothing to either gradient, so the encoder is free to hold that
/// unit wherever reconstruction wants it instead of being pulled back to the
/// prior. This is the "free bits" of Kingma et al., and what it is for is
/// posterior collapse: units the decoder has stopped reading drift to the
/// prior, the KL term is happy, and the capacity is gone.
///
/// The unit here is one element of the latent grid, which is what
/// `examples/kl_report.rs` counts as `active`. Read that report before
/// choosing a floor: the floor only does something for units already under it,
/// so a corpus whose units all sit well above the floor will not move, and the
/// weight on the whole term is the knob that matters there instead.
///
/// ```
/// # use rusting_brain::losses::{kl_divergence, kl_divergence_free_bits};
/// let (mean, log_variance) = ([0.1, 2.0], [-0.05, 0.4]);
/// // A floor of zero is the plain divergence, so a training loop can swap one
/// // for the other.
/// assert_eq!(
///     kl_divergence_free_bits(&mean, &log_variance, 0.0).0,
///     kl_divergence(&mean, &log_variance).0
/// );
/// // The quiet unit stops pulling; the loud one is untouched.
/// let (_, grad_mean, _) = kl_divergence_free_bits(&mean, &log_variance, 0.05);
/// assert_eq!(grad_mean[0], 0.0);
/// assert_eq!(grad_mean[1], 1.0);
/// ```
pub fn kl_divergence_free_bits(
    mean: &[f32],
    log_variance: &[f32],
    floor_nats: f32,
) -> (f32, Vec<f32>, Vec<f32>) {
    assert_eq!(mean.len(), log_variance.len());
    let count = mean.len().max(1) as f32;

    let mut value = 0.0;
    let mut grad_mean = Vec::with_capacity(mean.len());
    let mut grad_log_variance = Vec::with_capacity(mean.len());
    for (mean, log_variance) in mean.iter().zip(log_variance) {
        let variance = log_variance.exp();
        let nats = 0.5 * (mean * mean + variance - 1.0 - log_variance);
        if nats > floor_nats {
            value += nats;
            grad_mean.push(mean / count);
            grad_log_variance.push(0.5 * (variance - 1.0) / count);
        } else {
            value += floor_nats;
            grad_mean.push(0.0);
            grad_log_variance.push(0.0);
        }
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
    fn a_free_bits_floor_silences_the_units_under_it_and_nothing_else() {
        // One unit at the prior, one far from it.
        let (mean, log_variance) = ([0.0, 1.5], [0.0, 0.6]);
        let (_, plain_mean, plain_log_variance) = kl_divergence(&mean, &log_variance);
        let (value, grad_mean, grad_log_variance) =
            kl_divergence_free_bits(&mean, &log_variance, 0.2);

        // The quiet unit pays the floor and pulls on nothing.
        assert_eq!(grad_mean[0], 0.0);
        assert_eq!(grad_log_variance[0], 0.0);
        // The loud unit is charged and pulls exactly as it did before.
        assert_eq!(grad_mean[1], plain_mean[1]);
        assert_eq!(grad_log_variance[1], plain_log_variance[1]);

        // The reported value is the floor plus the loud unit, over two units.
        let loud = 0.5 * (1.5 * 1.5 + 0.6f32.exp() - 1.0 - 0.6);
        assert!((value - (0.2 + loud) / 2.0).abs() < 1e-6);
    }

    #[test]
    fn a_floor_every_unit_clears_changes_nothing_but_the_reported_value() {
        // The case a corpus with no collapsed units is in: every unit is well
        // above the floor, so the floor is a no-op on the gradients.
        let (mean, log_variance) = ([1.2, -0.9, 2.0], [0.5, 0.4, -0.7]);
        let (_, plain_mean, plain_log_variance) = kl_divergence(&mean, &log_variance);
        let (_, grad_mean, grad_log_variance) = kl_divergence_free_bits(&mean, &log_variance, 0.01);
        assert_eq!(grad_mean, plain_mean);
        assert_eq!(grad_log_variance, plain_log_variance);
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

// ---------------------------------------------------------------------------
// Losses on logits with per-sample weights, for behaviour cloning and
// advantage-weighted regression. Each reduces as `sum_i(w_i * loss_i) / batch`
// (Keras' `sum_over_batch_size`), so weights of one give the plain mean, and
// returns the loss with its gradient with respect to the input.
// ---------------------------------------------------------------------------

fn sample_weight(weights: Option<&[f32]>, sample: usize) -> f32 {
    weights.map_or(1.0, |weights| weights[sample])
}

fn check_weights(weights: Option<&[f32]>, batch: usize) {
    if let Some(weights) = weights {
        assert_eq!(weights.len(), batch, "one weight a sample");
    }
}

/// Sigmoid binary cross-entropy on `logits`, `[batch, keys]`, averaged over
/// the keys of each sample. Targets are smoothed to `y (1 - s) + s / 2`.
/// Stable for any logit: the loss is `max(z, 0) - z t + ln(1 + e^-|z|)`.
///
/// ```
/// use rusting_brain::{losses::bce_with_logits, matrix::Matrix};
/// let (loss, _) = bce_with_logits(&Matrix::new(1, 2), &Matrix::from_vec(1, 2, vec![1.0; 2]), 0.0, None);
/// assert!((loss - std::f32::consts::LN_2).abs() < 1e-6);
/// ```
pub fn bce_with_logits(
    logits: &Matrix,
    targets: &Matrix,
    smoothing: f32,
    weights: Option<&[f32]>,
) -> (f32, Matrix) {
    assert_eq!((logits.rows, logits.cols), (targets.rows, targets.cols));
    let (batch, keys) = (logits.rows, logits.cols);
    check_weights(weights, batch);
    let mut grad = Matrix::new(batch, keys);
    let mut total = 0f64;
    for sample in 0..batch {
        let weight = sample_weight(weights, sample);
        let (z, y) = (logits.row(sample), targets.row(sample));
        for (key, slot) in grad.row_mut(sample).iter_mut().enumerate() {
            let (z, t) = (z[key], y[key] * (1.0 - smoothing) + 0.5 * smoothing);
            let loss = z.max(0.0) - z * t + (-z.abs()).exp().ln_1p();
            total += (weight * loss / keys as f32) as f64;
            *slot = weight * (1.0 / (1.0 + (-z).exp()) - t) / (keys * batch) as f32;
        }
    }
    ((total / batch.max(1) as f64) as f32, grad)
}

/// Cross-entropy of `softmax(logits)` against soft `targets`, each row a
/// distribution over the classes.
pub fn soft_cross_entropy(
    logits: &Matrix,
    targets: &Matrix,
    weights: Option<&[f32]>,
) -> (f32, Matrix) {
    assert_eq!((logits.rows, logits.cols), (targets.rows, targets.cols));
    let batch = logits.rows;
    check_weights(weights, batch);
    let mut grad = Matrix::new(batch, logits.cols);
    let mut total = 0f64;
    for sample in 0..batch {
        let weight = sample_weight(weights, sample);
        let (z, t) = (logits.row(sample), targets.row(sample));
        let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum = max + z.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
        let mass: f32 = t.iter().sum();
        for (class, slot) in grad.row_mut(sample).iter_mut().enumerate() {
            let log_p = z[class] - log_sum;
            total -= (weight * t[class] * log_p) as f64;
            *slot = weight * (log_p.exp() * mass - t[class]) / batch as f32;
        }
    }
    ((total / batch.max(1) as f64) as f32, grad)
}

/// Huber loss with threshold `delta`: quadratic within it, linear outside.
pub fn huber(
    prediction: &[f32],
    target: &[f32],
    delta: f32,
    weights: Option<&[f32]>,
) -> (f32, Vec<f32>) {
    assert_eq!(prediction.len(), target.len());
    let batch = prediction.len();
    check_weights(weights, batch);
    let mut total = 0f64;
    let grad = (0..batch)
        .map(|sample| {
            let weight = sample_weight(weights, sample);
            let error = prediction[sample] - target[sample];
            let loss = match error.abs() <= delta {
                true => 0.5 * error * error,
                false => delta * (error.abs() - 0.5 * delta),
            };
            total += (weight * loss) as f64;
            weight * error.clamp(-delta, delta) / batch as f32
        })
        .collect();
    ((total / batch.max(1) as f64) as f32, grad)
}

#[cfg(test)]
mod logit_tests {
    use super::*;

    /// The largest gap between `grad` and central differences of `loss`,
    /// relative to the largest gradient.
    fn gap(loss: impl Fn(&[f32]) -> f32, input: &[f32], grad: &[f32]) -> f32 {
        let mut input = input.to_vec();
        let scale = grad.iter().fold(0f32, |m, g| m.max(g.abs()));
        let mut worst = 0f32;
        for index in 0..input.len() {
            let (original, h) = (input[index], 1e-2);
            input[index] = original + h;
            let high = loss(&input);
            input[index] = original - h;
            let low = loss(&input);
            input[index] = original;
            worst = worst.max(((high - low) / (2.0 * h) - grad[index]).abs() / scale);
        }
        worst
    }

    fn values(count: usize, seed: u32, spread: f32) -> Vec<f32> {
        (0..count)
            .map(|i| {
                ((((i as u32).wrapping_mul(2_654_435_761) ^ seed) % 1000) as f32 / 500.0 - 1.0)
                    * spread
            })
            .collect()
    }

    #[test]
    fn bce_with_logits_matches_finite_differences_and_stays_finite() {
        let z = values(12, 1, 4.0);
        let y = Matrix::from_vec(2, 6, vec![1., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0., 1.]);
        for weights in [None, Some(&[0.5f32, 2.0][..])] {
            let loss = |z: &[f32]| {
                bce_with_logits(&Matrix::from_vec(2, 6, z.to_vec()), &y, 0.1, weights).0
            };
            let (_, grad) = bce_with_logits(&Matrix::from_vec(2, 6, z.clone()), &y, 0.1, weights);
            assert!(gap(loss, &z, &grad.data) < 1e-2);
        }
        let extreme = Matrix::from_vec(1, 2, vec![100.0, -100.0]);
        let (loss, grad) =
            bce_with_logits(&extreme, &Matrix::from_vec(1, 2, vec![0.0, 1.0]), 0.0, None);
        assert!((loss - 100.0).abs() < 1e-3, "{loss}");
        assert_eq!(grad.data, vec![0.5, -0.5]);
    }

    #[test]
    fn soft_cross_entropy_matches_finite_differences_and_stays_finite() {
        let z = values(30, 2, 3.0);
        let mut t = vec![0f32; 30];
        (t[3], t[4], t[5], t[15], t[16]) = (0.1, 0.8, 0.1, 0.9, 0.1);
        let t = Matrix::from_vec(2, 15, t);
        for weights in [None, Some(&[1.5f32, 0.3][..])] {
            let loss =
                |z: &[f32]| soft_cross_entropy(&Matrix::from_vec(2, 15, z.to_vec()), &t, weights).0;
            let (_, grad) = soft_cross_entropy(&Matrix::from_vec(2, 15, z.clone()), &t, weights);
            assert!(gap(loss, &z, &grad.data) < 1e-2);
        }
        let (uniform, _) = soft_cross_entropy(
            &Matrix::new(1, 15),
            &Matrix::from_vec(1, 15, t.row(0).to_vec()),
            None,
        );
        assert!((uniform - 15f32.ln()).abs() < 1e-5);
        let extreme = Matrix::from_vec(1, 2, vec![100.0, -100.0]);
        let (loss, grad) =
            soft_cross_entropy(&extreme, &Matrix::from_vec(1, 2, vec![0.0, 1.0]), None);
        assert!((loss - 200.0).abs() < 1e-3 && grad.data.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn huber_matches_finite_differences_and_its_two_pieces() {
        let p = values(8, 3, 3.0);
        let t = values(8, 4, 1.0);
        for weights in [None, Some(&values(8, 5, 1.0)[..])] {
            let (_, grad) = huber(&p, &t, 1.0, weights);
            assert!(gap(|p| huber(p, &t, 1.0, weights).0, &p, &grad) < 1e-2);
        }
        assert_eq!(
            huber(&[3.0, 0.5], &[0.0, 0.0], 1.0, None).0,
            (2.5 + 0.125) / 2.0
        );
    }
}
