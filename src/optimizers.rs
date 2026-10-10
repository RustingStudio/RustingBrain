//! Parameter update rules.
//!
//! The optimizer is a value, not a trait object, and it is re-read from the
//! model on every step. Assigning a fresh one with a new learning rate is all
//! a warmup or cosine schedule needs to do.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Optimizer {
    Sgd {
        learning_rate: f32,
    },
    Adam {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        /// Decoupled (AdamW-style) weight decay applied to weights only, not
        /// biases. `0.0` reproduces the original unregularized Adam.
        #[serde(default)]
        weight_decay: f32,
    },
    /// Lion: the update is the *sign* of a momentum-smoothed gradient, so every
    /// parameter moves by exactly the learning rate.
    ///
    /// It keeps one moment where Adam keeps two, and the sign makes the step
    /// size independent of the gradient's magnitude, so the learning rate wants
    /// to be three to ten times smaller than an Adam rate and the weight decay
    /// correspondingly larger. `Param` still allocates both moment buffers,
    /// because the optimizer is a value that can be swapped between steps, so
    /// the memory Lion saves elsewhere is not saved here.
    ///
    /// The CPU and CUDA paths run it. Metal has no kernel for it and reports
    /// that rather than quietly taking an Adam step.
    Lion {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        /// Decoupled weight decay, as on [`Optimizer::Adam`].
        weight_decay: f32,
    },
}

/// The sign of `value`, and zero for zero.
///
/// `f32::signum` gives `1.0` for a positive zero, which would push a parameter
/// whose smoothed gradient has vanished.
pub(crate) fn sign(value: f32) -> f32 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

impl Optimizer {
    pub fn sgd(learning_rate: f32) -> Self {
        Self::Sgd { learning_rate }
    }

    pub fn adam(learning_rate: f32) -> Self {
        Self::Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
    }

    pub fn adam_with_weight_decay(learning_rate: f32, weight_decay: f32) -> Self {
        Self::Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay,
        }
    }

    /// Lion at `learning_rate`, with the betas from the paper.
    ///
    /// Start from roughly a tenth of the Adam rate that works for the same
    /// model: the update is a sign, so the rate alone sets the step size.
    pub fn lion(learning_rate: f32) -> Self {
        Self::Lion {
            learning_rate,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        }
    }

    pub fn lion_with_weight_decay(learning_rate: f32, weight_decay: f32) -> Self {
        Self::Lion {
            learning_rate,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay,
        }
    }

    /// Replaces the learning rate and keeps everything else.
    ///
    /// This is what a schedule should assign. Building a fresh
    /// `Optimizer::adam(rate)` each step silently drops the betas, the epsilon
    /// and the weight decay back to their defaults.
    pub fn set_learning_rate(&mut self, rate: f32) {
        match self {
            Self::Sgd { learning_rate }
            | Self::Adam { learning_rate, .. }
            | Self::Lion { learning_rate, .. } => *learning_rate = rate,
        }
    }

    /// The current learning rate, which is worth logging next to the loss.
    pub fn learning_rate(&self) -> f32 {
        match self {
            Self::Sgd { learning_rate }
            | Self::Adam { learning_rate, .. }
            | Self::Lion { learning_rate, .. } => *learning_rate,
        }
    }
}

/// Linear warmup, then a cosine decay to a fraction of the peak.
///
/// Full-size steps before Adam's moment estimates have settled can knock a
/// model somewhere it takes a long time to leave; full-size steps at the end
/// stop it settling at all. Every transformer run wants this curve, and it is
/// arithmetic rather than state, so it is a plain value the caller reads once
/// per step:
///
/// ```
/// # use rusting_brain::{Optimizer, Schedule};
/// let schedule = Schedule::warmup_cosine(3e-4, 2_000, 100_000);
/// let mut optimizer = Optimizer::adam(3e-4);
/// for step in 0..10 {
///     optimizer.set_learning_rate(schedule.rate(step));
///     // model.optimizer = optimizer.clone();
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Schedule {
    peak: f32,
    warmup: usize,
    steps: usize,
    floor: f32,
    /// `Some(fraction)` for warmup-stable-decay; `None` for cosine.
    linear_tail: Option<f32>,
}

impl Schedule {
    /// `warmup` steps rising to `peak`, then a cosine decay over the rest of
    /// `steps`, ending at a tenth of the peak.
    pub fn warmup_cosine(peak: f32, warmup: usize, steps: usize) -> Self {
        Self {
            peak,
            warmup,
            steps,
            floor: 0.1,
            linear_tail: None,
        }
    }

    /// Warmup-stable-decay: `warmup` steps rising to `peak`, the peak held,
    /// then a linear decay to the floor over the last `decay_fraction` of
    /// `steps`.
    ///
    /// Every rate before the decay starts is the peak whatever `steps` is, so
    /// a run can be extended by raising `steps` before its decay begins.
    pub fn warmup_stable_decay(
        peak: f32,
        warmup: usize,
        steps: usize,
        decay_fraction: f32,
    ) -> Self {
        Self {
            linear_tail: Some(decay_fraction.clamp(0.0, 1.0)),
            ..Self::warmup_cosine(peak, warmup, steps)
        }
    }

    /// The fraction of `peak` the decay ends at. The default is `0.1`, and
    /// `0.0` decays to nothing.
    pub fn floor(mut self, fraction: f32) -> Self {
        self.floor = fraction;
        self
    }

    /// The rate for `step`, counted from zero.
    ///
    /// Steps past the end of the schedule hold at the floor rather than going
    /// negative, so a run extended beyond its planned length keeps training.
    pub fn rate(&self, step: usize) -> f32 {
        if step < self.warmup {
            // `step + 1`, so the first step is not a forward and backward pass
            // at a learning rate of zero, which changes nothing.
            return self.peak * (step + 1) as f32 / self.warmup as f32;
        }
        if let Some(fraction) = self.linear_tail {
            let decay = (self.steps as f32 * fraction).round() as usize;
            let start = self.steps.saturating_sub(decay).max(self.warmup);
            if step < start {
                return self.peak;
            }
            let progress = match self.steps > start {
                true => ((step - start) as f32 / (self.steps - start) as f32).min(1.0),
                false => 1.0,
            };
            return self.peak * (1.0 - (1.0 - self.floor) * progress);
        }
        let decay = self.steps.saturating_sub(self.warmup);
        if decay == 0 {
            return self.peak * self.floor;
        }
        let progress = ((step - self.warmup) as f32 / decay as f32).min(1.0);
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        self.peak * (self.floor + (1.0 - self.floor) * cosine)
    }
}

/// The training-step trio: a global gradient norm, the optimizer step, and the
/// clipped step that combines them.
///
/// Every model in the crate accumulates gradients into a flat list of
/// [`Param`]s and then applies exactly this sequence. Two copies of gradient
/// clipping that drift apart is the kind of bug that shows up as a flat loss
/// curve six hours into a run and nowhere else, so there is one copy and the
/// models delegate to it.
///
/// ```
/// # use rusting_brain::optimizers::{Optimizer, step_clipped};
/// # use rusting_brain::param::Param;
/// # use rusting_brain::matrix::Matrix;
/// let mut weight = Param::zeros(1, 2);
/// weight.grad = Matrix::from_vec(1, 2, vec![3.0, 4.0]);
/// let norm = step_clipped(&mut [&mut weight], &Optimizer::adam(1e-2), 1, 1.0, 1.0)?;
/// assert!((norm - 5.0).abs() < 1e-6);
/// # Ok::<(), rusting_brain::NetworkError>(())
/// ```
use crate::param::Param;

/// The L2 norm of the accumulated gradients, over every parameter at once.
///
/// Worth logging on its own: a run that is about to diverge shows it in this
/// number one or two steps before the loss moves.
pub fn grad_norm(params: &mut [&mut Param]) -> Result<f32, crate::network::NetworkError> {
    let mut total = 0.0;
    for param in params.iter() {
        total += param.grad_sum_squares()?;
    }
    Ok(total.sqrt() as f32)
}

/// Applies the optimizer to every parameter.
///
/// `scale` divides the accumulated gradient, so a caller that ran several
/// batches before stepping passes `1.0 / batches`. `step` counts from 1, as
/// in [`Param::step`](crate::param::Param::step), which panics on 0.
pub fn apply_step(params: &mut [&mut Param], optimizer: &Optimizer, step: usize, scale: f32) {
    if params.iter().any(|param| param.is_on_device()) {
        // One stream, one cuBLAS handle: the launches would serialize anyway,
        // and a rayon pool around them only adds contention.
        for param in params.iter_mut() {
            param.step(optimizer, step, scale);
        }
        return;
    }
    params
        .par_iter_mut()
        .for_each(|param| param.step(optimizer, step, scale));
}

/// [`apply_step`] with the gradients clipped to a global norm of `max_norm`
/// first, and the pre-clip norm returned for logging.
///
/// Clipping is a uniform rescale of every gradient, and `scale` already
/// multiplies every gradient uniformly, so this folds the clip into that factor
/// rather than rewriting the gradient buffers.
///
/// A norm that is NaN or infinite skips the step: the gradients are cleared
/// and the parameters and optimizer moments are left as they were, because
/// one non-finite update would poison both for the rest of the run. The norm
/// is still returned, so the caller sees the skip.
pub fn step_clipped(
    params: &mut [&mut Param],
    optimizer: &Optimizer,
    step: usize,
    scale: f32,
    max_norm: f32,
) -> Result<f32, crate::network::NetworkError> {
    let norm = grad_norm(params)?;
    if !norm.is_finite() {
        zero_grad(params);
        return Ok(norm);
    }
    let clip = match norm > max_norm && norm > 0.0 {
        true => max_norm / norm,
        false => 1.0,
    };
    apply_step(params, optimizer, step, scale * clip);
    Ok(norm)
}

pub fn zero_grad(params: &mut [&mut Param]) {
    if params.iter().any(|param| param.is_on_device()) {
        for param in params.iter_mut() {
            param.zero_grad();
        }
        return;
    }
    params.par_iter_mut().for_each(|param| param.zero_grad());
}

/// Flags a training step whose loss jumps well above the recent average, so
/// the caller can drop it before the optimizer sees it.
///
/// The average is an exponential moving mean of the accepted losses. A loss
/// that is not finite, or more than `ratio` times that mean once `warmup`
/// losses have been accepted, is a spike: it is not folded into the mean, and
/// [`SpikeGuard::consecutive`] counts how many came in a row. Skip a spike by
/// clearing the gradients instead of stepping; after several in a row, roll
/// back to the last checkpoint and its optimizer state.
///
/// ```
/// use rusting_brain::SpikeGuard;
///
/// let mut guard = SpikeGuard::new(2.0, 0.99, 10);
/// for loss in [3.0, 2.9, 2.8, 2.9, 2.7, 2.8, 2.6, 2.7, 2.6, 2.5] {
///     assert!(!guard.is_spike(loss));
/// }
/// assert!(guard.is_spike(9.0));
/// assert!(guard.is_spike(f32::NAN));
/// assert_eq!(guard.consecutive(), 2);
/// assert!(!guard.is_spike(2.6));
/// assert_eq!(guard.consecutive(), 0);
///
/// // In a training loop:
/// // let loss = model.accumulate_step(&batch)?;
/// // if guard.is_spike(loss.lm_loss) {
/// //     model.zero_grad();
/// //     if guard.consecutive() >= 3 {
/// //         model = TransformerLm::load_bin("last.bin")?;
/// //         model.load_optimizer_state("last.opt")?;
/// //     }
/// // } else {
/// //     model.step_clipped(1.0, 1.0)?;
/// // }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpikeGuard {
    ratio: f32,
    beta: f32,
    warmup: usize,
    mean: f32,
    accepted: usize,
    consecutive: usize,
}

impl SpikeGuard {
    /// A loss above `ratio` times the moving mean is a spike. `beta` is the
    /// mean's decay per accepted loss (0.99 averages about the last hundred),
    /// and no loss is a spike on size alone until `warmup` have been accepted.
    pub fn new(ratio: f32, beta: f32, warmup: usize) -> Self {
        Self {
            ratio,
            beta,
            warmup,
            mean: 0.0,
            accepted: 0,
            consecutive: 0,
        }
    }

    /// Records `loss` and says whether it is a spike.
    pub fn is_spike(&mut self, loss: f32) -> bool {
        let spike =
            !loss.is_finite() || (self.accepted >= self.warmup && loss > self.ratio * self.mean);
        if spike {
            self.consecutive += 1;
            return true;
        }
        self.consecutive = 0;
        self.accepted += 1;
        // Bias-corrected, so the first losses are not pulled toward zero.
        let weight = (1.0 - self.beta) / (1.0 - self.beta.powi(self.accepted as i32));
        self.mean += weight * (loss - self.mean);
        false
    }

    /// Spikes seen in a row since the last accepted loss.
    pub fn consecutive(&self) -> usize {
        self.consecutive
    }

    /// The moving mean of the accepted losses, 0 before the first.
    pub fn mean(&self) -> f32 {
        self.mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spike_is_skipped_without_moving_the_mean() {
        let mut guard = SpikeGuard::new(2.0, 0.9, 3);
        // A large first loss is accepted: the warm-up has no mean to compare.
        assert!(!guard.is_spike(10.0));
        assert!((guard.mean() - 10.0).abs() < 1e-6);
        assert!(!guard.is_spike(4.0));
        assert!(!guard.is_spike(4.0));
        let mean = guard.mean();
        assert!(guard.is_spike(2.0 * mean + 0.1));
        assert!(guard.is_spike(f32::INFINITY));
        assert_eq!(guard.mean(), mean);
        assert_eq!(guard.consecutive(), 2);
        assert!(!guard.is_spike(2.0 * mean - 0.1));
        assert_eq!(guard.consecutive(), 0);
    }

    #[test]
    fn warmup_rises_to_the_peak_and_the_cosine_falls_to_the_floor() {
        let schedule = Schedule::warmup_cosine(1.0, 10, 110);

        assert!((schedule.rate(0) - 0.1).abs() < 1e-6);
        assert!((schedule.rate(9) - 1.0).abs() < 1e-6);
        // First step of the decay is still the peak: the cosine starts at one.
        assert!((schedule.rate(10) - 1.0).abs() < 1e-6);
        assert!((schedule.rate(60) - 0.55).abs() < 1e-3); // halfway, cosine 0.5
        assert!((schedule.rate(110) - 0.1).abs() < 1e-6);
        // Past the end, not below it.
        assert!((schedule.rate(10_000) - 0.1).abs() < 1e-6);
        // Monotonic across the decay.
        for step in 10..110 {
            assert!(schedule.rate(step) >= schedule.rate(step + 1));
        }
    }

    #[test]
    fn the_floor_is_adjustable_and_a_schedule_with_no_decay_is_flat() {
        assert!(Schedule::warmup_cosine(1.0, 0, 100).floor(0.0).rate(100) < 1e-6);
        // Warmup as long as the run: nothing left to decay over.
        assert!((Schedule::warmup_cosine(1.0, 100, 100).rate(100) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn warmup_stable_decay_holds_the_peak_then_falls_linearly_to_the_floor() {
        // Warmup 10, stable to step 80, linear decay over the last 20 steps.
        let schedule = Schedule::warmup_stable_decay(1.0, 10, 100, 0.2).floor(0.0);

        assert!((schedule.rate(0) - 0.1).abs() < 1e-6);
        for step in 9..=80 {
            assert_eq!(schedule.rate(step), 1.0);
        }
        assert!((schedule.rate(90) - 0.5).abs() < 1e-6);
        assert!(schedule.rate(100) < 1e-6);
        assert!(schedule.rate(10_000) < 1e-6);
        // Extending the run before the decay leaves every rate so far alone.
        let longer = Schedule::warmup_stable_decay(1.0, 10, 200, 0.2).floor(0.0);
        for step in 0..=80 {
            assert_eq!(schedule.rate(step), longer.rate(step));
        }
        // The default floor is a tenth, as for the cosine.
        assert!((Schedule::warmup_stable_decay(1.0, 10, 100, 0.2).rate(100) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn setting_the_rate_keeps_the_rest_of_the_optimizer() {
        let mut optimizer = Optimizer::adam_with_weight_decay(1e-3, 0.1);
        optimizer.set_learning_rate(2e-4);

        assert_eq!(optimizer.learning_rate(), 2e-4);
        assert_eq!(optimizer, Optimizer::adam_with_weight_decay(2e-4, 0.1));
    }
}
