//! Parameter update rules.
//!
//! The optimizer is a value, not a trait object, and it is re-read from the
//! model on every step. Assigning a fresh one with a new learning rate is all
//! a warmup or cosine schedule needs to do.

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
    /// CPU only: neither the CUDA nor the Metal path has a kernel for it, and
    /// both report that rather than quietly taking an Adam step.
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
        let decay = self.steps.saturating_sub(self.warmup);
        if decay == 0 {
            return self.peak * self.floor;
        }
        let progress = ((step - self.warmup) as f32 / decay as f32).min(1.0);
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        self.peak * (self.floor + (1.0 - self.floor) * cosine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn setting_the_rate_keeps_the_rest_of_the_optimizer() {
        let mut optimizer = Optimizer::adam_with_weight_decay(1e-3, 0.1);
        optimizer.set_learning_rate(2e-4);

        assert_eq!(optimizer.learning_rate(), 2e-4);
        assert_eq!(optimizer, Optimizer::adam_with_weight_decay(2e-4, 0.1));
    }
}
