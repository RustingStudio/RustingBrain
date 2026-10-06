//! The four-signal account body and its leaky-integrator dynamics.

/// Comfort baseline each signal decays back toward:
/// `[margin_stress, drawdown_velocity, slippage_pain, exposure_fatigue]`.
pub const SETPOINT: [f32; 4] = [0.10, 0.00, 0.05, 0.00];

/// Account physiology, every field in `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccountBody {
    /// Maintenance margin over margin balance; liquidation sits at one.
    pub margin_stress: f32,
    /// Leaky average of how fast equity is falling.
    pub drawdown_velocity: f32,
    /// Recent gap between requested and filled prices.
    pub slippage_pain: f32,
    /// Accumulated time spent holding leverage.
    pub exposure_fatigue: f32,
}

impl Default for AccountBody {
    /// A body at rest: every signal on its setpoint.
    fn default() -> Self {
        Self::from_array(SETPOINT)
    }
}

impl AccountBody {
    pub fn to_array(&self) -> [f32; 4] {
        [
            self.margin_stress,
            self.drawdown_velocity,
            self.slippage_pain,
            self.exposure_fatigue,
        ]
    }

    pub fn from_array(b: [f32; 4]) -> Self {
        Self {
            margin_stress: b[0],
            drawdown_velocity: b[1],
            slippage_pain: b[2],
            exposure_fatigue: b[3],
        }
    }

    /// One tick: `clamp(b - alpha * (b - s*) + delta, 0, 1)`.
    pub fn evolve(&self, alpha: [f32; 4], delta: [f32; 4]) -> Self {
        let b = self.to_array();
        Self::from_array(std::array::from_fn(|i| {
            (b[i] - alpha[i] * (b[i] - SETPOINT[i]) + delta[i]).clamp(0.0, 1.0)
        }))
    }

    /// Weighted distance from the setpoints, `sqrt(sum w_i (b_i - s*_i)^2)`.
    pub fn drive(&self, weights: [f32; 4]) -> f32 {
        let b = self.to_array();
        (0..4)
            .map(|i| weights[i] * (b[i] - SETPOINT[i]).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_body_relaxes_to_its_setpoints_and_stays_in_range() {
        let mut body = AccountBody::from_array([1.0, 1.0, 1.0, 1.0]);
        for _ in 0..500 {
            body = body.evolve([0.1; 4], [0.0; 4]);
        }
        assert!(body.drive([1.0; 4]) < 1e-4);

        let spiked = body.evolve([0.1; 4], [5.0, 5.0, -5.0, 0.0]);
        assert_eq!(spiked.margin_stress, 1.0);
        assert_eq!(spiked.slippage_pain, 0.0);
    }
}
