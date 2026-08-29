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
