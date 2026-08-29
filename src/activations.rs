use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Activation {
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
    Linear,
}

impl Activation {
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Activation::Relu => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Softmax | Activation::Linear => x,
        }
    }

    pub fn derivative(self, activated: f32) -> f32 {
        match self {
            Activation::Relu => {
                if activated > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Sigmoid => activated * (1.0 - activated),
            Activation::Tanh => 1.0 - activated * activated,
            Activation::Softmax | Activation::Linear => 1.0,
        }
    }

    pub fn apply_to_slice(self, values: &mut [f32]) {
        if self == Activation::Softmax {
            softmax(values);
            return;
        }

        for value in values {
            *value = self.apply(*value);
        }
    }
}

pub fn softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;

    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }

    if sum > 0.0 {
        for value in values {
            *value /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_outputs_probability_distribution() {
        let mut values = vec![1.0, 2.0, 3.0];
        Activation::Softmax.apply_to_slice(&mut values);

        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(values[2] > values[1]);
        assert!(values[1] > values[0]);
    }

    #[test]
    fn sigmoid_derivative_uses_activated_value() {
        let activated = Activation::Sigmoid.apply(0.0);
        assert_eq!(activated, 0.5);
        assert!((Activation::Sigmoid.derivative(activated) - 0.25).abs() < 1e-6);
    }
}
