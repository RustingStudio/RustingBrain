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
    },
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
        }
    }
}
