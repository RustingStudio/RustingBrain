//! Trainable tensors for the transformer stack.
//!
//! [`Network`](crate::network::Network) keeps its gradients in side tables and
//! stores them already negated, because its update rule adds them. The
//! transformer layers are a deeper tree of modules, so they carry their
//! gradient and optimizer state next to the weight instead and store the plain
//! derivative `dL/dw`, which the update rule subtracts.

use crate::matrix::Matrix;
use crate::optimizers::Optimizer;
use rand::{Rng, rngs::StdRng};
use serde::{Deserialize, Serialize};

/// A weight matrix plus its gradient and Adam moments.
///
/// Only the value is serialized: a snapshot restores the weights and restarts
/// the moment estimates from zero, which is what
/// [`Network::load_json`](crate::network::Network::load_json) already does for
/// dense networks.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(from = "Matrix", into = "Matrix")]
pub struct Param {
    pub value: Matrix,
    pub grad: Matrix,
    moment1: Matrix,
    moment2: Matrix,
    /// Device mirror, present between
    /// [`TransformerLm::to_cuda`](crate::transformer::TransformerLm::to_cuda)
    /// and `to_cpu`. While it is present the host `value`, `grad` and moments
    /// are stale: the device owns them.
    #[cfg(feature = "cuda")]
    pub(crate) device: Option<crate::gpu_transformer::DeviceParam>,
}

impl From<Matrix> for Param {
    fn from(value: Matrix) -> Self {
        Self::new(value)
    }
}

impl From<Param> for Matrix {
    fn from(param: Param) -> Self {
        param.value
    }
}

/// Compares weights only. Gradients and moments are transient training state,
/// and two models that predict identically are equal for every purpose a caller
/// has.
impl PartialEq for Param {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Param {
    pub fn new(value: Matrix) -> Self {
        let (rows, cols) = (value.rows, value.cols);
        Self {
            value,
            grad: Matrix::new(rows, cols),
            moment1: Matrix::new(rows, cols),
            moment2: Matrix::new(rows, cols),
            #[cfg(feature = "cuda")]
            device: None,
        }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(Matrix::new(rows, cols))
    }

    /// Every entry set to `value`. RMSNorm scales start at one.
    pub fn filled(rows: usize, cols: usize, value: f32) -> Self {
        Self::new(Matrix::from_vec(rows, cols, vec![value; rows * cols]))
    }

    /// He-style uniform initialization, matching the dense network's scheme so
    /// a transformer built with a given seed is as reproducible as a `Network`.
    pub fn he_uniform(rows: usize, cols: usize, fan_in: usize, rng: &mut StdRng) -> Self {
        let scale = (2.0 / fan_in as f32).sqrt();
        let data = (0..rows * cols)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        Self::new(Matrix::from_vec(rows, cols, data))
    }

    pub fn len(&self) -> usize {
        self.value.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.value.data.is_empty()
    }

    pub fn zero_grad(&mut self) {
        #[cfg(feature = "cuda")]
        if let Some(device) = &mut self.device {
            device.zero_grad();
            return;
        }
        self.grad.zeros();
    }

    /// Uploads the parameter and keeps it resident until [`Param::move_to_cpu`].
    #[cfg(feature = "cuda")]
    pub(crate) fn move_to_cuda(
        &mut self,
        context: &std::sync::Arc<crate::gpu_transformer::GpuContext>,
    ) -> Result<(), crate::network::NetworkError> {
        if self.device.is_none() {
            self.device = Some(crate::gpu_transformer::DeviceParam::new(
                context.clone(),
                &self.value,
            )?);
        }
        Ok(())
    }

    /// Copies the device value back and drops the mirror.
    #[cfg(feature = "cuda")]
    pub(crate) fn move_to_cpu(&mut self) -> Result<(), crate::network::NetworkError> {
        if let Some(device) = self.device.take() {
            device.download_value(&mut self.value)?;
            device.download_grad(&mut self.grad)?;
        }
        Ok(())
    }

    /// Refreshes the host value (and gradient) without giving up residency.
    #[cfg(feature = "cuda")]
    pub(crate) fn sync_from_device(&mut self) -> Result<(), crate::network::NetworkError> {
        if let Some(device) = &self.device {
            device.download_value(&mut self.value)?;
            device.download_grad(&mut self.grad)?;
        }
        Ok(())
    }

    /// The Adam moments, refreshed from the device first when the parameter is
    /// resident there.
    ///
    /// A snapshot carries weights only, so a run that stops and resumes has to
    /// carry the moments separately; see
    /// [`TransformerLm::save_optimizer_state`](crate::transformer::TransformerLm::save_optimizer_state).
    pub fn moments(&mut self) -> Result<(&Matrix, &Matrix), crate::network::NetworkError> {
        #[cfg(feature = "cuda")]
        if let Some(device) = &self.device {
            device.download_moments(&mut self.moment1, &mut self.moment2)?;
        }
        Ok((&self.moment1, &self.moment2))
    }

    /// Restores moments saved by an earlier run, uploading them when the
    /// parameter is already resident on a device.
    pub fn set_moments(
        &mut self,
        first: Matrix,
        second: Matrix,
    ) -> Result<(), crate::network::NetworkError> {
        self.moment1 = first;
        self.moment2 = second;
        #[cfg(feature = "cuda")]
        if let Some(device) = &mut self.device {
            device.upload_moments(&self.moment1, &self.moment2)?;
        }
        Ok(())
    }

    /// Applies one optimizer update and clears the gradient.
    ///
    /// `step` is the global Adam step count, already incremented; it is ignored
    /// by SGD. `scale` divides the accumulated gradient, which lets a caller
    /// accumulate over a batch and average once here.
    pub fn step(&mut self, optimizer: &Optimizer, step: usize, scale: f32) {
        #[cfg(feature = "cuda")]
        if let Some(device) = &mut self.device {
            device.step(optimizer, step, scale);
            return;
        }
        match *optimizer {
            Optimizer::Sgd { learning_rate } => {
                for (value, grad) in self.value.data.iter_mut().zip(&self.grad.data) {
                    *value -= learning_rate * grad * scale;
                }
            }
            Optimizer::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
            } => {
                let bias_correction1 = 1.0 - beta1.powi(step as i32);
                let bias_correction2 = 1.0 - beta2.powi(step as i32);

                for (((value, grad), m), v) in self
                    .value
                    .data
                    .iter_mut()
                    .zip(&self.grad.data)
                    .zip(&mut self.moment1.data)
                    .zip(&mut self.moment2.data)
                {
                    let grad = grad * scale;
                    *m = beta1 * *m + (1.0 - beta1) * grad;
                    *v = beta2 * *v + (1.0 - beta2) * grad * grad;

                    let m_hat = *m / bias_correction1;
                    let v_hat = *v / bias_correction2;
                    *value -= learning_rate * weight_decay * *value;
                    *value -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
                }
            }
        }

        self.zero_grad();
    }
}

/// A bias-free `y = x * W^T` projection.
///
/// Every projection in a Qwen-style decoder is bias-free, so the bias that
/// [`DenseLayer`](crate::network::DenseLayer) carries would only ever be zero
/// here. Rows of `x` are tokens; `weight` is `[out_features, in_features]`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Linear {
    pub weight: Param,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, rng: &mut StdRng) -> Self {
        Self {
            weight: Param::he_uniform(out_features, in_features, in_features, rng),
        }
    }

    pub fn in_features(&self) -> usize {
        self.weight.value.cols
    }

    pub fn out_features(&self) -> usize {
        self.weight.value.rows
    }

    /// `[tokens, in_features] -> [tokens, out_features]`.
    pub fn forward(&self, input: &Matrix) -> Matrix {
        debug_assert_eq!(input.cols, self.in_features());
        #[cfg(feature = "cuda")]
        if let Some(device) = &self.weight.device {
            return device.matmul_rhs_transposed(input);
        }
        let mut output = Matrix::new(input.rows, self.out_features());
        input.dot_rhs_transposed(&self.weight.value, &mut output);
        output
    }

    /// Accumulates the weight gradient and returns `dL/dinput`.
    pub fn backward(&mut self, input: &Matrix, grad_output: &Matrix) -> Matrix {
        debug_assert_eq!(grad_output.cols, self.out_features());
        #[cfg(feature = "cuda")]
        if let Some(device) = &mut self.weight.device {
            device.accumulate_grad(grad_output, input);
            return device.matmul(grad_output);
        }
        grad_output.dot_self_transposed_accumulate(input, &mut self.weight.grad);

        let mut grad_input = Matrix::new(grad_output.rows, self.in_features());
        grad_output.dot(&self.weight.value, &mut grad_input);
        grad_input
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        vec![&mut self.weight]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn linear_backward_matches_finite_differences() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut linear = Linear::new(3, 2, &mut rng);
        let input = Matrix::from_vec(2, 3, vec![0.5, -1.0, 2.0, 0.25, 0.75, -0.5]);

        // Scalar objective: the sum of the outputs, whose gradient is all ones.
        let grad_output = Matrix::from_vec(2, 2, vec![1.0; 4]);
        let grad_input = linear.backward(&input, &grad_output);

        let epsilon = 1e-3;
        for index in 0..input.data.len() {
            let mut bumped = input.clone();
            bumped.data[index] += epsilon;
            let high: f32 = linear.forward(&bumped).data.iter().sum();
            bumped.data[index] -= 2.0 * epsilon;
            let low: f32 = linear.forward(&bumped).data.iter().sum();
            let numeric = (high - low) / (2.0 * epsilon);
            assert!((grad_input.data[index] - numeric).abs() < 1e-2);
        }
    }

    #[test]
    fn param_round_trips_as_a_bare_matrix() {
        let param = Param::new(Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]));
        let json = serde_json::to_string(&param).unwrap();
        let restored: Param = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.value, param.value);
        assert!(restored.grad.data.iter().all(|&g| g == 0.0));
    }
}
