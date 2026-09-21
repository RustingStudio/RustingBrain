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
    /// Set by [`Param::quantize`]. While it is present `value` keeps its shape
    /// but holds no data, and the forward pass reads the bytes instead.
    #[serde(skip)]
    pub(crate) quantized: Option<crate::quantized::Quantized>,
    /// Set by [`Param::freeze`]. A frozen parameter accumulates no gradient and
    /// ignores the optimizer step, and its gradient and moment buffers are
    /// released, which is where LoRA's memory saving comes from: three
    /// weight-sized buffers per base parameter instead of four.
    ///
    /// Not serialized. A snapshot records the weights; what is frozen follows
    /// from [`TransformerConfig::lora`](crate::transformer::TransformerConfig),
    /// which is restored with the rest of the configuration.
    #[serde(skip)]
    frozen: bool,
    /// Set by
    /// [`TransformerLm::quantization_aware`](crate::transformer::TransformerLm::quantization_aware).
    /// The forward pass rounds the weight through the int8 grid while the
    /// stored weight and its gradient stay full precision, so training sees the
    /// rounding that [`Param::quantize`] will later make permanent.
    ///
    /// Not serialized: it is a property of a run, not of the weights.
    #[serde(skip)]
    fake_quantize: bool,
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
            quantized: None,
            frozen: false,
            fake_quantize: false,
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

    /// Replaces the weights with one byte per value and drops the gradient and
    /// both moments, which are dead weight in an inference process and four
    /// fifths of the memory. Returns the bytes the parameter now occupies.
    ///
    /// One-way: the shape survives but the `f32` values do not, so anything
    /// that trains or serializes the model afterwards has to refuse.
    pub(crate) fn quantize(&mut self) -> usize {
        if self.quantized.is_some() {
            return self
                .quantized
                .as_ref()
                .map_or(0, crate::quantized::Quantized::bytes);
        }
        let quantized = crate::quantized::Quantized::from_matrix(&self.value);
        let bytes = quantized.bytes();
        self.quantized = Some(quantized);
        self.value.data = Vec::new();
        self.grad = Matrix::new(0, 0);
        self.moment1 = Matrix::new(0, 0);
        self.moment2 = Matrix::new(0, 0);
        bytes
    }

    /// Stops the parameter training and releases its gradient and Adam
    /// moments, which is three quarters of what it costs to train.
    ///
    /// The weights stay, and a forward pass is unaffected. Reversed by
    /// [`Param::unfreeze`], which allocates the buffers again, empty.
    pub fn freeze(&mut self) {
        self.frozen = true;
        self.grad = Matrix::new(0, 0);
        self.moment1 = Matrix::new(0, 0);
        self.moment2 = Matrix::new(0, 0);
    }

    /// Gives the parameter its gradient and moment buffers back, zeroed.
    pub fn unfreeze(&mut self) {
        if !self.frozen {
            return;
        }
        self.frozen = false;
        let (rows, cols) = (self.value.rows, self.value.cols);
        self.grad = Matrix::new(rows, cols);
        self.moment1 = Matrix::new(rows, cols);
        self.moment2 = Matrix::new(rows, cols);
    }

    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Rounds this weight through the int8 grid in the forward pass, without
    /// changing what is stored or what the optimizer updates.
    pub fn set_fake_quantize(&mut self, enabled: bool) {
        self.fake_quantize = enabled;
    }

    /// Whether the forward pass rounds this weight.
    pub fn is_fake_quantized(&self) -> bool {
        self.fake_quantize
    }

    pub fn zero_grad(&mut self) {
        if self.frozen {
            return;
        }
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
                self.frozen,
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

    /// The Adam moments as they stand on the host, without consulting a
    /// device.
    ///
    /// [`Param::moments`] is the accessor to reach for. This one exists for a
    /// caller that has already refreshed a whole model and now wants to borrow
    /// every parameter at once, which the `&mut self` there rules out.
    pub(crate) fn host_moments(&self) -> (&Matrix, &Matrix) {
        (&self.moment1, &self.moment2)
    }

    /// Whether a device mirror is present, in which case the device owns the
    /// weights and writing to `value` on the host would be discarded.
    pub(crate) fn is_on_device(&self) -> bool {
        #[cfg(feature = "cuda")]
        return self.device.is_some();
        #[cfg(not(feature = "cuda"))]
        return false;
    }

    /// Restores moments saved by an earlier run, uploading them when the
    /// parameter is already resident on a device.
    pub fn set_moments(
        &mut self,
        first: Matrix,
        second: Matrix,
    ) -> Result<(), crate::network::NetworkError> {
        if self.frozen {
            return Ok(());
        }
        self.moment1 = first;
        self.moment2 = second;
        #[cfg(feature = "cuda")]
        if let Some(device) = &mut self.device {
            device.upload_moments(&self.moment1, &self.moment2)?;
        }
        Ok(())
    }

    /// The sum of the squared gradient entries, for a caller building a global
    /// gradient norm across every parameter.
    ///
    /// Accumulated in `f64`: a 100M-parameter model sums that many small
    /// squares into one number, and an `f32` accumulator loses the tail of
    /// them.
    pub fn grad_sum_squares(&self) -> Result<f64, crate::network::NetworkError> {
        #[cfg(feature = "cuda")]
        if let Some(device) = &self.device {
            let norm = f64::from(device.grad_norm()?);
            return Ok(norm * norm);
        }
        Ok(self
            .grad
            .data
            .iter()
            .map(|&g| f64::from(g) * f64::from(g))
            .sum())
    }

    /// Applies one optimizer update and clears the gradient.
    ///
    /// `step` is the global Adam step count, already incremented; it is ignored
    /// by SGD. `scale` divides the accumulated gradient, which lets a caller
    /// accumulate over a batch and average once here.
    pub fn step(&mut self, optimizer: &Optimizer, step: usize, scale: f32) {
        if self.frozen {
            return;
        }
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
            Optimizer::Lion {
                learning_rate,
                beta1,
                beta2,
                weight_decay,
            } => {
                for ((value, grad), m) in self
                    .value
                    .data
                    .iter_mut()
                    .zip(&self.grad.data)
                    .zip(&mut self.moment1.data)
                {
                    let grad = grad * scale;
                    // The step is taken from the moment as it was, and the
                    // moment is then updated with the other beta. The two
                    // betas and that ordering are what separate Lion from
                    // plain signed momentum.
                    let update = beta1 * *m + (1.0 - beta1) * grad;
                    *m = beta2 * *m + (1.0 - beta2) * grad;
                    *value -=
                        learning_rate * (crate::optimizers::sign(update) + weight_decay * *value);
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
    /// Present between
    /// [`TransformerLm::add_lora`](crate::transformer::TransformerLm::add_lora)
    /// and `merge_lora`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<Lora>,
}

/// A low-rank adapter over a [`Linear`]: `W + (alpha / rank) * up * down`.
///
/// `down` is `[rank, in_features]` and `up` is `[out_features, rank]`, so the
/// pair costs `rank * (in + out)` weights where the projection it adapts costs
/// `in * out`. `up` starts at zero, so attaching an adapter does not change
/// what the model predicts; the first optimizer step is what moves it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Lora {
    pub down: Param,
    pub up: Param,
    /// `alpha / rank`. Folded into the `[tokens, rank]` intermediate, which is
    /// the smallest matrix in the adapter, so neither pass scales a full-width
    /// one.
    pub scale: f32,
}

impl Lora {
    /// `down` is initialized like any other projection and `up` is zero, which
    /// is the usual asymmetry: two random factors would start the adapter at a
    /// random perturbation of a trained model.
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        rng: &mut StdRng,
    ) -> Self {
        Self {
            down: Param::he_uniform(rank, in_features, in_features, rng),
            up: Param::zeros(out_features, rank),
            scale: alpha / rank as f32,
        }
    }

    pub fn rank(&self) -> usize {
        self.down.value.rows
    }

    /// `scale * input * down^T`, the `[tokens, rank]` bottleneck both passes
    /// need. The backward pass recomputes it rather than caching it: it is one
    /// small GEMM, and caching it would give every projection in the model a
    /// per-call buffer to thread through.
    fn hidden(&self, input: &Matrix) -> Matrix {
        let mut hidden = Matrix::new(input.rows, self.rank());
        input.dot_rhs_transposed(&self.down.value, &mut hidden);
        for value in &mut hidden.data {
            *value *= self.scale;
        }
        hidden
    }

    fn add_forward(&self, input: &Matrix, output: &mut Matrix) {
        let hidden = self.hidden(input);
        let mut delta = Matrix::new(input.rows, self.up.value.rows);
        hidden.dot_rhs_transposed(&self.up.value, &mut delta);
        for (slot, value) in output.data.iter_mut().zip(&delta.data) {
            *slot += value;
        }
    }

    /// Accumulates both adapter gradients and adds the adapter's share of
    /// `dL/dinput` to what the base projection already wrote.
    fn backward(&mut self, input: &Matrix, grad_output: &Matrix, grad_input: &mut Matrix) {
        let hidden = self.hidden(input);
        grad_output.dot_self_transposed_accumulate(&hidden, &mut self.up.grad);

        let mut grad_hidden = Matrix::new(grad_output.rows, self.rank());
        grad_output.dot(&self.up.value, &mut grad_hidden);
        for value in &mut grad_hidden.data {
            *value *= self.scale;
        }

        grad_hidden.dot_self_transposed_accumulate(input, &mut self.down.grad);
        grad_hidden.dot_accumulate(&self.down.value, grad_input);
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        vec![&mut self.down, &mut self.up]
    }
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, rng: &mut StdRng) -> Self {
        Self {
            weight: Param::he_uniform(out_features, in_features, in_features, rng),
            lora: None,
        }
    }

    /// Attaches an adapter and freezes the base weight, so the projection keeps
    /// predicting what it did and only the adapter trains.
    pub fn attach_lora(&mut self, rank: usize, alpha: f32, rng: &mut StdRng) {
        self.weight.freeze();
        self.lora = Some(Lora::new(
            self.in_features(),
            self.out_features(),
            rank,
            alpha,
            rng,
        ));
    }

    /// Folds the adapter into the base weight and unfreezes it, leaving an
    /// ordinary projection that computes the same thing.
    pub fn merge_lora(&mut self) {
        let Some(lora) = self.lora.take() else {
            return;
        };
        let mut scaled = lora.down.value.clone();
        for value in &mut scaled.data {
            *value *= lora.scale;
        }
        lora.up
            .value
            .dot_accumulate(&scaled, &mut self.weight.value);
        self.weight.unfreeze();
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
        let mut output = self.base_forward(input);
        if let Some(lora) = &self.lora {
            lora.add_forward(input, &mut output);
        }
        output
    }

    /// The projection without its adapter, whichever of the three weight
    /// representations is live.
    fn base_forward(&self, input: &Matrix) -> Matrix {
        #[cfg(feature = "cuda")]
        if let Some(device) = &self.weight.device {
            return device.matmul_rhs_transposed(input);
        }
        if let Some(quantized) = &self.weight.quantized {
            return quantized.matmul_rhs_transposed(input);
        }
        if self.weight.fake_quantize {
            // The int8 matmul inference will run, rather than a rounded copy
            // multiplied in full precision: training then sees exactly the
            // arithmetic `quantize` leaves behind.
            //
            // NOTE: the weight is rounded again on every forward pass, one
            // pass over it per batch next to a GEMM over the same weight.
            // Caching the rounding would have to be invalidated by every
            // optimizer step.
            return crate::quantized::Quantized::from_matrix(&self.weight.value)
                .matmul_rhs_transposed(input);
        }
        let mut output = Matrix::new(input.rows, self.out_features());
        input.dot_rhs_transposed(&self.weight.value, &mut output);
        output
    }

    /// Accumulates the weight gradient and returns `dL/dinput`.
    pub fn backward(&mut self, input: &Matrix, grad_output: &Matrix) -> Matrix {
        debug_assert_eq!(grad_output.cols, self.out_features());
        let mut grad_input = self.base_backward(input, grad_output);
        if let Some(lora) = &mut self.lora {
            lora.backward(input, grad_output, &mut grad_input);
        }
        grad_input
    }

    /// The base projection's share of the backward pass.
    ///
    /// NOTE: a device-resident adapter accumulates into its host gradient
    /// here rather than its device mirror. Nothing reaches this: device
    /// training runs in `gpu_model`, which never calls a `Linear` method.
    fn base_backward(&mut self, input: &Matrix, grad_output: &Matrix) -> Matrix {
        #[cfg(feature = "cuda")]
        if let Some(device) = &mut self.weight.device {
            device.accumulate_grad(grad_output, input);
            return device.matmul(grad_output);
        }
        if !self.weight.is_frozen() {
            grad_output.dot_self_transposed_accumulate(input, &mut self.weight.grad);
        }
        let mut grad_input = Matrix::new(grad_output.rows, self.in_features());
        grad_output.dot(&self.weight.value, &mut grad_input);
        grad_input
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = vec![&mut self.weight];
        if let Some(lora) = &mut self.lora {
            params.extend(lora.params_mut());
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn a_lion_step_moves_by_the_learning_rate_and_updates_the_moment_after() {
        let mut param = Param::new(Matrix::from_vec(1, 1, vec![1.0]));
        let optimizer = Optimizer::lion(0.1);

        // update = 0.9 * 0 + 0.1 * 0.5, which is positive, so the step is the
        // whole learning rate however small the gradient was.
        param.grad.data[0] = 0.5;
        param.step(&optimizer, 1, 1.0);
        assert!((param.value.data[0] - 0.9).abs() < 1e-6);
        assert!(
            (param.moment1.data[0] - 0.005).abs() < 1e-7,
            "the moment took the second beta, not the first: {}",
            param.moment1.data[0]
        );
        assert_eq!(param.moment2.data[0], 0.0, "Lion keeps one moment");

        // 0.9 * 0.005 + 0.1 * -0.5 is negative, so the sign flips with the
        // gradient rather than being dragged by the old moment.
        param.grad.data[0] = -0.5;
        param.step(&optimizer, 2, 1.0);
        assert!((param.value.data[0] - 1.0).abs() < 1e-6);
        assert!((param.moment1.data[0] - -0.00005).abs() < 1e-7);
    }

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

    /// The adapter's three gradients against central differences: the base
    /// weight is frozen, so `down`, `up` and `dL/dinput` are the whole backward
    /// pass here.
    #[test]
    fn lora_backward_matches_finite_differences() {
        let mut rng = StdRng::seed_from_u64(11);
        let mut linear = Linear::new(4, 3, &mut rng);
        linear.attach_lora(2, 4.0, &mut rng);
        // `up` starts at zero, which makes every gradient through it zero and
        // the check vacuous, so move it somewhere generic first.
        let up = &mut linear.lora.as_mut().unwrap().up.value;
        for (index, value) in up.data.iter_mut().enumerate() {
            *value = 0.1 * (index as f32) - 0.25;
        }

        let input = Matrix::from_vec(2, 4, vec![0.5, -1.0, 2.0, 0.25, 0.75, -0.5, 1.5, -0.125]);
        let grad_output = Matrix::from_vec(2, 3, vec![1.0; 6]);
        let grad_input = linear.backward(&input, &grad_output);

        let epsilon = 1e-3;
        let objective =
            |linear: &Linear, input: &Matrix| -> f32 { linear.forward(input).data.iter().sum() };

        for index in 0..input.data.len() {
            let mut bumped = input.clone();
            bumped.data[index] += epsilon;
            let high = objective(&linear, &bumped);
            bumped.data[index] -= 2.0 * epsilon;
            let low = objective(&linear, &bumped);
            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (grad_input.data[index] - numeric).abs() < 1e-2,
                "grad_input[{index}]: {} vs {numeric}",
                grad_input.data[index]
            );
        }

        for weight in [0, 1] {
            let expected: Vec<f32> = {
                let lora = linear.lora.as_ref().unwrap();
                if weight == 0 {
                    &lora.down.grad
                } else {
                    &lora.up.grad
                }
            }
            .data
            .clone();

            for (index, &analytic) in expected.iter().enumerate() {
                let mut probe = linear.clone();
                {
                    let lora = probe.lora.as_mut().unwrap();
                    let slot = if weight == 0 {
                        &mut lora.down.value
                    } else {
                        &mut lora.up.value
                    };
                    slot.data[index] += epsilon;
                }
                let high = objective(&probe, &input);
                {
                    let lora = probe.lora.as_mut().unwrap();
                    let slot = if weight == 0 {
                        &mut lora.down.value
                    } else {
                        &mut lora.up.value
                    };
                    slot.data[index] -= 2.0 * epsilon;
                }
                let low = objective(&probe, &input);
                let numeric = (high - low) / (2.0 * epsilon);
                assert!(
                    (analytic - numeric).abs() < 1e-2,
                    "weight {weight} gradient[{index}]: {analytic} vs {numeric}"
                );
            }
        }
    }

    /// Attaching an adapter must not move the model, and merging one must not
    /// move it back.
    #[test]
    fn attaching_and_merging_an_adapter_preserve_the_output() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut linear = Linear::new(4, 3, &mut rng);
        let input = Matrix::from_vec(2, 4, vec![0.5, -1.0, 2.0, 0.25, 0.75, -0.5, 1.5, -0.125]);
        let before = linear.forward(&input);

        linear.attach_lora(2, 4.0, &mut rng);
        assert!(linear.weight.is_frozen());
        assert_eq!(linear.forward(&input), before);

        // Train the adapter by hand: one SGD step on a non-zero gradient.
        let grad_output = Matrix::from_vec(2, 3, vec![0.3; 6]);
        let up = &mut linear.lora.as_mut().unwrap().up.value;
        up.data.iter_mut().for_each(|value| *value = 0.05);
        linear.backward(&input, &grad_output);
        for param in linear.params_mut() {
            param.step(&Optimizer::sgd(0.1), 1, 1.0);
        }

        let adapted = linear.forward(&input);
        assert_ne!(adapted, before);

        linear.merge_lora();
        assert!(!linear.weight.is_frozen());
        assert!(linear.lora.is_none());
        for (merged, expected) in linear.forward(&input).data.iter().zip(&adapted.data) {
            assert!((merged - expected).abs() < 1e-5, "{merged} vs {expected}");
        }
    }

    /// A frozen weight keeps its value, accumulates nothing and ignores the
    /// optimizer.
    #[test]
    fn a_frozen_weight_does_not_move() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut linear = Linear::new(3, 2, &mut rng);
        let before = linear.weight.value.clone();
        linear.weight.freeze();
        assert_eq!(linear.weight.grad.data.len(), 0);

        let input = Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        linear.backward(&input, &Matrix::from_vec(1, 2, vec![1.0, 1.0]));
        linear.weight.step(&Optimizer::sgd(1.0), 1, 1.0);
        assert_eq!(linear.weight.value, before);
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
