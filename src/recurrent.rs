//! A GRU cell with backward through time.
//!
//! The equations are PyTorch's `nn.GRU`, gates in the order reset, update,
//! new:
//!
//! ```text
//! r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
//! z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
//! n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
//! h' = (1 - z) * n + z * h
//! ```
//!
//! A sequence batch is `[batch * t, features]` with rows batch-major: row
//! `b * t + s` is sample `b` at step `s`.

use rand::rngs::StdRng;

use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::param::{CudaDevice, Linear, Param};

#[derive(Clone, Debug)]
pub struct Gru {
    /// `[3 * hidden, input]`, gates stacked r, z, n.
    pub input_weight: Linear,
    /// `[3 * hidden, hidden]`.
    pub hidden_weight: Linear,
    /// `[1, 3 * hidden]`.
    pub input_bias: Param,
    pub hidden_bias: Param,
}

/// What [`Gru::backward_seq`] needs from [`Gru::forward_seq`].
#[derive(Clone, Debug)]
pub struct GruCache {
    input: Matrix,
    batch: usize,
    steps: Vec<StepCache>,
}

/// One step, every matrix `[batch, hidden]`.
#[derive(Clone, Debug)]
struct StepCache {
    h_prev: Matrix,
    r: Matrix,
    z: Matrix,
    n: Matrix,
    /// `W_hn h + b_hn`, which the reset gate multiplies.
    hidden_n: Matrix,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Gru {
    /// PyTorch's initialization: every weight and bias uniform in
    /// `±1/sqrt(hidden)`.
    pub fn new(input: usize, hidden: usize, rng: &mut StdRng) -> Self {
        let mut uniform = |rows, cols| {
            use rand::Rng;
            let bound = 1.0 / (hidden as f32).sqrt();
            let data = (0..rows * cols)
                .map(|_| rng.gen_range(-bound..bound))
                .collect();
            Param::new(Matrix::from_vec(rows, cols, data))
        };
        Self {
            input_weight: Linear {
                weight: uniform(3 * hidden, input),
                lora: None,
            },
            hidden_weight: Linear {
                weight: uniform(3 * hidden, hidden),
                lora: None,
            },
            input_bias: uniform(1, 3 * hidden),
            hidden_bias: uniform(1, 3 * hidden),
        }
    }

    pub fn hidden(&self) -> usize {
        self.hidden_bias.value.cols / 3
    }

    /// `x` and `h0` are `[batch * t, input]` and `[batch, hidden]` (zeros when
    /// absent). Returns every step's hidden state, `[batch * t, hidden]`.
    pub fn forward_seq(&self, x: &Matrix, batch: usize, h0: Option<&Matrix>) -> (Matrix, GruCache) {
        assert!(
            batch > 0 && x.rows.is_multiple_of(batch),
            "{} rows do not split into {batch} sequences",
            x.rows
        );
        let (hidden, t) = (self.hidden(), x.rows / batch);
        let projected = self.project_input(x);
        let mut h = h0.cloned().unwrap_or_else(|| Matrix::new(batch, hidden));
        let mut output = Matrix::new(x.rows, hidden);
        let mut steps = Vec::with_capacity(t);
        for s in 0..t {
            let mut gates = Matrix::new(batch, 3 * hidden);
            for b in 0..batch {
                gates.data[b * 3 * hidden..(b + 1) * 3 * hidden].copy_from_slice(
                    &projected.data[(b * t + s) * 3 * hidden..(b * t + s + 1) * 3 * hidden],
                );
            }
            let step = self.step_cached(&gates, h);
            for b in 0..batch {
                let row = &step.next[b * hidden..(b + 1) * hidden];
                output.data[(b * t + s) * hidden..(b * t + s + 1) * hidden].copy_from_slice(row);
            }
            h = Matrix::from_vec(batch, hidden, step.next);
            steps.push(step.cache);
        }
        let cache = GruCache {
            input: x.clone(),
            batch,
            steps,
        };
        (output, cache)
    }

    /// One step for `x` `[batch, input]` from `h` `[batch, hidden]`, for
    /// running a policy a frame at a time.
    pub fn step(&self, x: &Matrix, h: &Matrix) -> Matrix {
        let next = self.step_cached(&self.project_input(x), h.clone()).next;
        Matrix::from_vec(h.rows, h.cols, next)
    }

    /// `x W_i^T + b_i`, for every row at once.
    fn project_input(&self, x: &Matrix) -> Matrix {
        let mut projected = self.input_weight.forward(x);
        for row in projected.data.chunks_mut(self.input_bias.value.cols) {
            for (value, bias) in row.iter_mut().zip(&self.input_bias.value.data) {
                *value += bias;
            }
        }
        projected
    }

    /// One step given the input projection `[batch, 3 * hidden]`.
    fn step_cached(&self, projected: &Matrix, h_prev: Matrix) -> Step {
        let hidden = self.hidden();
        let batch = h_prev.rows;
        let recurrent = self.hidden_weight.forward(&h_prev);
        let mut r = Matrix::new(batch, hidden);
        let mut z = Matrix::new(batch, hidden);
        let mut n = Matrix::new(batch, hidden);
        let mut hidden_n = Matrix::new(batch, hidden);
        let mut next = vec![0.0; batch * hidden];
        for b in 0..batch {
            let gi = &projected.data[b * 3 * hidden..(b + 1) * 3 * hidden];
            let gh = &recurrent.data[b * 3 * hidden..(b + 1) * 3 * hidden];
            let bias = &self.hidden_bias.value.data;
            for j in 0..hidden {
                let i = b * hidden + j;
                r.data[i] = sigmoid(gi[j] + gh[j] + bias[j]);
                z.data[i] = sigmoid(gi[hidden + j] + gh[hidden + j] + bias[hidden + j]);
                hidden_n.data[i] = gh[2 * hidden + j] + bias[2 * hidden + j];
                n.data[i] = (gi[2 * hidden + j] + r.data[i] * hidden_n.data[i]).tanh();
                next[i] = (1.0 - z.data[i]) * n.data[i] + z.data[i] * h_prev.data[i];
            }
        }
        Step {
            next,
            cache: StepCache {
                h_prev,
                r,
                z,
                n,
                hidden_n,
            },
        }
    }

    /// Accumulates every parameter gradient and returns `dL/dx`
    /// `[batch * t, input]` and `dL/dh0` `[batch, hidden]`. `grad_out` is the
    /// gradient of every step's output; `grad_h_last` adds to the last one,
    /// for a loss that also reads the final state through another path.
    pub fn backward_seq(
        &mut self,
        cache: &GruCache,
        grad_out: &Matrix,
        grad_h_last: Option<&Matrix>,
    ) -> (Matrix, Matrix) {
        let (hidden, batch, t) = (self.hidden(), cache.batch, cache.steps.len());
        let mut carry = grad_h_last
            .cloned()
            .unwrap_or_else(|| Matrix::new(batch, hidden));
        let mut grad_projected = Matrix::new(batch * t, 3 * hidden);
        for (s, step) in cache.steps.iter().enumerate().rev() {
            let mut grad_recurrent = Matrix::new(batch, 3 * hidden);
            for b in 0..batch {
                let row = b * t + s;
                for j in 0..hidden {
                    let i = b * hidden + j;
                    let dh = grad_out.data[row * hidden + j] + carry.data[i];
                    let (r, z, n) = (step.r.data[i], step.z.data[i], step.n.data[i]);
                    let dn = dh * (1.0 - z) * (1.0 - n * n);
                    let dz = dh * (step.h_prev.data[i] - n) * z * (1.0 - z);
                    let dr = dn * step.hidden_n.data[i] * r * (1.0 - r);
                    carry.data[i] = dh * z;
                    let gi = &mut grad_projected.data[row * 3 * hidden..(row + 1) * 3 * hidden];
                    gi[j] = dr;
                    gi[hidden + j] = dz;
                    gi[2 * hidden + j] = dn;
                    let gh = &mut grad_recurrent.data[b * 3 * hidden..(b + 1) * 3 * hidden];
                    gh[j] = dr;
                    gh[hidden + j] = dz;
                    gh[2 * hidden + j] = dn * r;
                }
            }
            accumulate_bias(&mut self.hidden_bias, &grad_recurrent);
            let grad_h_prev = self.hidden_weight.backward(&step.h_prev, &grad_recurrent);
            for (c, g) in carry.data.iter_mut().zip(&grad_h_prev.data) {
                *c += g;
            }
        }
        accumulate_bias(&mut self.input_bias, &grad_projected);
        let grad_x = self.input_weight.backward(&cache.input, &grad_projected);
        (grad_x, carry)
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = self.input_weight.params_mut();
        params.extend(self.hidden_weight.params_mut());
        params.push(&mut self.input_bias);
        params.push(&mut self.hidden_bias);
        params
    }

    /// Moves the two gate GEMMs onto `device`. The biases and the gate
    /// arithmetic stay on the host.
    ///
    /// NOTE: each step copies `h` up and the gates down, fine for a
    /// 256-wide cell over 16 steps; a fused device step when a cell is wide
    /// enough for the copies to matter.
    pub fn to_cuda_on(&mut self, device: &CudaDevice) -> Result<(), NetworkError> {
        self.input_weight.to_cuda_on(device)?;
        self.hidden_weight.to_cuda_on(device)
    }

    pub fn to_cpu(&mut self) -> Result<(), NetworkError> {
        self.input_weight.to_cpu()?;
        self.hidden_weight.to_cpu()
    }
}

struct Step {
    next: Vec<f32>,
    cache: StepCache,
}

fn accumulate_bias(bias: &mut Param, grad: &Matrix) {
    for row in grad.data.chunks(grad.cols) {
        for (g, value) in bias.grad.data.iter_mut().zip(row) {
            *g += value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    const BATCH: usize = 2;
    const STEPS: usize = 16;

    fn random(rows: usize, cols: usize, rng: &mut StdRng) -> Matrix {
        Matrix::from_vec(
            rows,
            cols,
            (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect(),
        )
    }

    struct Case {
        gru: Gru,
        x: Matrix,
        h0: Matrix,
        w_out: Matrix,
        w_last: Matrix,
    }

    impl Case {
        /// `sum(out * w_out) + sum(h_last * w_last)`, in f64.
        fn loss(&self) -> f64 {
            let (out, _) = self.gru.forward_seq(&self.x, BATCH, Some(&self.h0));
            let hidden = self.gru.hidden();
            let mut total: f64 = out
                .data
                .iter()
                .zip(&self.w_out.data)
                .map(|(a, b)| (a * b) as f64)
                .sum();
            for b in 0..BATCH {
                let row = b * STEPS + STEPS - 1;
                for j in 0..hidden {
                    total += (out.data[row * hidden + j] * self.w_last.data[b * hidden + j]) as f64;
                }
            }
            total
        }

        /// Entry `i` of `x` (`which` 0), `h0` (1) or parameter `which - 2`.
        fn slot(&mut self, which: usize, i: usize) -> &mut f32 {
            match which {
                0 => &mut self.x.data[i],
                1 => &mut self.h0.data[i],
                p => {
                    &mut self
                        .gru
                        .params_mut()
                        .into_iter()
                        .nth(p - 2)
                        .unwrap()
                        .value
                        .data[i]
                }
            }
        }

        fn central_difference(&mut self, which: usize, i: usize) -> f32 {
            let eps = 1e-2;
            let original = *self.slot(which, i);
            *self.slot(which, i) = original + eps;
            let plus = self.loss();
            *self.slot(which, i) = original - eps;
            let minus = self.loss();
            *self.slot(which, i) = original;
            ((plus - minus) / (2.0 * eps as f64)) as f32
        }
    }

    #[test]
    fn every_gradient_matches_finite_differences_over_sixteen_steps() {
        let mut rng = StdRng::seed_from_u64(7);
        let (input, hidden) = (3, 4);
        let mut case = Case {
            gru: Gru::new(input, hidden, &mut rng),
            x: random(BATCH * STEPS, input, &mut rng),
            h0: random(BATCH, hidden, &mut rng),
            w_out: random(BATCH * STEPS, hidden, &mut rng),
            w_last: random(BATCH, hidden, &mut rng),
        };
        let (_, cache) = case.gru.forward_seq(&case.x, BATCH, Some(&case.h0));
        let (w_out, w_last) = (case.w_out.clone(), case.w_last.clone());
        let (grad_x, grad_h0) = case.gru.backward_seq(&cache, &w_out, Some(&w_last));
        let mut analytic = vec![grad_x.data, grad_h0.data];
        analytic.extend(case.gru.params_mut().iter().map(|p| p.grad.data.clone()));

        let mut worst = 0.0f32;
        for (which, grads) in analytic.iter().enumerate() {
            for (i, &grad) in grads.iter().enumerate() {
                worst = worst.max((grad - case.central_difference(which, i)).abs());
            }
        }
        println!("worst GRU gradient gap {worst:.2e}");
        assert!(worst < 1e-3, "worst gap {worst}");
    }

    /// The step a policy runs frame by frame is the sequence's step.
    #[test]
    fn stepping_matches_the_sequence() {
        let mut rng = StdRng::seed_from_u64(3);
        let gru = Gru::new(5, 6, &mut rng);
        let x = random(STEPS, 5, &mut rng);
        let (out, _) = gru.forward_seq(&x, 1, None);
        let mut h = Matrix::new(1, 6);
        for s in 0..STEPS {
            h = gru.step(
                &Matrix::from_vec(1, 5, x.data[s * 5..(s + 1) * 5].to_vec()),
                &h,
            );
            // A 16-row GEMM and a 1-row one may round the last bit apart.
            for (a, b) in h.data.iter().zip(&out.data[s * 6..(s + 1) * 6]) {
                assert!((a - b).abs() < 1e-6, "{a} vs {b}");
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn device_gate_gemms_match_the_host_or_skip_without_a_device() {
        let device = match crate::cuda_training::cuda_doctor(0, 512) {
            Ok(_) => CudaDevice::new(0, 512).expect("the CUDA doctor passed"),
            Err(NetworkError::Cuda(message))
                if message.contains("NO_DEVICE") || message.contains("no CUDA-capable device") =>
            {
                crate::cuda_training::skip_without_cuda();
                return;
            }
            Err(error) => panic!("CUDA is present but the CUDA doctor failed: {error}"),
        };
        let mut rng = StdRng::seed_from_u64(11);
        let (input, hidden, batch) = (48, 256, 4);
        let mut host = Gru::new(input, hidden, &mut rng);
        let mut gpu = host.clone();
        gpu.to_cuda_on(&device).unwrap();
        let x = random(batch * STEPS, input, &mut rng);
        let grad_out = random(batch * STEPS, hidden, &mut rng);

        let (expected, host_cache) = host.forward_seq(&x, batch, None);
        let (actual, gpu_cache) = gpu.forward_seq(&x, batch, None);
        let (host_dx, host_dh0) = host.backward_seq(&host_cache, &grad_out, None);
        let (gpu_dx, gpu_dh0) = gpu.backward_seq(&gpu_cache, &grad_out, None);
        gpu.to_cpu().unwrap();

        let gap = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let mut worst = gap(&expected.data, &actual.data)
            .max(gap(&host_dx.data, &gpu_dx.data))
            .max(gap(&host_dh0.data, &gpu_dh0.data));
        for (h, g) in host.params_mut().into_iter().zip(gpu.params_mut()) {
            worst = worst.max(gap(&h.grad.data, &g.grad.data));
        }
        println!("worst device GRU gap {worst:.2e}");
        assert!(worst < 1e-3, "worst gap {worst}");
    }
}
