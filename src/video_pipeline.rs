//! Wan 2.2 TI2V-5B, prompt (and optionally a first frame) in, video out.
//!
//! The same shape as [`crate::pipeline::ImagePipeline`]: `load_at`, then
//! `try_cuda`, then a run with a per-step callback that can cancel it. The
//! pieces are umT5 ([`T5Encoder`]), the Wan VAE ([`WanVaeEncoder`] for the
//! first frame, [`WanVaeDecoder`] for the video) and the DiT ([`WanDit`]),
//! sampled with diffusers' `UniPCMultistepScheduler` as Wan configures it
//! ([`UniPc`]) under classifier-free guidance.
//!
//! Text-to-video and text+image-to-video are one loop, the way diffusers'
//! `expand_timesteps` pipelines run it: with an image, latent frame 0 is the
//! encoded image at every DiT call and its tokens sit at t = 0, and the
//! sampled frame 0 is replaced by the image at the end.
//!
//! Memory on a 12 GB card. umT5 never goes to the device: its int8 copy is
//! about 6.4 GB there and does not fit beside the 7.7 GB the DiT needs at
//! 704x1280x17, and a host encode costs 0.4-1.9 s against a 50 s reload. A
//! caller short on host memory sets [`VideoPipeline::encoder`] to `None` once
//! its prompts are encoded. The DiT and the VAE decoder share the device;
//! before a decode that would not fit beside the DiT, the DiT is dropped and
//! the next run reads it again from disk.

use crate::conv::FeatureMap;
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::t5::{T5Config, T5Encoder};
use crate::tokenizer::Unigram;
use crate::transformer::Precision;
use crate::wan_dit::{WanDit, WanDitConfig};
use crate::wan_vae::{WanVaeConfig, WanVaeDecoder, WanVaeEncoder};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::path::{Path, PathBuf};

/// Pixels per latent pixel on each side.
const SPATIAL: usize = 16;
/// Video frames per latent frame after the first.
const TEMPORAL: usize = 4;

/// What a run does that a request does not choose.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoConfig {
    pub steps: usize,
    /// Classifier-free guidance scale; at 1.0 or below there is no
    /// unconditional pass.
    pub guidance: f32,
    /// Tokens every prompt is padded to.
    pub prompt_length: usize,
}

impl Default for VideoConfig {
    /// Wan 2.2's own defaults.
    fn default() -> Self {
        Self {
            steps: 50,
            guidance: 5.0,
            prompt_length: 512,
        }
    }
}

/// The size and first frame of one clip.
#[derive(Clone, Copy, Debug)]
pub struct VideoRequest<'a> {
    pub width: usize,
    pub height: usize,
    /// `1 + 4k` frames.
    pub frames: usize,
    pub seed: Option<u64>,
    /// The first frame for image-to-video: RGB in `[-1, 1]` at exactly
    /// `width` x `height`. The caller resizes.
    pub image: Option<&'a FeatureMap>,
}

/// A decoded clip: `frames` of `width * height * 3` interleaved RGB bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct Video {
    pub width: usize,
    pub height: usize,
    pub frames: Vec<Vec<u8>>,
}

/// The latent grid `[frames, height, width]` of a clip, or why there is none.
///
/// Sides must be a multiple of 32 (the VAE's 16 times the DiT's 2x2 patch),
/// and the frame count `1 + 4k`.
pub fn latent_grid(width: usize, height: usize, frames: usize) -> Result<[usize; 3], NetworkError> {
    let side = SPATIAL * 2;
    if width == 0 || height == 0 || !width.is_multiple_of(side) || !height.is_multiple_of(side) {
        return Err(NetworkError::InvalidConfig(format!(
            "a Wan clip is a multiple of {side} on each side, not {width}x{height}"
        )));
    }
    if frames == 0 || !(frames - 1).is_multiple_of(TEMPORAL) {
        return Err(NetworkError::InvalidConfig(format!(
            "a Wan clip has 1 + 4k frames, not {frames}"
        )));
    }
    Ok([
        (frames - 1) / TEMPORAL + 1,
        height / SPATIAL,
        width / SPATIAL,
    ])
}

/// The first frame must be the clip's own size: the latent grid is fixed by
/// the request, and a mismatch would condition on the wrong pixels.
fn check_image(image: &FeatureMap, width: usize, height: usize) -> Result<(), NetworkError> {
    if image.channels != 3 || image.width != width || image.height != height {
        return Err(NetworkError::InvalidConfig(format!(
            "the first frame is {}x{}x{} and the clip is 3x{height}x{width}",
            image.channels, image.height, image.width
        )));
    }
    Ok(())
}

/// diffusers' `UniPCMultistepScheduler` as Wan 2.2 configures it: flow
/// prediction converted to x0, order 2 with the bh2 B(h), the corrector on
/// every step, lower order on the last step, flow sigmas shifted by
/// `flow_shift`, and a final sigma of zero. The arithmetic is f32 in the same
/// order as diffusers'.
#[derive(Clone, Debug)]
pub struct UniPc {
    timesteps: Vec<f32>,
    sigmas: Vec<f32>,
    /// The last two x0 predictions, oldest first.
    outputs: [Option<Vec<f32>>; 2],
    last_sample: Option<Vec<f32>>,
    this_order: usize,
    lower_order_nums: usize,
    index: usize,
}

impl UniPc {
    pub fn new(steps: usize, shift: f64, train_steps: usize) -> Result<Self, NetworkError> {
        if steps == 0 || train_steps == 0 || shift.is_nan() || shift <= 0.0 {
            return Err(NetworkError::InvalidConfig(format!(
                "UniPC needs steps, training steps and a positive shift, not {steps}, \
                 {train_steps}, {shift}"
            )));
        }
        // numpy's linspace(1, 1 / train, steps + 1)[:-1], in f64.
        let train = train_steps as f64;
        let delta = (1.0 / train - 1.0) / steps as f64;
        let mut sigmas: Vec<f64> = (0..steps)
            .map(|i| i as f64 * delta + 1.0)
            .map(|s| shift * s / (1.0 + (shift - 1.0) * s))
            .collect();
        if (sigmas[0] - 1.0).abs() < 1e-6 {
            sigmas[0] -= 1e-6;
        }
        // The integer timestep the model sees: `astype(int64)` truncates.
        let timesteps = sigmas.iter().map(|s| (s * train).trunc() as f32).collect();
        let sigmas = sigmas
            .iter()
            .map(|&s| s as f32)
            .chain(std::iter::once(0.0))
            .collect();
        Ok(Self {
            timesteps,
            sigmas,
            outputs: [None, None],
            last_sample: None,
            this_order: 0,
            lower_order_nums: 0,
            index: 0,
        })
    }

    /// The DiT's timestep at each step.
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    /// One sigma per step and the final zero.
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// One step: the denoiser's velocity at `sample`, the next sample.
    pub fn step(&mut self, velocity: &[f32], sample: &[f32]) -> Result<Vec<f32>, NetworkError> {
        if velocity.len() != sample.len() {
            return Err(NetworkError::InvalidInput {
                expected: sample.len(),
                actual: velocity.len(),
            });
        }
        let sigma = self.sigma(self.index)?;
        let x0 = sample
            .iter()
            .zip(velocity)
            .map(|(x, v)| x - sigma * v)
            .collect();
        self.step_x0(x0, sample)
    }

    fn sigma(&self, index: usize) -> Result<f32, NetworkError> {
        match index < self.timesteps.len() {
            true => Ok(self.sigmas[index]),
            false => Err(NetworkError::InvalidConfig(format!(
                "UniPC has {} steps and was asked for another",
                self.timesteps.len()
            ))),
        }
    }

    /// [`Self::step`] from the x0 prediction, which is what the test that
    /// replays diffusers' BF16 conversion needs.
    pub(crate) fn step_x0(
        &mut self,
        x0: Vec<f32>,
        sample: &[f32],
    ) -> Result<Vec<f32>, NetworkError> {
        let i = self.index;
        self.sigma(i)?;
        let mut sample = sample.to_vec();
        if i > 0
            && let Some(last) = &self.last_sample
        {
            sample = self.corrector(&x0, last, &sample, self.this_order);
        }
        self.outputs[0] = self.outputs[1].take();
        self.outputs[1] = Some(x0);
        let this_order = 2.min(self.timesteps.len() - i);
        self.this_order = this_order.min(self.lower_order_nums + 1);
        let next = self.predictor(&sample, self.this_order);
        self.last_sample = Some(sample);
        self.lower_order_nums = (self.lower_order_nums + 1).min(2);
        self.index += 1;
        Ok(next)
    }

    /// UniP from sigma `i` to `i + 1`.
    fn predictor(&self, x: &[f32], order: usize) -> Vec<f32> {
        let i = self.index;
        let m0 = self.outputs[1]
            .as_deref()
            .expect("step stored the prediction");
        let (sigma_t, sigma_s0) = (self.sigmas[i + 1], self.sigmas[i]);
        let h = lambda(sigma_t) - lambda(sigma_s0);
        let (h_phi_1, b_h, _) = coefficients(h, order);
        let alpha_t = 1.0 - sigma_t;
        let ratio = sigma_t / sigma_s0;
        let (c0, cb) = (alpha_t * h_phi_1, alpha_t * b_h);
        match order {
            1 => x
                .iter()
                .zip(m0)
                .map(|(x, m0)| ratio * x - c0 * m0)
                .collect(),
            _ => {
                let m1 = self.outputs[0].as_deref().expect("order 2 has a history");
                let rk = (lambda(self.sigmas[i - 1]) - lambda(sigma_s0)) / h;
                x.iter()
                    .zip(m0)
                    .zip(m1)
                    .map(|((x, m0), m1)| {
                        let d1 = (m1 - m0) / rk;
                        (ratio * x - c0 * m0) - cb * (0.5 * d1)
                    })
                    .collect()
            }
        }
    }

    /// UniC: corrects `this_sample` (the last predictor's output, at sigma
    /// `i`) with the x0 just predicted there, `model_t`.
    fn corrector(&self, model_t: &[f32], x: &[f32], this_sample: &[f32], order: usize) -> Vec<f32> {
        let i = self.index;
        let m0 = self.outputs[1]
            .as_deref()
            .expect("a corrector follows a step");
        let (sigma_t, sigma_s0) = (self.sigmas[i], self.sigmas[i - 1]);
        let h = lambda(sigma_t) - lambda(sigma_s0);
        let (h_phi_1, b_h, b) = coefficients(h, order);
        let alpha_t = 1.0 - sigma_t;
        let ratio = sigma_t / sigma_s0;
        let (c0, cb) = (alpha_t * h_phi_1, alpha_t * b_h);
        debug_assert_eq!(this_sample.len(), x.len());
        match order {
            1 => x
                .iter()
                .zip(m0)
                .zip(model_t)
                .map(|((x, m0), mt)| (ratio * x - c0 * m0) - cb * (0.5 * (mt - m0)))
                .collect(),
            _ => {
                let m1 = self.outputs[0].as_deref().expect("order 2 has a history");
                let rk = (lambda(self.sigmas[i - 2]) - lambda(sigma_s0)) / h;
                let rhos = solve2([[1.0, 1.0], [rk, 1.0]], [b[0], b[1]]);
                x.iter()
                    .zip(m0)
                    .zip(m1)
                    .zip(model_t)
                    .map(|(((x, m0), m1), mt)| {
                        let d1 = (m1 - m0) / rk;
                        (ratio * x - c0 * m0) - cb * (rhos[0] * d1 + rhos[1] * (mt - m0))
                    })
                    .collect()
            }
        }
    }
}

/// `log(alpha) - log(sigma)` for flow sigmas; infinite at sigma zero, which
/// the order-1 last step turns into "return x0".
fn lambda(sigma: f32) -> f32 {
    (1.0 - sigma).ln() - sigma.ln()
}

/// `h_phi_1`, `B(h)` (bh2) and the `b` vector of diffusers' UniPC updates.
fn coefficients(h: f32, order: usize) -> (f32, f32, Vec<f32>) {
    let hh = -h;
    let h_phi_1 = hh.exp_m1();
    let mut h_phi_k = h_phi_1 / hh - 1.0;
    let b_h = hh.exp_m1();
    let mut factorial = 1.0f32;
    let mut b = Vec::with_capacity(order);
    for i in 1..=order {
        b.push(h_phi_k * factorial / b_h);
        factorial *= (i + 1) as f32;
        h_phi_k = h_phi_k / hh - 1.0 / factorial;
    }
    (h_phi_1, b_h, b)
}

/// A 2x2 solve with partial pivoting, as `torch.linalg.solve` does it.
fn solve2(mut a: [[f32; 2]; 2], mut b: [f32; 2]) -> [f32; 2] {
    if a[1][0].abs() > a[0][0].abs() {
        a.swap(0, 1);
        b.swap(0, 1);
    }
    let l = a[1][0] / a[0][0];
    let x1 = (b[1] - l * b[0]) / (a[1][1] - l * a[0][1]);
    [(b[0] - a[0][1] * x1) / a[0][0], x1]
}

fn flatten(maps: &[FeatureMap]) -> Vec<f32> {
    maps.iter()
        .flat_map(|map| map.data.iter().copied())
        .collect()
}

fn refill(maps: &mut [FeatureMap], data: &[f32]) {
    let mut at = 0;
    for map in maps {
        let size = map.data.len();
        map.data.copy_from_slice(&data[at..at + size]);
        at += size;
    }
}

/// Wan 2.2 TI2V-5B from a diffusers checkpoint directory.
pub struct VideoPipeline {
    pub config: VideoConfig,
    /// umT5, on the host. `None` after the caller dropped it to save memory;
    /// [`Self::condition`] then refuses.
    pub encoder: Option<T5Encoder>,
    tokenizer: Unigram,
    /// `None` until the first run or [`Self::to_cuda`], and again after a
    /// large decode needed its device memory.
    denoiser: Option<WanDit>,
    dit_path: PathBuf,
    dit_config: WanDitConfig,
    precision: Precision,
    decoder: WanVaeDecoder,
    image_encoder: WanVaeEncoder,
    flow_shift: f64,
    train_steps: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<(usize, std::sync::Arc<crate::gpu_transformer::GpuContext>)>,
}

impl VideoPipeline {
    /// [`Self::load_at`] with int8 weights: the F32 DiT alone is 20 GB.
    pub fn load(directory: impl AsRef<Path>) -> Result<Self, NetworkError> {
        Self::load_at(directory, Precision::Q8)
    }

    /// Reads umT5, the tokenizer and the VAE, and checks every config. The
    /// DiT is read by [`Self::to_cuda`] or the first run, so the caller can
    /// drop the encoder before both are in host memory at once.
    pub fn load_at(
        directory: impl AsRef<Path>,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        let dir = directory.as_ref();
        let index = read_json(&dir.join("model_index.json"))?;
        if index.get("expand_timesteps").and_then(|v| v.as_bool()) != Some(true)
            || index.get("transformer_2").is_some_and(|v| !v.is_null())
        {
            return Err(NetworkError::InvalidConfig(
                "only Wan 2.2 TI2V-5B (one transformer, expand_timesteps) is implemented".into(),
            ));
        }
        let (flow_shift, train_steps) =
            read_scheduler(&dir.join("scheduler/scheduler_config.json"))?;
        let dit_config = WanDitConfig::from_file(dir.join("transformer/config.json"))?;
        let vae = WanVaeConfig::from_file(dir.join("vae/config.json"))?;
        if vae.z_dim != dit_config.in_channels
            || dit_config.out_channels != dit_config.in_channels
            || dit_config.patch != [1, 2, 2]
        {
            return Err(NetworkError::InvalidConfig(format!(
                "a {}-channel VAE does not feed a {}->{}-channel DiT with patch {:?}",
                vae.z_dim, dit_config.in_channels, dit_config.out_channels, dit_config.patch
            )));
        }
        let t5 = T5Config::from_file(dir.join("text_encoder/config.json"))?;
        if t5.d_model != dit_config.text_dim {
            return Err(NetworkError::InvalidConfig(format!(
                "umT5 makes {} features and the DiT reads {}",
                t5.d_model, dit_config.text_dim
            )));
        }
        let encoder = T5Encoder::load_at(
            dir.join("text_encoder/model.safetensors.index.json"),
            t5,
            precision,
        )?;
        let weights = dir.join("vae/diffusion_pytorch_model.safetensors");
        Ok(Self {
            config: VideoConfig::default(),
            encoder: Some(encoder),
            tokenizer: Unigram::from_file(dir.join("tokenizer/tokenizer.json"))?,
            denoiser: None,
            dit_path: dir.join("transformer/diffusion_pytorch_model.safetensors.index.json"),
            dit_config,
            precision,
            decoder: WanVaeDecoder::load(&weights, vae.clone())?,
            image_encoder: WanVaeEncoder::load(&weights, vae)?,
            flow_shift,
            train_steps,
            #[cfg(feature = "cuda")]
            gpu: None,
        })
    }

    /// Moves the VAE decoder and the DiT onto `device`; umT5 and the VAE
    /// encoder stay on the host (see the module docs).
    ///
    /// Fails closed: whatever did not move still runs on the host.
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&mut self, device: usize) -> Result<(), NetworkError> {
        if self.gpu.is_some() {
            return Ok(());
        }
        let image = crate::cuda_image::ImageGpu::new(device)?;
        let context = crate::gpu_transformer::GpuContext::new(device)?;
        let mut dit = match self.denoiser.take() {
            Some(dit) => dit,
            None => WanDit::load_at(&self.dit_path, self.dit_config.clone(), self.precision)?,
        };
        let uploaded = dit.attach_device(&context);
        self.denoiser = Some(dit);
        uploaded?;
        self.decoder.attach_device(&image)?;
        self.gpu = Some((device, context));
        Ok(())
    }

    /// The same, reporting whether it worked rather than why it did not.
    #[cfg(feature = "cuda")]
    pub fn try_cuda(&mut self, device: usize) -> bool {
        self.to_cuda(device).is_ok()
    }

    /// Without the feature there is no device to move to.
    #[cfg(not(feature = "cuda"))]
    pub fn try_cuda(&mut self, _device: usize) -> bool {
        false
    }

    /// A prompt's umT5 states, `[prompt_length, text_dim]`, zero past its
    /// tokens.
    pub fn condition(&self, prompt: &str) -> Result<Matrix, NetworkError> {
        let encoder = self
            .encoder
            .as_ref()
            .ok_or_else(|| NetworkError::InvalidConfig("the prompt encoder was dropped".into()))?;
        encoder.encode_real(&self.tokenizer, prompt, self.config.prompt_length)
    }

    /// The first frame as the DiT reads it: one normalised
    /// `[48, height / 16, width / 16]` latent frame.
    pub fn encode_image(&self, image: &FeatureMap) -> Result<FeatureMap, NetworkError> {
        latent_grid(image.width, image.height, 1)?;
        check_image(image, image.width, image.height)?;
        let mut latent = self.image_encoder.encode(std::slice::from_ref(image))?;
        Ok(latent.remove(0))
    }

    /// Standard normal noise at the latent grid of `request`, from `seed`.
    pub fn noise(&self, request: &VideoRequest) -> Result<Vec<FeatureMap>, NetworkError> {
        let [frames, height, width] = latent_grid(request.width, request.height, request.frames)?;
        let mut rng = match request.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        let channels = self.dit_config.in_channels;
        (0..frames)
            .map(|_| {
                let data = (0..channels * height * width)
                    .map(|_| crate::diffusion::gaussian(&mut rng))
                    .collect();
                FeatureMap::from_vec(channels, height, width, data)
            })
            .collect()
    }

    fn denoiser(&mut self) -> Result<&WanDit, NetworkError> {
        if self.denoiser.is_none() {
            #[allow(unused_mut)]
            let mut dit = WanDit::load_at(&self.dit_path, self.dit_config.clone(), self.precision)?;
            #[cfg(feature = "cuda")]
            if let Some((_, context)) = &self.gpu {
                dit.attach_device(context)?;
            }
            self.denoiser = Some(dit);
        }
        Ok(self.denoiser.as_ref().expect("loaded above"))
    }

    /// Samples from `noise` (one unit-scale map per latent frame) under the
    /// prompt states `positive` and, when guidance is on, `negative`.
    /// `first` is the encoded first frame ([`Self::encode_image`]) for
    /// image-to-video. `on_step` sees each step's latent and stops the run
    /// when it returns `false`. Returns the final latent, still normalised.
    pub fn sample_from_noise(
        &mut self,
        positive: &Matrix,
        negative: Option<&Matrix>,
        first: Option<&FeatureMap>,
        mut noise: Vec<FeatureMap>,
        mut on_step: impl FnMut(usize, &[FeatureMap]) -> bool,
    ) -> Result<Vec<FeatureMap>, NetworkError> {
        let channels = self.dit_config.in_channels;
        let Some(frame) = noise.first() else {
            return Err(NetworkError::InvalidConfig(
                "the noise has no frames".into(),
            ));
        };
        let (height, width) = (frame.height, frame.width);
        if noise
            .iter()
            .any(|map| map.channels != channels || map.height != height || map.width != width)
        {
            return Err(NetworkError::InvalidConfig(format!(
                "the noise is not {channels}x{height}x{width} in every frame"
            )));
        }
        if let Some(first) = first
            && (first.channels != channels || first.height != height || first.width != width)
        {
            return Err(NetworkError::InvalidConfig(format!(
                "the first frame's latent is {}x{}x{} and the noise {channels}x{height}x{width}",
                first.channels, first.height, first.width
            )));
        }
        let guidance = self.config.guidance;
        let negative = negative.filter(|_| guidance > 1.0);
        let mut scheduler = UniPc::new(self.config.steps, self.flow_shift, self.train_steps)?;
        for step in 0..self.config.steps {
            let t = scheduler.timesteps()[step];
            let mut input = noise.clone();
            let mut times = vec![t; noise.len()];
            if let Some(first) = first {
                input[0] = first.clone();
                times[0] = 0.0;
            }
            let dit = self.denoiser()?;
            let mut velocity = flatten(&dit.forward(&input, &times, positive)?);
            if let Some(negative) = negative {
                let unconditional = flatten(&dit.forward(&input, &times, negative)?);
                for (v, u) in velocity.iter_mut().zip(&unconditional) {
                    *v = u + guidance * (*v - u);
                }
            }
            let next = scheduler.step(&velocity, &flatten(&noise))?;
            refill(&mut noise, &next);
            if !on_step(step, &noise) {
                break;
            }
        }
        if let Some(first) = first {
            noise[0] = first.clone();
        }
        Ok(noise)
    }

    /// The video of a normalised latent, RGB frames in `[-1, 1]`.
    pub fn decode(&mut self, latent: &[FeatureMap]) -> Result<Vec<FeatureMap>, NetworkError> {
        let mut latent = latent.to_vec();
        self.decoder.config.denormalize(&mut latent);
        #[cfg(feature = "cuda")]
        if let Some(frame) = latent.first() {
            self.make_room(frame.width * SPATIAL, frame.height * SPATIAL)?;
        }
        self.decoder.decode(&latent)
    }

    /// Drops the device DiT when a decode at this size would not fit beside
    /// it. The decoder's need is W2's fit, `1163 MiB + 7.4e-3 MiB` per output
    /// pixel of one frame (the causal decode holds a frame at a time), plus a
    /// 512 MiB margin.
    // NOTE: the next run reads the DiT from disk again (about 35 s); keep
    // its host int8 copy instead if large clips come back to back and the
    // 4.6 GB of host memory is there.
    #[cfg(feature = "cuda")]
    fn make_room(&mut self, width: usize, height: usize) -> Result<(), NetworkError> {
        let Some((device, _)) = &self.gpu else {
            return Ok(());
        };
        let device = *device;
        if self.decoder.device.is_none() || self.denoiser.is_none() {
            return Ok(());
        }
        let need = (1163.0 + 7.4e-3 * (width * height) as f64 + 512.0) * 1048576.0;
        crate::cuda_training::release_freed_memory(device)?;
        let free = crate::cuda_image::memory_info().map_or(0, |(free, _)| free);
        if (free as f64) < need {
            self.denoiser = None;
            crate::cuda_training::release_freed_memory(device)?;
        }
        Ok(())
    }

    /// The whole run from prompt states: noise, sampling, decode.
    pub fn generate_conditioned(
        &mut self,
        positive: &Matrix,
        negative: Option<&Matrix>,
        request: &VideoRequest,
        on_step: impl FnMut(usize, &[FeatureMap]) -> bool,
    ) -> Result<Video, NetworkError> {
        let noise = self.noise(request)?;
        let first = match request.image {
            Some(image) => {
                check_image(image, request.width, request.height)?;
                Some(self.encode_image(image)?)
            }
            None => None,
        };
        let latent = self.sample_from_noise(positive, negative, first.as_ref(), noise, on_step)?;
        let frames = self.decode(&latent)?;
        Ok(Video {
            width: request.width,
            height: request.height,
            frames: frames.iter().map(crate::vae::to_rgb8).collect(),
        })
    }

    /// The whole run: prompts in, video out. `negative` is only encoded when
    /// guidance is on.
    pub fn generate(
        &mut self,
        prompt: &str,
        negative: &str,
        request: &VideoRequest,
        on_step: impl FnMut(usize, &[FeatureMap]) -> bool,
    ) -> Result<Video, NetworkError> {
        latent_grid(request.width, request.height, request.frames)?;
        let positive = self.condition(prompt)?;
        let negative = match self.config.guidance > 1.0 {
            true => Some(self.condition(negative)?),
            false => None,
        };
        self.generate_conditioned(&positive, negative.as_ref(), request, on_step)
    }
}

fn read_json(path: &Path) -> Result<serde_json::Value, NetworkError> {
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

/// The flow shift and training steps of a scheduler config, refusing every
/// variant [`UniPc`] does not implement.
fn read_scheduler(path: &Path) -> Result<(f64, usize), NetworkError> {
    let json = read_json(path)?;
    let text = |key: &str| json.get(key).and_then(|v| v.as_str());
    let flag = |key: &str| json.get(key).and_then(|v| v.as_bool());
    let supported = text("_class_name") == Some("UniPCMultistepScheduler")
        && json.get("solver_order").and_then(|v| v.as_u64()) == Some(2)
        && text("prediction_type") == Some("flow_prediction")
        && text("solver_type") == Some("bh2")
        && text("final_sigmas_type") == Some("zero")
        && flag("predict_x0") == Some(true)
        && flag("use_flow_sigmas") == Some(true)
        && flag("lower_order_final") == Some(true)
        && flag("use_dynamic_shifting") != Some(true)
        && flag("thresholding") != Some(true)
        && json.get("solver_p").is_none_or(|v| v.is_null())
        && json.get("shift_terminal").is_none_or(|v| v.is_null())
        && json
            .get("disable_corrector")
            .and_then(|v| v.as_array())
            .is_none_or(|v| v.is_empty());
    let shift = json.get("flow_shift").and_then(|v| v.as_f64());
    let train = json.get("num_train_timesteps").and_then(|v| v.as_u64());
    match (supported, shift, train) {
        (true, Some(shift), Some(train)) => Ok((shift, train as usize)),
        _ => Err(NetworkError::InvalidConfig(format!(
            "{} is not the UniPC variant Wan 2.2 ships",
            path.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unipc_matches_diffusers_schedule() {
        // diffusers 0.42.1, Wan 2.2's scheduler config, 6 steps.
        let unipc = UniPc::new(6, 5.0, 1000).unwrap();
        assert_eq!(
            unipc.timesteps(),
            [999.0, 961.0, 909.0, 833.0, 714.0, 501.0]
        );
        // The f32 sigmas as Python prints them.
        let want = [
            0.9999989867210388f64,
            0.9615828394889832,
            0.9092148542404175,
            0.8336109519004822,
            0.7148972749710083,
            0.5014970302581787,
            0.0,
        ]
        .map(|sigma| sigma as f32);
        assert_eq!(unipc.sigmas(), want);
    }

    /// With the exact velocity of a straight path, every prediction is the
    /// same x0 and UniPC must land on the path at each sigma, then on x0.
    #[test]
    fn unipc_follows_an_exact_flow() {
        let (x0, noise) = ([0.5f32, -1.25, 2.0], [1.0f32, 0.3, -0.7]);
        let mut unipc = UniPc::new(6, 5.0, 1000).unwrap();
        let sigmas = unipc.sigmas().to_vec();
        let mut sample: Vec<f32> = (0..3)
            .map(|i| (1.0 - sigmas[0]) * x0[i] + sigmas[0] * noise[i])
            .collect();
        for step in 0..6 {
            let velocity: Vec<f32> = (0..3).map(|i| noise[i] - x0[i]).collect();
            sample = unipc.step(&velocity, &sample).unwrap();
            let s = sigmas[step + 1];
            for i in 0..3 {
                let want = (1.0 - s) * x0[i] + s * noise[i];
                assert!((sample[i] - want).abs() < 1e-5, "step {step}: {sample:?}");
            }
        }
        assert!(unipc.step(&[0.0; 3], &sample).is_err());
    }

    #[test]
    fn solve2_pivots() {
        let [x, y] = solve2([[1.0, 1.0], [-3.0, 1.0]], [2.0, -2.0]);
        assert!((x - 1.0).abs() < 1e-6 && (y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn shapes_are_checked() {
        assert_eq!(latent_grid(256, 256, 5).unwrap(), [2, 16, 16]);
        assert_eq!(latent_grid(1280, 704, 121).unwrap(), [31, 44, 80]);
        for (w, h, f) in [
            (0, 256, 5),
            (256, 48, 5),
            (240, 256, 5),
            (256, 256, 4),
            (256, 256, 0),
        ] {
            assert!(
                matches!(latent_grid(w, h, f), Err(NetworkError::InvalidConfig(_))),
                "{w}x{h}x{f}"
            );
        }
        let image = FeatureMap::new(3, 256, 224);
        assert!(check_image(&image, 224, 256).is_ok());
        assert!(check_image(&image, 256, 256).is_err());
        assert!(check_image(&FeatureMap::new(4, 256, 224), 224, 256).is_err());
    }

    #[test]
    fn scheduler_config_is_checked() {
        let dir = std::env::temp_dir().join(format!("rb-unipc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("scheduler_config.json");
        let good = r#"{"_class_name": "UniPCMultistepScheduler", "solver_order": 2,
            "prediction_type": "flow_prediction", "solver_type": "bh2",
            "final_sigmas_type": "zero", "predict_x0": true, "use_flow_sigmas": true,
            "lower_order_final": true, "use_dynamic_shifting": false, "thresholding": false,
            "solver_p": null, "disable_corrector": [], "flow_shift": 5.0,
            "num_train_timesteps": 1000}"#;
        std::fs::write(&path, good).unwrap();
        assert_eq!(read_scheduler(&path).unwrap(), (5.0, 1000));
        std::fs::write(&path, good.replace("bh2", "bh1")).unwrap();
        assert!(read_scheduler(&path).is_err());
        std::fs::write(&path, good.replace("[]", "[0]")).unwrap();
        assert!(read_scheduler(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `x` rounded to BF16, round to nearest even, as torch does.
    pub(crate) fn bf16(x: f32) -> f32 {
        let bits = x.to_bits();
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
        f32::from_bits(rounded)
    }

    /// The reference's guidance and x0 conversion, BF16 the way diffusers
    /// runs them on a BF16 transformer's output.
    pub(crate) fn bf16_x0(
        cond: &[f32],
        uncond: &[f32],
        sample: &[f32],
        sigma: f32,
        g: f32,
    ) -> Vec<f32> {
        cond.iter()
            .zip(uncond)
            .zip(sample)
            .map(|((c, u), x)| {
                let v = bf16(u + bf16(g * bf16(c - u)));
                x - bf16(sigma * v)
            })
            .collect()
    }

    /// UniPC alone against diffusers: the scheduler fed the reference's own
    /// DiT outputs must walk the reference's latents. Needs only
    /// `RB_WAN_REF` (the directory with RustingFrame's `w4.safetensors`).
    #[test]
    #[ignore = "needs the output of RustingFrame's tools/wan_diffusers_bench.py w4"]
    fn wan22_unipc_replays_diffusers() {
        use crate::wan_vae::tests::relative;
        let dir = PathBuf::from(std::env::var("RB_WAN_REF").expect("RB_WAN_REF"));
        let mut reference =
            crate::safetensors::SafeTensors::open(dir.join("w4.safetensors")).unwrap();
        let mut read = |name: &str| reference.tensor(name).unwrap().0;
        let mut worst: f32 = 0.0;
        for mode in ["t2v", "ti2v"] {
            let mut unipc = UniPc::new(6, 5.0, 1000).unwrap();
            assert_eq!(unipc.timesteps(), read(&format!("{mode}.timesteps")));
            assert_eq!(unipc.sigmas(), read(&format!("{mode}.sigmas")));
            let (mut emulated, mut plain) = (
                read(&format!("{mode}.noise")),
                read(&format!("{mode}.noise")),
            );
            let mut float = UniPc::new(6, 5.0, 1000).unwrap();
            for step in 0..6 {
                let cond = read(&format!("{mode}.step{step}.cond"));
                let uncond = read(&format!("{mode}.step{step}.uncond"));
                let want = read(&format!("{mode}.step{step}.latent"));
                let sigma = unipc.sigmas()[step];
                let x0 = bf16_x0(&cond, &uncond, &emulated, sigma, 5.0);
                emulated = unipc.step_x0(x0, &emulated).unwrap();
                let velocity: Vec<f32> = cond
                    .iter()
                    .zip(&uncond)
                    .map(|(c, u)| u + 5.0 * (c - u))
                    .collect();
                plain = float.step(&velocity, &plain).unwrap();
                let (error, f32_error) = (relative(&emulated, &want), relative(&plain, &want));
                eprintln!(
                    "unipc {mode} step {step}: BF16-emulated {error:.3e}, plain f32 {f32_error:.3e}"
                );
                worst = worst.max(error);
            }
        }
        assert!(worst < 1e-5, "UniPC replay {worst:.3e} over the 1e-5 gate");
    }

    /// Peak resident host memory so far, in GiB.
    fn peak_rss_gib() -> f64 {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|status| {
                let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
                line.split_whitespace().nth(1)?.parse::<f64>().ok()
            })
            .map_or(0.0, |kib| kib / 1048576.0)
    }

    /// Device memory in use by every process, pool reservations included, in
    /// MiB; zero without a device.
    fn device_used_mib() -> f64 {
        #[cfg(feature = "cuda")]
        if let Some((free, total)) = crate::cuda_image::memory_info() {
            return (total - free) as f64 / 1048576.0;
        }
        0.0
    }

    /// Writes `video` as `{name}.mp4` and `{name}_frameNN.png` under `dir`
    /// with ffmpeg, the way RustingFrame's reference clips were written.
    fn save_clip(video: &[FeatureMap], dir: &Path, name: &str) {
        use std::io::Write;
        let (width, height) = (video[0].width, video[0].height);
        let bytes: Vec<u8> = video.iter().flat_map(crate::vae::to_rgb8).collect();
        let outputs = [
            vec![
                "-c:v".to_string(),
                "libx264".into(),
                "-pix_fmt".into(),
                "yuv420p".into(),
                "-crf".into(),
                "16".into(),
                format!("{name}.mp4"),
            ],
            vec![format!("{name}_frame%02d.png")],
        ];
        for output in outputs {
            let mut child = std::process::Command::new("ffmpeg")
                .args([
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                ])
                .args(["-s", &format!("{width}x{height}"), "-r", "24", "-i", "-"])
                .args(output)
                .current_dir(dir)
                .stdin(std::process::Stdio::piped())
                .spawn()
                .expect("ffmpeg");
            child.stdin.take().unwrap().write_all(&bytes).unwrap();
            assert!(child.wait().unwrap().success(), "ffmpeg failed on {name}");
        }
    }

    /// Wan 2.2 TI2V-5B end to end against diffusers on the output of
    /// RustingFrame's `tools/wan_diffusers_bench.py w4` (256x256, 5 frames,
    /// 6 steps, guidance 5, BF16 DiT and VAE): umT5 states, the first frame's
    /// latent, every DiT call teacher-forced from the reference's own input,
    /// each scheduler step from the reference's previous latent, then a
    /// free-running run from the reference noise to the final latent and the
    /// decoded video, for text-to-video and image-to-video.
    ///
    /// `RB_WAN_REF` is the reference directory (the clips are written there
    /// too), `RB_WAN_MODEL` the model (default the local copy),
    /// `RB_WAN_CUDA=1` runs the DiT and the decoder on the device, and
    /// `RB_WAN_BENCH=WxHxF` times one more clip at that size. Host memory:
    /// umT5 and the VAE first (about 12 GB), then the VAE and the int8 DiT
    /// until upload (about 8 GB).
    #[test]
    #[ignore = "needs Wan2.2-TI2V-5B and the output of RustingFrame's tools/wan_diffusers_bench.py w4"]
    fn wan22_video_pipeline_matches_diffusers() {
        use crate::wan_vae::tests::{frames, relative, stack};
        use std::time::Instant;
        let env = |name: &str| std::env::var(name).ok();
        let dir = PathBuf::from(env("RB_WAN_REF").expect("RB_WAN_REF"));
        let model = PathBuf::from(
            env("RB_WAN_MODEL")
                .unwrap_or_else(|| "/home/vasylt/Code/ai/ml/models/Wan2.2-TI2V-5B".into()),
        );
        let cuda = env("RB_WAN_CUDA").as_deref() == Some("1");
        let meta = read_json(&dir.join("reference.json")).unwrap();
        let prompts = &meta["w4"]["prompts"];
        let mut reference =
            crate::safetensors::SafeTensors::open(dir.join("w4.safetensors")).unwrap();
        let mut read = |name: &str| reference.tensor(name).unwrap();
        let mut report = Vec::new();
        let mut gate = |stage: &str, error: f32, limit: f32| {
            eprintln!("wan video {stage}: relative L2 {error:.3e} (gate {limit:.0e})");
            report.push((stage.to_string(), error, limit));
        };
        let base_mib = device_used_mib();
        let mut peak_mib = base_mib;

        // umT5 on the host, then dropped before the DiT is read.
        let started = Instant::now();
        let mut pipeline = VideoPipeline::load(&model).unwrap();
        eprintln!(
            "wan video load (umT5 Q8, tokenizer, VAE): {:?}",
            started.elapsed()
        );
        pipeline.config.steps = 6;
        let embeds = |data: Vec<f32>, shape: Vec<usize>| Matrix::from_vec(shape[0], shape[1], data);
        let mut ours = Vec::new();
        for (key, name) in [
            ("t2v", "t2v.prompt_embeds"),
            ("ti2v", "ti2v.prompt_embeds"),
            ("negative", "t2v.negative_prompt_embeds"),
        ] {
            let started = Instant::now();
            let states = pipeline.condition(prompts[key].as_str().unwrap()).unwrap();
            eprintln!("wan video umT5 {key}: {:?}", started.elapsed());
            gate(
                &format!("umT5 {key}"),
                relative(&states.data, &read(name).0),
                5e-2,
            );
            ours.push(states);
        }
        let [our_negative, our_ti2v, our_t2v] =
            [ours.pop(), ours.pop(), ours.pop()].map(Option::unwrap);
        pipeline.encoder = None;

        let (image, shape) = read("ti2v.image");
        let image = FeatureMap::from_vec(shape[0], shape[1], shape[2], image).unwrap();
        let started = Instant::now();
        let our_first = pipeline.encode_image(&image).unwrap();
        eprintln!(
            "wan video first-frame encode (host): {:?}",
            started.elapsed()
        );
        let (condition, shape) = read("ti2v.condition");
        let ref_first = frames(&condition, &shape).remove(0);
        gate(
            "ti2v first-frame latent",
            relative(&our_first.data, &ref_first.data),
            5e-2,
        );

        if cuda {
            let started = Instant::now();
            assert!(
                pipeline.try_cuda(0),
                "RB_WAN_CUDA=1 needs the cuda feature and a device"
            );
            eprintln!(
                "wan video DiT load + upload, decoder upload: {:?}",
                started.elapsed()
            );
            peak_mib = peak_mib.max(device_used_mib());
        } else {
            pipeline.denoiser().unwrap();
        }
        eprintln!("wan video peak RSS after load: {:.2} GiB", peak_rss_gib());

        for mode in ["t2v", "ti2v"] {
            let (data, shape) = read(&format!("{mode}.prompt_embeds"));
            let positive = embeds(data, shape);
            let (data, shape) = read(&format!("{mode}.negative_prompt_embeds"));
            let negative = embeds(data, shape);
            let first = (mode == "ti2v").then_some(&ref_first);

            // Teacher-forced: each DiT call on the reference's own input, each
            // step from the reference's previous latent.
            let mut unipc = UniPc::new(6, 5.0, 1000).unwrap();
            let mut sample = read(&format!("{mode}.noise")).0;
            for step in 0..6 {
                let (input, shape) = read(&format!("{mode}.step{step}.dit_in"));
                let input = frames(&input, &shape);
                let per_token = read(&format!("{mode}.step{step}.dit_t")).0;
                let per_frame = per_token.len() / input.len();
                let times: Vec<f32> = per_token.iter().step_by(per_frame).copied().collect();
                let dit = pipeline.denoiser().unwrap();
                let started = Instant::now();
                let cond = stack(&dit.forward(&input, &times, &positive).unwrap());
                let uncond = stack(&dit.forward(&input, &times, &negative).unwrap());
                eprintln!("wan video {mode} DiT call: {:?}", started.elapsed() / 2);
                peak_mib = peak_mib.max(device_used_mib());
                let want_cond = read(&format!("{mode}.step{step}.cond")).0;
                let want_uncond = read(&format!("{mode}.step{step}.uncond")).0;
                gate(
                    &format!("{mode} step {step} cond"),
                    relative(&cond, &want_cond),
                    6e-2,
                );
                gate(
                    &format!("{mode} step {step} uncond"),
                    relative(&uncond, &want_uncond),
                    6e-2,
                );
                let guided: Vec<f32> = cond
                    .iter()
                    .zip(&uncond)
                    .map(|(c, u)| u + 5.0 * (c - u))
                    .collect();
                let want_guided: Vec<f32> = want_cond
                    .iter()
                    .zip(&want_uncond)
                    .map(|(c, u)| u + 5.0 * (c - u))
                    .collect();
                eprintln!(
                    "wan video {mode} step {step} guided velocity: relative L2 {:.3e}",
                    relative(&guided, &want_guided)
                );
                let next = unipc.step(&guided, &sample).unwrap();
                let want = read(&format!("{mode}.step{step}.latent")).0;
                gate(
                    &format!("{mode} step {step} latent"),
                    relative(&next, &want),
                    6e-2,
                );
                sample = want;
            }

            // Free-running from the reference noise, prompt states and first
            // frame.
            let (noise, noise_shape) = read(&format!("{mode}.noise"));
            let started = Instant::now();
            let latent = pipeline
                .sample_from_noise(
                    &positive,
                    Some(&negative),
                    first,
                    frames(&noise, &noise_shape),
                    |_, _| true,
                )
                .unwrap();
            eprintln!("wan video {mode} 6 steps: {:?}", started.elapsed());
            let (final_latent, shape) = read(&format!("{mode}.final_latent"));
            gate(
                &format!("{mode} final latent"),
                relative(&stack(&latent), &final_latent),
                1.5e-1,
            );
            let want_video = read(&format!("{mode}.video")).0;
            let started = Instant::now();
            let video = stack(&pipeline.decode(&latent).unwrap());
            eprintln!("wan video {mode} decode: {:?}", started.elapsed());
            peak_mib = peak_mib.max(device_used_mib());
            gate(
                &format!("{mode} video"),
                relative(&video, &want_video),
                1.5e-1,
            );
            let decoded = stack(&pipeline.decode(&frames(&final_latent, &shape)).unwrap());
            gate(
                &format!("{mode} decode of the reference latent"),
                relative(&decoded, &want_video),
                3e-2,
            );

            // The whole pipeline on our own prompt states and first frame,
            // from the reference noise; this is the clip that is saved.
            let (ours_positive, ours_first) = match mode {
                "t2v" => (&our_t2v, None),
                _ => (&our_ti2v, Some(&our_first)),
            };
            let latent = pipeline
                .sample_from_noise(
                    ours_positive,
                    Some(&our_negative),
                    ours_first,
                    frames(&noise, &noise_shape),
                    |_, _| true,
                )
                .unwrap();
            gate(
                &format!("{mode} final latent, our states"),
                relative(&stack(&latent), &final_latent),
                2e-1,
            );
            let clip = pipeline.decode(&latent).unwrap();
            gate(
                &format!("{mode} video, our states"),
                relative(&stack(&clip), &want_video),
                2e-1,
            );
            save_clip(&clip, &dir, &format!("rb_w4_{mode}"));

            // A whole clip from our RNG, timed: noise, first frame, sampling,
            // decode (the prompt encode is timed above).
            let request = VideoRequest {
                width: 256,
                height: 256,
                frames: 5,
                seed: Some(0),
                image: (mode == "ti2v").then_some(&image),
            };
            let started = Instant::now();
            let mut marks = Vec::new();
            let video = pipeline
                .generate_conditioned(ours_positive, Some(&our_negative), &request, |_, _| {
                    marks.push(started.elapsed());
                    true
                })
                .unwrap();
            eprintln!(
                "wan video {mode} whole 256x256x5 clip: {:?} (steps end at {marks:?})",
                started.elapsed()
            );
            assert_eq!(video.frames.len(), 5);
            assert_eq!(video.frames[0].len(), 256 * 256 * 3);
        }

        if let Some(size) = env("RB_WAN_BENCH") {
            let [width, height, count] = size
                .split('x')
                .map(|part| part.parse().unwrap())
                .collect::<Vec<usize>>()[..]
            else {
                panic!("RB_WAN_BENCH is WxHxF")
            };
            let request = VideoRequest {
                width,
                height,
                frames: count,
                seed: Some(0),
                image: None,
            };
            let started = Instant::now();
            let mut marks = Vec::new();
            let latent = pipeline
                .sample_from_noise(
                    &our_t2v,
                    Some(&our_negative),
                    None,
                    pipeline.noise(&request).unwrap(),
                    |_, _| {
                        marks.push(started.elapsed());
                        true
                    },
                )
                .unwrap();
            peak_mib = peak_mib.max(device_used_mib());
            eprintln!("wan video {size}: 6 steps end at {marks:?}");
            let started = Instant::now();
            let clip = pipeline.decode(&latent).unwrap();
            peak_mib = peak_mib.max(device_used_mib());
            eprintln!(
                "wan video {size} decode: {:?}, DiT kept: {}",
                started.elapsed(),
                pipeline.denoiser.is_some()
            );
            save_clip(&clip, &dir, &format!("rb_w4_t2v_{size}"));
        }
        eprintln!(
            "wan video peak RSS {:.2} GiB, device memory in use at most {peak_mib:.0} MiB \
             ({base_mib:.0} MiB before the run)",
            peak_rss_gib()
        );

        let failures: Vec<_> = report
            .iter()
            .filter(|(_, error, limit)| error.is_nan() || error > limit)
            .collect();
        assert!(failures.is_empty(), "over the gate: {failures:?}");
    }
}
