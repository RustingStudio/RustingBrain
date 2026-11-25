//! Running a diffusion image model: the sampling loop and the noise schedules.
//!
//! Every image generator published since 2022 is the same loop. Start from
//! noise, ask a network what to subtract, subtract a fraction of it, repeat
//! twenty to fifty times. What differs between families is the network, not the
//! loop: FLUX and Stable Diffusion 3 predict a *velocity* along a straight path
//! from noise to image, while Stable Diffusion 1.5 and SDXL predict the *noise*
//! that was added. Both are covered here by [`Scheduler`], and a model plugs in
//! by implementing [`Denoiser`].
//!
//! The loop is deliberately ignorant of what a latent is. A transformer model
//! hands it `[patches, channels]`, a convolutional one `[channels, height *
//! width]`, and nothing here reads either dimension: every operation the
//! schedules perform is elementwise. That is what lets one loop drive model
//! families that otherwise share no code.
//!
//! What a `Denoiser` implementation must do is in that trait's documentation.

use crate::matrix::Matrix;
use crate::network::NetworkError;
use rand::{Rng, SeedableRng, rngs::StdRng};

/// A network that predicts what to remove from a noisy latent.
///
/// Implementors hold the weights, the text conditioning and whatever else they
/// need; the loop only ever asks for one prediction at a time.
pub trait Denoiser {
    /// The prediction for `latents` at noise level `sigma`.
    ///
    /// `sigma` runs from 1.0 (pure noise) down to 0.0 (a clean image) for a
    /// flow-matching model, and is the same quantity the schedule's
    /// [`Scheduler::sigmas`] produced. Return whatever the model was trained to
    /// predict: [`Scheduler::FlowMatch`] expects a velocity, [`Scheduler::Ddim`]
    /// expects noise. The returned matrix has the shape of `latents`.
    fn denoise(&mut self, latents: &Matrix, sigma: f32) -> Result<Matrix, NetworkError>;

    /// The same prediction with the conditioning dropped, for
    /// classifier-free guidance.
    ///
    /// The default is no guidance at all: a distilled model — FLUX schnell,
    /// FLUX.2 klein, SDXL-Turbo — has the guidance baked into its weights and
    /// running it twice per step would halve the speed for nothing. Implement
    /// it only for a model that was trained with a dropped condition.
    fn denoise_unconditional(
        &mut self,
        _latents: &Matrix,
        _sigma: f32,
    ) -> Result<Option<Matrix>, NetworkError> {
        Ok(None)
    }
}

/// How the noise level falls from step to step, and what a prediction means.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scheduler {
    /// Rectified flow, as FLUX, FLUX.2 and Stable Diffusion 3 use it.
    ///
    /// The path from noise to image is a straight line, the model predicts the
    /// direction along it, and a step is `x += (next - current) * velocity`.
    /// `shift` bends the spacing of the noise levels towards the noisy end,
    /// where the large-scale structure of the image is decided: 1.0 is an even
    /// spacing, and published models ship values between 1.0 and 3.0, rising
    /// with resolution.
    ///
    /// A `terminal` above zero stretches the levels so the last one before
    /// the clean end lands on it, as Qwen-Image's `shift_terminal` asks.
    FlowMatch { shift: f32, terminal: f32 },
    /// DDIM over the scaled-linear beta schedule of the Stable Diffusion 1.x
    /// and XL line, where the model predicts the noise.
    Ddim {
        beta_start: f32,
        beta_end: f32,
        train_steps: usize,
    },
}

impl Scheduler {
    /// Flow matching at the shift a model's own configuration names.
    pub fn flow_match(shift: f32) -> Self {
        Self::FlowMatch {
            shift,
            terminal: 0.0,
        }
    }

    /// The betas Stable Diffusion 1.x, 2.x and XL were all trained with.
    pub fn ddim() -> Self {
        Self::Ddim {
            beta_start: 0.00085,
            beta_end: 0.012,
            train_steps: 1000,
        }
    }

    /// The `steps + 1` noise levels a run passes through, from noisiest to
    /// zero.
    ///
    /// The extra entry is the clean end: a step reads the level it is at and
    /// the one it is going to, so `steps` steps need one more level than that.
    pub fn sigmas(&self, steps: usize) -> Result<Vec<f32>, NetworkError> {
        if steps == 0 {
            return Err(NetworkError::InvalidConfig(
                "a sampling run of no steps produces noise".into(),
            ));
        }

        Ok(match *self {
            Self::FlowMatch { .. } => {
                let levels: Vec<f32> = (0..steps)
                    .map(|step| 1.0 - step as f32 / steps as f32)
                    .collect();
                return self.sigmas_from(&levels);
            }
            Self::Ddim {
                beta_start,
                beta_end,
                train_steps,
            } => {
                let alphas = alphas_cumprod(beta_start, beta_end, train_steps);
                // `sigma = sqrt((1 - alpha) / alpha)`, the Karras-style
                // continuous level that the same Euler update works on.
                let mut sigmas: Vec<f32> = (0..steps)
                    .map(|step| {
                        let index = train_steps - 1 - step * train_steps / steps;
                        let alpha = alphas[index];
                        ((1.0 - alpha) / alpha).sqrt()
                    })
                    .collect();
                sigmas.push(0.0);
                sigmas
            }
        })
    }

    /// [`Scheduler::sigmas`] over levels the caller names, noisiest first and
    /// before the shift, as a distilled checkpoint's `sample_sigmas` are. The
    /// clean end is added. Only flow matching takes them.
    pub fn sigmas_from(&self, levels: &[f32]) -> Result<Vec<f32>, NetworkError> {
        let Self::FlowMatch { shift, terminal } = *self else {
            return Err(NetworkError::InvalidConfig(
                "only a flow-matching schedule takes its levels from the checkpoint".into(),
            ));
        };
        // The shift pushes levels towards one, so more of the run is spent
        // where the image's structure is still being set.
        let mut sigmas: Vec<f32> = levels
            .iter()
            .map(|level| shift * level / (1.0 + (shift - 1.0) * level))
            .collect();
        if terminal > 0.0
            && let Some(&last) = sigmas.last()
        {
            let scale = (1.0 - last) / (1.0 - terminal);
            for sigma in &mut sigmas {
                *sigma = 1.0 - (1.0 - *sigma) / scale;
            }
        }
        sigmas.push(0.0);
        Ok(sigmas)
    }

    /// The noise range of [`Scheduler::sigmas`] for a DDIM schedule, spaced
    /// the way Karras et al. (2022) space it: even steps in `sigma^(1/7)`, so
    /// more of the run is spent at low noise, where detail is decided. This
    /// is the "Karras" in "DPM++ 2M Karras". Run it with
    /// [`sample_with_sigmas`].
    ///
    /// A flow-matching schedule has no such spacing; its `shift` does that
    /// job, and it is refused here.
    pub fn karras_sigmas(&self, steps: usize) -> Result<Vec<f32>, NetworkError> {
        let Self::Ddim {
            beta_start,
            beta_end,
            train_steps,
        } = *self
        else {
            return Err(NetworkError::InvalidConfig(
                "Karras spacing is for a DDIM schedule; flow matching spaces its levels with `shift`"
                    .into(),
            ));
        };
        self.sigmas(steps)?;
        const RHO: f32 = 7.0;
        let alphas = alphas_cumprod(beta_start, beta_end, train_steps);
        let root = |alpha: f32| ((1.0 - alpha) / alpha).sqrt().powf(1.0 / RHO);
        let (high, low) = (root(alphas[train_steps - 1]), root(alphas[0]));
        let mut sigmas: Vec<f32> = (0..steps)
            .map(|step| {
                let fraction = step as f32 / (steps - 1).max(1) as f32;
                (high + fraction * (low - high)).powf(RHO)
            })
            .collect();
        sigmas.push(0.0);
        Ok(sigmas)
    }

    /// The noise levels a latent consistency model (LCM) was distilled at,
    /// for [`sample_consistency`].
    ///
    /// An LCM learns its jump to the image only at the 50 timesteps its
    /// teacher ran, every twentieth of a 1000-step schedule; this picks `steps`
    /// of them evenly from the noisy end, as diffusers' `LCMScheduler` does:
    /// four steps are timesteps 999, 759, 499 and 259. More than 50 steps is
    /// refused, and so is a flow-matching schedule.
    pub fn lcm_sigmas(&self, steps: usize) -> Result<Vec<f32>, NetworkError> {
        const ORIGINAL_STEPS: usize = 50;
        let Self::Ddim {
            beta_start,
            beta_end,
            train_steps,
        } = *self
        else {
            return Err(NetworkError::InvalidConfig(
                "LCM timesteps are for a DDIM schedule".into(),
            ));
        };
        self.sigmas(steps)?;
        let stride = train_steps / ORIGINAL_STEPS;
        if steps > ORIGINAL_STEPS || stride == 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "an LCM runs at most {ORIGINAL_STEPS} steps over a schedule of at least {ORIGINAL_STEPS} timesteps"
            )));
        }
        let alphas = alphas_cumprod(beta_start, beta_end, train_steps);
        let mut sigmas: Vec<f32> = (0..steps)
            .map(|step| {
                // `floor(linspace(0, 50, steps))` into the teacher's timesteps,
                // noisiest first.
                let index = step * ORIGINAL_STEPS / steps;
                let alpha = alphas[(ORIGINAL_STEPS - index) * stride - 1];
                ((1.0 - alpha) / alpha).sqrt()
            })
            .collect();
        sigmas.push(0.0);
        Ok(sigmas)
    }

    /// One update: `latents` moves from `sigma` to `next_sigma` given the
    /// model's `prediction`.
    pub fn step(
        &self,
        latents: &mut Matrix,
        prediction: &Matrix,
        sigma: f32,
        next_sigma: f32,
    ) -> Result<(), NetworkError> {
        if latents.data.len() != prediction.data.len() {
            return Err(NetworkError::InvalidTarget {
                expected: latents.data.len(),
                actual: prediction.data.len(),
            });
        }

        match self {
            Self::FlowMatch { .. } => {
                // The prediction is the direction from noise to image, so the
                // whole step is a move along it.
                let delta = next_sigma - sigma;
                for (latent, velocity) in latents.data.iter_mut().zip(&prediction.data) {
                    *latent += delta * velocity;
                }
            }
            Self::Ddim { .. } => {
                // The prediction is the noise: remove it to see the image the
                // model believes is underneath, then add back as much as the
                // next level calls for. That is Euler on `d = (x - x0) / sigma`.
                for (latent, noise) in latents.data.iter_mut().zip(&prediction.data) {
                    let clean = *latent - sigma * noise;
                    *latent = clean + next_sigma * noise;
                }
            }
        }
        Ok(())
    }

    /// Puts `latents` at the noise level `sigma`, which is how image-to-image
    /// starts: an encoded picture is noised part of the way back up the
    /// schedule, and the run then denoises it from there.
    ///
    /// The two schedules disagree about what a level means. A flow-matching
    /// level is the position along the straight line between noise and image,
    /// so it mixes; a DDIM level is how much noise sits on top of a clean
    /// image, so it adds.
    pub fn add_noise(
        &self,
        latents: &mut Matrix,
        noise: &Matrix,
        sigma: f32,
    ) -> Result<(), NetworkError> {
        if latents.data.len() != noise.data.len() {
            return Err(NetworkError::InvalidTarget {
                expected: latents.data.len(),
                actual: noise.data.len(),
            });
        }
        for (latent, noise) in latents.data.iter_mut().zip(&noise.data) {
            *latent = match self {
                Self::FlowMatch { .. } => (1.0 - sigma) * *latent + sigma * noise,
                Self::Ddim { .. } => *latent + sigma * noise,
            };
        }
        Ok(())
    }
}

/// How one step of the loop is taken.
///
/// The schedule says where the noise levels are; the solver says how to move
/// between two of them. All three read the model once per step, so the choice
/// costs nothing but arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Solver {
    /// One Euler step per level, which is the schedule's own rule. What the
    /// distilled models are tuned for.
    #[default]
    Euler,
    /// Euler with part of each step's noise put back, which is what
    /// "ancestral" means. It gives more variation between seeds and needs more
    /// steps to settle.
    EulerAncestral,
    /// DPM++ 2M: a second-order multistep solver that reuses the previous
    /// step's prediction rather than reading the model twice, so a step costs
    /// what an Euler step costs. It reaches a usable image in fewer steps,
    /// which is what a guided model is usually run with.
    DpmPlusPlus2m,
}

/// `alpha_bar` for the scaled-linear beta schedule: the betas are spaced
/// linearly in their square roots, which is what Stable Diffusion trained on.
pub(crate) fn alphas_cumprod(beta_start: f32, beta_end: f32, steps: usize) -> Vec<f32> {
    let (start, end) = (beta_start.sqrt(), beta_end.sqrt());
    let mut product = 1.0;
    (0..steps)
        .map(|step| {
            let fraction = step as f32 / (steps - 1).max(1) as f32;
            let beta = (start + fraction * (end - start)).powi(2);
            product *= 1.0 - beta;
            product
        })
        .collect()
}

/// How a run is set up: how many steps, how strongly to follow the prompt, and
/// where the starting noise comes from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingConfig {
    pub steps: usize,
    /// How one step is taken. Euler unless a model asks for more.
    pub solver: Solver,
    /// Classifier-free guidance scale. 1.0 is off, and it stays off unless the
    /// model implements [`Denoiser::denoise_unconditional`]. A distilled model
    /// wants it off; an SDXL-style model wants 5.0 to 8.0.
    pub guidance: f32,
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            steps: 28,
            solver: Solver::default(),
            guidance: 1.0,
            seed: None,
        }
    }
}

/// Standard normal noise of the given shape, which is where a run starts.
pub fn noise(rows: usize, cols: usize, seed: Option<u64>) -> Matrix {
    let mut rng = match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };
    let data = (0..rows * cols).map(|_| gaussian(&mut rng)).collect();
    Matrix::from_vec(rows, cols, data)
}

/// One standard normal sample.
///
/// Box-Muller, because `rand` without the distributions feature has no normal
/// of its own and this is cheaper than the feature.
pub fn gaussian<R: Rng>(rng: &mut R) -> f32 {
    let first: f32 = rng.gen_range(f32::MIN_POSITIVE..1.0);
    let second: f32 = rng.gen_range(0.0..1.0);
    (-2.0 * first.ln()).sqrt() * (std::f32::consts::TAU * second).cos()
}

/// Runs the whole loop and returns the latents it ends on.
///
/// `latents` is the starting noise, which [`noise`] produces at the shape the
/// model wants. The result is still a latent: turning it into pixels is the
/// decoder's job, and that is model-specific.
///
/// ```
/// # use rusting_brain::{Denoiser, Matrix, NetworkError, SamplingConfig, Scheduler, sample};
/// // A model that always points straight at a solid grey image.
/// struct Grey;
/// impl Denoiser for Grey {
///     fn denoise(&mut self, latents: &Matrix, sigma: f32) -> Result<Matrix, NetworkError> {
///         let velocity = latents.data.iter().map(|value| (value - 0.5) / sigma.max(1e-3));
///         Ok(Matrix::from_vec(latents.rows, latents.cols, velocity.collect()))
///     }
/// }
///
/// let start = Matrix::from_vec(1, 4, vec![2.0, -1.0, 0.0, 3.0]);
/// let config = SamplingConfig { steps: 50, ..SamplingConfig::default() };
/// let image = sample(&mut Grey, Scheduler::flow_match(1.0), start, &config, |_, _| true)?;
/// assert!(image.data.iter().all(|value| (value - 0.5).abs() < 0.05));
/// # Ok::<(), rusting_brain::NetworkError>(())
/// ```
pub fn sample<D: Denoiser>(
    denoiser: &mut D,
    scheduler: Scheduler,
    latents: Matrix,
    config: &SamplingConfig,
    on_step: impl FnMut(usize, &Matrix) -> bool,
) -> Result<Matrix, NetworkError> {
    sample_from(denoiser, scheduler, latents, config, 0, on_step)
}

/// The same loop, started partway down the schedule.
///
/// `first` is the step to begin at, counted from the noisiest end. Zero is a
/// whole run from pure noise, which is what [`sample`] does; a later step is
/// what image-to-image uses, having already put `latents` at that step's noise
/// level with [`Scheduler::add_noise`]. The callback still sees the absolute
/// step number, so a progress bar reads the same either way.
pub fn sample_from<D: Denoiser>(
    denoiser: &mut D,
    scheduler: Scheduler,
    latents: Matrix,
    config: &SamplingConfig,
    first: usize,
    on_step: impl FnMut(usize, &Matrix) -> bool,
) -> Result<Matrix, NetworkError> {
    let sigmas = scheduler.sigmas(config.steps)?;
    sample_with_sigmas(
        denoiser, scheduler, latents, &sigmas, config, first, on_step,
    )
}

/// [`sample_from`] over noise levels the caller chose, such as
/// [`Scheduler::karras_sigmas`].
///
/// `sigmas` runs from noisiest to zero, one more entry than there are steps,
/// and sets the step count: `config.steps` is not read.
pub fn sample_with_sigmas<D: Denoiser>(
    denoiser: &mut D,
    scheduler: Scheduler,
    mut latents: Matrix,
    sigmas: &[f32],
    config: &SamplingConfig,
    first: usize,
    mut on_step: impl FnMut(usize, &Matrix) -> bool,
) -> Result<Matrix, NetworkError> {
    let Some(steps) = sigmas.len().checked_sub(1).filter(|&steps| steps > 0) else {
        return Err(NetworkError::InvalidConfig(
            "a sampling run needs at least two noise levels".into(),
        ));
    };
    // DPM++ is a multistep solver: it reuses the image the last step believed
    // was underneath, and the step size it was taken over.
    let mut previous: Option<(Vec<f32>, f32)> = None;

    for step in first.min(steps)..steps {
        let (sigma, next) = (sigmas[step], sigmas[step + 1]);
        let prediction = predict(denoiser, &latents, sigma, config.guidance)?;

        match config.solver {
            Solver::Euler => scheduler.step(&mut latents, &prediction, sigma, next)?,
            Solver::EulerAncestral => {
                // How much of the next level to reach by stepping, and how
                // much to reach by putting noise back.
                let up = (next * next * (sigma * sigma - next * next) / (sigma * sigma))
                    .max(0.0)
                    .sqrt()
                    .min(next);
                let down = (next * next - up * up).max(0.0).sqrt();
                scheduler.step(&mut latents, &prediction, sigma, down)?;
                if up > 0.0 {
                    // Every step needs its own noise, and a seeded run still
                    // has to repeat, so the step number goes into the seed.
                    let seed = config
                        .seed
                        .map(|seed| seed ^ (step as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    let extra = noise(latents.rows, latents.cols, seed);
                    for (latent, value) in latents.data.iter_mut().zip(&extra.data) {
                        *latent += up * value;
                    }
                }
            }
            Solver::DpmPlusPlus2m => {
                // Both schedules predict the direction away from the image, so
                // the image the model believes is underneath is the same
                // arithmetic for either.
                let denoised: Vec<f32> = latents
                    .data
                    .iter()
                    .zip(&prediction.data)
                    .map(|(latent, prediction)| latent - sigma * prediction)
                    .collect();
                // The step in log-noise, which is what the solver is written
                // in. It is infinite on the last step, where the update below
                // lands exactly on the denoised image.
                let height = (sigma / next).ln();
                let ratio = next / sigma;
                let mut target = denoised.clone();
                if let Some((before, last)) = &previous
                    && next > 0.0
                {
                    // Second order: extrapolate through the last step's answer
                    // rather than trusting this one alone.
                    let slope = 0.5 * height / last;
                    for (target, before) in target.iter_mut().zip(before) {
                        *target += slope * (*target - before);
                    }
                }
                for (latent, target) in latents.data.iter_mut().zip(&target) {
                    *latent = ratio * *latent + (1.0 - ratio) * target;
                }
                previous = Some((denoised, height));
            }
        }
        if !on_step(step, &latents) {
            break;
        }
    }

    Ok(latents)
}

/// The model's prediction at `sigma`, guided when `guidance` is not 1.0 and
/// the model has an unconditional prediction to push away from.
fn predict<D: Denoiser>(
    denoiser: &mut D,
    latents: &Matrix,
    sigma: f32,
    guidance: f32,
) -> Result<Matrix, NetworkError> {
    let mut prediction = denoiser.denoise(latents, sigma)?;
    if guidance != 1.0
        && let Some(unconditional) = denoiser.denoise_unconditional(latents, sigma)?
    {
        if unconditional.data.len() != prediction.data.len() {
            return Err(NetworkError::InvalidTarget {
                expected: prediction.data.len(),
                actual: unconditional.data.len(),
            });
        }
        // Push away from what the model would have predicted with no
        // prompt, which is what "follow the prompt harder" means.
        for (conditional, unconditional) in prediction.data.iter_mut().zip(&unconditional.data) {
            *conditional = unconditional + guidance * (*conditional - unconditional);
        }
    }
    Ok(prediction)
}

/// Inpainting: a [`Denoiser`] that repaints only part of a latent and keeps
/// the rest of `image`.
///
/// `keep` holds one weight a latent row (a packed latent's rows are its
/// positions): 1.0 keeps the picture there, 0.0 repaints it, and anything
/// between blends the two predictions, which softens a seam. Where it keeps,
/// the prediction is the one whose clean image is exactly `image`, so those
/// rows walk their own noise path down to the picture under every solver and
/// schedule, while the model still sees them and paints the rest to match.
///
/// This is the blending kind of inpainting, which any model runs. A UNet
/// trained for it reads the mask itself instead; see
/// [`crate::ImagePipeline::generate_inpaint`], which picks between the two.
pub struct Inpaint<'a, D> {
    pub denoiser: &'a mut D,
    pub image: &'a Matrix,
    pub keep: &'a [f32],
}

impl<D: Denoiser> Inpaint<'_, D> {
    fn blend(
        &self,
        latents: &Matrix,
        sigma: f32,
        mut prediction: Matrix,
    ) -> Result<Matrix, NetworkError> {
        if self.image.data.len() != latents.data.len() || self.keep.len() != latents.rows {
            return Err(NetworkError::InvalidConfig(format!(
                "inpainting a {}x{} latent needs an image of that shape and {} weights, \
                 not {}x{} and {}",
                latents.rows,
                latents.cols,
                latents.rows,
                self.image.rows,
                self.image.cols,
                self.keep.len()
            )));
        }
        let cols = latents.cols;
        for (i, value) in prediction.data.iter_mut().enumerate() {
            let keep = self.keep[i / cols];
            let known = (latents.data[i] - self.image.data[i]) / sigma.max(1e-6);
            *value = keep * known + (1.0 - keep) * *value;
        }
        Ok(prediction)
    }
}

impl<D: Denoiser> Denoiser for Inpaint<'_, D> {
    fn denoise(&mut self, latents: &Matrix, sigma: f32) -> Result<Matrix, NetworkError> {
        let prediction = self.denoiser.denoise(latents, sigma)?;
        self.blend(latents, sigma, prediction)
    }

    fn denoise_unconditional(
        &mut self,
        latents: &Matrix,
        sigma: f32,
    ) -> Result<Option<Matrix>, NetworkError> {
        match self.denoiser.denoise_unconditional(latents, sigma)? {
            Some(prediction) => self.blend(latents, sigma, prediction).map(Some),
            None => Ok(None),
        }
    }
}

/// Multistep consistency sampling, the loop a latent consistency model (LCM)
/// and its relatives (LCM-LoRA, TCD) run in two to eight steps.
///
/// Each step takes the model's answer as the finished image, then noises that
/// image afresh down to the next level with [`Scheduler::add_noise`]; the last
/// step returns the image itself. Unlike the [`Solver`]s, no step follows the
/// path the previous one was on, which is what a model distilled to jump
/// straight to the image wants. Run it over [`Scheduler::lcm_sigmas`] for an
/// LCM; `config.solver` is not read, and `config.seed` seeds the noise.
///
/// NOTE: diffusers also blends a `c_skip` share of the noisy input back
/// into the answer, but below `1e-5` at every LCM timestep, so it is left out.
pub fn sample_consistency<D: Denoiser>(
    denoiser: &mut D,
    scheduler: Scheduler,
    mut latents: Matrix,
    sigmas: &[f32],
    config: &SamplingConfig,
    mut on_step: impl FnMut(usize, &Matrix) -> bool,
) -> Result<Matrix, NetworkError> {
    let Some(steps) = sigmas.len().checked_sub(1).filter(|&steps| steps > 0) else {
        return Err(NetworkError::InvalidConfig(
            "a sampling run needs at least two noise levels".into(),
        ));
    };
    for step in 0..steps {
        let (sigma, next) = (sigmas[step], sigmas[step + 1]);
        let prediction = predict(denoiser, &latents, sigma, config.guidance)?;
        // Either schedule's prediction points away from the image, so moving
        // all the way to zero noise lands on the model's answer.
        scheduler.step(&mut latents, &prediction, sigma, 0.0)?;
        if next > 0.0 {
            let seed = config
                .seed
                .map(|seed| seed ^ (step as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let fresh = noise(latents.rows, latents.cols, seed);
            scheduler.add_noise(&mut latents, &fresh, next)?;
        }
        if !on_step(step, &latents) {
            break;
        }
    }
    Ok(latents)
}

/// UniPC (Zhao et al., 2023), the multistep predictor-corrector solver many
/// Stable Diffusion front ends default to: second order, `bh2`, predicting
/// the image, falling back to first order on the first and last step, as
/// diffusers' `UniPCMultistepScheduler` runs by default.
///
/// Every step first corrects the latent it arrived at with the model's new
/// answer, then predicts the next one from the last two answers, so it still
/// reads the model once per step. It is a function rather than a [`Solver`]
/// because it carries more state between steps. `config.solver` is not read.
///
/// For a DDIM schedule this matches diffusers step for step when it is given
/// the same noise levels; diffusers spaces its timesteps slightly differently
/// from [`Scheduler::sigmas`].
pub fn sample_unipc<D: Denoiser>(
    denoiser: &mut D,
    scheduler: Scheduler,
    mut latents: Matrix,
    sigmas: &[f32],
    config: &SamplingConfig,
    mut on_step: impl FnMut(usize, &Matrix) -> bool,
) -> Result<Matrix, NetworkError> {
    let Some(steps) = sigmas.len().checked_sub(1).filter(|&steps| steps > 0) else {
        return Err(NetworkError::InvalidConfig(
            "a sampling run needs at least two noise levels".into(),
        ));
    };
    // The images the model gave on the last two steps, newest last, and the
    // latent and order of the last prediction, which the corrector redoes.
    let mut answers: Vec<Vec<f32>> = Vec::with_capacity(2);
    let mut last: Option<(Vec<f32>, usize)> = None;
    for step in 0..steps {
        let (sigma, next) = (sigmas[step], sigmas[step + 1]);
        let prediction = predict(denoiser, &latents, sigma, config.guidance)?;
        let mut answer = latents.clone();
        scheduler.step(&mut answer, &prediction, sigma, 0.0)?;

        if let Some((before, order)) = &last {
            let earlier = (*order == 2).then(|| (&answers[0][..], sigmas[step - 2]));
            latents.data = unipc_update(
                scheduler,
                before,
                (sigmas[step - 1], sigma),
                &answers[answers.len() - 1],
                earlier,
                Some(&answer.data),
            );
        }
        if answers.len() == 2 {
            answers.remove(0);
        }
        answers.push(answer.data);

        let order = if step == 0 || step + 1 == steps { 1 } else { 2 };
        let earlier = (order == 2).then(|| (&answers[0][..], sigmas[step - 1]));
        let predicted = unipc_update(
            scheduler,
            &latents.data,
            (sigma, next),
            &answers[answers.len() - 1],
            earlier,
            None,
        );
        last = Some((std::mem::replace(&mut latents.data, predicted), order));
        if !on_step(step, &latents) {
            break;
        }
    }
    Ok(latents)
}

/// The signal and noise scales of a latent at `sigma`: a DDIM latent is the
/// image plus `sigma` of noise, a flow-matching one mixes the two.
fn signal_and_noise(scheduler: Scheduler, sigma: f32) -> (f64, f64) {
    let sigma = f64::from(sigma);
    match scheduler {
        Scheduler::FlowMatch { .. } => (1.0 - sigma, sigma),
        Scheduler::Ddim { .. } => (1.0, sigma),
    }
}

/// One UniPC `bh2` update of `x` from level `from` to level `to`, given the
/// model's newest answer `newest`, the answer before it with its level when
/// the update is second order, and, for the corrector, the answer at `to`.
fn unipc_update(
    scheduler: Scheduler,
    x: &[f32],
    (from, to): (f32, f32),
    newest: &[f32],
    earlier: Option<(&[f32], f32)>,
    answer_at_to: Option<&[f32]>,
) -> Vec<f32> {
    let log_snr = |sigma| {
        let (signal, noise) = signal_and_noise(scheduler, sigma);
        signal.ln() - noise.ln()
    };
    let (signal_to, noise_to) = signal_and_noise(scheduler, to);
    let (_, noise_from) = signal_and_noise(scheduler, from);
    let h = log_snr(to) - log_snr(from);
    let phi = (-h).exp_m1();
    // bh2: B(h) = e^-h - 1, and the right-hand side of the order conditions.
    let phi_2 = phi / -h - 1.0;
    let first = phi_2 / phi;
    let second = 2.0 * (phi_2 / -h - 0.5) / phi;
    // `earlier` enters through its slope against the newest answer, scaled by
    // where its level sits on this step's log-SNR axis.
    let ratio = earlier.map(|(_, sigma)| (log_snr(sigma) - log_snr(from)) / h);
    let (weight_earlier, weight_new) = match (answer_at_to, ratio) {
        (None, _) => (0.5, 0.0),
        (Some(_), None) => (0.0, 0.5),
        (Some(_), Some(ratio)) => {
            let weight = (first - second) / (1.0 - ratio);
            (weight, first - weight)
        }
    };
    let keep = noise_to / noise_from;
    (0..x.len())
        .map(|i| {
            let newest = f64::from(newest[i]);
            let mut correction = 0.0;
            if let (Some((answers, _)), Some(ratio)) = (earlier, ratio) {
                correction += weight_earlier * (f64::from(answers[i]) - newest) / ratio;
            }
            if let Some(at_to) = answer_at_to {
                correction += weight_new * (f64::from(at_to[i]) - newest);
            }
            let value = keep * f64::from(x[i]) - signal_to * phi * (newest + correction);
            value as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model whose image is a fixed target: the velocity towards it is
    /// `(target - x) / sigma`, which is exactly what a rectified-flow model
    /// trained on that one image would learn.
    struct Straight {
        target: Vec<f32>,
        calls: usize,
    }

    impl Denoiser for Straight {
        fn denoise(&mut self, latents: &Matrix, sigma: f32) -> Result<Matrix, NetworkError> {
            self.calls += 1;
            let data = latents
                .data
                .iter()
                .zip(&self.target)
                .map(|(value, target)| (value - target) / sigma.max(1e-6))
                .collect();
            Ok(Matrix::from_vec(latents.rows, latents.cols, data))
        }
    }

    #[test]
    fn a_terminal_stretches_the_levels_to_end_on_it() {
        let scheduler = Scheduler::FlowMatch {
            shift: 2.0,
            terminal: 0.02,
        };
        let sigmas = scheduler.sigmas(40).unwrap();
        assert_eq!(sigmas.len(), 41);
        assert!((sigmas[0] - 1.0).abs() < 1e-6);
        assert!((sigmas[39] - 0.02).abs() < 1e-6);
        assert_eq!(sigmas[40], 0.0);
        assert!(sigmas.windows(2).all(|pair| pair[0] > pair[1]));
        assert_eq!(
            Scheduler::flow_match(2.0).sigmas(4).unwrap()[3],
            2.0 * 0.25 / 1.25
        );
        // Named levels take the same shift and stretch as spaced ones.
        assert_eq!(
            scheduler.sigmas_from(&[1.0, 0.75, 0.5, 0.25]).unwrap(),
            scheduler.sigmas(4).unwrap()
        );
        assert_eq!(
            Scheduler::flow_match(1.0).sigmas_from(&[1.0, 0.9, 0.4]).unwrap(),
            [1.0, 0.9, 0.4, 0.0]
        );
    }

    #[test]
    fn flow_matching_walks_a_straight_path_to_the_image() {
        let target = vec![0.5, -0.25, 1.0, 0.0];
        let mut model = Straight {
            target: target.clone(),
            calls: 0,
        };
        let start = Matrix::from_vec(1, 4, vec![2.0, -3.0, 0.5, 1.5]);
        let config = SamplingConfig {
            steps: 30,
            ..SamplingConfig::default()
        };

        let image = sample(
            &mut model,
            Scheduler::flow_match(1.0),
            start,
            &config,
            |_, _| true,
        )
        .unwrap();

        assert_eq!(model.calls, 30);
        for (value, target) in image.data.iter().zip(&target) {
            assert!((value - target).abs() < 0.05, "{value} vs {target}");
        }
    }

    #[test]
    fn lcm_sigmas_sit_on_the_timesteps_diffusers_picks() {
        let alphas = alphas_cumprod(0.00085, 0.012, 1000);
        let level = |t: usize| ((1.0 - alphas[t]) / alphas[t]).sqrt();
        let sigmas = Scheduler::ddim().lcm_sigmas(4).unwrap();
        let expected: Vec<f32> = [999, 759, 499, 259]
            .map(level)
            .into_iter()
            .chain([0.0])
            .collect();
        assert_eq!(sigmas, expected);
        assert_eq!(
            Scheduler::ddim().lcm_sigmas(1).unwrap(),
            vec![level(999), 0.0]
        );
        assert_eq!(Scheduler::ddim().lcm_sigmas(50).unwrap()[49], level(19));
        assert!(Scheduler::ddim().lcm_sigmas(51).is_err());
        assert!(Scheduler::ddim().lcm_sigmas(0).is_err());
        assert!(Scheduler::flow_match(1.0).lcm_sigmas(4).is_err());
    }

    /// A noise-predicting model whose idea of the image is half its input, and
    /// which keeps every input it was shown.
    struct Halving {
        inputs: Vec<Vec<f32>>,
    }

    impl Denoiser for Halving {
        fn denoise(&mut self, latents: &Matrix, sigma: f32) -> Result<Matrix, NetworkError> {
            self.inputs.push(latents.data.clone());
            let data = latents.data.iter().map(|x| 0.5 * x / sigma).collect();
            Ok(Matrix::from_vec(latents.rows, latents.cols, data))
        }
    }

    #[test]
    fn consistency_sampling_renoises_each_answer_to_the_next_level() {
        let sigmas = Scheduler::ddim().lcm_sigmas(4).unwrap();
        let config = SamplingConfig {
            seed: Some(3),
            ..SamplingConfig::default()
        };
        let run = |seed| {
            let mut model = Halving { inputs: Vec::new() };
            let start = noise(64, 64, Some(1));
            let config = SamplingConfig { seed, ..config };
            let image = sample_consistency(
                &mut model,
                Scheduler::ddim(),
                start,
                &sigmas,
                &config,
                |_, _| true,
            )
            .unwrap();
            (image, model.inputs)
        };
        let (image, inputs) = run(Some(3));
        assert_eq!(inputs.len(), 4);
        // Each input after the first is the last answer plus fresh noise at
        // exactly the next level.
        for step in 1..4 {
            let residual: Vec<f32> = inputs[step]
                .iter()
                .zip(&inputs[step - 1])
                .map(|(x, before)| x - 0.5 * before)
                .collect();
            let spread =
                (residual.iter().map(|r| r * r).sum::<f32>() / residual.len() as f32).sqrt();
            assert!(
                (spread / sigmas[step] - 1.0).abs() < 0.05,
                "step {step}: noise {spread} vs level {}",
                sigmas[step]
            );
        }
        // The last step returns the answer with no noise on it.
        for (value, input) in image.data.iter().zip(&inputs[3]) {
            assert!((value - 0.5 * input).abs() < 1e-4);
        }
        assert_eq!(run(Some(3)).0, image);
        assert_ne!(run(Some(4)).0, image);

        // A flow model that knows its image lands on it whatever the noise.
        let target = vec![0.5, -0.25, 1.0, 0.0];
        let mut model = Straight {
            target: target.clone(),
            calls: 0,
        };
        let start = Matrix::from_vec(1, 4, vec![2.0, -3.0, 0.5, 1.5]);
        let image = sample_consistency(
            &mut model,
            Scheduler::flow_match(1.0),
            start,
            &[1.0, 0.6, 0.3, 0.0],
            &config,
            |_, _| true,
        )
        .unwrap();
        assert_eq!(model.calls, 3);
        for (value, target) in image.data.iter().zip(&target) {
            assert!((value - target).abs() < 1e-5, "{value} vs {target}");
        }
    }

    /// A noise-predicting model with a nonlinear term, so the solver's
    /// higher-order terms change the result.
    struct Wavy;

    impl Denoiser for Wavy {
        fn denoise(&mut self, latents: &Matrix, sigma: f32) -> Result<Matrix, NetworkError> {
            let data = latents
                .data
                .iter()
                .map(|x| sigma * x / (1.0 + sigma * sigma) + 0.1 * x.sin())
                .collect();
            Ok(Matrix::from_vec(latents.rows, latents.cols, data))
        }
    }

    #[test]
    fn unipc_matches_the_diffusers_scheduler() {
        // diffusers 0.40 `UniPCMultistepScheduler` at its defaults over the
        // Stable Diffusion betas, driving the same model from
        // `x_T = [2, -3, 0.5, 1.5]` (our latent is `x_T / sqrt(alpha_bar)`).
        let alphas = alphas_cumprod(0.00085, 0.012, 1000);
        let cases: [(&[usize], [f32; 4]); 3] = [
            (
                &[999, 500],
                [1.393_945_2, -1.558_015_2, -0.032_679_02, 0.594_333_2],
            ),
            (
                &[999, 874, 749, 624, 500, 375, 250, 125],
                [1.819_281_8, -2.845_583, 0.296_030_2, 1.358_274],
            ),
            (
                &[
                    999, 949, 899, 849, 799, 749, 699, 649, 599, 549, 500, 450, 400, 350, 300, 250,
                    200, 150, 100, 50,
                ],
                [1.961_916_3, -2.946_663, 0.386_013, 1.407_398_7],
            ),
        ];
        for (timesteps, expected) in cases {
            let mut sigmas: Vec<f32> = timesteps
                .iter()
                .map(|&t| ((1.0 - alphas[t]) / alphas[t]).sqrt())
                .collect();
            sigmas.push(0.0);
            let scale = alphas[999].sqrt();
            let start = Matrix::from_vec(1, 4, [2.0, -3.0, 0.5, 1.5].map(|x| x / scale).to_vec());
            let image = sample_unipc(
                &mut Wavy,
                Scheduler::ddim(),
                start,
                &sigmas,
                &SamplingConfig::default(),
                |_, _| true,
            )
            .unwrap();
            for (value, expected) in image.data.iter().zip(expected) {
                assert!(
                    (value - expected).abs() < 1e-4,
                    "{} steps: {value} vs diffusers {expected}",
                    timesteps.len()
                );
            }
        }

        // From pure noise at level 1.0 a flow model that knows its image
        // lands on it.
        let target = vec![0.5, -0.25, 1.0, 0.0];
        let mut model = Straight {
            target: target.clone(),
            calls: 0,
        };
        let start = Matrix::from_vec(1, 4, vec![2.0, -3.0, 0.5, 1.5]);
        let sigmas = Scheduler::flow_match(3.0).sigmas(6).unwrap();
        let image = sample_unipc(
            &mut model,
            Scheduler::flow_match(3.0),
            start,
            &sigmas,
            &SamplingConfig::default(),
            |_, _| true,
        )
        .unwrap();
        assert_eq!(model.calls, 6);
        for (value, target) in image.data.iter().zip(&target) {
            assert!((value - target).abs() < 1e-4, "{value} vs {target}");
        }
    }

    #[test]
    fn karras_sigmas_span_the_stable_diffusion_range() {
        let sigmas = Scheduler::ddim().karras_sigmas(10).unwrap();
        // The range diffusers reports for Stable Diffusion's betas.
        assert_eq!(sigmas.len(), 11);
        assert!((sigmas[0] - 14.6146).abs() < 1e-2, "{}", sigmas[0]);
        assert!((sigmas[9] - 0.0292).abs() < 1e-4, "{}", sigmas[9]);
        assert_eq!(sigmas[10], 0.0);
        assert!(sigmas.windows(2).all(|pair| pair[0] > pair[1]));
        // Denser at the low end than the DDIM spacing over the same range.
        let ddim = Scheduler::ddim().sigmas(10).unwrap();
        assert!(sigmas[5] < ddim[5]);
        assert!(Scheduler::flow_match(1.0).karras_sigmas(10).is_err());

        let target = vec![0.5, -0.25, 1.0, 0.0];
        let mut model = Straight {
            target: target.clone(),
            calls: 0,
        };
        let start = Matrix::from_vec(1, 4, vec![2.0, -3.0, 0.5, 1.5]);
        let config = SamplingConfig {
            solver: Solver::DpmPlusPlus2m,
            ..SamplingConfig::default()
        };
        let image = sample_with_sigmas(
            &mut model,
            Scheduler::ddim(),
            start,
            &sigmas,
            &config,
            0,
            |_, _| true,
        )
        .unwrap();
        assert_eq!(model.calls, 10);
        for (value, target) in image.data.iter().zip(&target) {
            assert!((value - target).abs() < 1e-4, "{value} vs {target}");
        }
    }

    /// A model whose idea of the image drifts with the noise level, so the
    /// path from noise to image bends and a first-order step has something to
    /// miss. A real model behaves this way; `Straight` does not.
    struct Curved {
        target: Vec<f32>,
        calls: usize,
    }

    impl Denoiser for Curved {
        fn denoise(&mut self, latents: &Matrix, sigma: f32) -> Result<Matrix, NetworkError> {
            self.calls += 1;
            let data = latents
                .data
                .iter()
                .zip(&self.target)
                .map(|(value, target)| {
                    // Where the model thinks the image is depends on how noisy
                    // it believes the input to be, so the path bends and where
                    // it ends depends on the route taken.
                    let clean = target + 0.5 * sigma * (value - target);
                    (value - clean) / sigma.max(1e-6)
                })
                .collect();
            Ok(Matrix::from_vec(latents.rows, latents.cols, data))
        }
    }

    #[test]
    fn the_higher_order_solver_lands_on_the_image_in_fewer_steps() {
        let target = vec![0.5, -0.25, 1.0, 0.0];
        let start = Matrix::from_vec(1, 4, vec![2.0, -3.0, 0.5, 1.5]);
        let run = |solver: Solver, steps: usize| {
            let mut model = Curved {
                target: target.clone(),
                calls: 0,
            };
            let config = SamplingConfig {
                steps,
                solver,
                ..SamplingConfig::default()
            };
            let image = sample(
                &mut model,
                Scheduler::flow_match(1.0),
                start.clone(),
                &config,
                |_, _| true,
            )
            .unwrap();
            // One model call per step whichever solver is running: the second
            // order comes from the last step's answer, not from a second pass.
            assert_eq!(model.calls, steps);
            image
        };

        // What the path actually ends on, to the accuracy of five hundred
        // steps. A solver is good when it lands near this on few steps, which
        // is not the same as landing near the model's image.
        let settled = run(Solver::Euler, 500);
        let error = |solver: Solver, steps: usize| {
            run(solver, steps)
                .data
                .iter()
                .zip(&settled.data)
                .map(|(value, settled)| (value - settled).abs())
                .fold(0.0f32, f32::max)
        };

        assert!(error(Solver::DpmPlusPlus2m, 4) < error(Solver::Euler, 4));
        assert!(error(Solver::DpmPlusPlus2m, 8) < error(Solver::Euler, 8));
        // And it still converges where Euler does.
        assert!(error(Solver::DpmPlusPlus2m, 30) < 0.01);
    }

    #[test]
    fn the_ancestral_solver_repeats_on_a_seed_and_wanders_without_one() {
        let target = vec![0.5, -0.25, 1.0, 0.0];
        let start = Matrix::from_vec(1, 4, vec![2.0, -3.0, 0.5, 1.5]);
        // The path, not the destination: this model names the image exactly,
        // so every run lands on it whatever noise was put back along the way.
        // What the seed decides is the route.
        let run = |seed: Option<u64>| {
            let mut model = Straight {
                target: target.clone(),
                calls: 0,
            };
            let config = SamplingConfig {
                steps: 12,
                solver: Solver::EulerAncestral,
                seed,
                ..SamplingConfig::default()
            };
            let mut route = Vec::new();
            let landed = sample(
                &mut model,
                Scheduler::ddim(),
                start.clone(),
                &config,
                |_, latents| {
                    route.extend_from_slice(&latents.data);
                    true
                },
            )
            .unwrap();
            (route, landed)
        };

        assert_eq!(run(Some(7)).0, run(Some(7)).0);
        assert_ne!(run(Some(7)).0, run(Some(8)).0);
        // The noise it puts back is gone by the end, so it still lands on the
        // image rather than near it.
        for (value, target) in run(Some(7)).1.data.iter().zip(&target) {
            assert!((value - target).abs() < 0.05, "{value} vs {target}");
        }
    }

    #[test]
    fn the_levels_fall_to_zero_and_the_shift_slows_the_noisy_end() {
        for scheduler in [Scheduler::flow_match(1.0), Scheduler::flow_match(3.0)] {
            let sigmas = scheduler.sigmas(20).unwrap();
            assert_eq!(sigmas.len(), 21);
            assert!((sigmas[0] - 1.0).abs() < 1e-6);
            assert_eq!(*sigmas.last().unwrap(), 0.0);
            for pair in sigmas.windows(2) {
                assert!(pair[0] > pair[1], "{pair:?}");
            }
        }
        // A shift above one holds the run at high noise for longer, so every
        // level sits above the evenly spaced one.
        let even = Scheduler::flow_match(1.0).sigmas(20).unwrap();
        let shifted = Scheduler::flow_match(3.0).sigmas(20).unwrap();
        assert!(
            even.iter()
                .zip(&shifted)
                .skip(1)
                .take(19)
                .all(|(even, shifted)| shifted > even)
        );

        let ddim = Scheduler::ddim().sigmas(20).unwrap();
        assert_eq!(ddim.len(), 21);
        assert_eq!(*ddim.last().unwrap(), 0.0);
        for pair in ddim.windows(2) {
            assert!(pair[0] > pair[1], "{pair:?}");
        }
        assert!(Scheduler::ddim().sigmas(0).is_err());
    }

    #[test]
    fn a_noise_predicting_model_lands_on_the_same_image() {
        // The DDIM update reads a prediction of the noise rather than of the
        // direction, so the same target needs a different model.
        struct Noisy {
            target: Vec<f32>,
        }
        impl Denoiser for Noisy {
            fn denoise(&mut self, latents: &Matrix, sigma: f32) -> Result<Matrix, NetworkError> {
                let data = latents
                    .data
                    .iter()
                    .zip(&self.target)
                    .map(|(value, target)| (value - target) / sigma.max(1e-6))
                    .collect();
                Ok(Matrix::from_vec(latents.rows, latents.cols, data))
            }
        }

        let target = vec![0.25, -0.75, 0.0];
        let image = sample(
            &mut Noisy {
                target: target.clone(),
            },
            Scheduler::ddim(),
            Matrix::from_vec(1, 3, vec![3.0, 1.0, -2.0]),
            &SamplingConfig {
                steps: 12,
                ..SamplingConfig::default()
            },
            |_, _| true,
        )
        .unwrap();

        for (value, target) in image.data.iter().zip(&target) {
            assert!((value - target).abs() < 1e-3, "{value} vs {target}");
        }
    }

    #[test]
    fn inpainting_keeps_the_picture_and_repaints_the_rest() {
        // The model paints a solid 0.5; the picture is two rows of its own.
        let image = Matrix::from_vec(3, 2, vec![1.0, -1.0, 2.0, -2.0, 0.0, 0.0]);
        let keep = [1.0, 0.0, 0.5];
        for (scheduler, solver) in [
            (Scheduler::flow_match(1.0), Solver::Euler),
            (Scheduler::flow_match(3.0), Solver::DpmPlusPlus2m),
            (Scheduler::ddim(), Solver::EulerAncestral),
        ] {
            let mut model = Straight {
                target: vec![0.5; 6],
                calls: 0,
            };
            let start = noise(3, 2, Some(7));
            let mut inpaint = Inpaint {
                denoiser: &mut model,
                image: &image,
                keep: &keep,
            };
            let config = SamplingConfig {
                steps: 20,
                solver,
                seed: Some(3),
                ..SamplingConfig::default()
            };
            let out = sample(&mut inpaint, scheduler, start, &config, |_, _| true).unwrap();
            let expected = [1.0, -1.0, 0.5, 0.5, 0.25, 0.25];
            for (value, expected) in out.data.iter().zip(expected) {
                assert!(
                    (value - expected).abs() < 1e-3,
                    "{scheduler:?} {solver:?}: {:?}",
                    out.data
                );
            }
        }
        let mut model = Straight {
            target: vec![0.5; 6],
            calls: 0,
        };
        let mut wrong = Inpaint {
            denoiser: &mut model,
            image: &image,
            keep: &keep[..2],
        };
        assert!(wrong.denoise(&image, 0.5).is_err());
    }

    #[test]
    fn guidance_pushes_away_from_the_unconditional_prediction() {
        struct Guided;
        impl Denoiser for Guided {
            fn denoise(&mut self, latents: &Matrix, _: f32) -> Result<Matrix, NetworkError> {
                Ok(Matrix::from_vec(
                    latents.rows,
                    latents.cols,
                    vec![2.0; latents.data.len()],
                ))
            }
            fn denoise_unconditional(
                &mut self,
                latents: &Matrix,
                _: f32,
            ) -> Result<Option<Matrix>, NetworkError> {
                Ok(Some(Matrix::from_vec(
                    latents.rows,
                    latents.cols,
                    vec![1.0; latents.data.len()],
                )))
            }
        }

        // One step of flow matching from sigma 1.0 to 0.0 moves by minus the
        // prediction, so the guided prediction is visible in the result:
        // 1 + 3 * (2 - 1) = 4.
        let start = Matrix::from_vec(1, 2, vec![0.0, 0.0]);
        let config = SamplingConfig {
            steps: 1,
            guidance: 3.0,
            ..SamplingConfig::default()
        };
        let guided = sample(
            &mut Guided,
            Scheduler::flow_match(1.0),
            start.clone(),
            &config,
            |_, _| true,
        )
        .unwrap();
        assert_eq!(guided.data, vec![-4.0, -4.0]);

        // A model with no unconditional branch ignores the scale rather than
        // failing, which is what a distilled model needs.
        struct Distilled;
        impl Denoiser for Distilled {
            fn denoise(&mut self, latents: &Matrix, _: f32) -> Result<Matrix, NetworkError> {
                Ok(Matrix::from_vec(
                    latents.rows,
                    latents.cols,
                    vec![2.0; latents.data.len()],
                ))
            }
        }
        let plain = sample(
            &mut Distilled,
            Scheduler::flow_match(1.0),
            start,
            &config,
            |_, _| true,
        )
        .unwrap();
        assert_eq!(plain.data, vec![-2.0, -2.0]);
    }

    #[test]
    fn the_callback_can_stop_a_run_early() {
        let mut model = Straight {
            target: vec![0.0; 4],
            calls: 0,
        };
        sample(
            &mut model,
            Scheduler::flow_match(1.0),
            Matrix::new(1, 4),
            &SamplingConfig {
                steps: 20,
                ..SamplingConfig::default()
            },
            |step, _| step < 4,
        )
        .unwrap();

        assert_eq!(model.calls, 5);
    }

    #[test]
    fn seeded_noise_repeats_and_looks_normal() {
        let first = noise(8, 16, Some(9));
        assert_eq!(first.data, noise(8, 16, Some(9)).data);
        assert_ne!(first.data, noise(8, 16, Some(10)).data);

        let mean = first.data.iter().sum::<f32>() / first.data.len() as f32;
        let variance = first
            .data
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / first.data.len() as f32;
        assert!(mean.abs() < 0.2, "{mean}");
        assert!((variance - 1.0).abs() < 0.3, "{variance}");
    }
}
