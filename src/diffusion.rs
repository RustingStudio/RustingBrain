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
    FlowMatch { shift: f32 },
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
        Self::FlowMatch { shift }
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
            Self::FlowMatch { shift } => (0..=steps)
                .map(|step| {
                    let level = 1.0 - step as f32 / steps as f32;
                    // The shift pushes levels towards one, so more of the run
                    // is spent where the image's structure is still being set.
                    shift * level / (1.0 + (shift - 1.0) * level)
                })
                .collect(),
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
    // Box-Muller: two uniforms into two normals, and `rand` without the
    // distributions feature has no normal of its own.
    let mut data = Vec::with_capacity(rows * cols);
    while data.len() < rows * cols {
        let first: f32 = rng.gen_range(f32::MIN_POSITIVE..1.0);
        let second: f32 = rng.gen_range(0.0..1.0);
        let radius = (-2.0 * first.ln()).sqrt();
        let angle = std::f32::consts::TAU * second;
        data.push(radius * angle.cos());
        if data.len() < rows * cols {
            data.push(radius * angle.sin());
        }
    }
    Matrix::from_vec(rows, cols, data)
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
    mut latents: Matrix,
    config: &SamplingConfig,
    first: usize,
    mut on_step: impl FnMut(usize, &Matrix) -> bool,
) -> Result<Matrix, NetworkError> {
    let sigmas = scheduler.sigmas(config.steps)?;
    // DPM++ is a multistep solver: it reuses the image the last step believed
    // was underneath, and the step size it was taken over.
    let mut previous: Option<(Vec<f32>, f32)> = None;

    for step in first.min(config.steps)..config.steps {
        let (sigma, next) = (sigmas[step], sigmas[step + 1]);
        let mut prediction = denoiser.denoise(&latents, sigma)?;

        if config.guidance != 1.0
            && let Some(unconditional) = denoiser.denoise_unconditional(&latents, sigma)?
        {
            if unconditional.data.len() != prediction.data.len() {
                return Err(NetworkError::InvalidTarget {
                    expected: prediction.data.len(),
                    actual: unconditional.data.len(),
                });
            }
            // Push away from what the model would have predicted with no
            // prompt, which is what "follow the prompt harder" means.
            for (conditional, unconditional) in prediction.data.iter_mut().zip(&unconditional.data)
            {
                *conditional = unconditional + config.guidance * (*conditional - unconditional);
            }
        }

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
            sample(
                &mut model,
                Scheduler::ddim(),
                start.clone(),
                &config,
                |_, _| true,
            )
            .unwrap()
        };

        assert_eq!(run(Some(7)).data, run(Some(7)).data);
        assert_ne!(run(Some(7)).data, run(Some(8)).data);
        // The noise it puts back is gone by the end, so it still lands on the
        // image rather than near it.
        for (value, target) in run(Some(7)).data.iter().zip(&target) {
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
