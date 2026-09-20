//! Prompt in, image out: the four parts of a text-to-image model wired
//! together.
//!
//! The parts are the prompt encoder ([`PromptEncoder`], which carries its own
//! tokenizer), the denoiser ([`ImageDenoiser`]) and the decoder
//! ([`VaeDecoder`]), and each is useful on its own. This is the seam that runs
//! them in order:
//!
//! 1. The prompt becomes token ids, and the ids become hidden states.
//! 2. A noise latent is drawn at the resolution the image asks for.
//! 3. The sampler walks the latent from noise to image, asking the denoiser
//!    for a direction at each step.
//! 4. The latent is unpacked from its patch grid and decoded to pixels.
//!
//! Step four is where the packing lives. A transformer denoiser works on
//! patches, not pixels, so a `[channels, height, width]` latent is folded into
//! `[patches, channels * patch * patch]` before sampling and unfolded after —
//! the pixel shuffle from [`crate::conv`], in both directions. A UNet reads the
//! latent itself, so its patch is one and the folding is a formality.
//!
//! Which schedule a run walks is part of the model too: a rectified-flow
//! transformer moves along a straight line, and the Stable Diffusion line
//! predicts noise along the beta schedule it was trained on, with the empty
//! prompt denoised beside the real one when
//! [`PipelineConfig::guidance`] is above one.
//!
//! Memory: the parts are loaded independently, so a caller that cannot hold
//! all of them at once can drop the encoder after [`ImagePipeline::condition`]
//! and the denoiser after [`ImagePipeline::sample_latent`]. That is what
//! makes a 4B model fit a 12 GB card.

use crate::clip::ClipTextEncoder;
use crate::conv::{FeatureMap, pixel_shuffle, pixel_unshuffle};
use crate::diffusion::{Denoiser, SamplingConfig, Scheduler, Solver, noise, sample, sample_from};
use crate::matrix::Matrix;
use crate::mmdit::{Conditioning, Dit};
use crate::network::NetworkError;
use crate::t5::T5Encoder;
use crate::text_encoder::TextEncoder;
use crate::tokenizer::{Bpe, Unigram};
use crate::transformer::Precision;
use crate::unet::{Unet, UnetConfig};
use crate::vae::{VaeDecoder, VaeEncoder};

/// How a run is put together: the sampler's settings and the latent's packing.
#[derive(Clone, Debug, PartialEq)]
pub struct PipelineConfig {
    pub steps: usize,
    /// How one sampling step is taken. Euler unless a caller asks for more:
    /// [`Solver::DpmPlusPlus2m`] reaches a usable image in fewer steps on a
    /// guided model, at the same cost per step.
    pub solver: Solver,
    /// Classifier-free guidance, which a distilled model leaves at 1.0.
    pub guidance: f32,
    /// The distilled guidance scale a FLUX-style model embeds, which is a
    /// different thing: it is an input to the network, not a second pass.
    pub embedded_guidance: f32,
    /// How the schedule bends towards the noisy end. Published models raise it
    /// with resolution.
    pub shift: f32,
    /// The models that record `use_dynamic_shifting` raise the shift with the
    /// image's size instead of holding it fixed, and this is how far.
    pub dynamic_shift: Option<DynamicShift>,
    /// The side of a patch, in latent pixels. Two, for every model in this
    /// family.
    pub patch: usize,
    /// Channels in the latent, before packing.
    pub latent_channels: usize,
    /// The beta schedule a Stable Diffusion 1.x, 2.x or XL model was trained
    /// on, which is what makes the run a DDIM one. A rectified-flow model
    /// leaves this empty and walks a straight line instead.
    pub betas: Option<(f32, f32, usize)>,
}

impl Default for PipelineConfig {
    /// A distilled rectified-flow model, which is what the klein and schnell
    /// checkpoints are.
    fn default() -> Self {
        Self {
            steps: 28,
            solver: Solver::default(),
            guidance: 1.0,
            embedded_guidance: 3.5,
            shift: 3.0,
            dynamic_shift: None,
            patch: 2,
            latent_channels: 16,
            betas: None,
        }
    }
}

/// How a model bends its schedule by the size of the image.
///
/// The shift is interpolated between two sequence lengths and exponentiated,
/// which is what the published schedulers do when they record
/// `use_dynamic_shifting`: a larger image gets a schedule that spends longer at
/// the noisy end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicShift {
    pub base_shift: f32,
    pub max_shift: f32,
    pub base_seq_len: usize,
    pub max_seq_len: usize,
}

impl DynamicShift {
    /// The shift for an image of this many patches.
    pub fn shift(&self, patches: usize) -> f32 {
        let span = self.max_seq_len.saturating_sub(self.base_seq_len).max(1) as f32;
        let slope = (self.max_shift - self.base_shift) / span;
        let offset = self.base_shift - slope * self.base_seq_len as f32;
        (slope * patches as f32 + offset).exp()
    }
}

/// How long a T5 prompt is padded to when its tokenizer does not say, which is
/// what the FLUX.1 and Stable Diffusion 3 pipelines use.
const DEFAULT_PROMPT_TOKENS: usize = 512;

/// A text-to-image model.
pub struct ImagePipeline {
    pub encoder: PromptEncoder,
    /// A second, smaller encoder that produces the pooled prompt vector.
    ///
    /// FLUX.1 and Stable Diffusion 3 read their per-token states from a large
    /// encoder and their pooled vector from a CLIP tower beside it, each with
    /// its own tokenizer. Models that pool their own states leave this empty.
    pub pooled: Option<PooledEncoder>,
    pub denoiser: ImageDenoiser,
    pub decoder: VaeDecoder,
    /// The encoder half of the autoencoder, which only image-to-image needs.
    ///
    /// [`ImagePipeline::load`] attaches one when the checkpoint's VAE file
    /// holds it, which nearly all of them do. Text-to-image never touches it,
    /// so a file without one still loads.
    pub image_encoder: Option<VaeEncoder>,
    pub config: PipelineConfig,
}

/// The model at the middle of a pipeline.
///
/// A transformer denoiser works over patches of the latent and a UNet works
/// over the latent itself, so the pipeline packs for one and not for the
/// other. Everything else about a run is the same.
// One of these exists per pipeline and it holds a model of gigabytes behind
// its pointers, so a kilobyte of difference between the variants is not worth
// an allocation.
#[allow(clippy::large_enum_variant)]
pub enum ImageDenoiser {
    /// A rectified-flow transformer: FLUX, FLUX.2, Stable Diffusion 3.
    Transformer(Dit),
    /// A convolutional UNet: Stable Diffusion 1.x, 2.x and XL.
    Unet(Unet),
}

impl From<Dit> for ImageDenoiser {
    fn from(denoiser: Dit) -> Self {
        Self::Transformer(denoiser)
    }
}

impl From<Unet> for ImageDenoiser {
    fn from(denoiser: Unet) -> Self {
        Self::Unet(denoiser)
    }
}

impl ImageDenoiser {
    /// How many channels it reads at each position it works over.
    pub fn in_channels(&self) -> usize {
        match self {
            Self::Transformer(denoiser) => denoiser.config().in_channels,
            Self::Unet(denoiser) => denoiser.config().in_channels,
        }
    }

    /// How wide a prompt state it reads.
    pub fn text_dim(&self) -> usize {
        match self {
            Self::Transformer(denoiser) => denoiser.config().text_dim,
            Self::Unet(denoiser) => denoiser.config().cross_dim,
        }
    }

    /// How wide a pooled prompt vector it takes, if it takes one.
    pub fn pooled_dim(&self) -> Option<usize> {
        match self {
            Self::Transformer(denoiser) => denoiser.pooled_dim(),
            Self::Unet(denoiser) => denoiser.pooled_dim(),
        }
    }

    /// Hands it the prompt to follow.
    pub fn set_conditioning(&mut self, conditioning: Conditioning) {
        match self {
            Self::Transformer(denoiser) => denoiser.set_conditioning(conditioning),
            Self::Unet(denoiser) => denoiser.set_conditioning(conditioning),
        }
    }

    /// Hands it the prompt to push away from, for classifier-free guidance.
    pub fn set_unconditional(&mut self, conditioning: Conditioning) {
        match self {
            Self::Transformer(denoiser) => denoiser.set_unconditional(conditioning),
            Self::Unet(denoiser) => denoiser.set_unconditional(conditioning),
        }
    }
}

impl Denoiser for ImageDenoiser {
    fn denoise(&mut self, latents: &Matrix, sigma: f32) -> Result<Matrix, NetworkError> {
        match self {
            Self::Transformer(denoiser) => denoiser.denoise(latents, sigma),
            Self::Unet(denoiser) => denoiser.denoise(latents, sigma),
        }
    }

    fn denoise_unconditional(
        &mut self,
        latents: &Matrix,
        sigma: f32,
    ) -> Result<Option<Matrix>, NetworkError> {
        match self {
            Self::Transformer(denoiser) => denoiser.denoise_unconditional(latents, sigma),
            Self::Unet(denoiser) => denoiser.denoise_unconditional(latents, sigma),
        }
    }
}

/// A CLIP tower and the tokenizer it was trained with, used for the pooled
/// prompt vector alone.
pub struct PooledEncoder {
    pub tokenizer: Bpe,
    pub encoder: ClipTextEncoder,
}

/// The encoder a pipeline turns its prompt into hidden states with, holding the
/// tokenizer it was trained with.
///
/// Which one a model uses is part of the model: FLUX.2 encodes with a
/// LLaMA-family decoder, FLUX.1 and Stable Diffusion 3 with T5.
#[allow(clippy::large_enum_variant)]
pub enum PromptEncoder {
    /// A LLaMA-family decoder — Qwen2, Qwen3, LLaMA, Mistral — over byte-level
    /// byte-pair encoding.
    Sequence {
        tokenizer: Bpe,
        encoder: TextEncoder,
    },
    /// A T5 encoder over sentencepiece, padded to the length the model was
    /// tuned for.
    T5 {
        tokenizer: Unigram,
        encoder: T5Encoder,
        length: usize,
    },
    /// One CLIP tower, which is how Stable Diffusion 1.x and 2.x read a
    /// prompt. `skip` is how many layers from the end of the tower to leave
    /// off: zero for 1.x, one for the 2.x line.
    Clip {
        tokenizer: Bpe,
        encoder: ClipTextEncoder,
        skip: usize,
    },
    /// Two CLIP towers side by side, which is how Stable Diffusion XL reads
    /// one: the per-token states are laid beside each other and the pooled
    /// vector comes from the second tower alone.
    ClipPair {
        tokenizer: Bpe,
        encoder: ClipTextEncoder,
        second_tokenizer: Bpe,
        second: ClipTextEncoder,
        skip: usize,
    },
    /// Two CLIP towers and a T5 encoder, which is how Stable Diffusion 3 reads
    /// a prompt. The CLIP states are laid side by side and padded out to T5's
    /// width, the T5 states follow them in the sequence, and the pooled vector
    /// is both towers' pooled vectors side by side.
    ClipPairAndT5 {
        tokenizer: Bpe,
        encoder: ClipTextEncoder,
        second_tokenizer: Bpe,
        second: ClipTextEncoder,
        skip: usize,
        third_tokenizer: Unigram,
        third: T5Encoder,
        length: usize,
    },
}

impl PromptEncoder {
    /// Every CLIP tower this encoder holds, in the order it reads them.
    #[cfg(feature = "cuda")]
    pub(crate) fn towers(&mut self) -> Vec<&mut ClipTextEncoder> {
        match self {
            Self::Sequence { .. } | Self::T5 { .. } => Vec::new(),
            Self::Clip { encoder, .. } => vec![encoder],
            Self::ClipPair {
                encoder, second, ..
            }
            | Self::ClipPairAndT5 {
                encoder, second, ..
            } => vec![encoder, second],
        }
    }

    /// How wide one hidden state is.
    pub fn d_model(&self) -> usize {
        match self {
            Self::Sequence { encoder, .. } => encoder.config().d_model,
            Self::T5 { encoder, .. } => encoder.config().d_model,
            Self::Clip { encoder, .. } => encoder.config().d_model,
            // Side by side, so the widths add.
            Self::ClipPair {
                encoder, second, ..
            } => encoder.config().d_model + second.config().d_model,
            // The CLIP states are padded out to T5's width and stacked on top
            // of the T5 ones, so the width is T5's alone.
            Self::ClipPairAndT5 { third, .. } => third.config().d_model,
        }
    }

    /// How wide a pooled prompt vector this encoder produces, if it produces
    /// one.
    ///
    /// T5 does not: the models that use it carry a CLIP tower for this, which
    /// is what [`ImagePipeline::with_pooled`] takes.
    pub fn pooled_dim(&self) -> Option<usize> {
        let projected = |encoder: &ClipTextEncoder| {
            encoder
                .config()
                .projection_dim
                .unwrap_or(encoder.config().d_model)
        };
        match self {
            Self::Sequence { encoder, .. } => Some(encoder.config().d_model),
            Self::T5 { .. } => None,
            Self::Clip { encoder, .. } => Some(projected(encoder)),
            Self::ClipPair { second, .. } => Some(projected(second)),
            Self::ClipPairAndT5 {
                encoder, second, ..
            } => Some(projected(encoder) + projected(second)),
        }
    }

    /// The hidden states for a prompt, and the pooled vector where the encoder
    /// produces one.
    pub fn encode(&self, prompt: &str) -> Result<(Matrix, Option<Vec<f32>>), NetworkError> {
        match self {
            Self::Sequence { tokenizer, encoder } => {
                let hidden = encoder.encode(tokenizer, prompt)?;
                let pooled = encoder.pool(&hidden);
                Ok((hidden, Some(pooled)))
            }
            Self::T5 {
                tokenizer,
                encoder,
                length,
            } => Ok((encoder.encode(tokenizer, prompt, *length)?, None)),
            Self::Clip {
                tokenizer,
                encoder,
                skip,
            } => {
                let ids = encoder.tokenize(tokenizer, prompt)?;
                let (states, normalized) = encoder.forward_skipping(&ids, *skip)?;
                let pooled = encoder.pool(&ids, &normalized)?;
                Ok((states, Some(pooled)))
            }
            Self::ClipPair {
                tokenizer,
                encoder,
                second_tokenizer,
                second,
                skip,
            } => {
                let first = encoder.tokenize(tokenizer, prompt)?;
                let (left, _) = encoder.forward_skipping(&first, *skip)?;
                let ids = second.tokenize(second_tokenizer, prompt)?;
                let (right, normalized) = second.forward_skipping(&ids, *skip)?;
                if left.rows != right.rows {
                    return Err(NetworkError::InvalidTarget {
                        expected: left.rows,
                        actual: right.rows,
                    });
                }
                let pooled = second.pool(&ids, &normalized)?;
                Ok((beside(&left, &right), Some(pooled)))
            }
            Self::ClipPairAndT5 {
                tokenizer,
                encoder,
                second_tokenizer,
                second,
                skip,
                third_tokenizer,
                third,
                length,
            } => {
                let first = encoder.tokenize(tokenizer, prompt)?;
                let (left, left_states) = encoder.forward_skipping(&first, *skip)?;
                let ids = second.tokenize(second_tokenizer, prompt)?;
                let (right, right_states) = second.forward_skipping(&ids, *skip)?;
                if left.rows != right.rows {
                    return Err(NetworkError::InvalidTarget {
                        expected: left.rows,
                        actual: right.rows,
                    });
                }
                let clip = beside(&left, &right);
                let states = third.encode(third_tokenizer, prompt, *length)?;
                if clip.cols > states.cols {
                    return Err(NetworkError::InvalidConfig(format!(
                        "the CLIP towers produce {}-wide states and T5 produces {}-wide ones, so \
                         they do not stack",
                        clip.cols, states.cols
                    )));
                }
                let mut pooled = encoder.pool(&first, &left_states)?;
                pooled.extend(second.pool(&ids, &right_states)?);
                Ok((stacked(&clip, &states), Some(pooled)))
            }
        }
    }
}

/// The CLIP states padded out to the width of the T5 ones and laid above them,
/// which is the one sequence Stable Diffusion 3 attends over.
fn stacked(clip: &Matrix, states: &Matrix) -> Matrix {
    let mut output = Matrix::new(clip.rows + states.rows, states.cols);
    for row in 0..clip.rows {
        output.row_mut(row)[..clip.cols].copy_from_slice(clip.row(row));
    }
    for row in 0..states.rows {
        output
            .row_mut(clip.rows + row)
            .copy_from_slice(states.row(row));
    }
    output
}

/// Two sets of states of the same height, laid side by side.
fn beside(left: &Matrix, right: &Matrix) -> Matrix {
    let mut output = Matrix::new(left.rows, left.cols + right.cols);
    for row in 0..left.rows {
        let target = output.row_mut(row);
        target[..left.cols].copy_from_slice(left.row(row));
        target[left.cols..].copy_from_slice(right.row(row));
    }
    output
}

impl ImagePipeline {
    /// Wires the loaded parts together.
    pub fn new(
        encoder: PromptEncoder,
        denoiser: impl Into<ImageDenoiser>,
        decoder: VaeDecoder,
        config: PipelineConfig,
    ) -> Result<Self, NetworkError> {
        let denoiser = denoiser.into();
        let packed = config.latent_channels * config.patch * config.patch;
        if denoiser.in_channels() != packed {
            return Err(NetworkError::InvalidConfig(format!(
                "the denoiser reads {} channels per patch, and {} latent channels in {}x{} \
                 patches pack to {packed}",
                denoiser.in_channels(),
                config.latent_channels,
                config.patch,
                config.patch
            )));
        }
        if denoiser.text_dim() != encoder.d_model() {
            return Err(NetworkError::InvalidConfig(format!(
                "the denoiser reads {}-wide prompt states and the encoder produces {}-wide ones",
                denoiser.text_dim(),
                encoder.d_model()
            )));
        }
        // A denoiser that takes a pooled prompt vector has to be handed one of
        // the width it was trained on. An encoder that pools its own states is
        // checked here; one that does not is checked when a tower is attached.
        if let Some(width) = encoder.pooled_dim() {
            pooled_width(&denoiser, width)?;
        }
        if decoder.config().latent_channels != config.latent_channels {
            return Err(NetworkError::InvalidConfig(format!(
                "the decoder reads {} latent channels and the pipeline packs {}",
                decoder.config().latent_channels,
                config.latent_channels
            )));
        }
        Ok(Self {
            encoder,
            pooled: None,
            denoiser,
            decoder,
            image_encoder: None,
            config,
        })
    }

    /// Hands the pipeline a separate encoder for the pooled prompt vector.
    ///
    /// The per-token states keep coming from the main encoder; only the pooled
    /// vector changes hands. Its width is what the denoiser is checked against
    /// from here on.
    pub fn with_pooled(
        mut self,
        tokenizer: Bpe,
        encoder: ClipTextEncoder,
    ) -> Result<Self, NetworkError> {
        let width = encoder
            .config()
            .projection_dim
            .unwrap_or(encoder.config().d_model);
        pooled_width(&self.denoiser, width)?;
        self.pooled = Some(PooledEncoder { tokenizer, encoder });
        Ok(self)
    }

    /// Loads a model from the directory it was published in.
    ///
    /// The layout is the one a Hugging Face repository has: a `transformer`,
    /// `vae`, `text_encoder` and `tokenizer` beside each other, each with its
    /// own `config.json` and its own weights, sharded or not. Everything the
    /// parts need is read from those files, so hosting a new model is a
    /// download rather than a code change.
    ///
    /// ```no_run
    /// use rusting_brain::ImagePipeline;
    ///
    /// let mut pipeline = ImagePipeline::load("models/FLUX.2-klein-4B")?;
    /// let image = pipeline.generate("a lighthouse in fog", 512, 512, Some(7), |_, _| true)?;
    /// let pixels = rusting_brain::vae::to_rgb8(&image); // or `save_png` with the images feature
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn load(directory: impl AsRef<std::path::Path>) -> Result<Self, NetworkError> {
        Self::load_at(directory, Precision::F32)
    }

    /// The same, holding the denoiser and the encoder at a chosen precision.
    ///
    /// [`Precision::Q8`] is a quarter of the memory those two need, measured
    /// here at about a tenth slower per step. It is the difference between a
    /// model that does not fit and one that does; the decoder stays in `f32`,
    /// because it is small and it is what the eye sees.
    pub fn load_at(
        directory: impl AsRef<std::path::Path>,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        let directory = directory.as_ref();
        let part = |name: &str| directory.join(name);

        let decoder_config = crate::vae::VaeConfig::from_file(part("vae/config.json"))?;
        let scheduler = read_json(&part("scheduler/scheduler_config.json"));
        // Which denoiser a model has is which directory it published it in.
        let unet = part("unet/config.json").exists();

        // A model with two encoders reads its per-token states from the second
        // one. A model with one reads and pools the same states.
        let two = part("text_encoder_2/config.json").exists();
        // A model with three reads them from the third and keeps the first two
        // for the CLIP half of its prompt, which is Stable Diffusion 3.
        let three = part("text_encoder_3/config.json").exists();
        let states = match (three, two) {
            (true, _) => "text_encoder_3",
            (false, true) => "text_encoder_2",
            (false, false) => "text_encoder",
        };
        let states_config = part(&format!("{states}/config.json"));
        // Which encoder this is, is written in its own configuration.
        let states_json = read_json(&states_config);
        let t5 = states_json.as_ref().is_some_and(|json| {
            json.get("model_type").and_then(|kind| kind.as_str()) == Some("t5")
                || architecture(json, "T5")
        });
        let clip = states_json.as_ref().is_some_and(|json| {
            json.get("model_type")
                .and_then(|kind| kind.as_str())
                .is_some_and(|kind| kind.starts_with("clip"))
                || architecture(json, "CLIP")
        });

        let denoiser_config = match unet {
            true => None,
            false => Some(crate::mmdit::DitConfig::from_file(part(
                "transformer/config.json",
            ))?),
        };
        // A UNet works on the latent itself; a transformer works on patches of
        // it, and the patch size is not written down anywhere. It is whatever
        // squares the latent's channels up to the width the denoiser reads.
        let patch = match &denoiser_config {
            None => 1,
            Some(config) => {
                let packed = config.in_channels / decoder_config.latent_channels.max(1);
                let patch = (packed as f64).sqrt().round() as usize;
                if patch == 0
                    || patch * patch * decoder_config.latent_channels != config.in_channels
                {
                    return Err(NetworkError::InvalidConfig(format!(
                        "a denoiser reading {} channels per patch does not hold a square patch \
                         of a {}-channel latent",
                        config.in_channels, decoder_config.latent_channels
                    )));
                }
                patch
            }
        };

        let defaults = PipelineConfig::default();
        let config = PipelineConfig {
            patch,
            latent_channels: decoder_config.latent_channels,
            shift: scheduler
                .as_ref()
                .and_then(|scheduler| number(scheduler, "shift"))
                .unwrap_or(defaults.shift),
            dynamic_shift: scheduler.as_ref().and_then(dynamic_shift),
            // The Stable Diffusion line predicts noise along a beta schedule
            // and was trained with the prompt dropped part of the time, so it
            // is sampled with guidance over more steps than a distilled model
            // needs.
            betas: unet.then(|| ddim_betas(scheduler.as_ref())),
            // A published scheduler names the solver it was tuned with.
            solver: scheduler.as_ref().map_or(defaults.solver, solver_named),
            steps: match unet {
                true => 50,
                false => defaults.steps,
            },
            guidance: match unet {
                true => 7.5,
                false => defaults.guidance,
            },
            ..defaults
        };

        let tower = |name: &str| -> Result<ClipTextEncoder, NetworkError> {
            ClipTextEncoder::load_at(
                weights(&part(name))?,
                "",
                crate::clip::ClipTextConfig::from_file(part(&format!("{name}/config.json")))?,
                precision,
            )
        };
        // How long a T5 prompt is padded to is what the model was tuned for,
        // and its tokenizer says what that is.
        let padded_to = |words: &std::path::Path| -> usize {
            read_json(&words.join("tokenizer_config.json"))
                .as_ref()
                .and_then(|json| number(json, "model_max_length"))
                .filter(|length| *length >= 1.0)
                .map_or(DEFAULT_PROMPT_TOKENS, |length| length as usize)
        };
        let encoder = match (clip, two) {
            // Stable Diffusion 3: two CLIP towers padded out to T5's width and
            // stacked on top of the T5 states, all in one sequence.
            _ if three => PromptEncoder::ClipPairAndT5 {
                tokenizer: Bpe::from_file(part("tokenizer/tokenizer.json"))?,
                encoder: tower("text_encoder")?,
                second_tokenizer: Bpe::from_file(part("tokenizer_2/tokenizer.json"))?,
                second: tower("text_encoder_2")?,
                skip: 1,
                third_tokenizer: Unigram::from_file(part("tokenizer_3/tokenizer.json"))?,
                third: crate::t5::T5Encoder::load_at(
                    weights(&part(states))?,
                    crate::t5::T5Config::from_file(&states_config)?,
                    precision,
                )?,
                length: padded_to(&part("tokenizer_3")),
            },
            // Stable Diffusion XL: two towers, read second to last because
            // that is the layer it was trained against.
            (true, true) => PromptEncoder::ClipPair {
                tokenizer: Bpe::from_file(part("tokenizer/tokenizer.json"))?,
                encoder: tower("text_encoder")?,
                second_tokenizer: Bpe::from_file(part("tokenizer_2/tokenizer.json"))?,
                second: tower(states)?,
                skip: 1,
            },
            (true, false) => PromptEncoder::Clip {
                tokenizer: Bpe::from_file(part("tokenizer/tokenizer.json"))?,
                encoder: tower("text_encoder")?,
                skip: 0,
            },
            (false, _) => {
                let mut encoder_file =
                    crate::safetensors::ShardedSafeTensors::open(weights(&part(states))?)?;
                let words = match two {
                    true => part("tokenizer_2"),
                    false => part("tokenizer"),
                };
                let encoder = match t5 {
                    true => PromptEncoder::T5 {
                        tokenizer: Unigram::from_file(words.join("tokenizer.json"))?,
                        encoder: crate::t5::T5Encoder::read_at(
                            &mut encoder_file,
                            crate::t5::T5Config::from_file(&states_config)?,
                            precision,
                        )?,
                        length: padded_to(&words),
                    },
                    false => {
                        // A language model's own file writes its layers under
                        // `model.`; a repacked one writes them bare. The file
                        // says which.
                        let prefix = match encoder_file
                            .names()
                            .any(|name| name.starts_with("model.layers."))
                        {
                            true => "model.",
                            false => "",
                        };
                        PromptEncoder::Sequence {
                            tokenizer: Bpe::from_file(words.join("tokenizer.json"))?,
                            encoder: TextEncoder::read_at(
                                &mut encoder_file,
                                prefix,
                                crate::text_encoder::TextEncoderConfig::from_file(&states_config)?,
                                precision,
                            )?,
                        }
                    }
                };
                drop(encoder_file);
                encoder
            }
        };

        let denoiser: ImageDenoiser = match denoiser_config {
            Some(denoiser_config) => Dit::load_at(
                weights(&part("transformer"))?,
                "",
                denoiser_config,
                precision,
            )?
            .into(),
            None => {
                let mut model = Unet::load_at(
                    weights(&part("unet"))?,
                    UnetConfig::from_file(part("unet/config.json"))?,
                    precision,
                )?;
                // The noise level a sampler hands the model only means a step
                // index against the schedule the model was trained on.
                if let Some((beta_start, beta_end, train_steps)) = config.betas {
                    model.set_schedule(Scheduler::Ddim {
                        beta_start,
                        beta_end,
                        train_steps,
                    });
                }
                model.into()
            }
        };

        // One file holds both halves of the autoencoder. Text-to-image needs
        // only the decoder, so a file without an encoder still loads.
        let mut vae = crate::safetensors::ShardedSafeTensors::open(weights(&part("vae"))?)?;
        let decoder = VaeDecoder::read(&mut vae, "decoder.", decoder_config.clone())?;
        let image_encoder = VaeEncoder::read(&mut vae, "encoder.", decoder_config).ok();
        drop(vae);

        let mut pipeline = Self::new(encoder, denoiser, decoder, config)?;
        pipeline.image_encoder = image_encoder;

        // A model whose second encoder is not a CLIP tower pools through the
        // first one, which is.
        match two && !clip && !three {
            false => Ok(pipeline),
            true => pipeline.with_pooled(
                Bpe::from_file(part("tokenizer/tokenizer.json"))?,
                tower("text_encoder")?,
            ),
        }
    }

    /// Moves everything that has a device path onto CUDA device `device`.
    ///
    /// That is the UNet, the VAE decoder and every CLIP tower. A transformer
    /// denoiser and a T5 encoder stay on the host, because neither has a
    /// device mirror yet; a pipeline built from those loads and runs exactly
    /// as it did.
    ///
    /// Fails closed. Whatever was uploaded before the failure is dropped with
    /// the error, and every part that did not move is still the CPU one, so a
    /// pipeline that returns an error here is still a working pipeline.
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&mut self, device: usize) -> Result<(), NetworkError> {
        let gpu = crate::cuda_image::ImageGpu::new(device)?;
        if let ImageDenoiser::Unet(unet) = &mut self.denoiser {
            unet.attach_device(gpu.clone())?;
        }
        self.decoder.attach_device(gpu.clone())?;
        for tower in self.encoder.towers() {
            tower.attach_device(gpu.clone())?;
        }
        if let Some(pooled) = &mut self.pooled {
            pooled.encoder.attach_device(gpu)?;
        }
        Ok(())
    }

    /// The same, reporting whether it worked rather than why it did not.
    ///
    /// This is the automatic fallback: a machine with no device, a driver that
    /// will not load, or a model that does not fit leaves the pipeline on the
    /// CPU and answers `false`.
    #[cfg(feature = "cuda")]
    pub fn try_cuda(&mut self, device: usize) -> bool {
        self.to_cuda(device).is_ok()
    }

    /// Without the feature there is no device to move to, and saying so is
    /// cheaper for a caller than a `cfg` of its own.
    #[cfg(not(feature = "cuda"))]
    pub fn try_cuda(&mut self, _device: usize) -> bool {
        false
    }

    /// How many pixels one latent pixel becomes: the decoder's ladder.
    pub fn upscale(&self) -> usize {
        self.decoder.config().upscale()
    }

    /// The latent grid an image of this size needs, in patches.
    ///
    /// The image's sides have to be a multiple of the decoder's upscale times
    /// the patch size — sixteen pixels for a typical model — because nothing
    /// downstream can represent a fraction of a patch.
    pub fn patch_grid(&self, width: usize, height: usize) -> Result<(usize, usize), NetworkError> {
        let block = self.upscale() * self.config.patch;
        if width == 0 || height == 0 || width % block != 0 || height % block != 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "a {width}x{height} image does not divide into {block}-pixel patches"
            )));
        }
        Ok((width / block, height / block))
    }

    /// Runs the prompt through the tokenizer and the text encoder.
    ///
    /// Held separately from [`ImagePipeline::sample_latent`] so that a caller
    /// short of memory can drop the encoder in between.
    pub fn condition(
        &self,
        prompt: &str,
        width: usize,
        height: usize,
    ) -> Result<Conditioning, NetworkError> {
        let (columns, rows) = self.patch_grid(width, height)?;
        let (text, own) = self.encoder.encode(prompt)?;
        let pooled = match &self.pooled {
            Some(part) => part.encoder.encode(&part.tokenizer, prompt)?.1,
            // A denoiser that wants no pooled vector is handed none, which is
            // what an encoder that does not pool leaves behind.
            None => own.unwrap_or_default(),
        };
        if let Some(width) = self.denoiser.pooled_dim()
            && pooled.len() != width
        {
            return Err(NetworkError::InvalidConfig(format!(
                "the denoiser takes a {width}-wide pooled prompt and this encoder produces \
                 none: the model wants a CLIP tower beside it"
            )));
        }
        Ok(Conditioning {
            text,
            pooled,
            height: rows,
            width: columns,
            guidance: self.config.embedded_guidance,
        })
    }

    /// Walks a latent from noise to image, returning it still packed.
    pub fn sample_latent(
        &mut self,
        conditioning: Conditioning,
        seed: Option<u64>,
        on_step: impl FnMut(usize, &Matrix) -> bool,
    ) -> Result<Matrix, NetworkError> {
        self.sample_latent_guided(conditioning, None, seed, on_step)
    }

    /// The same run with a second conditioning to push away from.
    ///
    /// That is what classifier-free guidance is: the empty prompt is denoised
    /// alongside the real one and the step is taken along the difference. It
    /// costs a second pass of the denoiser per step, which is why a distilled
    /// model leaves [`PipelineConfig::guidance`] at 1.0 and passes `None`.
    pub fn sample_latent_guided(
        &mut self,
        conditioning: Conditioning,
        unconditional: Option<Conditioning>,
        seed: Option<u64>,
        on_step: impl FnMut(usize, &Matrix) -> bool,
    ) -> Result<Matrix, NetworkError> {
        let patches = conditioning.height * conditioning.width;
        let channels = self.denoiser.in_channels();
        self.denoiser.set_conditioning(conditioning);
        if let Some(unconditional) = unconditional {
            self.denoiser.set_unconditional(unconditional);
        }

        let scheduler = self.scheduler(patches);
        // A run starts at the noisiest level the schedule holds, which is one
        // for flow matching and far above it for DDIM.
        let mut start = noise(patches, channels, seed);
        let first = scheduler.sigmas(self.config.steps)?[0];
        start.data.iter_mut().for_each(|value| *value *= first);

        let settings = SamplingConfig {
            steps: self.config.steps,
            solver: self.config.solver,
            guidance: self.config.guidance,
            seed,
        };
        sample(&mut self.denoiser, scheduler, start, &settings, on_step)
    }

    /// The schedule this pipeline runs on, at the shift its latent size asks
    /// for. DDIM where the configuration names betas, flow matching where it
    /// does not.
    fn scheduler(&self, patches: usize) -> Scheduler {
        match self.config.betas {
            Some((beta_start, beta_end, train_steps)) => Scheduler::Ddim {
                beta_start,
                beta_end,
                train_steps,
            },
            None => Scheduler::flow_match(match self.config.dynamic_shift {
                Some(dynamic) => dynamic.shift(patches),
                None => self.config.shift,
            }),
        }
    }

    /// Walks a latent that already holds a picture, which is image-to-image.
    ///
    /// `latent` is a packed latent, as [`ImagePipeline::pack`] produces from
    /// what the image encoder returned. `strength` says how much of the
    /// schedule to redo: 1.0 noises the latent all the way up and ignores the
    /// picture, 0.0 leaves it alone, and the useful range sits between 0.3 and
    /// 0.8.
    pub fn sample_latent_from(
        &mut self,
        conditioning: Conditioning,
        unconditional: Option<Conditioning>,
        latent: &Matrix,
        strength: f32,
        seed: Option<u64>,
        on_step: impl FnMut(usize, &Matrix) -> bool,
    ) -> Result<Matrix, NetworkError> {
        let patches = conditioning.height * conditioning.width;
        let channels = self.denoiser.in_channels();
        if latent.rows != patches || latent.cols != channels {
            return Err(NetworkError::InvalidConfig(format!(
                "the starting latent is {}x{} and the run wants {patches}x{channels}",
                latent.rows, latent.cols
            )));
        }
        self.denoiser.set_conditioning(conditioning);
        if let Some(unconditional) = unconditional {
            self.denoiser.set_unconditional(unconditional);
        }

        let scheduler = self.scheduler(patches);
        let sigmas = scheduler.sigmas(self.config.steps)?;
        // Full strength starts at the top of the schedule, none of it starts
        // at the bottom, where there is no work left to do.
        let first = ((1.0 - strength.clamp(0.0, 1.0)) * self.config.steps as f32).round() as usize;
        let first = first.min(self.config.steps);
        let mut start = latent.clone();
        scheduler.add_noise(&mut start, &noise(patches, channels, seed), sigmas[first])?;

        let settings = SamplingConfig {
            steps: self.config.steps,
            solver: self.config.solver,
            guidance: self.config.guidance,
            seed,
        };
        sample_from(
            &mut self.denoiser,
            scheduler,
            start,
            &settings,
            first,
            on_step,
        )
    }

    /// Unpacks a sampled latent and decodes it to an image in `[-1, 1]`.
    pub fn decode(
        &self,
        latent: &Matrix,
        columns: usize,
        rows: usize,
    ) -> Result<FeatureMap, NetworkError> {
        let packed = FeatureMap::from_tokens(latent, rows, columns)?;
        let unpacked = pixel_shuffle(&packed, self.config.patch)?;
        self.decoder.decode(&unpacked)
    }

    /// Packs a latent into the patch grid a denoiser reads, which is the
    /// inverse of what [`ImagePipeline::decode`] undoes.
    pub fn pack(&self, latent: &FeatureMap) -> Result<Matrix, NetworkError> {
        Ok(pixel_unshuffle(latent, self.config.patch)?.to_tokens())
    }

    /// The whole run: prompt in, image out.
    ///
    /// The callback sees each sampling step and ends the run early when it
    /// returns `false`, which is what a progress bar or a cancel button needs.
    pub fn generate(
        &mut self,
        prompt: &str,
        width: usize,
        height: usize,
        seed: Option<u64>,
        on_step: impl FnMut(usize, &Matrix) -> bool,
    ) -> Result<FeatureMap, NetworkError> {
        let conditioning = self.condition(prompt, width, height)?;
        let (columns, rows) = (conditioning.width, conditioning.height);
        // Guidance above one asks for the empty prompt as well.
        let unconditional = match self.config.guidance != 1.0 {
            true => Some(self.condition("", width, height)?),
            false => None,
        };
        let latent = self.sample_latent_guided(conditioning, unconditional, seed, on_step)?;
        self.decode(&latent, columns, rows)
    }

    /// The same run started from a picture instead of from noise.
    ///
    /// `image` is in `[-1, 1]`, three channels, at a size
    /// [`ImagePipeline::patch_grid`] accepts. `strength` says how far back up
    /// the schedule to push it before denoising: low values keep the
    /// composition and repaint the detail, high values keep little but the
    /// layout. Needs [`ImagePipeline::image_encoder`], which
    /// [`ImagePipeline::load`] attaches when the checkpoint has one.
    pub fn generate_from_image(
        &mut self,
        prompt: &str,
        image: &FeatureMap,
        strength: f32,
        seed: Option<u64>,
        on_step: impl FnMut(usize, &Matrix) -> bool,
    ) -> Result<FeatureMap, NetworkError> {
        let encoder = self.image_encoder.as_ref().ok_or_else(|| {
            NetworkError::InvalidConfig(
                "this checkpoint's autoencoder holds no encoder, so it cannot start from an image"
                    .into(),
            )
        })?;
        let latent = self.pack(&encoder.encode(image)?)?;

        let conditioning = self.condition(prompt, image.width, image.height)?;
        let (columns, rows) = (conditioning.width, conditioning.height);
        let unconditional = match self.config.guidance != 1.0 {
            true => Some(self.condition("", image.width, image.height)?),
            false => None,
        };
        let latent = self.sample_latent_from(
            conditioning,
            unconditional,
            &latent,
            strength,
            seed,
            on_step,
        )?;
        self.decode(&latent, columns, rows)
    }
}

/// The solver a published `scheduler_config.json` names, by the class it was
/// saved as. Anything else is Euler, which every schedule steps with.
fn solver_named(scheduler: &serde_json::Value) -> Solver {
    match scheduler.get("_class_name").and_then(|name| name.as_str()) {
        Some(name) if name.contains("DPMSolverMultistep") => Solver::DpmPlusPlus2m,
        Some(name) if name.contains("EulerAncestral") => Solver::EulerAncestral,
        _ => Solver::Euler,
    }
}

/// The weights inside one component's directory: the shard index if the
/// component is sharded, and the single file if it is not.
fn weights(directory: &std::path::Path) -> Result<std::path::PathBuf, NetworkError> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let path = entry?.path();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if name.ends_with(".safetensors.index.json") {
            // An index names every shard, so it is the whole component and
            // there is nothing else to look at.
            return Ok(path);
        }
        if name.ends_with(".safetensors") {
            files.push(path);
        }
    }
    files.sort();
    match files.len() {
        1 => Ok(files.remove(0)),
        0 => Err(NetworkError::InvalidDataset(format!(
            "{} holds no .safetensors weights",
            directory.display()
        ))),
        count => Err(NetworkError::InvalidDataset(format!(
            "{} holds {count} .safetensors files and no index saying how they go together",
            directory.display()
        ))),
    }
}

/// A configuration file, if there is one. A model that ships no scheduler is
/// not an error: the defaults cover it.
fn read_json(path: &std::path::Path) -> Option<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

/// Whether a configuration names an architecture of this family.
fn architecture(json: &serde_json::Value, family: &str) -> bool {
    json.get("architectures")
        .and_then(|names| names.as_array())
        .is_some_and(|names| {
            names
                .iter()
                .any(|name| name.as_str().is_some_and(|name| name.starts_with(family)))
        })
}

/// The beta schedule a scheduler names, or the one the whole Stable Diffusion
/// line was trained with.
fn ddim_betas(scheduler: Option<&serde_json::Value>) -> (f32, f32, usize) {
    let default = (0.00085, 0.012, 1000);
    let Some(json) = scheduler else {
        return default;
    };
    (
        number(json, "beta_start").unwrap_or(default.0),
        number(json, "beta_end").unwrap_or(default.1),
        json.get("num_train_timesteps")
            .and_then(|value| value.as_u64())
            .map_or(default.2, |value| value as usize),
    )
}

fn number(json: &serde_json::Value, name: &str) -> Option<f32> {
    json.get(name)
        .and_then(|value| value.as_f64())
        .map(|value| value as f32)
}

/// The size-dependent shift, for the schedulers that say they use one.
fn dynamic_shift(scheduler: &serde_json::Value) -> Option<DynamicShift> {
    if !scheduler
        .get("use_dynamic_shifting")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
    {
        return None;
    }
    let length = |name: &str, fallback: usize| -> usize {
        scheduler
            .get(name)
            .and_then(|value| value.as_u64())
            .map_or(fallback, |value| value as usize)
    };
    Some(DynamicShift {
        base_shift: number(scheduler, "base_shift").unwrap_or(0.5),
        max_shift: number(scheduler, "max_shift").unwrap_or(1.15),
        base_seq_len: length("base_image_seq_len", 256),
        max_seq_len: length("max_image_seq_len", 4096),
    })
}

/// Checks that a pooled prompt of this width is the one the denoiser was
/// trained on. A denoiser that takes no pooled vector accepts anything.
fn pooled_width(denoiser: &ImageDenoiser, width: usize) -> Result<(), NetworkError> {
    match denoiser.pooled_dim() {
        Some(pooled) if pooled != width => Err(NetworkError::InvalidConfig(format!(
            "the denoiser takes a {pooled}-wide pooled prompt and the encoder produces a \
             {width}-wide one: this model wants a prompt encoder that is not this one"
        ))),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mmdit::DitConfig;
    use crate::text_encoder::TextEncoderConfig;
    use crate::vae::{VaeConfig, to_rgb8};

    /// The encoder shape the synthetic pipeline is built with.
    fn denoiser_encoder_config() -> TextEncoderConfig {
        TextEncoderConfig {
            vocab_size: 320,
            d_model: 16,
            layers: 1,
            num_heads: 4,
            kv_heads: 2,
            head_dim: 4,
            mlp_dim: 32,
            rope_theta: 10_000.0,
            eps: 1e-6,
            causal: true,
        }
    }

    /// A whole pipeline at a size that runs in milliseconds, built from the
    /// same synthetic checkpoints the parts' own tests use.
    fn pipeline(name: &str) -> (ImagePipeline, PipelineConfig) {
        let config = PipelineConfig {
            steps: 3,
            guidance: 1.0,
            embedded_guidance: 3.5,
            shift: 3.0,
            dynamic_shift: None,
            patch: 2,
            latent_channels: 4,
            betas: None,
            solver: Solver::default(),
        };

        let vae = VaeConfig {
            latent_channels: 4,
            out_channels: 3,
            block_out_channels: vec![8, 16],
            layers_per_block: 1,
            norm_groups: 4,
            scaling_factor: 0.5,
            shift_factor: 0.25,
        };
        let text = denoiser_encoder_config();
        let dit = DitConfig {
            in_channels: config.latent_channels * config.patch * config.patch,
            out_channels: config.latent_channels * config.patch * config.patch,
            d_model: 16,
            num_heads: 2,
            double_blocks: 1,
            single_blocks: 1,
            mlp_ratio: 2.0,
            text_dim: text.d_model,
            pooled_dim: text.d_model,
            axes_dim: vec![4, 4],
            theta: 10_000.0,
            guidance: false,
            eps: 1e-6,
        };

        let directory = std::env::temp_dir().join(format!("rusting_brain_pipeline_{name}"));
        std::fs::create_dir_all(&directory).unwrap();
        let vae_path = directory.join("vae.safetensors");
        let text_path = directory.join("text.safetensors");
        let dit_path = directory.join("dit.safetensors");
        crate::vae::tests::checkpoint(&vae, &vae_path);
        crate::text_encoder::tests::checkpoint(&text, &text_path, true);
        crate::mmdit::tests::checkpoint(&dit, &dit_path);

        let tokenizer = crate::tokenizer::tests::tokenizer(&[], &[], None);
        let encoder = TextEncoder::load(&text_path, "model.", text).unwrap();
        let denoiser = Dit::load(&dit_path, "", dit).unwrap();
        let decoder = VaeDecoder::load(&vae_path, "decoder.", vae).unwrap();
        std::fs::remove_dir_all(&directory).ok();

        (
            ImagePipeline::new(
                PromptEncoder::Sequence { tokenizer, encoder },
                denoiser,
                decoder,
                config.clone(),
            )
            .unwrap(),
            config,
        )
    }

    /// Writes the parts of a pipeline into a directory laid out the way a
    /// published repository is, with a CLIP tower beside them when one is
    /// asked for.
    fn publish(
        reference: &ImagePipeline,
        directory: &std::path::Path,
        clip: Option<&crate::clip::ClipTextConfig>,
        t5: Option<&crate::t5::T5Config>,
    ) {
        let dit = match &reference.denoiser {
            ImageDenoiser::Transformer(denoiser) => denoiser.config().clone(),
            ImageDenoiser::Unet(_) => unreachable!("the test pipeline is built with a transformer"),
        };
        publish_parts(&dit, reference.decoder.config(), directory, clip, t5, None);
    }

    /// The same from configurations alone, with an optional second CLIP tower,
    /// which is the three-encoder shape.
    fn publish_parts(
        dit: &DitConfig,
        vae: &VaeConfig,
        directory: &std::path::Path,
        clip: Option<&crate::clip::ClipTextConfig>,
        t5: Option<&crate::t5::T5Config>,
        second_clip: Option<&crate::clip::ClipTextConfig>,
    ) {
        let text = denoiser_encoder_config();
        // With two encoders the per-token states come from the second one and
        // the first is the CLIP tower, each with its own tokenizer. With three
        // they come from the third.
        let (states, words) = match (second_clip.is_some(), clip.is_some()) {
            (true, _) => ("text_encoder_3", "tokenizer_3"),
            (false, true) => ("text_encoder_2", "tokenizer_2"),
            (false, false) => ("text_encoder", "tokenizer"),
        };

        for name in ["transformer", "vae", states, words, "scheduler"] {
            std::fs::create_dir_all(directory.join(name)).unwrap();
        }
        crate::mmdit::tests::checkpoint(
            dit,
            &directory.join("transformer/diffusion_pytorch_model.safetensors"),
        );
        crate::vae::tests::checkpoint(
            vae,
            &directory.join("vae/diffusion_pytorch_model.safetensors"),
        );
        match t5 {
            None => {
                crate::text_encoder::tests::checkpoint(
                    &text,
                    &directory.join(format!("{states}/model.safetensors")),
                    true,
                );
                crate::tokenizer::tests::write(
                    &directory.join(format!("{words}/tokenizer.json")),
                    &[],
                    &[],
                    None,
                );
            }
            Some(t5) => {
                crate::t5::tests::checkpoint(
                    t5,
                    &directory.join(format!("{states}/model.safetensors")),
                );
                std::fs::write(
                    directory.join(format!("{words}/tokenizer.json")),
                    crate::tokenizer::tests::unigram_json(),
                )
                .unwrap();
            }
        }

        let write = |name: &str, json: String| {
            std::fs::write(directory.join(name), json).unwrap();
        };
        write(
            "transformer/config.json",
            format!(
                r#"{{"in_channels": {}, "num_attention_heads": {}, "attention_head_dim": {},
                    "num_layers": {}, "num_single_layers": {}, "joint_attention_dim": {},
                    "pooled_projection_dim": {}, "axes_dims_rope": {:?}, "mlp_ratio": {}}}"#,
                dit.in_channels,
                dit.num_heads,
                dit.head_dim(),
                dit.double_blocks,
                dit.single_blocks,
                dit.text_dim,
                dit.pooled_dim,
                dit.axes_dim,
                dit.mlp_ratio,
            ),
        );
        write(
            "vae/config.json",
            format!(
                r#"{{"latent_channels": {}, "out_channels": {}, "block_out_channels": {:?},
                    "layers_per_block": {}, "norm_num_groups": {}, "scaling_factor": {},
                    "shift_factor": {}}}"#,
                vae.latent_channels,
                vae.out_channels,
                vae.block_out_channels,
                vae.layers_per_block,
                vae.norm_groups,
                vae.scaling_factor,
                vae.shift_factor,
            ),
        );
        match t5 {
            None => write(
                &format!("{states}/config.json"),
                format!(
                    r#"{{"vocab_size": {}, "hidden_size": {}, "num_hidden_layers": {},
                        "num_attention_heads": {}, "num_key_value_heads": {}, "head_dim": {},
                        "intermediate_size": {}, "rope_theta": {}}}"#,
                    text.vocab_size,
                    text.d_model,
                    text.layers,
                    text.num_heads,
                    text.kv_heads,
                    text.head_dim,
                    text.mlp_dim,
                    text.rope_theta,
                ),
            ),
            Some(t5) => {
                write(
                    &format!("{states}/config.json"),
                    format!(
                        r#"{{"model_type": "t5", "vocab_size": {}, "d_model": {}, "d_kv": {},
                            "num_heads": {}, "num_layers": {}, "d_ff": {},
                            "relative_attention_num_buckets": {},
                            "relative_attention_max_distance": {},
                            "feed_forward_proj": "gated-gelu"}}"#,
                        t5.vocab_size,
                        t5.d_model,
                        t5.head_dim,
                        t5.num_heads,
                        t5.layers,
                        t5.mlp_dim,
                        t5.buckets,
                        t5.max_distance,
                    ),
                );
                // How long a prompt is padded to is the tokenizer's business.
                write(
                    &format!("{words}/tokenizer_config.json"),
                    r#"{"model_max_length": 6}"#.into(),
                );
            }
        }
        write(
            "scheduler/scheduler_config.json",
            r#"{"_class_name": "FlowMatchEulerDiscreteScheduler", "shift": 2.5,
                "use_dynamic_shifting": true, "base_shift": 0.5, "max_shift": 1.15,
                "base_image_seq_len": 256, "max_image_seq_len": 4096}"#
                .into(),
        );

        for (tower, clip) in [("", clip), ("_2", second_clip)] {
            let Some(clip) = clip else { continue };
            let (encoder, tokenizer) =
                (format!("text_encoder{tower}"), format!("tokenizer{tower}"));
            for name in [&encoder, &tokenizer] {
                std::fs::create_dir_all(directory.join(name)).unwrap();
            }
            crate::clip::tests::checkpoint(
                clip,
                &directory.join(format!("{encoder}/model.safetensors")),
                false,
            );
            crate::tokenizer::tests::write(
                &directory.join(format!("{tokenizer}/tokenizer.json")),
                &[],
                &[],
                None,
            );
            write(
                &format!("{encoder}/config.json"),
                format!(
                    r#"{{"vocab_size": {}, "hidden_size": {}, "num_hidden_layers": {},
                        "num_attention_heads": {}, "intermediate_size": {},
                        "max_position_embeddings": {}, "hidden_act": "quick_gelu",
                        "bos_token_id": {}, "eos_token_id": {}}}"#,
                    clip.vocab_size,
                    clip.d_model,
                    clip.layers,
                    clip.num_heads,
                    clip.mlp_dim,
                    clip.max_positions,
                    clip.bos_token,
                    clip.eos_token,
                ),
            );
        }
    }

    #[test]
    fn a_published_directory_loads_without_being_told_its_shape() {
        let (reference, config) = pipeline("directory_reference");
        let directory = std::env::temp_dir().join("rusting_brain_pipeline_directory");
        publish(&reference, &directory, None, None);

        let loaded = ImagePipeline::load(&directory).unwrap();
        std::fs::remove_dir_all(&directory).ok();

        assert_eq!(
            loaded.denoiser.in_channels(),
            reference.denoiser.in_channels()
        );
        assert_eq!(loaded.denoiser.text_dim(), reference.denoiser.text_dim());
        assert_eq!(loaded.decoder.config(), reference.decoder.config());
        assert_eq!(loaded.encoder.d_model(), denoiser_encoder_config().d_model);
        assert!(matches!(loaded.encoder, PromptEncoder::Sequence { .. }));
        assert!(loaded.pooled.is_none());
        assert_eq!(loaded.config.patch, config.patch);
        assert_eq!(loaded.config.latent_channels, config.latent_channels);
        assert_eq!(loaded.config.shift, 2.5);

        // A published model that bends its schedule by size says so, and the
        // fixed shift is then only the fallback.
        let dynamic = loaded.config.dynamic_shift.expect("the scheduler says so");
        assert_eq!(dynamic.base_seq_len, 256);
        assert!((dynamic.shift(256) - 0.5f32.exp()).abs() < 1e-5);
        assert!((dynamic.shift(4096) - 1.15f32.exp()).abs() < 1e-5);
        assert!(dynamic.shift(1024) > dynamic.shift(256));
    }

    /// Writes a directory shaped the way a Stable Diffusion release is: a
    /// UNet, one CLIP tower, and a beta schedule.
    fn publish_unet(
        directory: &std::path::Path,
        unet: &UnetConfig,
        clip: &crate::clip::ClipTextConfig,
        second: Option<&crate::clip::ClipTextConfig>,
    ) {
        let vae = VaeConfig {
            latent_channels: 4,
            out_channels: 3,
            block_out_channels: vec![8, 16],
            layers_per_block: 1,
            norm_groups: 4,
            ..VaeConfig::default()
        };
        for name in ["unet", "vae", "text_encoder", "tokenizer", "scheduler"] {
            std::fs::create_dir_all(directory.join(name)).unwrap();
        }
        crate::unet::tests::checkpoint(
            unet,
            &directory.join("unet/diffusion_pytorch_model.safetensors"),
        );
        crate::vae::tests::checkpoint(
            &vae,
            &directory.join("vae/diffusion_pytorch_model.safetensors"),
        );
        crate::clip::tests::checkpoint(
            clip,
            &directory.join("text_encoder/model.safetensors"),
            false,
        );
        crate::tokenizer::tests::write(&directory.join("tokenizer/tokenizer.json"), &[], &[], None);

        let write = |name: &str, json: String| {
            std::fs::write(directory.join(name), json).unwrap();
        };
        // An XL-shaped model names the projection its pooled vector and its
        // crop go through; a 1.x one has neither.
        let pooled = match unet.pooled_dim {
            Some(width) => format!(
                r#", "addition_time_embed_dim": {}, "projection_class_embeddings_input_dim": {}"#,
                unet.addition_time_dim,
                width + 6 * unet.addition_time_dim
            ),
            None => String::new(),
        };
        write(
            "unet/config.json",
            format!(
                r#"{{"in_channels": {}, "out_channels": {}, "block_out_channels": {:?},
                    "attention_head_dim": {:?}, "cross_attention_dim": {},
                    "norm_num_groups": {}, "norm_eps": {}{pooled}}}"#,
                unet.in_channels,
                unet.out_channels,
                unet.block_channels,
                unet.heads,
                unet.cross_dim,
                unet.norm_groups,
                unet.eps,
            ),
        );
        write(
            "vae/config.json",
            format!(
                r#"{{"latent_channels": {}, "out_channels": {}, "block_out_channels": {:?},
                    "layers_per_block": {}, "norm_num_groups": {}, "scaling_factor": {},
                    "shift_factor": {}}}"#,
                vae.latent_channels,
                vae.out_channels,
                vae.block_out_channels,
                vae.layers_per_block,
                vae.norm_groups,
                vae.scaling_factor,
                vae.shift_factor,
            ),
        );
        write(
            "text_encoder/config.json",
            format!(
                r#"{{"architectures": ["CLIPTextModel"], "vocab_size": {}, "hidden_size": {},
                    "num_hidden_layers": {}, "num_attention_heads": {}, "intermediate_size": {},
                    "max_position_embeddings": {}, "hidden_act": "quick_gelu",
                    "bos_token_id": {}, "eos_token_id": {}}}"#,
                clip.vocab_size,
                clip.d_model,
                clip.layers,
                clip.num_heads,
                clip.mlp_dim,
                clip.max_positions,
                clip.bos_token,
                clip.eos_token,
            ),
        );
        write(
            "scheduler/scheduler_config.json",
            r#"{"beta_start": 0.00085, "beta_end": 0.012, "num_train_timesteps": 1000,
                "beta_schedule": "scaled_linear"}"#
                .into(),
        );

        // The second tower of an XL-shaped model, which is the one its pooled
        // vector comes from.
        if let Some(second) = second {
            for name in ["text_encoder_2", "tokenizer_2"] {
                std::fs::create_dir_all(directory.join(name)).unwrap();
            }
            crate::clip::tests::checkpoint(
                second,
                &directory.join("text_encoder_2/model.safetensors"),
                true,
            );
            crate::tokenizer::tests::write(
                &directory.join("tokenizer_2/tokenizer.json"),
                &[],
                &[],
                None,
            );
            write(
                "text_encoder_2/config.json",
                format!(
                    r#"{{"architectures": ["CLIPTextModelWithProjection"], "vocab_size": {},
                        "hidden_size": {}, "num_hidden_layers": {}, "num_attention_heads": {},
                        "intermediate_size": {}, "max_position_embeddings": {},
                        "hidden_act": "quick_gelu", "projection_dim": {},
                        "bos_token_id": {}, "eos_token_id": {}}}"#,
                    second.vocab_size,
                    second.d_model,
                    second.layers,
                    second.num_heads,
                    second.mlp_dim,
                    second.max_positions,
                    second.projection_dim.unwrap_or(second.d_model),
                    second.bos_token,
                    second.eos_token,
                ),
            );
        }
    }

    #[test]
    fn a_unet_directory_loads_and_runs_to_an_image() {
        let unet = UnetConfig {
            cross_dim: 16,
            ..crate::unet::tests::tiny()
        };
        let clip = crate::clip::ClipTextConfig {
            vocab_size: 320,
            bos_token: 300,
            eos_token: 301,
            ..crate::clip::tests::tiny()
        };
        let directory = std::env::temp_dir().join("rusting_brain_pipeline_unet");
        publish_unet(&directory, &unet, &clip, None);

        let mut loaded = ImagePipeline::load(&directory).unwrap();
        std::fs::remove_dir_all(&directory).ok();

        assert!(matches!(loaded.denoiser, ImageDenoiser::Unet(_)));
        assert!(matches!(
            loaded.encoder,
            PromptEncoder::Clip { skip: 0, .. }
        ));
        // A UNet reads the latent itself, so nothing is packed.
        assert_eq!(loaded.config.patch, 1);
        assert_eq!(loaded.config.betas, Some((0.00085, 0.012, 1000)));
        assert_eq!(loaded.config.guidance, 7.5);

        loaded.config.steps = 2;
        let image = loaded
            .generate("a lighthouse", 8, 8, Some(4), |_, _| true)
            .unwrap();
        assert_eq!((image.channels, image.height, image.width), (3, 8, 8));
        assert!(image.data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn a_picture_can_start_the_run_instead_of_noise() {
        let unet = UnetConfig {
            cross_dim: 16,
            ..crate::unet::tests::tiny()
        };
        let clip = crate::clip::ClipTextConfig {
            vocab_size: 320,
            bos_token: 300,
            eos_token: 301,
            ..crate::clip::tests::tiny()
        };
        let directory = std::env::temp_dir().join("rusting_brain_pipeline_img2img");
        publish_unet(&directory, &unet, &clip, None);

        let mut loaded = ImagePipeline::load(&directory).unwrap();
        std::fs::remove_dir_all(&directory).ok();

        // The published autoencoder holds both halves, so the encoder is there.
        assert!(loaded.image_encoder.is_some());
        loaded.config.steps = 4;

        let start = FeatureMap::from_vec(
            3,
            8,
            8,
            (0..192).map(|value| (value as f32 * 0.05).sin()).collect(),
        )
        .unwrap();
        let image = loaded
            .generate_from_image("a lighthouse", &start, 0.5, Some(4), |_, _| true)
            .unwrap();
        assert_eq!((image.channels, image.height, image.width), (3, 8, 8));
        assert!(image.data.iter().all(|value| value.is_finite()));

        // No strength at all means no sampling steps, so what comes back is
        // the picture put through the autoencoder and nothing else.
        let latent = loaded
            .pack(
                &loaded
                    .image_encoder
                    .as_ref()
                    .unwrap()
                    .encode(&start)
                    .unwrap(),
            )
            .unwrap();
        let conditioning = loaded.condition("a lighthouse", 8, 8).unwrap();
        let untouched = loaded
            .sample_latent_from(conditioning, None, &latent, 0.0, Some(4), |_, _| true)
            .unwrap();
        assert_eq!(untouched.data, latent.data);

        // A pipeline without the encoder half says so rather than guessing.
        loaded.image_encoder = None;
        assert!(
            loaded
                .generate_from_image("a lighthouse", &start, 0.5, Some(4), |_, _| true)
                .is_err()
        );
    }

    #[test]
    fn two_clip_towers_are_read_side_by_side_the_way_an_xl_model_wants() {
        let clip = crate::clip::ClipTextConfig {
            vocab_size: 320,
            bos_token: 300,
            eos_token: 301,
            ..crate::clip::tests::tiny()
        };
        let second = crate::clip::ClipTextConfig {
            projection_dim: Some(8),
            ..clip.clone()
        };
        let unet = UnetConfig {
            // Side by side, the two towers are twice as wide as one.
            cross_dim: clip.d_model * 2,
            pooled_dim: Some(8),
            ..crate::unet::tests::tiny()
        };
        let directory = std::env::temp_dir().join("rusting_brain_pipeline_xl");
        publish_unet(&directory, &unet, &clip, Some(&second));

        let mut loaded = ImagePipeline::load(&directory).unwrap();
        std::fs::remove_dir_all(&directory).ok();

        assert!(matches!(
            loaded.encoder,
            PromptEncoder::ClipPair { skip: 1, .. }
        ));
        // The pooled vector comes from the second tower's projection, not from
        // a tower attached beside the pipeline.
        assert!(loaded.pooled.is_none());
        assert_eq!(loaded.encoder.pooled_dim(), Some(8));

        let conditioning = loaded.condition("a lighthouse", 8, 8).unwrap();
        assert_eq!(conditioning.text.cols, unet.cross_dim);
        assert_eq!(conditioning.pooled.len(), 8);

        loaded.config.steps = 2;
        let image = loaded
            .generate("a lighthouse", 8, 8, Some(4), |_, _| true)
            .unwrap();
        assert_eq!((image.channels, image.height, image.width), (3, 8, 8));
        assert!(image.data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn a_model_with_two_encoders_pools_through_its_clip_tower() {
        let (reference, _) = pipeline("directory_two");
        let clip = crate::clip::ClipTextConfig {
            vocab_size: 320,
            bos_token: 300,
            eos_token: 301,
            ..crate::clip::tests::tiny()
        };
        let directory = std::env::temp_dir().join("rusting_brain_pipeline_two_encoders");
        publish(&reference, &directory, Some(&clip), None);

        let loaded = ImagePipeline::load(&directory).unwrap();
        std::fs::remove_dir_all(&directory).ok();

        // The states still come from the large encoder, and only the pooled
        // vector changes hands.
        assert_eq!(loaded.encoder.d_model(), reference.encoder.d_model());
        assert_eq!(
            loaded.pooled.as_ref().map(|part| part.encoder.config()),
            Some(&clip)
        );

        let conditioning = loaded.condition("a lighthouse", 32, 32).unwrap();
        let alone = reference.condition("a lighthouse", 32, 32).unwrap();
        assert_eq!(conditioning.pooled.len(), clip.d_model);
        assert_eq!(conditioning.text.data, alone.text.data);
        assert_ne!(conditioning.pooled, alone.pooled);
    }

    #[test]
    fn a_prompt_becomes_an_image_of_the_size_that_was_asked_for() {
        let (mut pipeline, _) = pipeline("size");
        // Two latent pixels per patch and two decoder rungs: sixteen pixels a
        // patch, so a 32x16 image is a 2x1 patch grid.
        assert_eq!(pipeline.upscale(), 2);
        assert_eq!(pipeline.patch_grid(32, 16).unwrap(), (8, 4));

        let mut steps = 0;
        let image = pipeline
            .generate("a small red square", 32, 16, Some(3), |_, _| {
                steps += 1;
                true
            })
            .unwrap();

        assert_eq!(steps, 3);
        assert_eq!(image.channels, 3);
        assert_eq!((image.height, image.width), (16, 32));
        assert!(image.data.iter().all(|value| value.is_finite()));
        assert_eq!(to_rgb8(&image).len(), 32 * 16 * 3);
    }

    #[test]
    fn the_same_seed_repeats_and_a_different_prompt_does_not() {
        let (mut pipeline, _) = pipeline("seed");
        let first = pipeline
            .generate("a cat", 32, 32, Some(11), |_, _| true)
            .unwrap();
        let again = pipeline
            .generate("a cat", 32, 32, Some(11), |_, _| true)
            .unwrap();
        assert_eq!(first.data, again.data);

        let other = pipeline
            .generate("a completely different prompt", 32, 32, Some(11), |_, _| {
                true
            })
            .unwrap();
        assert_ne!(first.data, other.data);

        let seeded = pipeline
            .generate("a cat", 32, 32, Some(12), |_, _| true)
            .unwrap();
        assert_ne!(first.data, seeded.data);
    }

    #[test]
    fn packing_a_latent_and_unpacking_it_are_inverses() {
        let (pipeline, config) = pipeline("pack");
        let latent = FeatureMap::from_vec(
            config.latent_channels,
            4,
            6,
            (0..config.latent_channels * 24)
                .map(|index| index as f32)
                .collect(),
        )
        .unwrap();

        let packed = pipeline.pack(&latent).unwrap();
        assert_eq!(packed.rows, 2 * 3);
        assert_eq!(packed.cols, config.latent_channels * 4);

        let unpacked = pixel_shuffle(&FeatureMap::from_tokens(&packed, 2, 3).unwrap(), 2).unwrap();
        assert_eq!(unpacked, latent);
    }

    #[test]
    fn a_size_that_does_not_divide_into_patches_is_refused() {
        let (mut pipeline, _) = pipeline("refuse");
        assert!(pipeline.patch_grid(30, 32).is_err());
        assert!(pipeline.patch_grid(0, 32).is_err());
        assert!(pipeline.generate("x", 18, 18, None, |_, _| true).is_err());
    }

    #[test]
    fn parts_that_do_not_fit_together_are_refused_at_the_seam() {
        let (pipeline, mut config) = pipeline("seam");
        config.latent_channels = 8;
        let ImagePipeline {
            encoder,
            denoiser,
            decoder,
            ..
        } = pipeline;
        assert!(ImagePipeline::new(encoder, denoiser, decoder, config.clone()).is_err());

        // A denoiser wanting a pooled prompt of a width this encoder does not
        // produce is the other half of the same check.
        let (pooled_pipeline, config) = super::tests::pipeline("seam_pooled");
        let ImagePipeline {
            encoder: wide,
            denoiser,
            decoder,
            ..
        } = pooled_pipeline;
        let tokenizer = match wide {
            PromptEncoder::Sequence { tokenizer, .. } => tokenizer,
            other => unreachable!(
                "the test pipeline is built with one, not {:?}",
                other.d_model()
            ),
        };
        let narrow = TextEncoderConfig {
            d_model: 8,
            head_dim: 2,
            ..denoiser_encoder_config()
        };
        let path = std::env::temp_dir().join("rusting_brain_pipeline_seam_encoder.safetensors");
        crate::text_encoder::tests::checkpoint(&narrow, &path, true);
        let encoder = TextEncoder::load(&path, "model.", narrow).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(denoiser.pooled_dim(), Some(16));
        assert!(
            ImagePipeline::new(
                PromptEncoder::Sequence { tokenizer, encoder },
                denoiser,
                decoder,
                config
            )
            .is_err()
        );
    }
    #[test]
    fn the_scheduler_names_the_solver_the_model_was_published_with() {
        let (reference, _) = pipeline("directory_solver");
        let directory = std::env::temp_dir().join("rusting_brain_pipeline_solver");
        publish(&reference, &directory, None, None);
        std::fs::write(
            directory.join("scheduler/scheduler_config.json"),
            r#"{"_class_name": "DPMSolverMultistepScheduler", "shift": 1.0}"#,
        )
        .unwrap();

        let loaded = ImagePipeline::load(&directory).unwrap();
        std::fs::remove_dir_all(&directory).ok();
        assert_eq!(loaded.config.solver, Solver::DpmPlusPlus2m);
    }

    #[test]
    fn three_encoders_are_read_the_way_stable_diffusion_three_publishes_them() {
        let clip = crate::clip::ClipTextConfig {
            vocab_size: 320,
            bos_token: 300,
            eos_token: 301,
            ..crate::clip::tests::tiny()
        };
        // The two towers are laid side by side and padded out to T5's width,
        // so T5 has to be at least as wide as both of them together.
        let t5 = crate::t5::T5Config {
            d_model: 3 * clip.d_model,
            ..crate::t5::tests::tiny()
        };
        let vae = VaeConfig {
            latent_channels: 4,
            out_channels: 3,
            block_out_channels: vec![8, 16],
            layers_per_block: 1,
            norm_groups: 4,
            scaling_factor: 0.5,
            shift_factor: 0.25,
        };
        let dit = DitConfig {
            in_channels: 16,
            out_channels: 16,
            d_model: 16,
            num_heads: 2,
            double_blocks: 1,
            single_blocks: 1,
            mlp_ratio: 2.0,
            text_dim: t5.d_model,
            pooled_dim: 2 * clip.d_model,
            axes_dim: vec![4, 4],
            theta: 10_000.0,
            guidance: false,
            eps: 1e-6,
        };

        let directory = std::env::temp_dir().join("rusting_brain_pipeline_sd3");
        publish_parts(&dit, &vae, &directory, Some(&clip), Some(&t5), Some(&clip));

        let mut loaded = ImagePipeline::load(&directory).unwrap();
        std::fs::remove_dir_all(&directory).ok();

        assert!(matches!(
            loaded.encoder,
            PromptEncoder::ClipPairAndT5 { length: 6, .. }
        ));
        // The pooled vector comes from both towers, so no separate one is
        // attached beside them.
        assert!(loaded.pooled.is_none());
        assert_eq!(loaded.encoder.d_model(), t5.d_model);
        assert_eq!(loaded.encoder.pooled_dim(), Some(2 * clip.d_model));

        let conditioning = loaded.condition("abc", 8, 8).unwrap();
        assert_eq!(conditioning.text.cols, t5.d_model);
        // The CLIP states sit above the T5 ones in one sequence.
        assert_eq!(conditioning.text.rows, clip.max_positions + 6);
        assert_eq!(conditioning.pooled.len(), 2 * clip.d_model);
        // The padding is real: nothing of the CLIP half reaches past its width.
        assert!(
            conditioning.text.row(0)[2 * clip.d_model..]
                .iter()
                .all(|value| *value == 0.0)
        );

        loaded.config.steps = 1;
        let image = loaded
            .generate("a lighthouse", 8, 8, Some(4), |_, _| true)
            .unwrap();
        assert_eq!((image.channels, image.height, image.width), (3, 8, 8));
        assert!(image.data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn a_t5_encoder_is_read_where_the_directory_holds_one() {
        let (reference, _) = pipeline("directory_t5");
        let clip = crate::clip::ClipTextConfig {
            vocab_size: 320,
            bos_token: 300,
            eos_token: 301,
            ..crate::clip::tests::tiny()
        };
        let t5 = crate::t5::tests::tiny();
        let directory = std::env::temp_dir().join("rusting_brain_pipeline_t5");
        publish(&reference, &directory, Some(&clip), Some(&t5));

        let loaded = ImagePipeline::load(&directory).unwrap();
        std::fs::remove_dir_all(&directory).ok();

        let length = match &loaded.encoder {
            PromptEncoder::T5 { length, .. } => *length,
            _ => panic!("the directory says T5"),
        };
        // The tokenizer's own configuration is where the padded length is
        // written down.
        assert_eq!(length, 6);

        let conditioning = loaded.condition("abc", 32, 32).unwrap();
        assert_eq!(conditioning.text.rows, length);
        assert_eq!(conditioning.text.cols, t5.d_model);
        assert_eq!(conditioning.pooled.len(), clip.d_model);
    }
}
