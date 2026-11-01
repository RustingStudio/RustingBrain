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
use crate::diffusion::{
    Denoiser, Inpaint, SamplingConfig, Scheduler, Solver, noise, sample, sample_from,
};
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
    /// FLUX.2's pipeline ignores the four numbers above and takes the
    /// exponent from a fit over the image size and the step count instead,
    /// `compute_empirical_mu` in diffusers.
    pub empirical: bool,
}

impl DynamicShift {
    /// The shift for an image of this many patches, sampled in `steps`.
    pub fn shift(&self, patches: usize, steps: usize) -> f32 {
        if self.empirical {
            return empirical_mu(patches, steps).exp();
        }
        let span = self.max_seq_len.saturating_sub(self.base_seq_len).max(1) as f32;
        let slope = (self.max_shift - self.base_shift) / span;
        let offset = self.base_shift - slope * self.base_seq_len as f32;
        (slope * patches as f32 + offset).exp()
    }
}

/// diffusers' `compute_empirical_mu` for FLUX.2: two lines in the sequence
/// length, fitted at 10 and at 200 steps, and a line between them in the
/// step count. Long sequences take the 200-step line alone.
fn empirical_mu(patches: usize, steps: usize) -> f32 {
    let (a1, b1) = (8.738_095_24e-05_f64, 1.898_333_33);
    let (a2, b2) = (0.000_169_27_f64, 0.456_666_66);
    let length = patches as f64;
    let at_200 = a2 * length + b2;
    if patches > 4300 {
        return at_200 as f32;
    }
    let at_10 = a1 * length + b1;
    let slope = (at_200 - at_10) / 190.0;
    let offset = at_200 - 200.0 * slope;
    (slope * steps as f64 + offset) as f32
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
    /// How many latent channels it denoises at each position it works over.
    /// An inpainting UNet reads more than that, the mask and the masked
    /// picture after them.
    pub fn in_channels(&self) -> usize {
        match self {
            Self::Transformer(denoiser) => denoiser.config().in_channels,
            Self::Unet(denoiser) => denoiser.config().out_channels,
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
        /// Set for a model that reads its prompt through a chat template and
        /// from several layers at once, which is FLUX.2. Left empty, the
        /// prompt is read bare and from the last layer.
        chat: Option<ChatPrompt>,
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

/// How FLUX.2 reads a prompt through its Qwen3 encoder: wrapped in the
/// chat template with thinking switched off, padded to a fixed length, and
/// taken from the states after several layers, laid side by side per token.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatPrompt {
    /// What comes before the prompt and after it.
    pub prefix: String,
    pub suffix: String,
    /// How many tokens the sequence is truncated or padded to.
    pub length: usize,
    /// The token the padding is made of.
    pub pad: u32,
    /// Which states are read, as how many layers have run: 9 is the state
    /// after the ninth layer. No final norm is applied.
    pub layers: Vec<usize>,
}

impl ChatPrompt {
    /// `Flux2KleinPipeline`'s: Qwen3's template for one user turn and an
    /// empty thought, 512 tokens, the states after layers 9, 18 and 27.
    pub fn flux2(pad: u32) -> Self {
        Self {
            prefix: "<|im_start|>user\n".into(),
            suffix: "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".into(),
            length: 512,
            pad,
            layers: vec![9, 18, 27],
        }
    }

    /// The token ids for `prompt`, padded, and how many of them are real.
    pub fn tokenize(&self, tokenizer: &Bpe, prompt: &str) -> Result<(Vec<u32>, usize), NetworkError> {
        let mut ids = tokenizer.encode(&format!("{}{prompt}{}", self.prefix, self.suffix))?;
        ids.truncate(self.length);
        let real = ids.len();
        ids.resize(self.length, self.pad);
        Ok((ids, real))
    }
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

    /// The T5 encoder this encoder holds, if any.
    #[cfg(feature = "cuda")]
    pub(crate) fn t5(&mut self) -> Option<&mut T5Encoder> {
        match self {
            Self::T5 { encoder, .. } => Some(encoder),
            Self::ClipPairAndT5 { third, .. } => Some(third),
            _ => None,
        }
    }

    /// How wide one hidden state is.
    pub fn d_model(&self) -> usize {
        match self {
            Self::Sequence { encoder, chat, .. } => {
                encoder.config().d_model * chat.as_ref().map_or(1, |chat| chat.layers.len())
            }
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
            // FLUX.2 pools nothing: its denoiser takes no pooled vector.
            Self::Sequence { chat: Some(_), .. } => None,
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
            Self::Sequence {
                tokenizer,
                encoder,
                chat: Some(chat),
            } => {
                let (ids, real) = chat.tokenize(tokenizer, prompt)?;
                Ok((encoder.forward_layers(&ids, real, &chat.layers)?, None))
            }
            Self::Sequence {
                tokenizer, encoder, ..
            } => {
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
        Self::load_with_loras(directory, precision, &[])
    }

    /// The same, with LoRA files merged into the denoiser and the CLIP text
    /// encoders at the strength paired with each; see
    /// [`crate::safetensors::ShardedSafeTensors::add_lora`]. A file must adapt
    /// the denoiser.
    pub fn load_with_loras(
        directory: impl AsRef<std::path::Path>,
        precision: Precision,
        loras: &[(std::path::PathBuf, f32)],
    ) -> Result<Self, NetworkError> {
        let directory = directory.as_ref();
        let part = |name: &str| directory.join(name);

        let decoder_config = crate::vae::VaeConfig::from_file(part("vae/config.json"))?;
        let scheduler = read_json(&part("scheduler/scheduler_config.json"));
        // FLUX.2 is told apart by its pipeline's name: its prompt, schedule
        // and latent all differ from FLUX.1's in ways no part's own
        // configuration records.
        let flux2 = read_json(&part("model_index.json")).is_some_and(|json| {
            json.get("_class_name")
                .and_then(|name| name.as_str())
                .is_some_and(|name| name.starts_with("Flux2"))
        });
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
            dynamic_shift: scheduler.as_ref().and_then(dynamic_shift).map(|dynamic| {
                DynamicShift {
                    empirical: flux2,
                    ..dynamic
                }
            }),
            // The Stable Diffusion line predicts noise along a beta schedule
            // and was trained with the prompt dropped part of the time, so it
            // is sampled with guidance over more steps than a distilled model
            // needs.
            betas: unet.then(|| ddim_betas(scheduler.as_ref())),
            // A published scheduler names the solver it was tuned with.
            solver: scheduler.as_ref().map_or(defaults.solver, solver_named),
            // klein is distilled to four steps.
            steps: match (unet, flux2) {
                (true, _) => 50,
                (false, true) => 4,
                (false, false) => defaults.steps,
            },
            guidance: match unet {
                true => 7.5,
                false => defaults.guidance,
            },
            ..defaults
        };

        let tower = |name: &str| -> Result<ClipTextEncoder, NetworkError> {
            let mut file = crate::safetensors::ShardedSafeTensors::open(weights(&part(name))?)?;
            // kohya numbers XL's two towers and leaves 1.x's unnumbered.
            let prefixes: &[&str] = match name {
                "text_encoder" => &[name, "lora_te", "lora_te1"],
                _ => &[name, "lora_te2"],
            };
            for (path, scale) in loras {
                file.add_lora(path, prefixes, *scale)?;
            }
            ClipTextEncoder::read_at(
                &mut file,
                "",
                crate::clip::ClipTextConfig::from_file(part(&format!("{name}/config.json")))?,
                precision,
            )
        };
        // A CLIP tower and its tokenizer, padding with whatever token the
        // tokenizer names: the UNet reads the padded positions too.
        let clip_tower =
            |words: &str, name: &str| -> Result<(Bpe, ClipTextEncoder), NetworkError> {
                let tokenizer = Bpe::from_file(part(&format!("{words}/tokenizer.json")))?;
                let mut encoder = tower(name)?;
                let pad =
                    read_json(&part(&format!("{words}/tokenizer_config.json"))).and_then(|json| {
                        let pad = json.get("pad_token")?;
                        tokenizer.token_id(pad.get("content").unwrap_or(pad).as_str()?)
                    });
                if let Some(pad) = pad {
                    encoder.set_pad_token(pad);
                }
                Ok((tokenizer, encoder))
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
            _ if three => {
                let (tokenizer, encoder) = clip_tower("tokenizer", "text_encoder")?;
                let (second_tokenizer, second) = clip_tower("tokenizer_2", "text_encoder_2")?;
                PromptEncoder::ClipPairAndT5 {
                    tokenizer,
                    encoder,
                    second_tokenizer,
                    second,
                    skip: 1,
                    third_tokenizer: Unigram::from_file(part("tokenizer_3/tokenizer.json"))?,
                    third: crate::t5::T5Encoder::load_at(
                        weights(&part(states))?,
                        crate::t5::T5Config::from_file(&states_config)?,
                        precision,
                    )?,
                    length: padded_to(&part("tokenizer_3")),
                }
            }
            // Stable Diffusion XL: two towers, read second to last because
            // that is the layer it was trained against.
            (true, true) => {
                let (tokenizer, encoder) = clip_tower("tokenizer", "text_encoder")?;
                let (second_tokenizer, second) = clip_tower("tokenizer_2", states)?;
                PromptEncoder::ClipPair {
                    tokenizer,
                    encoder,
                    second_tokenizer,
                    second,
                    skip: 1,
                }
            }
            (true, false) => {
                let (tokenizer, encoder) = clip_tower("tokenizer", "text_encoder")?;
                PromptEncoder::Clip {
                    tokenizer,
                    encoder,
                    skip: 0,
                }
            }
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
                        let tokenizer = Bpe::from_file(words.join("tokenizer.json"))?;
                        let mut encoder_config =
                            crate::text_encoder::TextEncoderConfig::from_file(&states_config)?;
                        let chat = match flux2 {
                            false => None,
                            true => {
                                let pad = read_json(&words.join("tokenizer_config.json"))
                                    .and_then(|json| {
                                        let pad = json.get("pad_token")?;
                                        tokenizer
                                            .token_id(pad.get("content").unwrap_or(pad).as_str()?)
                                    })
                                    .ok_or_else(|| {
                                        NetworkError::InvalidConfig(
                                            "FLUX.2's tokenizer names no padding token".into(),
                                        )
                                    })?;
                                Some(ChatPrompt::flux2(pad))
                            }
                        };
                        // Layers past the deepest one read are never run, so
                        // they are not loaded either.
                        if let Some(deepest) =
                            chat.as_ref().and_then(|chat| chat.layers.iter().max())
                        {
                            encoder_config.layers = encoder_config.layers.min(*deepest);
                        }
                        PromptEncoder::Sequence {
                            tokenizer,
                            encoder: TextEncoder::read_at(
                                &mut encoder_file,
                                prefix,
                                encoder_config,
                                precision,
                            )?,
                            chat,
                        }
                    }
                };
                drop(encoder_file);
                encoder
            }
        };

        let denoiser_file = |name: &str| {
            let mut file = crate::safetensors::ShardedSafeTensors::open(weights(&part(name))?)?;
            // kohya calls every denoiser, FLUX's DiT too, `lora_unet`.
            let prefixes = [name, &format!("lora_{name}"), "lora_unet", ""];
            for (path, scale) in loras {
                if file.add_lora(path, &prefixes, *scale)? == 0 {
                    return Err(NetworkError::InvalidDataset(format!(
                        "{} adapts nothing in the {name}",
                        path.display()
                    )));
                }
            }
            Ok::<_, NetworkError>(file)
        };
        let denoiser: ImageDenoiser = match denoiser_config {
            Some(denoiser_config) => Dit::read_at(
                &mut denoiser_file("transformer")?,
                "",
                denoiser_config,
                precision,
            )?
            .into(),
            None => {
                let mut model = Unet::read_at(
                    &mut denoiser_file("unet")?,
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

    /// Adds a textual inversion file as `token`, which prompts can then use
    /// like a word. Every CLIP tower the pipeline reads with, the pooled one
    /// included, takes the file's tensor as wide as it is; see
    /// [`crate::clip::read_embedding`].
    pub fn add_embedding(
        &mut self,
        path: impl AsRef<std::path::Path>,
        token: &str,
    ) -> Result<(), NetworkError> {
        let mut towers = match &mut self.encoder {
            PromptEncoder::Sequence { .. } | PromptEncoder::T5 { .. } => Vec::new(),
            PromptEncoder::Clip {
                tokenizer, encoder, ..
            } => vec![(tokenizer, encoder)],
            PromptEncoder::ClipPair {
                tokenizer,
                encoder,
                second_tokenizer,
                second,
                ..
            }
            | PromptEncoder::ClipPairAndT5 {
                tokenizer,
                encoder,
                second_tokenizer,
                second,
                ..
            } => {
                vec![(tokenizer, encoder), (second_tokenizer, second)]
            }
        };
        if let Some(part) = &mut self.pooled {
            towers.push((&mut part.tokenizer, &mut part.encoder));
        }
        if towers.is_empty() {
            return Err(NetworkError::InvalidConfig(
                "textual inversion needs a CLIP text encoder".into(),
            ));
        }
        // Every tower's vectors are read before any is added, so a file that
        // fits one tower and not another changes nothing.
        let vectors = towers
            .iter()
            .map(|(_, encoder)| crate::clip::read_embedding(&path, encoder.config().d_model))
            .collect::<Result<Vec<_>, _>>()?;
        for ((tokenizer, encoder), vectors) in towers.into_iter().zip(vectors) {
            encoder.add_token(tokenizer, token, &vectors)?;
        }
        Ok(())
    }

    /// Moves everything that has a device path onto CUDA device `device`.
    ///
    /// That is the denoiser, both halves of the VAE, every CLIP tower, a gated
    /// T5 encoder and a LLaMA-family one (FLUX.2's Qwen3) where it fits. The
    /// UNet runs on the FP16 image kernels; a transformer denoiser and both
    /// language-model encoders run on the BF16 ones, because T5 1.1's
    /// activations pass FP16's range. The original ungated T5 stays on the
    /// host.
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
        if let Some(encoder) = &mut self.image_encoder {
            encoder.attach_device(gpu.clone())?;
        }
        for tower in self.encoder.towers() {
            tower.attach_device(gpu.clone())?;
        }
        if let Some(pooled) = &mut self.pooled {
            pooled.encoder.attach_device(gpu)?;
        }
        let sequence = matches!(self.encoder, PromptEncoder::Sequence { .. });
        let t5 = self.encoder.t5().filter(|t5| t5.config().gated);
        if sequence || t5.is_some() || matches!(self.denoiser, ImageDenoiser::Transformer(_)) {
            let context = crate::gpu_transformer::GpuContext::new(device)?;
            if let Some(t5) = t5 {
                t5.attach_device(&context)?;
            }
            if let ImageDenoiser::Transformer(dit) = &mut self.denoiser {
                dit.attach_device(&context)?;
            }
            // After the denoiser, and allowed to fail: a prompt encoder that
            // does not fit beside it (FLUX.2-klein's Qwen3 in BF16 on a 12 GB
            // card) or has no device path stays on the host, where it worked.
            if let PromptEncoder::Sequence { encoder, .. } = &mut self.encoder {
                encoder.attach_device(&context).ok();
            }
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

    /// Whether a sequence prompt encoder runs on the device.
    ///
    /// [`ImagePipeline::to_cuda`] succeeds even when this encoder does not fit
    /// beside the denoiser and stays on the host, where FLUX.2-klein's prompt
    /// encode takes about 12 s instead of 0.7 s on an RTX 3060. This is how a
    /// caller finds out. `false` for every other encoder and without the
    /// `cuda` feature.
    pub fn encoder_on_device(&self) -> bool {
        #[cfg(feature = "cuda")]
        if let PromptEncoder::Sequence { encoder, .. } = &self.encoder {
            return encoder.on_device();
        }
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
        if width == 0
            || height == 0
            || !width.is_multiple_of(block)
            || !height.is_multiple_of(block)
        {
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
        let start = noise(patches, self.denoiser.in_channels(), seed);
        self.sample_latent_from_noise(conditioning, unconditional, start, seed, on_step)
    }

    /// [`ImagePipeline::sample_latent_guided`] from noise the caller drew,
    /// packed as the denoiser reads it and at unit scale: the run scales it
    /// to the schedule's first level. This is how a run is lined up with
    /// another implementation's, which draws its noise its own way.
    pub fn sample_latent_from_noise(
        &mut self,
        conditioning: Conditioning,
        unconditional: Option<Conditioning>,
        mut start: Matrix,
        seed: Option<u64>,
        on_step: impl FnMut(usize, &Matrix) -> bool,
    ) -> Result<Matrix, NetworkError> {
        let patches = conditioning.height * conditioning.width;
        let channels = self.denoiser.in_channels();
        if start.rows != patches || start.cols != channels {
            return Err(NetworkError::InvalidConfig(format!(
                "the noise is {}x{} and the run wants {patches}x{channels}",
                start.rows, start.cols
            )));
        }
        self.denoiser.set_conditioning(conditioning);
        if let Some(unconditional) = unconditional {
            self.denoiser.set_unconditional(unconditional);
        }

        let scheduler = self.scheduler(patches);
        // A run starts at the noisiest level the schedule holds, which is one
        // for flow matching and far above it for DDIM.
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
                Some(dynamic) => dynamic.shift(patches, self.config.steps),
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
        self.sample_masked(
            conditioning,
            unconditional,
            latent,
            None,
            strength,
            seed,
            on_step,
        )
    }

    /// Repaints the rows of `latent` where `keep` is 0.0 and keeps the
    /// picture where it is 1.0, which is inpainting. `keep` holds one weight
    /// a packed latent row, a patch of the picture, and a value between
    /// blends; see [`Inpaint`]. `strength` is as in
    /// [`ImagePipeline::sample_latent_from`]: 1.0 repaints from pure noise,
    /// less starts from the picture already there.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_latent_inpaint(
        &mut self,
        conditioning: Conditioning,
        unconditional: Option<Conditioning>,
        latent: &Matrix,
        keep: &[f32],
        strength: f32,
        seed: Option<u64>,
        on_step: impl FnMut(usize, &Matrix) -> bool,
    ) -> Result<Matrix, NetworkError> {
        let keep = Some(keep);
        self.sample_masked(
            conditioning,
            unconditional,
            latent,
            keep,
            strength,
            seed,
            on_step,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_masked(
        &mut self,
        conditioning: Conditioning,
        unconditional: Option<Conditioning>,
        latent: &Matrix,
        keep: Option<&[f32]>,
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
        match keep {
            Some(keep) => {
                let mut inpaint = Inpaint {
                    denoiser: &mut self.denoiser,
                    image: latent,
                    keep,
                };
                sample_from(&mut inpaint, scheduler, start, &settings, first, on_step)
            }
            None => sample_from(
                &mut self.denoiser,
                scheduler,
                start,
                &settings,
                first,
                on_step,
            ),
        }
    }

    /// Unpacks a sampled latent and decodes it to an image in `[-1, 1]`.
    pub fn decode(
        &self,
        latent: &Matrix,
        columns: usize,
        rows: usize,
    ) -> Result<FeatureMap, NetworkError> {
        let packed = match self.decoder.patch_norm() {
            // FLUX.2's denoiser works on batch-normalised patches, so the
            // statistics are put back before the autoencoder sees them.
            Some((mean, std)) => {
                let mut latent = latent.clone();
                normalise_patches(&mut latent, mean, std, false)?;
                FeatureMap::from_tokens(&latent, rows, columns)?
            }
            None => FeatureMap::from_tokens(latent, rows, columns)?,
        };
        let unpacked = pixel_shuffle(&packed, self.config.patch)?;
        self.decoder.decode(&unpacked)
    }

    /// Packs a latent into the patch grid a denoiser reads, which is the
    /// inverse of what [`ImagePipeline::decode`] undoes.
    pub fn pack(&self, latent: &FeatureMap) -> Result<Matrix, NetworkError> {
        let mut packed = pixel_unshuffle(latent, self.config.patch)?.to_tokens();
        if let Some((mean, std)) = self.decoder.patch_norm() {
            normalise_patches(&mut packed, mean, std, true)?;
        }
        Ok(packed)
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

    /// Repaints `image` where `keep` is 0.0 and keeps it where it is 1.0,
    /// which is inpainting. `keep` is one channel at the image's size;
    /// `strength` is as in [`ImagePipeline::generate_from_image`], and 1.0
    /// repaints from pure noise.
    ///
    /// A checkpoint trained for inpainting (a UNet reading more channels than
    /// it predicts) is handed the mask and the latent of the picture with the
    /// repainted part blanked, the way diffusers does it. Any other model is
    /// steered through [`Inpaint`], with the mask averaged over each latent
    /// cell so its edge stays soft.
    pub fn generate_inpaint(
        &mut self,
        prompt: &str,
        image: &FeatureMap,
        keep: &FeatureMap,
        strength: f32,
        seed: Option<u64>,
        on_step: impl FnMut(usize, &Matrix) -> bool,
    ) -> Result<FeatureMap, NetworkError> {
        if (keep.channels, keep.height, keep.width) != (1, image.height, image.width) {
            return Err(NetworkError::InvalidConfig(format!(
                "the mask is {}x{}x{} and the picture {}x{}: it wants one channel at the \
                 picture's size",
                keep.channels, keep.height, keep.width, image.height, image.width
            )));
        }
        let encoder = self.image_encoder.as_ref().ok_or_else(|| {
            NetworkError::InvalidConfig(
                "this checkpoint's autoencoder holds no encoder, so it cannot inpaint".into(),
            )
        })?;
        let encoded = encoder.encode(image)?;
        let latent = self.pack(&encoded)?;
        let trained = matches!(&self.denoiser, ImageDenoiser::Unet(unet)
            if unet.config().in_channels > unet.config().out_channels);
        // The mask on the latent's own grid: one value a packed row, which a
        // UNet's latent position and a transformer's patch both are.
        let conditioning = self.condition(prompt, image.width, image.height)?;
        let (columns, rows) = (conditioning.width, conditioning.height);
        let (step_y, step_x) = (keep.height / rows, keep.width / columns);
        let cells = (0..rows * columns).map(|cell| {
            let (y, x) = (cell / columns * step_y, cell % columns * step_x);
            (0..step_y).flat_map(move |dy| (0..step_x).map(move |dx| (y + dy, x + dx)))
        });
        let unconditional = match self.config.guidance != 1.0 {
            true => Some(self.condition("", image.width, image.height)?),
            false => None,
        };

        let sampled = if trained {
            // Binarized and sampled at each cell's corner, as diffusers
            // resizes the mask; 1.0 marks what to repaint.
            let mut extra = FeatureMap::new(1, rows, columns);
            for (value, mut cell) in extra.data.iter_mut().zip(cells) {
                let (y, x) = cell.next().expect("a cell holds a pixel");
                *value = if keep.data[y * keep.width + x] < 0.5 {
                    1.0
                } else {
                    0.0
                };
            }
            let mut blanked = image.clone();
            let plane = image.height * image.width;
            for (index, value) in blanked.data.iter_mut().enumerate() {
                if keep.data[index % plane] < 0.5 {
                    *value = 0.0;
                }
            }
            let encoder = self.image_encoder.as_ref().expect("checked above");
            let extra = crate::unet::concatenate(&extra, &encoder.encode(&blanked)?);
            let ImageDenoiser::Unet(unet) = &mut self.denoiser else {
                unreachable!("only a UNet reads the mask")
            };
            unet.set_inpaint(Some(extra));
            let sampled = self.sample_latent_from(
                conditioning,
                unconditional,
                &latent,
                strength,
                seed,
                on_step,
            );
            if let ImageDenoiser::Unet(unet) = &mut self.denoiser {
                unet.set_inpaint(None);
            }
            sampled
        } else {
            let pixels = (step_y * step_x) as f32;
            let weights: Vec<f32> = cells
                .map(|cell| {
                    cell.map(|(y, x)| keep.data[y * keep.width + x])
                        .sum::<f32>()
                        / pixels
                })
                .collect();
            self.sample_latent_inpaint(
                conditioning,
                unconditional,
                &latent,
                &weights,
                strength,
                seed,
                on_step,
            )
        }?;
        self.decode(&sampled, columns, rows)
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

/// `(x - mean) / std` per packed column when `forward`, and its inverse
/// otherwise: the batch norm FLUX.2's autoencoder keeps over patches.
fn normalise_patches(
    latent: &mut Matrix,
    mean: &[f32],
    std: &[f32],
    forward: bool,
) -> Result<(), NetworkError> {
    if mean.len() != latent.cols || std.len() != latent.cols {
        return Err(NetworkError::InvalidConfig(format!(
            "the autoencoder normalises {}-channel patches and the latent has {}",
            mean.len(),
            latent.cols
        )));
    }
    for row in latent.data.chunks_mut(mean.len()) {
        for ((value, mean), std) in row.iter_mut().zip(mean).zip(std) {
            *value = match forward {
                true => (*value - mean) / std,
                false => *value * std + mean,
            };
        }
    }
    Ok(())
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
        empirical: false,
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
            batch_norm_eps: 1e-4,
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
                PromptEncoder::Sequence {
                    tokenizer,
                    encoder,
                    chat: None,
                },
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
        assert!((dynamic.shift(256, 28) - 0.5f32.exp()).abs() < 1e-5);
        assert!((dynamic.shift(4096, 28) - 1.15f32.exp()).abs() < 1e-5);
        assert!(dynamic.shift(1024, 28) > dynamic.shift(256, 28));
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
    fn inpainting_runs_through_a_trained_unet_and_any_other_model() {
        let clip = crate::clip::ClipTextConfig {
            vocab_size: 320,
            bos_token: 300,
            eos_token: 301,
            ..crate::clip::tests::tiny()
        };
        let picture = FeatureMap::from_vec(
            3,
            8,
            8,
            (0..192).map(|value| (value as f32 * 0.05).sin()).collect(),
        )
        .unwrap();
        let mut keep = FeatureMap::new(1, 8, 8);
        keep.data[32..].fill(1.0);
        // A plain UNet, then one trained for inpainting with the mask and the
        // masked picture's four channels after the latent.
        for in_channels in [4, 9] {
            let unet = UnetConfig {
                cross_dim: 16,
                in_channels,
                ..crate::unet::tests::tiny()
            };
            let directory =
                std::env::temp_dir().join(format!("rusting_brain_inpaint_{in_channels}"));
            publish_unet(&directory, &unet, &clip, None);
            let mut loaded = ImagePipeline::load(&directory).unwrap();
            std::fs::remove_dir_all(&directory).ok();
            loaded.config.steps = 2;

            let image = loaded
                .generate_inpaint("a lighthouse", &picture, &keep, 1.0, Some(4), |_, _| true)
                .unwrap();
            assert_eq!((image.channels, image.height, image.width), (3, 8, 8));
            assert!(image.data.iter().all(|value| value.is_finite()));
            let wrong = FeatureMap::new(1, 4, 4);
            assert!(
                loaded
                    .generate_inpaint("a lighthouse", &picture, &wrong, 1.0, None, |_, _| true)
                    .is_err()
            );
            // The trained model's mask goes with the run, so a plain run after
            // it has none and says so.
            let plain = loaded.generate("a lighthouse", 8, 8, Some(4), |_, _| true);
            assert_eq!(plain.is_err(), in_channels == 9);
        }
    }

    #[test]
    fn a_lora_file_changes_the_models_it_is_loaded_into() {
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
        let directory = std::env::temp_dir().join("rusting_brain_pipeline_lora");
        publish_unet(&directory, &unet, &clip, None);
        // Rank-one adapters on `conv_in` and on the text encoder's first
        // feed-forward layer.
        let file = |name: &str, adapters: &[(&str, usize, usize)]| {
            let path = directory.join(name);
            let tensors = adapters
                .iter()
                .flat_map(|(stem, inputs, outputs)| {
                    [
                        (
                            format!("{stem}.lora_down.weight"),
                            vec![1, *inputs],
                            vec![0.1; *inputs],
                        ),
                        (
                            format!("{stem}.lora_up.weight"),
                            vec![*outputs, 1],
                            vec![0.5; *outputs],
                        ),
                    ]
                })
                .map(|(name, shape, values)| (name, (shape, values)))
                .collect();
            crate::safetensors::write_checkpoint(&path, &tensors);
            path
        };
        let conv_in = ("lora_unet_conv_in", 36, 8);
        let fc1 = ("lora_te_text_model_encoder_layers_0_mlp_fc1", 16, 32);
        let unet_only = file("unet.safetensors", &[conv_in]);
        let both = file("both.safetensors", &[conv_in, fc1]);
        let text_only = file("text.safetensors", &[fc1]);
        let run = |loras: &[(std::path::PathBuf, f32)]| {
            let mut loaded = ImagePipeline::load_with_loras(&directory, Precision::F32, loras)?;
            loaded.config.steps = 2;
            loaded.generate("a lighthouse", 8, 8, Some(4), |_, _| true)
        };
        let plain = run(&[]).unwrap();
        assert_eq!(run(&[(both.clone(), 0.0)]).unwrap().data, plain.data);
        let steered = run(&[(unet_only, 1.0)]).unwrap();
        assert_ne!(steered.data, plain.data);
        // The text encoder's half changes the prompt, and so the picture.
        assert_ne!(run(&[(both, 1.0)]).unwrap().data, steered.data);
        assert!(run(&[(text_only, 1.0)]).is_err());
        std::fs::remove_dir_all(&directory).ok();
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

        // Inpainting from pure noise keeps the rows it is told to keep and
        // repaints the others.
        let mut keep = vec![1.0; latent.rows];
        keep[0] = 0.0;
        let conditioning = loaded.condition("a lighthouse", 8, 8).unwrap();
        let painted = loaded
            .sample_latent_inpaint(conditioning, None, &latent, &keep, 1.0, Some(4), |_, _| {
                true
            })
            .unwrap();
        for (row, &keep) in keep.iter().enumerate() {
            let same = painted
                .row(row)
                .iter()
                .zip(latent.row(row))
                .all(|(a, b)| (a - b).abs() < 1e-4);
            assert_eq!(same, keep == 1.0, "row {row}");
        }

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
        // XL's second tokenizer pads with `!`, and its UNet reads the padding.
        std::fs::write(
            directory.join("tokenizer_2/tokenizer_config.json"),
            r#"{"pad_token": "!"}"#,
        )
        .unwrap();

        let mut loaded = ImagePipeline::load(&directory).unwrap();
        std::fs::remove_dir_all(&directory).ok();

        let PromptEncoder::ClipPair {
            tokenizer,
            encoder,
            second_tokenizer,
            second,
            skip: 1,
        } = &loaded.encoder
        else {
            panic!("an XL model reads two CLIP towers");
        };
        let pad = second_tokenizer.token_id("!").unwrap();
        let ids = second.tokenize(second_tokenizer, "").unwrap();
        assert!(ids[2..].iter().all(|id| *id == pad));
        assert_eq!(
            *encoder.tokenize(tokenizer, "").unwrap().last().unwrap(),
            clip.eos_token
        );
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
    fn a_host_pipeline_says_its_encoder_is_not_on_the_device() {
        let (pipeline, _) = pipeline("encoder_host");
        assert!(matches!(pipeline.encoder, PromptEncoder::Sequence { .. }));
        assert!(!pipeline.encoder_on_device());
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
                PromptEncoder::Sequence {
                    tokenizer,
                    encoder,
                    chat: None,
                },
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

    /// The parts of a tiny Stable Diffusion 3: two CLIP towers, a T5 encoder,
    /// a VAE and a denoiser of the given width, heads and rotary axes.
    fn sd3_parts(
        d_model: usize,
        num_heads: usize,
        axes_dim: Vec<usize>,
    ) -> (
        crate::clip::ClipTextConfig,
        crate::t5::T5Config,
        VaeConfig,
        DitConfig,
    ) {
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
            batch_norm_eps: 1e-4,
        };
        let dit = DitConfig {
            in_channels: 16,
            out_channels: 16,
            d_model,
            num_heads,
            double_blocks: 1,
            single_blocks: 1,
            mlp_ratio: 2.0,
            text_dim: t5.d_model,
            pooled_dim: 2 * clip.d_model,
            axes_dim,
            theta: 10_000.0,
            guidance: false,
            eps: 1e-6,
        };
        (clip, t5, vae, dit)
    }

    #[test]
    fn three_encoders_are_read_the_way_stable_diffusion_three_publishes_them() {
        let (clip, t5, vae, dit) = sd3_parts(16, 2, vec![4, 4]);

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

    /// A Stable Diffusion 3 pipeline on the device: T5 and the denoiser on
    /// the BF16 kernels, the CLIP towers and the decoder on the FP16 ones.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_transformer_pipeline_on_the_device_matches_the_host_or_skips_without_device() {
        if crate::cuda_training::cuda_doctor(0, 8192).is_err() {
            crate::cuda_training::skip_without_cuda();
            return;
        }
        // Heads of 64, which the fused attention takes.
        let (clip, t5, vae, dit) = sd3_parts(128, 2, vec![32, 32]);
        let directory = std::env::temp_dir().join("rusting_brain_pipeline_sd3_cuda");
        publish_parts(&dit, &vae, &directory, Some(&clip), Some(&t5), Some(&clip));
        let mut host = ImagePipeline::load(&directory).unwrap();
        let mut device = ImagePipeline::load(&directory).unwrap();
        std::fs::remove_dir_all(&directory).ok();
        device.to_cuda(0).unwrap();
        assert!(matches!(&device.denoiser, ImageDenoiser::Transformer(dit) if dit.on_device()));

        let relative = |actual: &[f32], expected: &[f32]| {
            let error: f32 = actual
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).powi(2))
                .sum::<f32>()
                .sqrt();
            error / expected.iter().map(|v| v * v).sum::<f32>().sqrt()
        };
        let expected = host.condition("abc", 8, 8).unwrap();
        let actual = device.condition("abc", 8, 8).unwrap();
        let error = relative(&actual.text.data, &expected.text.data);
        // BF16 never reproduces the host bit for bit, so an exact match means
        // the host path ran.
        assert!(
            0.0 < error && error < 2e-2,
            "prompt states: relative error {error}"
        );

        host.config.steps = 2;
        device.config.steps = 2;
        let expected = host
            .generate("a lighthouse", 8, 8, Some(4), |_, _| true)
            .unwrap();
        let actual = device
            .generate("a lighthouse", 8, 8, Some(4), |_, _| true)
            .unwrap();
        let error = relative(&actual.data, &expected.data);
        assert!(0.0 < error && error < 5e-2, "image: relative error {error}");
    }

    /// Writes a Stable Diffusion 1.x-shaped pipeline (one CLIP tower, a
    /// two-level UNet, an autoencoder with its encoder) into `RB_TINY_SD`:
    /// the model `scripts/unet_lora_trained_check.py` swaps a diffusers UNet
    /// into and runs `train_unet_lora` on.
    #[test]
    #[ignore]
    fn publish_a_tiny_sd() {
        let directory = std::path::PathBuf::from(std::env::var("RB_TINY_SD").unwrap());
        let clip = crate::clip::ClipTextConfig {
            vocab_size: 320,
            bos_token: 300,
            eos_token: 301,
            ..crate::clip::tests::tiny()
        };
        let unet = UnetConfig {
            cross_dim: clip.d_model,
            ..crate::unet::tests::tiny()
        };
        publish_unet(&directory, &unet, &clip, None);
        let loaded = ImagePipeline::load(&directory).unwrap();
        assert!(loaded.image_encoder.is_some());
    }

    /// Writes a FLUX.1-shaped pipeline (CLIP and T5, two of each block kind)
    /// with 64-wide heads, which the device DiT takes, into `RB_TINY_FLUX`:
    /// the model `scripts/dit_lora_trained_check.py` runs `train_dit_lora` on.
    #[test]
    #[ignore]
    fn publish_a_tiny_flux() {
        let directory = std::path::PathBuf::from(std::env::var("RB_TINY_FLUX").unwrap());
        let clip = crate::clip::ClipTextConfig {
            vocab_size: 320,
            bos_token: 300,
            eos_token: 301,
            ..crate::clip::tests::tiny()
        };
        let t5 = crate::t5::tests::tiny();
        let (_, _, vae, dit) = sd3_parts(128, 2, vec![16, 24, 24]);
        let dit = DitConfig {
            double_blocks: 2,
            single_blocks: 2,
            mlp_ratio: 4.0,
            text_dim: t5.d_model,
            pooled_dim: clip.d_model,
            ..dit
        };
        publish_parts(&dit, &vae, &directory, Some(&clip), Some(&t5), None);
        ImagePipeline::load(&directory).unwrap();
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

    #[test]
    fn flux2_takes_its_shift_from_the_empirical_fit() {
        let dynamic = DynamicShift {
            base_shift: 0.5,
            max_shift: 1.15,
            base_seq_len: 256,
            max_seq_len: 4096,
            empirical: true,
        };
        // diffusers' `compute_empirical_mu(1024, 4)`, and the levels its
        // scheduler builds from it for a 512x512 image in four steps.
        assert!((empirical_mu(1024, 4) - 2.030_689_7).abs() < 1e-5);
        let sigmas = Scheduler::flow_match(dynamic.shift(1024, 4))
            .sigmas(4)
            .unwrap();
        for (got, want) in sigmas
            .iter()
            .zip([1.0, 0.958_085_4, 0.883_982, 0.717_496_6, 0.0])
        {
            assert!((got - want).abs() < 1e-5, "{sigmas:?}");
        }
        // Past 4300 patches the 200-step line holds alone, whatever the
        // step count.
        let long = (0.000_169_27 * 5000.0 + 0.456_666_66) as f32;
        assert!((empirical_mu(5000, 4) - long).abs() < 1e-6);
        assert_eq!(empirical_mu(5000, 4), empirical_mu(5000, 50));
        // The fitted lines are not the configured ones.
        let configured = DynamicShift {
            empirical: false,
            ..dynamic
        };
        assert!((configured.shift(1024, 4).ln() - 2.030_689_7).abs() > 0.1);
    }

    #[test]
    fn patch_statistics_are_taken_out_and_put_back() {
        let mut latent = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (mean, std) = ([1.0, -1.0, 0.5], [2.0, 0.5, 1.0]);
        normalise_patches(&mut latent, &mean, &std, true).unwrap();
        assert_eq!(latent.data[..3], [0.0, 6.0, 2.5]);
        normalise_patches(&mut latent, &mean, &std, false).unwrap();
        assert_eq!(latent.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(normalise_patches(&mut latent, &mean[..2], &std[..2], true).is_err());
    }

    #[test]
    fn a_chat_prompt_is_wrapped_padded_and_counted() {
        let tokenizer = crate::tokenizer::tests::tokenizer(&[], &[], None);
        let chat = ChatPrompt {
            length: 128,
            ..ChatPrompt::flux2(7)
        };
        let wrapped = tokenizer
            .encode(&format!("{}fog{}", chat.prefix, chat.suffix))
            .unwrap();
        let (ids, real) = chat.tokenize(&tokenizer, "fog").unwrap();
        assert_eq!(ids.len(), 128);
        assert_eq!(real, wrapped.len());
        assert_eq!(ids[..real], wrapped[..]);
        assert!(ids[real..].iter().all(|&id| id == 7));
        // Too long a prompt is cut, and every token left is real.
        let short = ChatPrompt { length: 5, ..chat };
        let (ids, real) = short.tokenize(&tokenizer, "fog").unwrap();
        assert_eq!((ids.as_slice(), real), (&wrapped[..5], 5));
    }

    /// Relative L2 distance of `got` from `want`.
    fn relative(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len());
        let (gap, norm) = got.iter().zip(want).fold((0.0f64, 0.0f64), |(g, n), (a, b)| {
            (g + ((a - b) as f64).powi(2), n + (*b as f64).powi(2))
        });
        (gap / norm).sqrt() as f32
    }

    /// FLUX.2-klein-4B against `scripts/flux2_reference.py`, stage by stage.
    /// `RB_FLUX2_REF` is the script's output, `RB_FLUX2_MODEL` the model
    /// (default the local copy), `RB_FLUX2_ENCODER=f32` adds a run of the
    /// prompt encoder in F32 on its own (about 12 GB) and `RB_FLUX2_CUDA=1`
    /// moves the pipeline to the device. The denoiser is Q8: in F32 it alone
    /// needs about 16 GB.
    #[test]
    #[ignore = "needs FLUX.2-klein-4B and the output of scripts/flux2_reference.py"]
    fn flux2_klein_matches_diffusers() {
        let env = |name: &str| std::env::var(name).ok();
        let dir = std::path::PathBuf::from(env("RB_FLUX2_REF").expect("RB_FLUX2_REF"));
        let model = std::path::PathBuf::from(
            env("RB_FLUX2_MODEL")
                .unwrap_or_else(|| "/home/vasylt/Code/ai/ml/models/FLUX.2-klein-4B".into()),
        );
        let settings: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
                .unwrap();
        let prompt = settings["prompt"].as_str().unwrap();
        let mut reference =
            crate::safetensors::SafeTensors::open(dir.join("flux2.safetensors")).unwrap();
        let mut read = |name: &str| reference.tensor(name).unwrap().0;
        let mut report = Vec::new();
        let mut gate = |stage: &str, error: f32, limit: f32| {
            eprintln!("flux2 {stage}: relative L2 {error:.3e} (gate {limit:.0e})");
            report.push((stage.to_string(), error, limit));
        };

        // Token ids, exactly.
        let tokenizer = Bpe::from_file(model.join("tokenizer/tokenizer.json")).unwrap();
        let chat = ChatPrompt::flux2(tokenizer.token_id("<|endoftext|>").unwrap());
        let (ids, real) = chat.tokenize(&tokenizer, prompt).unwrap();
        let want_ids: Vec<u32> = read("input_ids").iter().map(|&id| id as u32).collect();
        let want_real = read("attention_mask").iter().filter(|&&m| m > 0.0).count();
        assert_eq!(ids, want_ids);
        assert_eq!(real, want_real);
        let embeds = read("prompt_embeds");

        if env("RB_FLUX2_ENCODER").as_deref() == Some("f32") {
            let mut file =
                crate::safetensors::ShardedSafeTensors::open(weights(&model.join("text_encoder")).unwrap())
                    .unwrap();
            let mut config =
                TextEncoderConfig::from_file(model.join("text_encoder/config.json")).unwrap();
            config.layers = 27;
            let encoder = PromptEncoder::Sequence {
                tokenizer: Bpe::from_file(model.join("tokenizer/tokenizer.json")).unwrap(),
                encoder: TextEncoder::read_at(&mut file, "model.", config, Precision::F32).unwrap(),
                chat: Some(chat.clone()),
            };
            drop(file);
            let started = std::time::Instant::now();
            let (got, pooled) = encoder.encode(prompt).unwrap();
            eprintln!("flux2 encoder f32: {:?}", started.elapsed());
            assert!(pooled.is_none());
            gate("prompt embeds, encoder F32", relative(&got.data, &embeds), 1e-3);
        }

        let started = std::time::Instant::now();
        #[allow(unused_mut)]
        let mut pipeline = ImagePipeline::load_at(&model, Precision::Q8).unwrap();
        #[cfg(feature = "cuda")]
        if env("RB_FLUX2_CUDA").as_deref() == Some("1") {
            pipeline.to_cuda(0).unwrap();
            let PromptEncoder::Sequence { encoder, .. } = &pipeline.encoder else {
                panic!("FLUX.2 takes a sequence encoder")
            };
            assert!(encoder.on_device(), "to_cuda left the Q8 encoder on the host");
        }
        eprintln!("flux2 load: {:?}", started.elapsed());
        assert_eq!(pipeline.config.steps, 4);

        let (got, _) = pipeline.encoder.encode(prompt).unwrap();
        gate("prompt embeds, encoder Q8", relative(&got.data, &embeds), 0.03);

        let levels = pipeline.scheduler(1024).sigmas(4).unwrap();
        gate("sigmas", relative(&levels, &read("sigmas")), 1e-5);

        // The noise is NCHW and diffusers packs it as one token per
        // position, row-major, with the channels as columns.
        let grid = read("noise");
        let mut noise = Matrix::new(1024, 128);
        for channel in 0..128 {
            for token in 0..1024 {
                noise.data[token * 128 + channel] = grid[channel * 1024 + token];
            }
        }
        let conditioning = Conditioning {
            text: Matrix::from_vec(512, 7680, embeds),
            pooled: Vec::new(),
            height: 32,
            width: 32,
            guidance: pipeline.config.embedded_guidance,
        };

        pipeline.denoiser.set_conditioning(conditioning.clone());
        let started = std::time::Instant::now();
        let prediction = pipeline.denoiser.denoise(&noise, levels[0]).unwrap();
        eprintln!("flux2 one step: {:?}", started.elapsed());
        // Q8 rounding, not a wrong layer: `flux2_klein_blocks_match_diffusers`
        // puts the same blocks at 1e-6 in F32. Over C11's 0.006 for SDXL's
        // UNet, which is why these gates are Q8's own.
        gate("step-0 prediction", relative(&prediction.data, &read("step0_pred")), 0.02);

        let started = std::time::Instant::now();
        let latent = pipeline
            .sample_latent_from_noise(conditioning, None, noise, None, |_, _| true)
            .unwrap();
        eprintln!("flux2 four steps: {:?}", started.elapsed());
        let final_latents = read("final_latents");
        gate("final latent, 4 steps", relative(&latent.data, &final_latents), 0.05);

        // What the autoencoder is handed, from diffusers' own final latent.
        let want_latent = Matrix::from_vec(1024, 128, final_latents);
        let (mean, std) = pipeline.decoder.patch_norm().unwrap();
        let mut unnormalised = want_latent.clone();
        normalise_patches(&mut unnormalised, mean, std, false).unwrap();
        let unpacked =
            pixel_shuffle(&FeatureMap::from_tokens(&unnormalised, 32, 32).unwrap(), 2).unwrap();
        gate("autoencoder input", relative(&unpacked.data, &read("vae_latent")), 1e-5);

        let image = read("image");
        let started = std::time::Instant::now();
        let decoded = pipeline.decode(&want_latent, 32, 32).unwrap();
        eprintln!("flux2 decode: {:?}", started.elapsed());
        gate("decoded image, reference latent", relative(&decoded.data, &image), 0.012);
        let decoded = pipeline.decode(&latent, 32, 32).unwrap();
        gate("decoded image, whole chain", relative(&decoded.data, &image), 0.03);

        let failed: Vec<_> = report.iter().filter(|(_, error, limit)| error > limit).collect();
        assert!(failed.is_empty(), "over the gate: {failed:?}");
    }

    /// FLUX.2-klein's Qwen3 prompt encoder on the device against diffusers'
    /// prompt embeddings, all 512 positions of the three stacked layers, at
    /// Q8 (int8 weights) and, with `RB_FLUX2_ENCODER=f32`, from the F32 host
    /// encoder (BF16 weights, about 12 GB of host memory). Same variables as
    /// [`flux2_klein_matches_diffusers`]; skips without a device.
    #[test]
    #[ignore = "needs FLUX.2-klein-4B, the output of scripts/flux2_reference.py and a device"]
    fn flux2_klein_device_encoder_matches_diffusers() {
        #[cfg(feature = "cuda")]
        {
            let env = |name: &str| std::env::var(name).ok();
            let context = match crate::cuda_training::cuda_doctor(0, 8192) {
                Ok(_) => crate::gpu_transformer::GpuContext::new(0).unwrap(),
                Err(NetworkError::Cuda(message))
                    if message.contains("NO_DEVICE")
                        || message.contains("no CUDA-capable device") =>
                {
                    crate::cuda_training::skip_without_cuda();
                    return;
                }
                Err(error) => panic!("CUDA is present but the CUDA doctor failed: {error}"),
            };
            let dir = std::path::PathBuf::from(env("RB_FLUX2_REF").expect("RB_FLUX2_REF"));
            let model = std::path::PathBuf::from(
                env("RB_FLUX2_MODEL")
                    .unwrap_or_else(|| "/home/vasylt/Code/ai/ml/models/FLUX.2-klein-4B".into()),
            );
            let settings: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(dir.join("reference.json")).unwrap(),
            )
            .unwrap();
            let prompt = settings["prompt"].as_str().unwrap();
            let embeds = crate::safetensors::SafeTensors::open(dir.join("flux2.safetensors"))
                .unwrap()
                .tensor("prompt_embeds")
                .unwrap()
                .0;
            let tokenizer = Bpe::from_file(model.join("tokenizer/tokenizer.json")).unwrap();
            let chat = ChatPrompt::flux2(tokenizer.token_id("<|endoftext|>").unwrap());

            // Q8's own 0.025 on the host is the gate for int8 weights; BF16
            // weights from the F32 encoder sit near BF16's rounding.
            let mut runs = vec![(Precision::Q8, 0.025)];
            if env("RB_FLUX2_ENCODER").as_deref() == Some("f32") {
                runs.push((Precision::F32, 0.01));
            }
            let mut failed = Vec::new();
            for (precision, limit) in runs {
                let mut file = crate::safetensors::ShardedSafeTensors::open(
                    weights(&model.join("text_encoder")).unwrap(),
                )
                .unwrap();
                let mut config =
                    TextEncoderConfig::from_file(model.join("text_encoder/config.json")).unwrap();
                config.layers = 27;
                let mut encoder = PromptEncoder::Sequence {
                    tokenizer: tokenizer.clone(),
                    encoder: TextEncoder::read_at(&mut file, "model.", config, precision).unwrap(),
                    chat: Some(chat.clone()),
                };
                drop(file);
                let host = relative(&encoder.encode(prompt).unwrap().0.data, &embeds);
                eprintln!("flux2 device encoder {precision:?}: host relative L2 {host:.4e}");
                let PromptEncoder::Sequence { encoder: inner, .. } = &mut encoder else {
                    unreachable!()
                };
                let started = std::time::Instant::now();
                inner.attach_device(&context).unwrap();
                eprintln!("flux2 device encoder {precision:?}: upload {:?}", started.elapsed());
                let mut times = Vec::new();
                let mut got = None;
                for _ in 0..4 {
                    let started = std::time::Instant::now();
                    got = Some(encoder.encode(prompt).unwrap().0);
                    times.push(started.elapsed());
                }
                let error = relative(&got.unwrap().data, &embeds);
                eprintln!(
                    "flux2 device encoder {precision:?}: relative L2 {error:.4e} (gate {limit}), \
                     encodes {times:?}"
                );
                if error > limit {
                    failed.push((precision, error));
                }
            }
            assert!(failed.is_empty(), "over the gate: {failed:?}");
        }
    }

    /// The transformer cut to its first two double and two single blocks,
    /// in F32 and in Q8, against `scripts/flux2_blocks_reference.py`. The
    /// whole one in F32 does not fit beside everything else on this machine,
    /// and this separates a wrong layer from Q8's rounding. Same variables as
    /// [`flux2_klein_matches_diffusers`].
    #[test]
    #[ignore = "needs FLUX.2-klein-4B and the output of scripts/flux2_blocks_reference.py"]
    fn flux2_klein_blocks_match_diffusers() {
        let env = |name: &str| std::env::var(name).ok();
        let dir = std::path::PathBuf::from(env("RB_FLUX2_REF").expect("RB_FLUX2_REF"));
        let model = std::path::PathBuf::from(
            env("RB_FLUX2_MODEL")
                .unwrap_or_else(|| "/home/vasylt/Code/ai/ml/models/FLUX.2-klein-4B".into()),
        );
        let mut reference =
            crate::safetensors::SafeTensors::open(dir.join("flux2.safetensors")).unwrap();
        let grid = reference.tensor("noise").unwrap().0;
        let embeds = reference.tensor("prompt_embeds").unwrap().0;
        let want = crate::safetensors::SafeTensors::open(dir.join("flux2_blocks.safetensors"))
            .unwrap()
            .tensor("step0_pred_blocks")
            .unwrap()
            .0;
        let mut noise = Matrix::new(1024, 128);
        for channel in 0..128 {
            for token in 0..1024 {
                noise.data[token * 128 + channel] = grid[channel * 1024 + token];
            }
        }
        let config = DitConfig {
            double_blocks: 2,
            single_blocks: 2,
            ..DitConfig::from_file(model.join("transformer/config.json")).unwrap()
        };
        let path = weights(&model.join("transformer")).unwrap();
        // The device runs BF16 activations, and BF16 weights for an F32
        // model, so its F32 gate is BF16's.
        let device = cfg!(feature = "cuda") && env("RB_FLUX2_CUDA").as_deref() == Some("1");
        let f32_limit = if device { 0.01 } else { 1e-4 };
        for (precision, limit) in [(Precision::F32, f32_limit), (Precision::Q8, 0.02)] {
            #[allow(unused_mut)]
            let mut dit = Dit::load_at(&path, "", config.clone(), precision).unwrap();
            #[cfg(feature = "cuda")]
            if env("RB_FLUX2_CUDA").as_deref() == Some("1") {
                dit.attach_device(&crate::gpu_transformer::GpuContext::new(0).unwrap())
                    .unwrap();
            }
            dit.set_conditioning(Conditioning {
                text: Matrix::from_vec(512, 7680, embeds.clone()),
                pooled: Vec::new(),
                height: 32,
                width: 32,
                guidance: 0.0,
            });
            let got = dit.denoise(&noise, 1.0).unwrap();
            let error = relative(&got.data, &want);
            eprintln!("flux2 2+2 blocks {precision:?}: relative L2 {error:.3e} (gate {limit:.0e})");
            assert!(error <= limit, "{precision:?}: {error}");
        }
    }
}
