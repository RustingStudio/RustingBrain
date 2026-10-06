//! A variational autoencoder over shapes, from surface points to a signed
//! distance field.
//!
//! The encoder reads a cloud of points sampled off a mesh's surface, each with
//! its normal, and compresses the whole shape into a small grid of latent
//! vectors. The decoder answers one question: given a point in space, how far
//! is it from the surface, and on which side? Between the two,
//! [`crate::mesh::marching_tetrahedra`] turns that field back into triangles.
//!
//! The shape is a *set* at both ends. The surface points arrive in no
//! particular order and the latents are a learned collection rather than a
//! sequence, so nothing here uses rotary positions or a causal mask — position
//! enters only through [`fourier_features`], which is a function of where a
//! point actually is rather than of where it sits in an array.
//!
//! ```no_run
//! # use rusting_brain::shape_vae::{ShapeVae, ShapeVaeConfig};
//! # use rusting_brain::matrix::Matrix;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let surface = Matrix::new(2048, 6);
//! # let queries = Matrix::new(4096, 3);
//! let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0);
//! let model = ShapeVae::new(ShapeVaeConfig::default(), &mut rng)?;
//!
//! // A draw from the posterior, not its mean: a draw is the only kind of
//! // latent the decoder has ever been trained to answer for. See `encode`.
//! let latent = model.encode_sample(&surface, &mut rng)?;  // [latents, latent_dim]
//! let distances = model.decode(&latent, &queries, 65_536)?; // one per query point
//! # let _ = distances;
//! # Ok(())
//! # }
//! ```
//!
//! ponytail: no Eikonal regularization. It needs `dL/dquery_point`, which is a
//! second backward path through the decoder's input projection that nothing
//! else in the crate wants. The plan defers it until the surfaces come out
//! noisy, and this follows the plan.

use crate::attention::{AttentionCache, CrossAttentionCache, MultiHeadAttention};
use crate::batch::Layout;
use crate::ffn::{SwiGlu, SwiGluCache};
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::norm::RmsNorm;
use crate::param::{Linear, Param};
use crate::rope::Rope;
use rand::Rng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The shape of a [`ShapeVae`].
///
/// The defaults are the plan's: 2,048 surface points compressed into a
/// `[512, 64]` latent, which is 32,768 numbers for a whole shape.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct ShapeVaeConfig {
    pub d_model: usize,
    /// How many latent vectors the shape is compressed into.
    pub latents: usize,
    /// The width of one latent vector, which is much narrower than `d_model`.
    pub latent_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub d_ff: usize,
    /// Self-attention blocks over the latents, after the surface is read.
    pub encoder_blocks: usize,
    /// Octaves of [`fourier_features`]. The highest frequency is `2^(n-1)`
    /// cycles per unit, so this is the cap on the detail the field can hold.
    pub frequencies: usize,
    pub eps: f32,
    /// The per-unit KL floor a training loop hands to
    /// [`crate::losses::kl_divergence_free_bits`], in nats. Zero is the plain
    /// divergence.
    ///
    /// The model itself never reads this. It lives on the config because the
    /// config is what a checkpoint carries and what a resume compares, so a
    /// run cannot silently continue under a different floor than it started
    /// with. Old checkpoints, written before the field existed, load as zero
    /// and so keep the behaviour they were trained with.
    #[serde(default)]
    pub free_bits: f32,
}

impl Default for ShapeVaeConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            latents: 512,
            latent_dim: 64,
            num_heads: 8,
            head_dim: 64,
            d_ff: 2048,
            encoder_blocks: 4,
            frequencies: 8,
            eps: 1e-5,
            free_bits: 0.0,
        }
    }
}

impl ShapeVaeConfig {
    /// The width of one Fourier-encoded point: the raw coordinates, then a
    /// sine and a cosine per axis per octave.
    pub fn feature_dim(&self) -> usize {
        3 + 6 * self.frequencies
    }

    fn check(&self) -> Result<(), NetworkError> {
        if self.latents == 0 || self.latent_dim == 0 {
            return Err(NetworkError::InvalidConfig(
                "a shape autoencoder needs at least one latent vector of at least one dimension"
                    .into(),
            ));
        }
        if !self.free_bits.is_finite() || self.free_bits < 0.0 {
            return Err(NetworkError::InvalidConfig(format!(
                "a free-bits floor is a non-negative number of nats, got {}",
                self.free_bits
            )));
        }
        if self.head_dim % 2 != 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "head_dim must be even, got {}",
                self.head_dim
            )));
        }
        Ok(())
    }
}

/// Sinusoidal features for a set of 3-D points, `[n, 3] -> [n, 3 + 6 * bands]`.
///
/// Each row is the point itself followed by `sin` and `cos` of `2^k * pi * x`
/// for every axis and every octave. A network reading raw coordinates through a
/// few linear layers is biased hard towards smooth functions and cannot
/// represent a sharp surface; the high-frequency channels are what let it.
///
/// This is not [`Rope`], which rotates a *sequence position* into an existing
/// embedding. A point in space has no sequence position and three coordinates
/// rather than one.
///
/// ```
/// # use rusting_brain::shape_vae::fourier_features;
/// # use rusting_brain::matrix::Matrix;
/// let points = Matrix::from_vec(1, 3, vec![0.0, 0.0, 0.0]);
/// let features = fourier_features(&points, 2).unwrap();
/// assert_eq!(features.cols, 3 + 6 * 2);
/// // At the origin every sine is zero and every cosine is one.
/// assert_eq!(features.row(0), &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
/// ```
pub fn fourier_features(points: &Matrix, bands: usize) -> Result<Matrix, NetworkError> {
    if points.cols != 3 {
        return Err(NetworkError::InvalidConfig(format!(
            "points are 3-D, so a [n, 3] matrix was wanted and this one is [{}, {}]",
            points.rows, points.cols
        )));
    }

    let width = 3 + 6 * bands;
    let mut features = Matrix::new(points.rows, width);
    for row in 0..points.rows {
        let point = points.row(row);
        let out = features.row_mut(row);
        out[..3].copy_from_slice(point);
        for band in 0..bands {
            let frequency = (1u32 << band) as f32 * std::f32::consts::PI;
            for axis in 0..3 {
                let (sin, cos) = (frequency * point[axis]).sin_cos();
                out[3 + band * 6 + axis * 2] = sin;
                out[3 + band * 6 + axis * 2 + 1] = cos;
            }
        }
    }
    Ok(features)
}

/// Pre-norm self-attention and a SwiGLU feed-forward, both residual.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct Block {
    pub(crate) attention_norm: RmsNorm,
    pub(crate) attention: MultiHeadAttention,
    pub(crate) mlp_norm: RmsNorm,
    pub(crate) mlp: SwiGlu,
}

struct BlockCache {
    input: Matrix,
    attention: AttentionCache,
    /// The input plus the attention branch, which is what the feed-forward
    /// half normalizes and adds onto.
    residual: Matrix,
    mlp: SwiGluCache,
}

impl Block {
    fn new(config: &ShapeVaeConfig, rng: &mut StdRng) -> Result<Self, NetworkError> {
        let rope = Rope::new(config.head_dim, 1, 10000.0)?;
        let mut attention = MultiHeadAttention::new(
            config.d_model,
            config.num_heads,
            config.num_heads,
            config.head_dim,
            rope,
            rng,
        )?;
        // A set, not a sequence: every latent reads every other one, and none
        // of them has a position to rotate against.
        attention.set_causal(false);
        attention.set_rope_enabled(false);

        Ok(Self {
            attention_norm: RmsNorm::new(config.d_model, config.eps),
            attention,
            mlp_norm: RmsNorm::new(config.d_model, config.eps),
            mlp: SwiGlu::new(config.d_model, config.d_ff, rng),
        })
    }

    fn forward_train(&self, input: &Matrix) -> Result<(Matrix, BlockCache), NetworkError> {
        let normed = self.attention_norm.forward(input);
        let (attended, attention) = self.attention.forward_train(&normed, Layout::default())?;
        let residual = add(input, &attended);

        let normed = self.mlp_norm.forward(&residual);
        let (wide, mlp) = self.mlp.forward_train(&normed);
        let output = add(&residual, &wide);

        Ok((
            output,
            BlockCache {
                input: input.clone(),
                attention,
                residual,
                mlp,
            },
        ))
    }

    fn backward(
        &mut self,
        cache: &BlockCache,
        grad_output: &Matrix,
    ) -> Result<Matrix, NetworkError> {
        // Both halves are residual, so the gradient reaching the block's input
        // is the one arriving here plus whatever each branch sends back.
        let grad_wide = self.mlp.backward(&cache.mlp, grad_output);
        let grad_residual = add(
            grad_output,
            &self.mlp_norm.backward(&cache.residual, &grad_wide),
        );

        let grad_attended = self.attention.backward(&cache.attention, &grad_residual)?;
        Ok(add(
            &grad_residual,
            &self.attention_norm.backward(&cache.input, &grad_attended),
        ))
    }

    fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = self.attention_norm.params_mut();
        params.extend(self.attention.params_mut());
        params.extend(self.mlp_norm.params_mut());
        params.extend(self.mlp.params_mut());
        params
    }
}

/// What the encoder's backward pass needs.
///
/// Opaque on purpose, like [`crate::flow_transformer::FlowCache`]: a cache
/// from a model on a device holds device buffers and one from a host model
/// holds matrices, and a caller only carries it from the forward pass to the
/// backward one.
pub struct EncoderCache {
    inner: EncoderCached,
}

enum EncoderCached {
    Host(Box<HostEncoderCache>),
    #[cfg(feature = "cuda")]
    Device(Box<crate::gpu_shape::GpuEncoderCache>),
}

struct HostEncoderCache {
    features: Matrix,
    cross: CrossAttentionCache,
    blocks: Vec<BlockCache>,
    hidden: Matrix,
    normed: Matrix,
}

/// What the decoder's backward pass needs, for one chunk of query points.
pub struct DecoderCache {
    inner: DecoderCached,
}

enum DecoderCached {
    Host(Box<HostDecoderCache>),
    #[cfg(feature = "cuda")]
    Device(Box<crate::gpu_shape::GpuDecoderCache>),
}

struct HostDecoderCache {
    latent: Matrix,
    features: Matrix,
    cross: CrossAttentionCache,
    residual: Matrix,
    mlp: SwiGluCache,
    hidden: Matrix,
    normed: Matrix,
}

/// Surface points in, a signed distance field out.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShapeVae {
    pub(crate) config: ShapeVaeConfig,

    /// Encoder: the six input channels of a surface point, Fourier-encoded.
    pub(crate) surface_in: Linear,
    /// The learned latent queries, which are what actually reads the shape.
    pub(crate) latent_queries: Param,
    pub(crate) read_surface: MultiHeadAttention,
    pub(crate) blocks: Vec<Block>,
    pub(crate) encoder_norm: RmsNorm,
    /// To the mean and the log-variance at once, split down the middle.
    pub(crate) to_moments: Linear,

    /// Decoder: the latent widened back out to `d_model` to be attended to.
    pub(crate) from_latent: Linear,
    pub(crate) query_in: Linear,
    pub(crate) read_latent: MultiHeadAttention,
    pub(crate) decoder_norm: RmsNorm,
    pub(crate) decoder_mlp: SwiGlu,
    pub(crate) head_norm: RmsNorm,
    pub(crate) head: Linear,
    /// The texture stand-in: linear RGB at the query point, off the same
    /// hidden state the distance is read from. Raw, not squashed — a colour is
    /// trained towards 0 to 1 and clamped when it is written to a file, which
    /// costs one clamp instead of a sigmoid on every backward pass.
    pub(crate) color_head: Linear,
    /// One number added to every distance. The only bias in the model, and it
    /// earns its place: more of space is outside a shape than inside it, so the
    /// field has a non-zero mean the projections would otherwise have to fake.
    pub(crate) head_bias: Param,
    /// Set by [`ShapeVae::to_cuda`]. Never serialized: a snapshot is host
    /// data, and a restored model starts on the CPU.
    #[cfg(feature = "cuda")]
    #[serde(skip)]
    device: Option<std::sync::Arc<crate::gpu_transformer::GpuContext>>,
}

/// Compares the model, not where it happens to be running: two models with the
/// same weights are equal whether or not one of them holds a device context.
impl PartialEq for ShapeVae {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config
            && self.surface_in == other.surface_in
            && self.latent_queries == other.latent_queries
            && self.read_surface == other.read_surface
            && self.blocks == other.blocks
            && self.encoder_norm == other.encoder_norm
            && self.to_moments == other.to_moments
            && self.from_latent == other.from_latent
            && self.query_in == other.query_in
            && self.read_latent == other.read_latent
            && self.decoder_norm == other.decoder_norm
            && self.decoder_mlp == other.decoder_mlp
            && self.head_norm == other.head_norm
            && self.head == other.head
            && self.color_head == other.color_head
            && self.head_bias == other.head_bias
    }
}

impl ShapeVae {
    pub fn new(config: ShapeVaeConfig, rng: &mut StdRng) -> Result<Self, NetworkError> {
        config.check()?;
        let features = config.feature_dim();
        let rope = Rope::new(config.head_dim, 1, 10000.0)?;

        let mut blocks = Vec::with_capacity(config.encoder_blocks);
        for _ in 0..config.encoder_blocks {
            blocks.push(Block::new(&config, rng)?);
        }

        Ok(Self {
            // The three normal channels ride alongside the encoded position.
            surface_in: Linear::new(features + 3, config.d_model, rng),
            latent_queries: Param::he_uniform(config.latents, config.d_model, config.d_model, rng),
            read_surface: MultiHeadAttention::cross(
                config.d_model,
                config.d_model,
                config.num_heads,
                config.num_heads,
                config.head_dim,
                rope.clone(),
                rng,
            )?,
            blocks,
            encoder_norm: RmsNorm::new(config.d_model, config.eps),
            to_moments: Linear::new(config.d_model, 2 * config.latent_dim, rng),

            from_latent: Linear::new(config.latent_dim, config.d_model, rng),
            query_in: Linear::new(features, config.d_model, rng),
            read_latent: MultiHeadAttention::cross(
                config.d_model,
                config.d_model,
                config.num_heads,
                config.num_heads,
                config.head_dim,
                rope,
                rng,
            )?,
            decoder_norm: RmsNorm::new(config.d_model, config.eps),
            decoder_mlp: SwiGlu::new(config.d_model, config.d_ff, rng),
            head_norm: RmsNorm::new(config.d_model, config.eps),
            head: Linear::new(config.d_model, 1, rng),
            color_head: Linear::new(config.d_model, 3, rng),
            head_bias: Param::zeros(1, 1),
            #[cfg(feature = "cuda")]
            device: None,
            config,
        })
    }

    pub fn config(&self) -> &ShapeVaeConfig {
        &self.config
    }

    /// The mean and the log-variance of the latent distribution, each
    /// `[latents, latent_dim]`.
    ///
    /// `surface` is `[points, 6]`: a position and its outward normal.
    ///
    /// # The mean is not what the decoder was trained on
    ///
    /// Every training step decodes a *draw* from this distribution, never its
    /// mean, so the decoder has only ever seen latents with the posterior's
    /// noise on them. The mean sits in the middle of that cloud, in a spot no
    /// training step ever visited, and the reconstruction from it is visibly
    /// worse: on the reference model at 24,000 steps the posterior is wide —
    /// sigma around 0.53 against a mean magnitude of 0.23 — and reconstructing
    /// from the mean gives a clamped L1 of 0.029 where a draw gives 0.0025, an
    /// order of magnitude.
    ///
    /// So reconstruction goes through [`ShapeVae::encode_sample`]. Reach for
    /// the mean when the question is about the distribution itself — measuring
    /// how wide it is, feeding a flow model that learns to produce means, or
    /// checking a KL term — not when the answer is a mesh.
    pub fn encode(&self, surface: &Matrix) -> Result<(Matrix, Matrix), NetworkError> {
        let (mean, log_variance, _) = self.encode_train(surface)?;
        Ok((mean, log_variance))
    }

    /// One latent drawn from the posterior, which is what the decoder was
    /// trained to read.
    ///
    /// [`ShapeVae::encode`] followed by [`ShapeVae::sample`], in the order a
    /// training step does it. This is the call for reconstructing a shape: see
    /// the warning on `encode` for what the mean costs instead.
    pub fn encode_sample(
        &self,
        surface: &Matrix,
        rng: &mut StdRng,
    ) -> Result<Matrix, NetworkError> {
        let (mean, log_variance) = self.encode(surface)?;
        Ok(Self::sample(&mean, &log_variance, rng).0)
    }

    /// The same, keeping what the backward pass needs.
    pub fn encode_train(
        &self,
        surface: &Matrix,
    ) -> Result<(Matrix, Matrix, EncoderCache), NetworkError> {
        if surface.cols != 6 || surface.rows == 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "a surface point is a position and a normal, so [n, 6] was wanted and this is [{}, {}]",
                surface.rows, surface.cols
            )));
        }
        if surface.rows == 0 {
            return Err(NetworkError::InvalidConfig(
                "a shape cannot be encoded from no surface points".into(),
            ));
        }

        #[cfg(feature = "cuda")]
        if let Some(context) = &self.device {
            let (mean, log_variance, cache) =
                crate::gpu_shape::encode_train(self, context, surface)?;
            return Ok((
                mean,
                log_variance,
                EncoderCache {
                    inner: EncoderCached::Device(Box::new(cache)),
                },
            ));
        }

        let features = self.surface_features(surface)?;
        let tokens = self.surface_in.forward(&features);

        // The learned queries are the whole compression: however many surface
        // points came in, exactly `latents` rows come out.
        let queries = self.latent_queries.value.clone();
        let (crossed, cross) =
            self.read_surface
                .forward_train_cross(&queries, &tokens, queries.rows, tokens.rows)?;
        let residual = add(&queries, &crossed);

        let mut hidden = residual.clone();
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (output, cache) = block.forward_train(&hidden)?;
            hidden = output;
            blocks.push(cache);
        }

        let normed = self.encoder_norm.forward(&hidden);
        let moments = self.to_moments.forward(&normed);
        let (mean, log_variance) = split(&moments);

        Ok((
            mean,
            log_variance,
            EncoderCache {
                inner: EncoderCached::Host(Box::new(HostEncoderCache {
                    features,
                    cross,
                    blocks,
                    hidden,
                    normed,
                })),
            },
        ))
    }

    /// Encodes a packed CUDA batch with the same number of surface points in
    /// every shape. Rows stay grouped by shape in both returned matrices.
    ///
    /// Packing turns the batch into one set of projection GEMMs and batched
    /// attention launches. The ordinary host path intentionally remains
    /// shape-at-a-time; call this only after [`to_cuda`](Self::to_cuda).
    #[cfg(feature = "cuda")]
    pub fn encode_train_batch(
        &self,
        surface: &Matrix,
        shapes: usize,
    ) -> Result<(Matrix, Matrix, EncoderCache), NetworkError> {
        if surface.cols != 6 {
            return Err(NetworkError::InvalidConfig(format!(
                "a surface point is a position and a normal, so [n, 6] was wanted and this is [{}, {}]",
                surface.rows, surface.cols
            )));
        }
        let context = self.device_context()?;
        let (mean, log_variance, cache) =
            crate::gpu_shape::encode_train_batch(self, &context, surface, shapes)?;
        Ok((
            mean,
            log_variance,
            EncoderCache {
                inner: EncoderCached::Device(Box::new(cache)),
            },
        ))
    }

    /// Accumulates the encoder's weight gradients.
    ///
    /// Nothing is returned: the surface points are data, not a layer below.
    pub fn encode_backward(
        &mut self,
        cache: &EncoderCache,
        grad_mean: &Matrix,
        grad_log_variance: &Matrix,
    ) -> Result<(), NetworkError> {
        // One arm without the `cuda` feature, which is what the allow is for.
        #[allow(clippy::infallible_destructuring_match)]
        let cache = match &cache.inner {
            EncoderCached::Host(cache) => cache,
            #[cfg(feature = "cuda")]
            EncoderCached::Device(cache) => {
                // Cloning the handle, not the context: the borrow below is
                // `&mut self`, and the context lives in a field of it.
                let context = self.device_context()?;
                return crate::gpu_shape::encode_backward(
                    self,
                    &context,
                    cache,
                    grad_mean,
                    grad_log_variance,
                );
            }
        };
        let grad_moments = join(grad_mean, grad_log_variance);
        let grad_normed = self.to_moments.backward(&cache.normed, &grad_moments);
        let mut grad = self.encoder_norm.backward(&cache.hidden, &grad_normed);

        for (block, cache) in self.blocks.iter_mut().zip(&cache.blocks).rev() {
            grad = block.backward(cache, &grad)?;
        }

        let (grad_queries, grad_tokens) = self.read_surface.backward_cross(&cache.cross, &grad)?;
        // The queries sit on the residual path, so they take both shares.
        let grad_queries = add(&grad, &grad_queries);
        if !self.latent_queries.is_frozen() {
            for (slot, value) in self
                .latent_queries
                .grad
                .data
                .iter_mut()
                .zip(&grad_queries.data)
            {
                *slot += value;
            }
        }

        // The projection wants its weight gradient; its input gradient is the
        // gradient with respect to the surface points, which nothing reads.
        self.surface_in.backward(&cache.features, &grad_tokens);
        Ok(())
    }

    /// Draws a latent from the distribution the encoder described.
    ///
    /// `z = mean + exp(log_variance / 2) * noise`. The noise is returned
    /// because the backward pass needs it, and because a caller that wants a
    /// deterministic reconstruction passes the mean straight to
    /// [`decode`](Self::decode) and never calls this at all.
    pub fn sample<R: Rng>(mean: &Matrix, log_variance: &Matrix, rng: &mut R) -> (Matrix, Matrix) {
        let mut noise = Matrix::new(mean.rows, mean.cols);
        let mut latent = Matrix::new(mean.rows, mean.cols);
        for index in 0..mean.data.len() {
            // Box-Muller, the same two-uniforms trick the mesh sampler uses.
            let first: f32 = rng.gen_range(f32::EPSILON..1.0);
            let second: f32 = rng.gen_range(0.0..1.0);
            let sample = (-2.0 * first.ln()).sqrt() * (std::f32::consts::TAU * second).cos();
            noise.data[index] = sample;
            latent.data[index] = mean.data[index] + (0.5 * log_variance.data[index]).exp() * sample;
        }
        (latent, noise)
    }

    /// Splits `dL/dlatent` back into `dL/dmean` and `dL/dlog_variance`.
    ///
    /// The mean is on a straight path so it takes the gradient unchanged; the
    /// log-variance only reaches the latent through the scaled noise, which is
    /// exactly `latent - mean`.
    pub fn sample_backward(
        grad_latent: &Matrix,
        log_variance: &Matrix,
        noise: &Matrix,
    ) -> (Matrix, Matrix) {
        let mut grad_log_variance = Matrix::new(grad_latent.rows, grad_latent.cols);
        for index in 0..grad_latent.data.len() {
            grad_log_variance.data[index] = grad_latent.data[index]
                * 0.5
                * (0.5 * log_variance.data[index]).exp()
                * noise.data[index];
        }
        (grad_latent.clone(), grad_log_variance)
    }

    /// One signed distance per query point, decoded `chunk` points at a time.
    ///
    /// `queries` is `[n, 3]`. The chunk size is what bounds the memory: the
    /// attention scores are `[chunk, latents]`, and a 128³ marching grid asks
    /// about 2.1 million points that will not fit at once.
    pub fn decode(
        &self,
        latent: &Matrix,
        queries: &Matrix,
        chunk: usize,
    ) -> Result<Vec<f32>, NetworkError> {
        if chunk == 0 {
            return Err(NetworkError::InvalidConfig(
                "a decode chunk holds at least one query point".into(),
            ));
        }
        let kv = self.latent_keys(latent)?;

        let mut distances = Vec::with_capacity(queries.rows);
        for start in (0..queries.rows).step_by(chunk) {
            let rows = chunk.min(queries.rows - start);
            let slice = Matrix::from_vec(
                rows,
                3,
                queries.data[start * 3..(start + rows) * 3].to_vec(),
            );
            // ponytail: this builds the training cache and throws it away.
            // There is one cross-attention forward in the crate, the chunk is
            // already bounded, and a second cache-free copy of it would be the
            // only thing here that could drift out of step with the backward
            // pass.
            // On the device the same forward runs through `decode_train`,
            // which is the one entry point `gpu_shape` has; the cache it keeps
            // is freed with each chunk.
            #[cfg(feature = "cuda")]
            if self.device.is_some() {
                distances.extend(self.decode_train(latent, &slice)?.0);
                continue;
            }
            let (chunk_distances, _) = self.decode_chunk(latent, &kv, &slice)?;
            distances.extend(chunk_distances);
        }
        Ok(distances)
    }

    /// One chunk of query points, keeping what the backward pass needs.
    ///
    /// Training decodes a subsample of a mesh's query points per step, which
    /// is one chunk, so it calls this directly.
    pub fn decode_train(
        &self,
        latent: &Matrix,
        queries: &Matrix,
    ) -> Result<(Vec<f32>, DecoderCache), NetworkError> {
        #[cfg(feature = "cuda")]
        if let Some(context) = &self.device {
            let (distances, cache) =
                crate::gpu_shape::decode_train(self, context, latent, queries)?;
            return Ok((
                distances,
                DecoderCache {
                    inner: DecoderCached::Device(Box::new(cache)),
                },
            ));
        }

        let kv = self.latent_keys(latent)?;
        self.decode_chunk(latent, &kv, queries)
    }

    /// Decodes a packed CUDA batch with equal query counts per shape.
    /// Latent and query rows must both be contiguous by shape.
    #[cfg(feature = "cuda")]
    pub fn decode_train_batch(
        &self,
        latent: &Matrix,
        queries: &Matrix,
        shapes: usize,
    ) -> Result<(Vec<f32>, DecoderCache), NetworkError> {
        if queries.cols != 3 || queries.rows == 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "query points are 3-D, so [n, 3] was wanted and this is [{}, {}]",
                queries.rows, queries.cols
            )));
        }
        let context = self.device_context()?;
        let (distances, cache) =
            crate::gpu_shape::decode_train_batch(self, &context, latent, queries, shapes)?;
        Ok((
            distances,
            DecoderCache {
                inner: DecoderCached::Device(Box::new(cache)),
            },
        ))
    }

    /// The colour at each point of a decoded chunk, `[n, 3]`.
    ///
    /// The distance and the colour share everything but the last projection,
    /// so this reads a cache [`decode_train`](Self::decode_train) already
    /// built rather than decoding again.
    pub fn colors(&self, cache: &DecoderCache) -> Result<Matrix, NetworkError> {
        match &cache.inner {
            DecoderCached::Host(cache) => Ok(self.color_head.forward(&cache.normed)),
            #[cfg(feature = "cuda")]
            DecoderCached::Device(cache) => crate::gpu_shape::colors(self, cache),
        }
    }

    /// One colour per point, chunked like [`decode`](Self::decode).
    pub fn decode_colors(
        &self,
        latent: &Matrix,
        points: &Matrix,
        chunk: usize,
    ) -> Result<Vec<[f32; 3]>, NetworkError> {
        if chunk == 0 {
            return Err(NetworkError::InvalidConfig(
                "a decode chunk holds at least one query point".into(),
            ));
        }
        let kv = self.latent_keys(latent)?;

        let mut colors = Vec::with_capacity(points.rows);
        for start in (0..points.rows).step_by(chunk) {
            let rows = chunk.min(points.rows - start);
            let slice =
                Matrix::from_vec(rows, 3, points.data[start * 3..(start + rows) * 3].to_vec());
            #[cfg(feature = "cuda")]
            let (_, cache) = if self.device.is_some() {
                self.decode_train(latent, &slice)?
            } else {
                self.decode_chunk(latent, &kv, &slice)?
            };
            #[cfg(not(feature = "cuda"))]
            let (_, cache) = self.decode_chunk(latent, &kv, &slice)?;
            let decoded = self.colors(&cache)?;
            colors.extend(decoded.data.chunks_exact(3).map(|c| [c[0], c[1], c[2]]));
        }
        Ok(colors)
    }

    /// Accumulates the decoder's weight gradients and returns `dL/dlatent`.
    pub fn decode_backward(
        &mut self,
        cache: &DecoderCache,
        grad_distances: &[f32],
    ) -> Result<Matrix, NetworkError> {
        self.decode_backward_colored(cache, grad_distances, None)
    }

    /// The same, with the colour head's gradients as well.
    pub fn decode_backward_colored(
        &mut self,
        cache: &DecoderCache,
        grad_distances: &[f32],
        grad_colors: Option<&Matrix>,
    ) -> Result<Matrix, NetworkError> {
        // One arm without the `cuda` feature, which is what the allow is for.
        #[allow(clippy::infallible_destructuring_match)]
        let cache = match &cache.inner {
            DecoderCached::Host(cache) => cache,
            #[cfg(feature = "cuda")]
            DecoderCached::Device(cache) => {
                let context = self.device_context()?;
                return crate::gpu_shape::decode_backward(
                    self,
                    &context,
                    cache,
                    grad_distances,
                    grad_colors,
                );
            }
        };
        if grad_distances.len() != cache.normed.rows {
            return Err(NetworkError::InvalidConfig(format!(
                "{} query points were decoded and {} gradients came back",
                cache.normed.rows,
                grad_distances.len()
            )));
        }
        let grad_output = Matrix::from_vec(grad_distances.len(), 1, grad_distances.to_vec());
        if !self.head_bias.is_frozen() {
            self.head_bias.grad.data[0] += grad_distances.iter().sum::<f32>();
        }

        let mut grad_normed = self.head.backward(&cache.normed, &grad_output);
        if let Some(grad_colors) = grad_colors {
            if grad_colors.rows != cache.normed.rows || grad_colors.cols != 3 {
                return Err(NetworkError::InvalidConfig(format!(
                    "the colour gradient is [{}, {}] and should be [{}, 3]",
                    grad_colors.rows, grad_colors.cols, cache.normed.rows
                )));
            }
            grad_normed = add(
                &grad_normed,
                &self.color_head.backward(&cache.normed, grad_colors),
            );
        }
        let grad_hidden = self.head_norm.backward(&cache.hidden, &grad_normed);

        let grad_wide = self.decoder_mlp.backward(&cache.mlp, &grad_hidden);
        let grad_residual = add(
            &grad_hidden,
            &self.decoder_norm.backward(&cache.residual, &grad_wide),
        );

        let (grad_embedded, grad_kv) = self
            .read_latent
            .backward_cross(&cache.cross, &grad_residual)?;
        // The embedded query is on the residual path, like the latent queries
        // in the encoder.
        let grad_embedded = add(&grad_residual, &grad_embedded);
        self.query_in.backward(&cache.features, &grad_embedded);

        Ok(self.from_latent.backward(&cache.latent, &grad_kv))
    }

    /// A closure over a fixed latent, shaped for
    /// [`crate::mesh::marching_tetrahedra`], which hands it one grid slice at a
    /// time.
    pub fn field<'a>(
        &'a self,
        latent: &'a Matrix,
        chunk: usize,
    ) -> impl Fn(&Matrix) -> Result<Vec<f32>, NetworkError> + Sync + 'a {
        move |queries| self.decode(latent, queries, chunk)
    }

    /// Moves the projections the device path owns onto a CUDA device, where
    /// every later `encode_train`, `decode_train` and backward pass runs.
    ///
    /// The normalization scales, the latent queries and the distance bias stay
    /// on the host, for the reason [`crate::gpu_shape`] gives. Fails closed: no
    /// device is an error, never a silent fallback to the CPU.
    ///
    /// `memory_budget_mib` of 0 means no ceiling. The estimate it is checked
    /// against covers weights, gradients and Adam moments, not activations.
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&mut self, device: usize, memory_budget_mib: usize) -> Result<(), NetworkError> {
        self.to_cuda_with_precision(device, memory_budget_mib, true)
    }

    /// [`ShapeVae::to_cuda`] with the tensor cores under the caller's control.
    ///
    /// Mixed precision keeps every buffer and every accumulator in FP32 and
    /// only rounds the GEMM multiplier inputs, which is what lets cuBLAS pick a
    /// tensor-core kernel instead of the FP32 SIMT one. On an RTX 3060 that is
    /// where most of a training step goes, so training wants it on. A test that
    /// compares device numbers against the host path wants it off.
    #[cfg(feature = "cuda")]
    pub fn to_cuda_with_precision(
        &mut self,
        device: usize,
        memory_budget_mib: usize,
        mixed_precision: bool,
    ) -> Result<(), NetworkError> {
        let context = crate::gpu_transformer::GpuContext::with_precision(device, mixed_precision)?;
        crate::gpu_shape::to_cuda(self, &context, memory_budget_mib)?;
        self.device = Some(context);
        Ok(())
    }

    /// Copies every device parameter back and releases the device buffers.
    #[cfg(feature = "cuda")]
    pub fn to_cpu(&mut self) -> Result<(), NetworkError> {
        crate::gpu_shape::to_cpu(self)?;
        self.device = None;
        Ok(())
    }

    /// Whether the projections currently live on a device.
    pub fn on_device(&self) -> bool {
        #[cfg(feature = "cuda")]
        return self.device.is_some();
        #[cfg(not(feature = "cuda"))]
        false
    }

    /// The device this model is running on, for a backward pass that holds
    /// `&mut self` and cannot borrow the field.
    #[cfg(feature = "cuda")]
    pub(crate) fn device_context(
        &self,
    ) -> Result<std::sync::Arc<crate::gpu_transformer::GpuContext>, NetworkError> {
        self.device.clone().ok_or_else(|| {
            NetworkError::InvalidConfig(
                "this cache came from a device the model has since left".into(),
            )
        })
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = self.surface_in.params_mut();
        params.push(&mut self.latent_queries);
        params.extend(self.read_surface.params_mut());
        for block in &mut self.blocks {
            params.extend(block.params_mut());
        }
        params.extend(self.encoder_norm.params_mut());
        params.extend(self.to_moments.params_mut());

        params.extend(self.from_latent.params_mut());
        params.extend(self.query_in.params_mut());
        params.extend(self.read_latent.params_mut());
        params.extend(self.decoder_norm.params_mut());
        params.extend(self.decoder_mlp.params_mut());
        params.extend(self.head_norm.params_mut());
        params.extend(self.head.params_mut());
        params.extend(self.color_head.params_mut());
        params.push(&mut self.head_bias);
        params
    }

    pub fn num_parameters(&mut self) -> usize {
        self.params_mut()
            .iter()
            .map(|param| param.value.data.len())
            .sum()
    }

    /// Writes the weights, the Adam moments and the configuration.
    ///
    /// The caller's metadata is written beside the configuration, which is
    /// where a training loop puts its step counter. The key `config` belongs
    /// to the model and is overwritten.
    pub fn save<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
        metadata: &BTreeMap<String, String>,
    ) -> Result<(), NetworkError> {
        let mut metadata = metadata.clone();
        metadata.insert("config".to_string(), serde_json::to_string(&self.config)?);
        crate::checkpoint::save(
            path,
            &mut crate::checkpoint::positional(self.params_mut(), "vae"),
            &metadata,
        )
    }

    /// Rebuilds a model from what [`save`](Self::save) wrote, and returns the
    /// rest of the metadata.
    ///
    /// The shape comes out of the file, so a caller does not have to remember
    /// which configuration a checkpoint was trained with; `rng` only fills
    /// weights that the file then overwrites.
    pub fn load<P: AsRef<std::path::Path>>(
        path: P,
        rng: &mut StdRng,
    ) -> Result<(Self, BTreeMap<String, String>), NetworkError> {
        let mut metadata = crate::safetensors::SafeTensors::open(path.as_ref())?
            .metadata()
            .clone();
        let config = metadata.remove("config").ok_or_else(|| {
            NetworkError::InvalidSnapshot(
                "the checkpoint carries no shape autoencoder configuration".into(),
            )
        })?;
        let mut model = Self::new(serde_json::from_str(&config)?, rng)?;
        crate::checkpoint::load(
            path,
            &mut crate::checkpoint::positional(model.params_mut(), "vae"),
        )?;
        Ok((model, metadata))
    }

    /// The Fourier-encoded position of each surface point with its normal
    /// appended.
    pub(crate) fn surface_features(&self, surface: &Matrix) -> Result<Matrix, NetworkError> {
        let positions = Matrix::from_vec(
            surface.rows,
            3,
            surface
                .data
                .chunks_exact(6)
                .flat_map(|point| point[..3].iter().copied())
                .collect(),
        );
        let encoded = fourier_features(&positions, self.config.frequencies)?;

        let width = encoded.cols + 3;
        let mut features = Matrix::new(surface.rows, width);
        for row in 0..surface.rows {
            let out = features.row_mut(row);
            out[..encoded.cols].copy_from_slice(encoded.row(row));
            out[encoded.cols..].copy_from_slice(&surface.row(row)[3..]);
        }
        Ok(features)
    }

    /// The latent widened to `d_model`, which is what a query attends to.
    ///
    /// Computed once per shape and shared by every chunk.
    fn latent_keys(&self, latent: &Matrix) -> Result<Matrix, NetworkError> {
        if latent.rows != self.config.latents || latent.cols != self.config.latent_dim {
            return Err(NetworkError::InvalidConfig(format!(
                "the latent is [{}, {}] and this model's is [{}, {}]",
                latent.rows, latent.cols, self.config.latents, self.config.latent_dim
            )));
        }
        Ok(self.from_latent.forward(latent))
    }

    fn decode_chunk(
        &self,
        latent: &Matrix,
        kv: &Matrix,
        queries: &Matrix,
    ) -> Result<(Vec<f32>, DecoderCache), NetworkError> {
        let features = fourier_features(queries, self.config.frequencies)?;
        let embedded = self.query_in.forward(&features);

        let (crossed, cross) =
            self.read_latent
                .forward_train_cross(&embedded, kv, embedded.rows, kv.rows)?;
        let residual = add(&embedded, &crossed);

        let normed = self.decoder_norm.forward(&residual);
        let (wide, mlp) = self.decoder_mlp.forward_train(&normed);
        let hidden = add(&residual, &wide);

        let normed = self.head_norm.forward(&hidden);
        let distances = self.head.forward(&normed);
        let bias = self.head_bias.value.data[0];

        Ok((
            distances.data.iter().map(|value| value + bias).collect(),
            DecoderCache {
                inner: DecoderCached::Host(Box::new(HostDecoderCache {
                    latent: latent.clone(),
                    features,
                    cross,
                    residual,
                    mlp,
                    hidden,
                    normed,
                })),
            },
        ))
    }
}

fn add(left: &Matrix, right: &Matrix) -> Matrix {
    let mut out = left.clone();
    for (slot, value) in out.data.iter_mut().zip(&right.data) {
        *slot += value;
    }
    out
}

/// Splits a `[rows, 2 * half]` matrix down the middle of each row.
pub(crate) fn split(moments: &Matrix) -> (Matrix, Matrix) {
    let half = moments.cols / 2;
    let mut left = Matrix::new(moments.rows, half);
    let mut right = Matrix::new(moments.rows, half);
    for row in 0..moments.rows {
        let source = moments.row(row);
        left.row_mut(row).copy_from_slice(&source[..half]);
        right.row_mut(row).copy_from_slice(&source[half..]);
    }
    (left, right)
}

/// The inverse of [`split`].
pub(crate) fn join(left: &Matrix, right: &Matrix) -> Matrix {
    let mut out = Matrix::new(left.rows, left.cols + right.cols);
    for row in 0..left.rows {
        let slot = out.row_mut(row);
        slot[..left.cols].copy_from_slice(left.row(row));
        slot[left.cols..].copy_from_slice(right.row(row));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// Small enough that a finite-difference sweep is cheap, wide enough that
    /// every path is exercised: two heads, one self-attention block, two
    /// Fourier octaves.
    fn tiny() -> ShapeVaeConfig {
        ShapeVaeConfig {
            d_model: 16,
            latents: 4,
            latent_dim: 3,
            num_heads: 2,
            head_dim: 8,
            d_ff: 16,
            encoder_blocks: 1,
            frequencies: 2,
            eps: 1e-5,
            free_bits: 0.0,
        }
    }

    fn rows(count: usize, width: usize, salt: usize) -> Matrix {
        Matrix::from_vec(
            count,
            width,
            (0..count * width)
                .map(|i| ((i * 37 + salt * 11) % 23) as f32 / 23.0 - 0.5)
                .collect(),
        )
    }

    fn objective(values: &[f32], weights: &Matrix) -> f32 {
        values
            .iter()
            .zip(weights.data.iter().cycle())
            .map(|(value, weight)| value * weight)
            .sum()
    }

    #[test]
    fn the_latent_is_the_shape_the_config_asks_for() {
        let mut rng = StdRng::seed_from_u64(0);
        let model = ShapeVae::new(tiny(), &mut rng).unwrap();

        // Sixteen surface points and four hundred surface points compress to
        // the same size, which is the whole point of the learned queries.
        for points in [16, 400] {
            let (mean, log_variance) = model.encode(&rows(points, 6, 1)).unwrap();
            assert_eq!((mean.rows, mean.cols), (4, 3));
            assert_eq!((log_variance.rows, log_variance.cols), (4, 3));
        }
    }

    #[test]
    fn a_surface_that_is_not_points_and_normals_is_refused() {
        let mut rng = StdRng::seed_from_u64(1);
        let model = ShapeVae::new(tiny(), &mut rng).unwrap();
        assert!(model.encode(&rows(8, 3, 1)).is_err());
        assert!(model.encode(&rows(0, 6, 1)).is_err());
    }

    #[test]
    fn a_latent_of_the_wrong_shape_is_refused() {
        let mut rng = StdRng::seed_from_u64(2);
        let model = ShapeVae::new(tiny(), &mut rng).unwrap();
        let error = model
            .decode(&rows(4, 5, 1), &rows(6, 3, 2), 8)
            .unwrap_err()
            .to_string();
        assert!(error.contains("[4, 5]"), "{error}");
    }

    #[test]
    fn encoding_a_sample_draws_around_the_mean_and_repeats_for_a_seed() {
        let mut rng = StdRng::seed_from_u64(12);
        let model = ShapeVae::new(tiny(), &mut rng).unwrap();
        let surface = rows(16, 6, 2);

        let (mean, log_variance) = model.encode(&surface).unwrap();
        let first = model
            .encode_sample(&surface, &mut StdRng::seed_from_u64(1))
            .unwrap();
        let again = model
            .encode_sample(&surface, &mut StdRng::seed_from_u64(1))
            .unwrap();
        let other = model
            .encode_sample(&surface, &mut StdRng::seed_from_u64(2))
            .unwrap();

        assert_eq!(first.rows, mean.rows);
        assert_eq!(first.cols, mean.cols);
        assert_eq!(first.data, again.data, "a seed repeats its draw");
        assert_ne!(first.data, other.data, "another seed draws elsewhere");
        assert_ne!(first.data, mean.data, "a draw is not the mean");

        // Every draw sits within a few standard deviations of the mean.
        for index in 0..mean.data.len() {
            let sigma = (0.5 * log_variance.data[index]).exp();
            let offset = (first.data[index] - mean.data[index]).abs();
            assert!(offset < 6.0 * sigma, "{offset} against a sigma of {sigma}");
        }
    }

    #[test]
    fn a_checkpoint_written_before_the_free_bits_floor_existed_reads_as_zero() {
        // The `config` metadata a checkpoint from before the field carries.
        let older = r#"{"d_model":32,"latents":4,"latent_dim":8,"num_heads":2,
            "head_dim":16,"d_ff":64,"encoder_blocks":1,"frequencies":2,"eps":1e-5}"#;
        let config: ShapeVaeConfig = serde_json::from_str(older).unwrap();
        assert_eq!(config.free_bits, 0.0, "an older run keeps its behaviour");

        // And a floor that is not a number of nats is refused rather than
        // silently trained under.
        let mut broken = tiny();
        broken.free_bits = -1.0;
        assert!(matches!(
            ShapeVae::new(broken, &mut StdRng::seed_from_u64(0)),
            Err(NetworkError::InvalidConfig(_))
        ));
    }

    #[test]
    fn chunked_decoding_matches_decoding_in_one_go() {
        let mut rng = StdRng::seed_from_u64(3);
        let model = ShapeVae::new(tiny(), &mut rng).unwrap();
        let latent = rows(4, 3, 4);
        let queries = rows(37, 3, 5);

        let whole = model.decode(&latent, &queries, 64).unwrap();
        // A chunk size that does not divide the query count, so the last
        // chunk is short.
        let pieces = model.decode(&latent, &queries, 8).unwrap();

        assert_eq!(whole.len(), 37);
        for (whole, piece) in whole.iter().zip(&pieces) {
            assert!((whole - piece).abs() < 1e-5, "{whole} vs {piece}");
        }
    }

    #[test]
    fn a_chunk_size_of_zero_is_refused() {
        let mut rng = StdRng::seed_from_u64(4);
        let model = ShapeVae::new(tiny(), &mut rng).unwrap();
        assert!(model.decode(&rows(4, 3, 1), &rows(4, 3, 2), 0).is_err());
    }

    /// The decoder's weight gradients against a central difference.
    #[test]
    fn the_decoder_gradients_match_finite_differences() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut model = ShapeVae::new(tiny(), &mut rng).unwrap();
        let latent = rows(4, 3, 6);
        let queries = rows(9, 3, 7);
        let upstream = rows(1, 9, 8);

        let (distances, cache) = model.decode_train(&latent, &queries).unwrap();
        let grad: Vec<f32> = upstream.data[..distances.len()].to_vec();
        model.decode_backward(&cache, &grad).unwrap();

        let epsilon = 1e-3;
        // One weight from each stage of the decoder, plus the bias.
        let probes: [(&str, usize); 4] = [
            ("head", 3),
            ("query_in", 11),
            ("from_latent", 5),
            ("head_bias", 0),
        ];
        for (name, index) in probes {
            let analytic = match name {
                "head" => model.head.weight.grad.data[index],
                "query_in" => model.query_in.weight.grad.data[index],
                "from_latent" => model.from_latent.weight.grad.data[index],
                _ => model.head_bias.grad.data[index],
            };

            let mut shifted = |delta: f32| {
                let slot = match name {
                    "head" => &mut model.head.weight.value.data[index],
                    "query_in" => &mut model.query_in.weight.value.data[index],
                    "from_latent" => &mut model.from_latent.weight.value.data[index],
                    _ => &mut model.head_bias.value.data[index],
                };
                let original = *slot;
                *slot = original + delta;
                let moved = model.decode(&latent, &queries, 64).unwrap();
                let slot = match name {
                    "head" => &mut model.head.weight.value.data[index],
                    "query_in" => &mut model.query_in.weight.value.data[index],
                    "from_latent" => &mut model.from_latent.weight.value.data[index],
                    _ => &mut model.head_bias.value.data[index],
                };
                *slot = original;
                objective(&moved, &upstream)
            };

            let numeric = (shifted(epsilon) - shifted(-epsilon)) / (2.0 * epsilon);
            assert!(
                (analytic - numeric).abs() < 1e-2,
                "{name}[{index}]: {analytic} vs {numeric}"
            );
        }
    }

    /// The colour head rides on the distance head's hidden state, so its
    /// gradients have to be right both in its own weights and in the trunk.
    #[test]
    fn the_colour_gradients_match_finite_differences() {
        let mut rng = StdRng::seed_from_u64(15);
        let mut model = ShapeVae::new(tiny(), &mut rng).unwrap();
        let latent = rows(4, 3, 16);
        let queries = rows(9, 3, 17);
        let upstream = rows(9, 3, 18);

        let (distances, cache) = model.decode_train(&latent, &queries).unwrap();
        model
            .decode_backward_colored(&cache, &vec![0.0; distances.len()], Some(&upstream))
            .unwrap();

        let loss = |model: &ShapeVae| {
            let (_, cache) = model.decode_train(&latent, &queries).unwrap();
            let colors = model.colors(&cache).unwrap();
            colors
                .data
                .iter()
                .zip(&upstream.data)
                .map(|(color, weight)| color * weight)
                .sum::<f32>()
        };

        let epsilon = 1e-3;
        // The head's own weight, and one in the trunk both heads share.
        for (name, index) in [("color_head", 5), ("query_in", 11)] {
            let analytic = match name {
                "color_head" => model.color_head.weight.grad.data[index],
                _ => model.query_in.weight.grad.data[index],
            };

            let mut shifted = |delta: f32| {
                let slot = match name {
                    "color_head" => &mut model.color_head.weight.value.data[index],
                    _ => &mut model.query_in.weight.value.data[index],
                };
                let original = *slot;
                *slot = original + delta;
                let moved = loss(&model);
                let slot = match name {
                    "color_head" => &mut model.color_head.weight.value.data[index],
                    _ => &mut model.query_in.weight.value.data[index],
                };
                *slot = original;
                moved
            };

            let numeric = (shifted(epsilon) - shifted(-epsilon)) / (2.0 * epsilon);
            assert!(
                (analytic - numeric).abs() < 1e-2,
                "{name}[{index}]: {analytic} vs {numeric}"
            );
        }
    }

    /// The gradient the decoder hands back to the latent, which is what joins
    /// the two halves and, later, what the flow transformer trains against.
    #[test]
    fn the_latent_gradient_matches_finite_differences() {
        let mut rng = StdRng::seed_from_u64(6);
        let mut model = ShapeVae::new(tiny(), &mut rng).unwrap();
        let mut latent = rows(4, 3, 9);
        let queries = rows(7, 3, 10);
        let upstream = rows(1, 7, 11);

        let (distances, cache) = model.decode_train(&latent, &queries).unwrap();
        let grad: Vec<f32> = upstream.data[..distances.len()].to_vec();
        let grad_latent = model.decode_backward(&cache, &grad).unwrap();

        let epsilon = 1e-3;
        for index in [0, 4, 7, 11] {
            let original = latent.data[index];
            latent.data[index] = original + epsilon;
            let high = objective(&model.decode(&latent, &queries, 64).unwrap(), &upstream);
            latent.data[index] = original - epsilon;
            let low = objective(&model.decode(&latent, &queries, 64).unwrap(), &upstream);
            latent.data[index] = original;

            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (grad_latent.data[index] - numeric).abs() < 1e-2,
                "index {index}: {} vs {numeric}",
                grad_latent.data[index]
            );
        }
    }

    /// The whole loop: surface in, distances out, gradient back to an encoder
    /// weight through both halves.
    #[test]
    fn the_encoder_gradients_match_finite_differences() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut model = ShapeVae::new(tiny(), &mut rng).unwrap();
        let surface = rows(12, 6, 12);
        let queries = rows(6, 3, 13);
        let upstream = rows(1, 6, 14);

        let (mean, _, encoder) = model.encode_train(&surface).unwrap();
        let (distances, decoder) = model.decode_train(&mean, &queries).unwrap();
        let grad: Vec<f32> = upstream.data[..distances.len()].to_vec();
        let grad_mean = model.decode_backward(&decoder, &grad).unwrap();
        // Only the mean is used, so the log-variance takes no gradient here.
        let grad_log_variance = Matrix::new(grad_mean.rows, grad_mean.cols);
        model
            .encode_backward(&encoder, &grad_mean, &grad_log_variance)
            .unwrap();

        let epsilon = 1e-3;
        for index in [0, 9, 23] {
            let analytic = model.latent_queries.grad.data[index];
            let original = model.latent_queries.value.data[index];

            model.latent_queries.value.data[index] = original + epsilon;
            let high = reconstruct(&model, &surface, &queries, &upstream);
            model.latent_queries.value.data[index] = original - epsilon;
            let low = reconstruct(&model, &surface, &queries, &upstream);
            model.latent_queries.value.data[index] = original;

            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (analytic - numeric).abs() < 1e-2,
                "index {index}: {analytic} vs {numeric}"
            );
        }
    }

    fn reconstruct(model: &ShapeVae, surface: &Matrix, queries: &Matrix, upstream: &Matrix) -> f32 {
        let (mean, _) = model.encode(surface).unwrap();
        objective(&model.decode(&mean, queries, 64).unwrap(), upstream)
    }

    #[test]
    fn the_reparameterization_gradient_matches_finite_differences() {
        let mean = rows(3, 4, 15);
        let mut log_variance = rows(3, 4, 16);
        let noise = rows(3, 4, 17);
        let upstream = rows(3, 4, 18);

        let mut latent = Matrix::new(3, 4);
        let draw = |log_variance: &Matrix, latent: &mut Matrix| {
            for index in 0..latent.data.len() {
                latent.data[index] =
                    mean.data[index] + (0.5 * log_variance.data[index]).exp() * noise.data[index];
            }
            objective(&latent.data, &upstream)
        };
        draw(&log_variance, &mut latent);

        let (grad_mean, grad_log_variance) =
            ShapeVae::sample_backward(&upstream, &log_variance, &noise);
        // The mean is on a straight path.
        assert_eq!(grad_mean.data, upstream.data);

        let epsilon = 1e-3;
        for index in [0, 5, 11] {
            let original = log_variance.data[index];
            log_variance.data[index] = original + epsilon;
            let high = draw(&log_variance, &mut latent);
            log_variance.data[index] = original - epsilon;
            let low = draw(&log_variance, &mut latent);
            log_variance.data[index] = original;

            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (grad_log_variance.data[index] - numeric).abs() < 1e-4,
                "index {index}: {} vs {numeric}",
                grad_log_variance.data[index]
            );
        }
    }

    #[test]
    fn sampling_scatters_around_the_mean_and_collapses_onto_it() {
        let mut rng = StdRng::seed_from_u64(8);
        let mean = rows(8, 8, 19);

        // A variance of one spreads the samples out.
        let (spread, noise) = ShapeVae::sample(&mean, &Matrix::new(8, 8), &mut rng);
        let moved = spread
            .data
            .iter()
            .zip(&mean.data)
            .map(|(sample, mean)| (sample - mean).abs())
            .fold(0.0f32, f32::max);
        assert!(moved > 0.5, "the samples barely moved: {moved}");
        assert_eq!(noise.rows, 8);

        // A log-variance of -40 is a standard deviation of 2e-9, so the sample
        // is the mean to the last bit of an f32.
        let mut silent = Matrix::new(8, 8);
        silent.data.fill(-40.0);
        let (collapsed, _) = ShapeVae::sample(&mean, &silent, &mut rng);
        for (sample, mean) in collapsed.data.iter().zip(&mean.data) {
            assert!((sample - mean).abs() < 1e-6);
        }
    }

    #[test]
    fn fourier_features_separate_points_a_linear_map_would_confuse() {
        let points = Matrix::from_vec(2, 3, vec![0.1, 0.2, 0.3, 0.1, 0.2, 0.35]);
        let features = fourier_features(&points, 4).unwrap();
        assert_eq!(features.cols, 3 + 24);

        // The two points differ in one coordinate by 0.05. The raw channels
        // move by that much; the top octave moves by far more, which is what
        // the encoding is for.
        let raw = (features.row(0)[2] - features.row(1)[2]).abs();
        let high = (features.row(0)[3 + 18 + 4] - features.row(1)[3 + 18 + 4]).abs();
        assert!(high > 4.0 * raw, "{high} against {raw}");

        assert!(fourier_features(&Matrix::new(2, 4), 2).is_err());
    }

    #[test]
    fn the_field_closure_drives_marching_tetrahedra() {
        // The decoder is untrained, so the surface it describes is arbitrary.
        // What is being checked is that the two halves fit together: the
        // closure takes a grid slice and hands back one distance per row.
        let mut rng = StdRng::seed_from_u64(9);
        let mut model = ShapeVae::new(tiny(), &mut rng).unwrap();
        // Push the bias so the field crosses zero somewhere inside the box.
        model.head_bias.value.data[0] = -model
            .decode(&rows(4, 3, 20), &Matrix::new(1, 3), 8)
            .unwrap()[0];

        let latent = rows(4, 3, 20);
        let mesh = crate::mesh::marching_tetrahedra(
            model.field(&latent, 4096),
            12,
            ([-1.0; 3], [1.0; 3]),
            0.0,
        )
        .unwrap();
        assert!(!mesh.indices.is_empty(), "the field never crossed zero");
    }

    /// One training step end to end, which is the real reachability check: a
    /// layer missing from `params_mut` would sit at its initialization here.
    #[test]
    fn a_training_step_moves_every_parameter() {
        use crate::losses::{clamped_l1, kl_divergence};
        use crate::optimizers::Optimizer;

        let mut rng = StdRng::seed_from_u64(10);
        let mut model = ShapeVae::new(tiny(), &mut rng).unwrap();
        let surface = rows(12, 6, 21);
        let queries = rows(10, 3, 22);
        let targets: Vec<f32> = rows(1, 10, 23).data;

        let before: Vec<Vec<f32>> = model
            .params_mut()
            .iter()
            .map(|param| param.value.data.clone())
            .collect();

        let (mean, log_variance, encoder) = model.encode_train(&surface).unwrap();
        let (latent, noise) = ShapeVae::sample(&mean, &log_variance, &mut rng);
        let (distances, decoder) = model.decode_train(&latent, &queries).unwrap();

        let (_, grad_distances) = clamped_l1(&distances, &targets, 0.1);
        // Colours too, or the colour head is the one parameter with no path to
        // the loss and the check below fails on it.
        let grad_colors = rows(queries.rows, 3, 13);
        let grad_latent = model
            .decode_backward_colored(&decoder, &grad_distances, Some(&grad_colors))
            .unwrap();

        let (mut grad_mean, mut grad_log_variance) =
            ShapeVae::sample_backward(&grad_latent, &log_variance, &noise);
        let (_, kl_mean, kl_log_variance) = kl_divergence(&mean.data, &log_variance.data);
        for (slot, value) in grad_mean.data.iter_mut().zip(&kl_mean) {
            *slot += value * 1e-3;
        }
        for (slot, value) in grad_log_variance.data.iter_mut().zip(&kl_log_variance) {
            *slot += value * 1e-3;
        }
        model
            .encode_backward(&encoder, &grad_mean, &grad_log_variance)
            .unwrap();

        let optimizer = Optimizer::adam(1e-2);
        for param in model.params_mut() {
            param.step(&optimizer, 1, 1.0);
        }

        for (index, (param, before)) in model.params_mut().iter().zip(&before).enumerate() {
            let moved = param
                .value
                .data
                .iter()
                .zip(before)
                .any(|(after, before)| after != before);
            assert!(
                moved,
                "parameter {index} ({}x{}) never received a gradient",
                param.value.rows, param.value.cols
            );
        }
    }

    /// The stage's real gate: can it learn a shape at all?
    ///
    /// One sphere, overfitted. A model that cannot memorize a single shape has
    /// a broken gradient somewhere the finite-difference checks above did not
    /// reach, and no amount of data will fix it.
    #[test]
    fn a_single_sphere_can_be_overfitted() {
        use crate::losses::clamped_l1;
        use crate::mesh::Bvh;
        use crate::optimizers::Optimizer;

        let mut rng = StdRng::seed_from_u64(11);
        let sphere = unit_sphere();
        let bvh = Bvh::build(&sphere);

        let (points, normals) = bvh.sample_surface(256, &mut rng);
        let mut surface = Matrix::new(points.len(), 6);
        for (row, (point, normal)) in points.iter().zip(&normals).enumerate() {
            let slot = surface.row_mut(row);
            slot[..3].copy_from_slice(point);
            slot[3..].copy_from_slice(normal);
        }

        // Query points near the surface, where the clamped loss actually has
        // something to say.
        let mut query_points = Vec::new();
        for _ in 0..256 {
            let radius = 0.6 + rng.gen_range(0.0..0.8);
            let (point, _) = bvh.sample_surface(1, &mut rng);
            query_points.push([
                point[0][0] * radius,
                point[0][1] * radius,
                point[0][2] * radius,
            ]);
        }
        let targets = bvh.signed_distance(&query_points);
        let queries = Matrix::from_vec(
            query_points.len(),
            3,
            query_points.iter().flatten().copied().collect(),
        );

        let config = ShapeVaeConfig {
            d_model: 48,
            latents: 8,
            latent_dim: 8,
            num_heads: 3,
            head_dim: 16,
            d_ff: 96,
            encoder_blocks: 1,
            frequencies: 6,
            eps: 1e-5,
            free_bits: 0.0,
        };
        let mut model = ShapeVae::new(config, &mut rng).unwrap();
        let optimizer = Optimizer::adam(3e-3);

        // The mean is used directly: this is a reconstruction test, and the
        // noise would only add variance to a 200-step budget.
        let mut first = 0.0;
        let mut last = 0.0;
        for step in 1..=120 {
            let (mean, _, encoder) = model.encode_train(&surface).unwrap();
            let (distances, decoder) = model.decode_train(&mean, &queries).unwrap();
            let (loss, grad) = clamped_l1(&distances, &targets, 0.2);
            if step == 1 {
                first = loss;
            }
            last = loss;

            let grad_mean = model.decode_backward(&decoder, &grad).unwrap();
            let zero = Matrix::new(grad_mean.rows, grad_mean.cols);
            model.encode_backward(&encoder, &grad_mean, &zero).unwrap();
            for param in model.params_mut() {
                param.step(&optimizer, step, 1.0);
                param.grad.zeros();
            }
        }

        assert!(
            last < first * 0.25,
            "the loss went from {first} to {last}, which is not learning"
        );
        // A tenth of the sphere's radius, from a model with 8 latents and a
        // 120-step budget.
        assert!(last < 0.1, "final loss {last}");
    }

    /// A unit sphere, wound outward.
    fn unit_sphere() -> crate::mesh::Mesh {
        let (rings, segments) = (12, 24);
        let mut mesh = crate::mesh::Mesh::default();
        for ring in 0..=rings {
            let phi = std::f32::consts::PI * ring as f32 / rings as f32;
            for segment in 0..=segments {
                let theta = std::f32::consts::TAU * segment as f32 / segments as f32;
                mesh.positions
                    .push([phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin()]);
            }
        }
        let stride = segments + 1;
        for ring in 0..rings {
            for segment in 0..segments {
                let a = (ring * stride + segment) as u32;
                let b = a + 1;
                let c = a + stride as u32;
                let d = c + 1;
                mesh.indices.push([a, b, c]);
                mesh.indices.push([b, d, c]);
            }
        }
        mesh.recompute_normals();
        mesh
    }

    #[test]
    fn a_checkpoint_round_trip_restores_the_model_and_its_metadata() {
        let path = std::env::temp_dir().join(format!(
            "rusting-brain-shape-vae-{}.safetensors",
            std::process::id()
        ));
        let mut rng = StdRng::seed_from_u64(4);
        let mut model = ShapeVae::new(tiny(), &mut rng).unwrap();
        // A fresh model is only interesting once something has moved it.
        for (index, param) in model.params_mut().into_iter().enumerate() {
            param.value.data[0] += 0.25 + index as f32 / 100.0;
        }

        let metadata = BTreeMap::from([("step".to_string(), "42".to_string())]);
        model.save(&path, &metadata).unwrap();
        let (restored, read) = ShapeVae::load(&path, &mut StdRng::seed_from_u64(99)).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(restored, model, "the restored model is not the saved one");
        assert_eq!(read, metadata, "the caller's metadata did not survive");
    }
}
