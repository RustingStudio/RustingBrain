//! The flow-matching transformer that turns an image into a shape latent.
//!
//! This is the generative half of the pipeline. The shape autoencoder in
//! [`crate::shape_vae`] defines what a latent *means*; this model learns to
//! produce one from a picture. It is trained by flow matching: a clean latent
//! and a Gaussian one are mixed at a random point along the straight line
//! between them, and the model is asked for the constant velocity that joins
//! them. Sampling then integrates that velocity from noise back to a latent.
//!
//! The trunk is 512 latent slots carried through a stack of blocks, each of
//! which is self-attention over the slots, cross-attention to the frozen ViT
//! tokens, and a SwiGLU — every sub-layer modulated from the timestep by
//! AdaLN-single, which is the cheap variant: one shared projection for the
//! whole model plus a `[1, 3 * d_model]` offset per sub-layer.
//!
//! ```no_run
//! # use rusting_brain::flow_transformer::{FlowConfig, FlowTransformer};
//! # use rusting_brain::matrix::Matrix;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0);
//! let config = FlowConfig::default();
//! let mut model = FlowTransformer::new(config, &mut rng)?;
//!
//! # let tokens = Matrix::new(config.cond_tokens, config.cond_dim);
//! # let clean = Matrix::new(config.latents, config.latent_dim);
//! // One training step: noise, a timestep, the velocity the model should have
//! // predicted, and the gradient of the error in it.
//! let loss = model.train_step(&clean, &tokens, 1.0, &mut rng)?;
//! model.step_clipped(1.0, 1.0)?;
//! # let _ = loss;
//! # Ok(())
//! # }
//! ```
//!
//! NOTE: CPU only. The GPU kernels in `gpu_cross` and `gpu_transformer`
//! are reached through `gpu_model`, which is built around `TransformerLm`'s
//! layer stack and would need generalizing before anything else can use them.
//! That is its own stage, and this one is what it would be tested against.

use crate::adaln::{
    AdaLayerNorm, AdaLnCache, Modulation, ModulationCache, TimestepCache, TimestepEmbedding,
    gate_residual, gate_residual_backward,
};
use crate::attention::{AttentionCache, CrossAttentionCache, MultiHeadAttention};
use crate::batch::Layout;
use crate::diffusion::{Denoiser, Scheduler, gaussian};
use crate::ffn::{SwiGlu, SwiGluCache};
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::optimizers::Optimizer;
use crate::param::{Linear, Param};
use crate::rope::Rope;
use rand::Rng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The shape of a [`FlowTransformer`].
///
/// The defaults are the plan's: 512 latent slots of 64 dimensions, 16 blocks
/// of width 1,024, conditioned on a ViT-B tower's `[197, 768]` tokens.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct FlowConfig {
    pub d_model: usize,
    /// Latent slots, which must match the shape autoencoder's.
    pub latents: usize,
    /// The width of one latent slot, likewise.
    pub latent_dim: usize,
    /// The width of one conditioning token — 768 for ViT-B.
    pub cond_dim: usize,
    /// How many conditioning tokens there are — 197 for ViT-B/16 at 224.
    pub cond_tokens: usize,
    pub blocks: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub d_ff: usize,
    /// The width of the timestep conditioning the blocks are modulated from.
    pub d_cond: usize,
    /// The sinusoidal basis the timestep is expanded into. Even.
    pub frequencies: usize,
    pub eps: f32,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            d_model: 1024,
            latents: 512,
            latent_dim: 64,
            cond_dim: 768,
            cond_tokens: 197,
            blocks: 16,
            num_heads: 16,
            head_dim: 64,
            d_ff: 4096,
            d_cond: 512,
            frequencies: 256,
            eps: 1e-5,
        }
    }
}

/// A projection that starts at zero, so what it feeds starts at zero too.
fn zero_linear(in_features: usize, out_features: usize) -> Linear {
    Linear {
        weight: Param::zeros(out_features, in_features),
        lora: None,
    }
}

/// Self-attention over the slots, cross-attention to the condition, a SwiGLU.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct FlowBlock {
    pub(crate) self_norm: AdaLayerNorm,
    pub(crate) self_attention: MultiHeadAttention,
    pub(crate) cross_norm: AdaLayerNorm,
    pub(crate) cross_attention: MultiHeadAttention,
    pub(crate) mlp_norm: AdaLayerNorm,
    pub(crate) mlp: SwiGlu,
}

struct BlockCache {
    self_norm: AdaLnCache,
    self_attention: AttentionCache,
    self_gate: Matrix,
    self_branch: Matrix,

    cross_norm: AdaLnCache,
    cross_attention: CrossAttentionCache,
    cross_gate: Matrix,
    cross_branch: Matrix,

    mlp_norm: AdaLnCache,
    mlp: SwiGluCache,
    mlp_gate: Matrix,
    mlp_branch: Matrix,
}

impl FlowBlock {
    fn new(config: &FlowConfig, rng: &mut StdRng) -> Result<Self, NetworkError> {
        let rope = Rope::new(config.head_dim, 1, 10000.0)?;
        let mut self_attention = MultiHeadAttention::new(
            config.d_model,
            config.num_heads,
            config.num_heads,
            config.head_dim,
            rope.clone(),
            rng,
        )?;
        // The slots are an unordered set: every one reads every other, and
        // none of them has a position to rotate against.
        self_attention.set_causal(false);
        self_attention.set_rope_enabled(false);

        Ok(Self {
            self_norm: AdaLayerNorm::new(config.d_model, config.eps),
            self_attention,
            cross_norm: AdaLayerNorm::new(config.d_model, config.eps),
            cross_attention: MultiHeadAttention::cross(
                config.d_model,
                config.cond_dim,
                config.num_heads,
                config.num_heads,
                config.head_dim,
                rope,
                rng,
            )?,
            mlp_norm: AdaLayerNorm::new(config.d_model, config.eps),
            mlp: SwiGlu::new(config.d_model, config.d_ff, rng),
        })
    }

    fn forward_train(
        &self,
        hidden: &Matrix,
        triple: &Matrix,
        tokens: &Matrix,
        latents: usize,
        cond_tokens: usize,
    ) -> Result<(Matrix, BlockCache), NetworkError> {
        let (self_modulated, self_gate, self_norm) =
            self.self_norm.forward_train(hidden, triple, latents)?;
        let (self_branch, self_attention) = self.self_attention.forward_train(
            &self_modulated,
            Layout {
                seq_len: Some(latents),
                valid: None,
            },
        )?;
        let hidden = gate_residual(hidden, &self_branch, &self_gate, latents)?;

        let (cross_modulated, cross_gate, cross_norm) =
            self.cross_norm.forward_train(&hidden, triple, latents)?;
        let (cross_branch, cross_attention) = self.cross_attention.forward_train_cross(
            &cross_modulated,
            tokens,
            latents,
            cond_tokens,
        )?;
        let hidden = gate_residual(&hidden, &cross_branch, &cross_gate, latents)?;

        let (mlp_modulated, mlp_gate, mlp_norm) =
            self.mlp_norm.forward_train(&hidden, triple, latents)?;
        let (mlp_branch, mlp) = self.mlp.forward_train(&mlp_modulated);
        let output = gate_residual(&hidden, &mlp_branch, &mlp_gate, latents)?;

        Ok((
            output,
            BlockCache {
                self_norm,
                self_attention,
                self_gate,
                self_branch,
                cross_norm,
                cross_attention,
                cross_gate,
                cross_branch,
                mlp_norm,
                mlp,
                mlp_gate,
                mlp_branch,
            },
        ))
    }

    /// Returns `(dL/dhidden, dL/dtriple, dL/dtokens)`.
    ///
    /// The last is what a trainable image tower would read. This one is
    /// frozen, so the caller drops it — but the cross-attention projections
    /// still want their own weight gradients, which they take on the way.
    fn backward(
        &mut self,
        cache: &BlockCache,
        grad_output: &Matrix,
        latents: usize,
    ) -> Result<(Matrix, Matrix, Matrix), NetworkError> {
        // Each sub-layer is `hidden + gate * branch`, so the gradient reaching
        // the sub-layer's input is the one arriving plus the branch's share.
        let (grad_mlp_branch, grad_mlp_gate) =
            gate_residual_backward(&cache.mlp_branch, &cache.mlp_gate, grad_output, latents)?;
        let grad_mlp_modulated = self.mlp.backward(&cache.mlp, &grad_mlp_branch);
        let (grad_hidden, mut grad_triple) =
            self.mlp_norm
                .backward(&cache.mlp_norm, &grad_mlp_modulated, &grad_mlp_gate)?;
        let grad_hidden = add(grad_output, &grad_hidden);

        let (grad_cross_branch, grad_cross_gate) = gate_residual_backward(
            &cache.cross_branch,
            &cache.cross_gate,
            &grad_hidden,
            latents,
        )?;
        let (grad_cross_modulated, grad_tokens) = self
            .cross_attention
            .backward_cross(&cache.cross_attention, &grad_cross_branch)?;
        let (grad_into_cross, triple) =
            self.cross_norm
                .backward(&cache.cross_norm, &grad_cross_modulated, &grad_cross_gate)?;
        accumulate(&mut grad_triple, &triple);
        let grad_hidden = add(&grad_hidden, &grad_into_cross);

        let (grad_self_branch, grad_self_gate) =
            gate_residual_backward(&cache.self_branch, &cache.self_gate, &grad_hidden, latents)?;
        let grad_self_modulated = self
            .self_attention
            .backward(&cache.self_attention, &grad_self_branch)?;
        let (grad_into_self, triple) =
            self.self_norm
                .backward(&cache.self_norm, &grad_self_modulated, &grad_self_gate)?;
        accumulate(&mut grad_triple, &triple);

        Ok((add(&grad_hidden, &grad_into_self), grad_triple, grad_tokens))
    }

    fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = self.self_norm.params_mut();
        params.extend(self.self_attention.params_mut());
        params.extend(self.cross_norm.params_mut());
        params.extend(self.cross_attention.params_mut());
        params.extend(self.mlp_norm.params_mut());
        params.extend(self.mlp.params_mut());
        params
    }
}

/// What [`FlowTransformer::forward_train`] hands to its backward pass.
///
/// Opaque on purpose: a cache from a model on a device holds device buffers
/// and one from a host model holds matrices, and a caller only ever carries it
/// from the forward pass to the backward one.
pub struct FlowCache {
    inner: Cached,
}

enum Cached {
    Host(HostCache),
    #[cfg(feature = "cuda")]
    Device(crate::gpu_flow::GpuFlowCache),
}

struct HostCache {
    latent: Matrix,
    tokens: Matrix,
    timestep: TimestepCache,
    modulation: ModulationCache,
    blocks: Vec<BlockCache>,
    final_norm: AdaLnCache,
    final_modulated: Matrix,
    sequences: usize,
}

/// Noise and a picture in, a shape latent out.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlowTransformer {
    pub(crate) config: FlowConfig,
    pub(crate) timestep: TimestepEmbedding,
    pub(crate) modulation: Modulation,
    /// The noisy latent widened to the trunk.
    pub(crate) latent_in: Linear,
    /// One learned vector per slot, which is what gives an unordered set of
    /// latents an identity to specialize into.
    pub(crate) slots: Param,
    pub(crate) blocks: Vec<FlowBlock>,
    pub(crate) final_norm: AdaLayerNorm,
    /// Back down to the latent width. Zero-initialized, so an untrained model
    /// predicts zero velocity rather than a random one.
    pub(crate) latent_out: Linear,
    /// What the conditioning is replaced with when it is dropped, both during
    /// training and for the unconditional half of classifier-free guidance.
    null_tokens: Param,
    optimizer: Optimizer,
    optimizer_step: usize,
    /// Set by [`FlowTransformer::to_cuda`]. Never serialized: a snapshot is
    /// host data, and a restored model starts on the CPU.
    #[cfg(feature = "cuda")]
    #[serde(skip)]
    device: Option<std::sync::Arc<crate::gpu_transformer::GpuContext>>,
}

/// Compares the model, not where it happens to be running: two models with the
/// same weights are equal whether or not one of them holds a device context.
impl PartialEq for FlowTransformer {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config
            && self.timestep == other.timestep
            && self.modulation == other.modulation
            && self.latent_in == other.latent_in
            && self.slots == other.slots
            && self.blocks == other.blocks
            && self.final_norm == other.final_norm
            && self.latent_out == other.latent_out
            && self.null_tokens == other.null_tokens
            && self.optimizer == other.optimizer
            && self.optimizer_step == other.optimizer_step
    }
}

impl FlowTransformer {
    pub fn new(config: FlowConfig, rng: &mut StdRng) -> Result<Self, NetworkError> {
        if config.latents == 0 || config.latent_dim == 0 || config.cond_tokens == 0 {
            return Err(NetworkError::InvalidConfig(
                "a flow transformer needs at least one latent slot and one conditioning token"
                    .into(),
            ));
        }

        let mut blocks = Vec::with_capacity(config.blocks);
        for _ in 0..config.blocks {
            blocks.push(FlowBlock::new(&config, rng)?);
        }

        Ok(Self {
            config,
            timestep: TimestepEmbedding::new(
                config.frequencies,
                config.d_cond,
                config.d_cond,
                rng,
            )?,
            modulation: Modulation::new(config.d_cond, config.d_model),
            latent_in: Linear::new(config.latent_dim, config.d_model, rng),
            slots: Param::he_uniform(config.latents, config.d_model, config.d_model, rng),
            blocks,
            final_norm: AdaLayerNorm::new(config.d_model, config.eps),
            latent_out: zero_linear(config.d_model, config.latent_dim),
            null_tokens: Param::he_uniform(
                config.cond_tokens,
                config.cond_dim,
                config.cond_dim,
                rng,
            ),
            optimizer: Optimizer::adam(1e-4),
            optimizer_step: 0,
            #[cfg(feature = "cuda")]
            device: None,
        })
    }

    pub fn config(&self) -> &FlowConfig {
        &self.config
    }

    pub fn set_optimizer(&mut self, optimizer: Optimizer) {
        self.optimizer = optimizer;
    }

    /// The learned stand-in for a dropped condition, `[cond_tokens, cond_dim]`.
    pub fn null_tokens(&self) -> &Matrix {
        &self.null_tokens.value
    }

    /// The predicted velocity at `time`, `[sequences * latents, latent_dim]`.
    ///
    /// `time` runs from 1.0 at pure noise down to 0.0 at a clean latent, which
    /// is the same quantity [`Scheduler::sigmas`] produces.
    pub fn forward(
        &self,
        latent: &Matrix,
        tokens: &Matrix,
        times: &[f32],
    ) -> Result<Matrix, NetworkError> {
        Ok(self.forward_train(latent, tokens, times)?.0)
    }

    /// The same, keeping what the backward pass needs.
    pub fn forward_train(
        &self,
        latent: &Matrix,
        tokens: &Matrix,
        times: &[f32],
    ) -> Result<(Matrix, FlowCache), NetworkError> {
        #[cfg(feature = "cuda")]
        if let Some(context) = &self.device {
            let (velocity, cache) =
                crate::gpu_flow::forward_train(self, context, latent, tokens, times)?;
            return Ok((
                velocity,
                FlowCache {
                    inner: Cached::Device(cache),
                },
            ));
        }

        let sequences = self.check(latent, tokens, times)?;
        let config = self.config;

        let (conditioning, timestep) = self.timestep.forward_train(times);
        let (triple, modulation) = self.modulation.forward_train(&conditioning);

        // The slot identities are the same for every sequence in the batch.
        let mut embedded = self.latent_in.forward(latent);
        for row in 0..embedded.rows {
            let slot = self.slots.value.row(row % config.latents);
            for (value, identity) in embedded.row_mut(row).iter_mut().zip(slot) {
                *value += identity;
            }
        }

        let mut hidden = embedded;
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (output, cache) = block.forward_train(
                &hidden,
                &triple,
                tokens,
                config.latents,
                config.cond_tokens,
            )?;
            hidden = output;
            blocks.push(cache);
        }

        // The final layer takes a shift and a scale but no gate: there is no
        // residual left for a gate to open onto.
        let (final_modulated, _, final_norm) =
            self.final_norm
                .forward_train(&hidden, &triple, config.latents)?;
        let velocity = self.latent_out.forward(&final_modulated);

        Ok((
            velocity,
            FlowCache {
                inner: Cached::Host(HostCache {
                    latent: latent.clone(),
                    tokens: tokens.clone(),
                    timestep,
                    modulation,
                    blocks,
                    final_norm,
                    final_modulated,
                    sequences,
                }),
            },
        ))
    }

    /// Accumulates every weight gradient and returns `dL/dtokens`.
    ///
    /// The image tower is frozen, so the caller normally drops what comes
    /// back; it is returned rather than swallowed because the gradient is what
    /// a later stage would need to fine-tune the tower.
    pub fn backward(
        &mut self,
        cache: &FlowCache,
        grad_velocity: &Matrix,
    ) -> Result<Matrix, NetworkError> {
        // One arm without the `cuda` feature, which is what the allow is for.
        #[allow(clippy::infallible_destructuring_match)]
        let cache = match &cache.inner {
            Cached::Host(cache) => cache,
            #[cfg(feature = "cuda")]
            Cached::Device(cache) => {
                // Cloning the handle, not the context: the borrow below is
                // `&mut self`, and the context lives in a field of it.
                let context = self.device.clone().ok_or_else(|| {
                    NetworkError::InvalidConfig(
                        "this cache came from a device the model has since left".into(),
                    )
                })?;
                return crate::gpu_flow::backward(self, &context, cache, grad_velocity);
            }
        };
        let config = self.config;
        let grad_final = self
            .latent_out
            .backward(&cache.final_modulated, grad_velocity);
        // No gate came out of the final layer, so none goes back into it.
        let zero_gate = Matrix::new(cache.sequences, config.d_model);
        let (mut grad_hidden, mut grad_triple) =
            self.final_norm
                .backward(&cache.final_norm, &grad_final, &zero_gate)?;

        let mut grad_tokens = Matrix::new(cache.tokens.rows, cache.tokens.cols);
        for (block, cache) in self.blocks.iter_mut().zip(&cache.blocks).rev() {
            let (hidden, triple, tokens) = block.backward(cache, &grad_hidden, config.latents)?;
            grad_hidden = hidden;
            accumulate(&mut grad_triple, &triple);
            accumulate(&mut grad_tokens, &tokens);
        }

        // Every sequence's rows share one set of slot identities, so the slot
        // gradient sums over the batch.
        if !self.slots.is_frozen() {
            for row in 0..grad_hidden.rows {
                let target = self.slots.grad.row_mut(row % config.latents);
                for (slot, value) in target.iter_mut().zip(grad_hidden.row(row)) {
                    *slot += value;
                }
            }
        }
        self.latent_in.backward(&cache.latent, &grad_hidden);

        let grad_conditioning = self.modulation.backward(&cache.modulation, &grad_triple);
        self.timestep.backward(&cache.timestep, &grad_conditioning);

        Ok(grad_tokens)
    }

    /// One flow-matching step: mix, predict, score, and accumulate gradients.
    ///
    /// `clean` is the shape autoencoder's latent for the target mesh and
    /// `tokens` the frozen image tokens that describe it. `dropout` is the
    /// probability that a sequence's condition is replaced by
    /// [`null_tokens`](Self::null_tokens), which is what makes
    /// classifier-free guidance possible at sampling time; 0.1 is the usual
    /// value and 0.0 turns it off.
    ///
    /// Returns the mean squared error against the velocity that actually joins
    /// the two endpoints.
    pub fn train_step<R: Rng>(
        &mut self,
        clean: &Matrix,
        tokens: &Matrix,
        dropout: f32,
        rng: &mut R,
    ) -> Result<f32, NetworkError> {
        let sequences = self.check(clean, tokens, &[])?;
        let times = sample_timesteps(sequences, 1.0, rng);

        let mut noise = Matrix::new(clean.rows, clean.cols);
        for value in &mut noise.data {
            *value = gaussian(rng);
        }
        let target = flow_match_target(clean, &noise)?;

        // `x_t = (1 - t) * clean + t * noise`, one time per sequence.
        let mut noisy = clean.clone();
        for row in 0..noisy.rows {
            let time = times[row / self.config.latents];
            for (value, noise) in noisy.row_mut(row).iter_mut().zip(noise.row(row)) {
                *value = (1.0 - time) * *value + time * noise;
            }
        }

        let (tokens, dropped) = self.drop_condition(tokens, dropout, rng)?;
        let (velocity, cache) = self.forward_train(&noisy, &tokens, &times)?;

        let count = velocity.data.len().max(1) as f32;
        let mut loss = 0.0;
        let mut grad = Matrix::new(velocity.rows, velocity.cols);
        for index in 0..velocity.data.len() {
            let difference = velocity.data[index] - target.data[index];
            loss += difference * difference;
            grad.data[index] = 2.0 * difference / count;
        }
        let grad_tokens = self.backward(&cache, &grad)?;

        // Wherever the condition was dropped, the gradient the blocks handed
        // back to "the tokens" is the null embedding's, since that is what
        // stood in for them. Nothing else trains it.
        if !self.null_tokens.is_frozen() {
            let width = self.config.cond_tokens;
            for (sequence, _) in dropped.iter().enumerate().filter(|(_, hit)| **hit) {
                for token in 0..width {
                    let target = self.null_tokens.grad.row_mut(token);
                    for (slot, value) in target
                        .iter_mut()
                        .zip(grad_tokens.row(sequence * width + token))
                    {
                        *slot += value;
                    }
                }
            }
        }
        Ok(loss / count)
    }

    /// Replaces each sequence's conditioning tokens with the learned null
    /// embedding, independently, with probability `dropout`.
    ///
    /// The flags say which sequences were replaced, which is what lets the
    /// backward pass send those sequences' token gradient to the null
    /// embedding instead of throwing it away.
    pub fn drop_condition<R: Rng>(
        &self,
        tokens: &Matrix,
        dropout: f32,
        rng: &mut R,
    ) -> Result<(Matrix, Vec<bool>), NetworkError> {
        if !(0.0..=1.0).contains(&dropout) {
            return Err(NetworkError::InvalidConfig(format!(
                "a dropout probability is between 0 and 1, not {dropout}"
            )));
        }
        let sequences = tokens.rows / self.config.cond_tokens;
        if dropout == 0.0 {
            return Ok((tokens.clone(), vec![false; sequences]));
        }

        let mut out = tokens.clone();
        let mut hits = vec![false; sequences];
        for (sequence, hit) in hits.iter_mut().enumerate() {
            if rng.gen_range(0.0..1.0) >= dropout {
                continue;
            }
            *hit = true;
            for token in 0..self.config.cond_tokens {
                out.row_mut(sequence * self.config.cond_tokens + token)
                    .copy_from_slice(self.null_tokens.value.row(token));
            }
        }
        Ok((out, hits))
    }

    /// Applies the optimizer with the gradients clipped to a global norm, and
    /// returns the pre-clip norm.
    pub fn step_clipped(&mut self, scale: f32, max_norm: f32) -> Result<f32, NetworkError> {
        self.optimizer_step += 1;
        let step = self.optimizer_step;
        let optimizer = self.optimizer.clone();
        crate::optimizers::step_clipped(&mut self.params_mut(), &optimizer, step, scale, max_norm)
    }

    pub fn zero_grad(&mut self) {
        crate::optimizers::zero_grad(&mut self.params_mut());
    }

    pub fn grad_norm(&mut self) -> Result<f32, NetworkError> {
        crate::optimizers::grad_norm(&mut self.params_mut())
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = self.timestep.params_mut();
        params.extend(self.modulation.params_mut());
        params.extend(self.latent_in.params_mut());
        params.push(&mut self.slots);
        for block in &mut self.blocks {
            params.extend(block.params_mut());
        }
        params.extend(self.final_norm.params_mut());
        params.extend(self.latent_out.params_mut());
        params.push(&mut self.null_tokens);
        params
    }

    pub fn num_parameters(&mut self) -> usize {
        self.params_mut()
            .iter()
            .map(|param| param.value.data.len())
            .sum()
    }

    /// Writes the weights, the Adam moments and the configuration, as
    /// [`ShapeVae::save`](crate::shape_vae::ShapeVae::save) does.
    ///
    /// The optimizer's step counter goes in the metadata as well, so a resumed
    /// run restores Adam's bias correction exactly instead of taking a few
    /// oversized steps with a counter that restarted at zero.
    pub fn save<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
        metadata: &BTreeMap<String, String>,
    ) -> Result<(), NetworkError> {
        let mut metadata = metadata.clone();
        metadata.insert("config".to_string(), serde_json::to_string(&self.config)?);
        metadata.insert(
            "optimizer_step".to_string(),
            self.optimizer_step.to_string(),
        );
        crate::checkpoint::save(
            path,
            &mut crate::checkpoint::positional(self.params_mut(), "flow"),
            &metadata,
        )
    }

    /// Rebuilds a model from what [`save`](Self::save) wrote, and returns the
    /// rest of the metadata.
    pub fn load<P: AsRef<std::path::Path>>(
        path: P,
        rng: &mut StdRng,
    ) -> Result<(Self, BTreeMap<String, String>), NetworkError> {
        let mut metadata = crate::safetensors::SafeTensors::open(path.as_ref())?
            .metadata()
            .clone();
        let config = metadata.remove("config").ok_or_else(|| {
            NetworkError::InvalidSnapshot(
                "the checkpoint carries no flow transformer configuration".into(),
            )
        })?;
        let mut model = Self::new(serde_json::from_str(&config)?, rng)?;
        // A checkpoint from before the counter was written carries none, and
        // starting it at zero is what that run already did.
        model.optimizer_step = metadata
            .remove("optimizer_step")
            .and_then(|step| step.parse().ok())
            .unwrap_or(0);
        crate::checkpoint::load(
            path,
            &mut crate::checkpoint::positional(model.params_mut(), "flow"),
        )?;
        Ok((model, metadata))
    }

    /// Moves the blocks onto a CUDA device, where every later `forward_train`,
    /// `backward` and `train_step` runs.
    ///
    /// The attention projections and the feed-forwards move; the timestep
    /// embedding, the shared modulation, the latent projections, the slot
    /// identities and the normalization scales stay on the host, because the
    /// conditioning triple is small next to the cost of splitting four more
    /// modules across two paths. Fails closed: no device is an error, never a
    /// silent fallback to the CPU.
    ///
    /// `memory_budget_mib` of 0 means no ceiling. The estimate it is checked
    /// against covers weights, gradients and Adam moments, not activations.
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&mut self, device: usize, memory_budget_mib: usize) -> Result<(), NetworkError> {
        self.to_cuda_with_precision(device, memory_budget_mib, true)
    }

    /// [`FlowTransformer::to_cuda`] with the tensor cores under the caller's
    /// control, for the reason
    /// [`ShapeVae::to_cuda_with_precision`](crate::shape_vae::ShapeVae::to_cuda_with_precision)
    /// gives.
    #[cfg(feature = "cuda")]
    pub fn to_cuda_with_precision(
        &mut self,
        device: usize,
        memory_budget_mib: usize,
        mixed_precision: bool,
    ) -> Result<(), NetworkError> {
        let context = crate::gpu_transformer::GpuContext::with_precision(device, mixed_precision)?;
        crate::gpu_flow::to_cuda(self, &context, memory_budget_mib)?;
        self.device = Some(context);
        Ok(())
    }

    /// Copies every device parameter back and releases the device buffers.
    #[cfg(feature = "cuda")]
    pub fn to_cpu(&mut self) -> Result<(), NetworkError> {
        crate::gpu_flow::to_cpu(self)?;
        self.device = None;
        Ok(())
    }

    /// Refreshes the host copies of the device parameters, keeping residency.
    ///
    /// Call this before reading weights from a model that is training on a
    /// device: the host values go stale at the first device optimizer step.
    /// [`save`](Self::save) and [`crate::checkpoint::save`] already do it.
    #[cfg(feature = "cuda")]
    pub fn sync_from_device(&mut self) -> Result<(), NetworkError> {
        for param in self.params_mut() {
            param.sync_from_device()?;
        }
        Ok(())
    }

    /// Whether the blocks currently live on a device.
    pub fn on_device(&self) -> bool {
        #[cfg(feature = "cuda")]
        return self.device.is_some();
        #[cfg(not(feature = "cuda"))]
        false
    }

    /// Checks the shapes and returns how many sequences are packed in.
    pub(crate) fn check(
        &self,
        latent: &Matrix,
        tokens: &Matrix,
        times: &[f32],
    ) -> Result<usize, NetworkError> {
        let config = self.config;
        if latent.cols != config.latent_dim || latent.rows % config.latents != 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "the latent is [{}, {}] and this model wants a multiple of [{}, {}]",
                latent.rows, latent.cols, config.latents, config.latent_dim
            )));
        }
        if tokens.cols != config.cond_dim || tokens.rows % config.cond_tokens != 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "the condition is [{}, {}] and this model wants a multiple of [{}, {}]",
                tokens.rows, tokens.cols, config.cond_tokens, config.cond_dim
            )));
        }

        let sequences = latent.rows / config.latents;
        if tokens.rows / config.cond_tokens != sequences {
            return Err(NetworkError::InvalidConfig(format!(
                "{sequences} latents against {} conditions",
                tokens.rows / config.cond_tokens
            )));
        }
        if !times.is_empty() && times.len() != sequences {
            return Err(NetworkError::InvalidConfig(format!(
                "{} timesteps for {sequences} sequences",
                times.len()
            )));
        }
        Ok(sequences)
    }
}

/// The velocity that joins a clean latent to a noise sample.
///
/// Flow matching's whole trick: the path between the two endpoints is a
/// straight line, so the velocity along it is constant and the target does not
/// depend on where along the line the sample was taken.
pub fn flow_match_target(clean: &Matrix, noise: &Matrix) -> Result<Matrix, NetworkError> {
    if clean.rows != noise.rows || clean.cols != noise.cols {
        return Err(NetworkError::InvalidTarget {
            expected: clean.data.len(),
            actual: noise.data.len(),
        });
    }
    let mut target = noise.clone();
    for (value, clean) in target.data.iter_mut().zip(&clean.data) {
        *value -= clean;
    }
    Ok(target)
}

/// Training timesteps, logit-normal and shift-aware.
///
/// A uniform draw spends as much of the budget on the nearly-clean end, where
/// the model has almost nothing left to do, as on the middle, where the whole
/// problem is. The logit-normal draw concentrates on the middle instead.
///
/// `shift` then bends the result towards the noisy end, which is what a model
/// with many latents wants: the more dimensions the sample has, the more of
/// the work happens early. A shift of 1.0 leaves the draw alone.
pub fn sample_timesteps<R: Rng>(count: usize, shift: f32, rng: &mut R) -> Vec<f32> {
    (0..count)
        .map(|_| {
            let time = 1.0 / (1.0 + (-gaussian(rng)).exp());
            match shift == 1.0 {
                true => time,
                false => shift * time / (1.0 + (shift - 1.0) * time),
            }
        })
        .collect()
}

/// A [`Denoiser`] over a trained model and one image's tokens.
///
/// This is what plugs the flow transformer into [`crate::diffusion::sample`].
/// The guidance scale lives in [`crate::diffusion::SamplingConfig`]; what is
/// implemented here is the unconditional branch it needs.
pub struct FlowDenoiser<'a> {
    pub model: &'a FlowTransformer,
    pub tokens: &'a Matrix,
}

impl Denoiser for FlowDenoiser<'_> {
    fn denoise(&mut self, latents: &Matrix, sigma: f32) -> Result<Matrix, NetworkError> {
        self.model.forward(latents, self.tokens, &[sigma])
    }

    fn denoise_unconditional(
        &mut self,
        latents: &Matrix,
        sigma: f32,
    ) -> Result<Option<Matrix>, NetworkError> {
        let null = self.model.null_tokens().clone();
        Ok(Some(self.model.forward(latents, &null, &[sigma])?))
    }
}

/// The schedule this model is trained against, for a caller assembling a
/// sampling run.
pub fn scheduler(shift: f32) -> Scheduler {
    Scheduler::flow_match(shift)
}

fn add(left: &Matrix, right: &Matrix) -> Matrix {
    let mut out = left.clone();
    accumulate(&mut out, right);
    out
}

fn accumulate(target: &mut Matrix, source: &Matrix) {
    for (slot, value) in target.data.iter_mut().zip(&source.data) {
        *slot += value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffusion::{SamplingConfig, sample};
    use rand::SeedableRng;

    /// Small enough for a finite-difference sweep, wide enough that every path
    /// is exercised: two blocks, two heads, four latent slots, three tokens.
    fn tiny() -> FlowConfig {
        FlowConfig {
            d_model: 16,
            latents: 4,
            latent_dim: 3,
            cond_dim: 5,
            cond_tokens: 3,
            blocks: 2,
            num_heads: 2,
            head_dim: 8,
            d_ff: 16,
            d_cond: 8,
            frequencies: 4,
            eps: 1e-5,
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

    fn objective(values: &Matrix, weights: &Matrix) -> f32 {
        values
            .data
            .iter()
            .zip(weights.data.iter().cycle())
            .map(|(value, weight)| value * weight)
            .sum()
    }

    /// A fresh model is deliberately the identity — the AdaLN offsets, the
    /// modulation projection and the output projection are all zero, so every
    /// gate is shut and every gradient upstream of the output is exactly zero.
    /// That is right for training and useless for a gradient check, so the
    /// tests below shake the model awake first.
    fn awaken(model: &mut FlowTransformer, rng: &mut StdRng) {
        for param in model.params_mut() {
            for value in &mut param.value.data {
                *value += rng.gen_range(-0.3..0.3);
            }
        }
    }

    /// Reads one parameter's value out by its position in `params_mut`.
    fn probe(model: &mut FlowTransformer, param: usize, index: usize) -> (f32, f32) {
        let params = model.params_mut();
        (
            params[param].value.data[index],
            params[param].grad.data[index],
        )
    }

    fn poke(model: &mut FlowTransformer, param: usize, index: usize, value: f32) {
        model.params_mut()[param].value.data[index] = value;
    }

    #[test]
    fn the_velocity_is_the_shape_of_the_latent_it_was_given() {
        let mut rng = StdRng::seed_from_u64(0);
        let config = tiny();
        let model = FlowTransformer::new(config, &mut rng).unwrap();

        // Two sequences packed into one call, which is how a batch is fed.
        for sequences in [1, 2] {
            let velocity = model
                .forward(
                    &rows(sequences * config.latents, config.latent_dim, 1),
                    &rows(sequences * config.cond_tokens, config.cond_dim, 2),
                    &vec![0.5; sequences],
                )
                .unwrap();
            assert_eq!(
                (velocity.rows, velocity.cols),
                (sequences * config.latents, config.latent_dim)
            );
        }
    }

    #[test]
    fn shapes_that_do_not_line_up_are_refused_by_name() {
        let mut rng = StdRng::seed_from_u64(1);
        let config = tiny();
        let model = FlowTransformer::new(config, &mut rng).unwrap();
        let latent = rows(config.latents, config.latent_dim, 1);
        let tokens = rows(config.cond_tokens, config.cond_dim, 2);

        let error = model
            .forward(&rows(4, 7, 1), &tokens, &[0.5])
            .unwrap_err()
            .to_string();
        assert!(error.contains("[4, 7]"), "{error}");

        // One latent, two conditions.
        assert!(
            model
                .forward(
                    &latent,
                    &rows(2 * config.cond_tokens, config.cond_dim, 2),
                    &[0.5]
                )
                .is_err()
        );
        // Two timesteps, one sequence.
        assert!(model.forward(&latent, &tokens, &[0.1, 0.9]).is_err());
    }

    /// An untrained model predicts nothing at all, which is what makes the
    /// first training steps stable.
    #[test]
    fn a_fresh_model_predicts_no_motion() {
        let mut rng = StdRng::seed_from_u64(2);
        let config = tiny();
        let model = FlowTransformer::new(config, &mut rng).unwrap();
        let velocity = model
            .forward(
                &rows(config.latents, config.latent_dim, 3),
                &rows(config.cond_tokens, config.cond_dim, 4),
                &[0.7],
            )
            .unwrap();
        assert!(velocity.data.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn the_target_is_the_straight_line_between_the_endpoints() {
        let clean = rows(4, 3, 5);
        let noise = rows(4, 3, 6);
        let target = flow_match_target(&clean, &noise).unwrap();

        // Walking from the clean latent along the target for the whole unit
        // interval lands on the noise, which is the property the sampler
        // integrates backwards.
        for index in 0..target.data.len() {
            let landed = clean.data[index] + target.data[index];
            assert!((landed - noise.data[index]).abs() < 1e-6);
        }
        assert!(flow_match_target(&clean, &rows(4, 5, 6)).is_err());
    }

    /// The mixture the loss is taken at has to be the one the scheduler builds,
    /// or training and sampling are solving different problems.
    #[test]
    fn the_training_mixture_is_the_schedulers() {
        let clean = rows(4, 3, 7);
        let noise = rows(4, 3, 8);
        let scheduler = Scheduler::flow_match(1.0);

        for time in [0.1, 0.5, 0.9] {
            let mut mixed = clean.clone();
            scheduler.add_noise(&mut mixed, &noise, time).unwrap();
            for index in 0..mixed.data.len() {
                let ours = (1.0 - time) * clean.data[index] + time * noise.data[index];
                assert!((mixed.data[index] - ours).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn the_weight_gradients_match_finite_differences() {
        let mut rng = StdRng::seed_from_u64(3);
        let config = tiny();
        let mut model = FlowTransformer::new(config, &mut rng).unwrap();
        awaken(&mut model, &mut rng);

        let latent = rows(config.latents, config.latent_dim, 9);
        let tokens = rows(config.cond_tokens, config.cond_dim, 10);
        let times = [0.4];
        let upstream = rows(1, config.latent_dim, 11);

        let (velocity, cache) = model.forward_train(&latent, &tokens, &times).unwrap();
        let mut grad = Matrix::new(velocity.rows, velocity.cols);
        for (slot, weight) in grad.data.iter_mut().zip(upstream.data.iter().cycle()) {
            *slot = *weight;
        }
        model.backward(&cache, &grad).unwrap();

        let count = model.params_mut().len();
        let epsilon = 1e-3;
        // A spread across the stack: the timestep tower, the modulation, the
        // input projection, the slots, both blocks, the final layer, the
        // output projection.
        for param in [0, 2, 4, 5, 7, 12, count - 4, count - 2] {
            let index = 3 % model.params_mut()[param].value.data.len();
            let (original, analytic) = probe(&mut model, param, index);

            poke(&mut model, param, index, original + epsilon);
            let high = objective(&model.forward(&latent, &tokens, &times).unwrap(), &upstream);
            poke(&mut model, param, index, original - epsilon);
            let low = objective(&model.forward(&latent, &tokens, &times).unwrap(), &upstream);
            poke(&mut model, param, index, original);

            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (analytic - numeric).abs() < 1e-2,
                "parameter {param}[{index}]: {analytic} vs {numeric}"
            );
        }
    }

    /// The gradient handed back to the conditioning, which is what a later
    /// stage would need to unfreeze the image tower.
    #[test]
    fn the_token_gradient_matches_finite_differences() {
        let mut rng = StdRng::seed_from_u64(4);
        let config = tiny();
        let mut model = FlowTransformer::new(config, &mut rng).unwrap();
        awaken(&mut model, &mut rng);

        let latent = rows(config.latents, config.latent_dim, 12);
        let mut tokens = rows(config.cond_tokens, config.cond_dim, 13);
        let times = [0.6];
        let upstream = rows(1, config.latent_dim, 14);

        let (velocity, cache) = model.forward_train(&latent, &tokens, &times).unwrap();
        let mut grad = Matrix::new(velocity.rows, velocity.cols);
        for (slot, weight) in grad.data.iter_mut().zip(upstream.data.iter().cycle()) {
            *slot = *weight;
        }
        let grad_tokens = model.backward(&cache, &grad).unwrap();
        assert_eq!(
            (grad_tokens.rows, grad_tokens.cols),
            (tokens.rows, tokens.cols)
        );

        let epsilon = 1e-3;
        for index in [0, 4, 9, 14] {
            let original = tokens.data[index];
            tokens.data[index] = original + epsilon;
            let high = objective(&model.forward(&latent, &tokens, &times).unwrap(), &upstream);
            tokens.data[index] = original - epsilon;
            let low = objective(&model.forward(&latent, &tokens, &times).unwrap(), &upstream);
            tokens.data[index] = original;

            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (grad_tokens.data[index] - numeric).abs() < 1e-2,
                "token {index}: {} vs {numeric}",
                grad_tokens.data[index]
            );
        }
    }

    #[test]
    fn the_timestep_actually_changes_the_prediction() {
        let mut rng = StdRng::seed_from_u64(5);
        let config = tiny();
        let mut model = FlowTransformer::new(config, &mut rng).unwrap();
        awaken(&mut model, &mut rng);

        let latent = rows(config.latents, config.latent_dim, 15);
        let tokens = rows(config.cond_tokens, config.cond_dim, 16);
        let early = model.forward(&latent, &tokens, &[0.05]).unwrap();
        let late = model.forward(&latent, &tokens, &[0.95]).unwrap();

        let drift: f32 = early
            .data
            .iter()
            .zip(&late.data)
            .map(|(early, late)| (early - late).abs())
            .sum();
        assert!(drift > 1e-3, "the timestep changed nothing: {drift}");
    }

    #[test]
    fn the_timestep_sampler_stays_inside_the_interval_and_shifts() {
        let mut rng = StdRng::seed_from_u64(6);
        let plain = sample_timesteps(2000, 1.0, &mut rng);
        let shifted = sample_timesteps(2000, 3.0, &mut rng);

        assert!(plain.iter().all(|time| *time > 0.0 && *time < 1.0));
        assert!(shifted.iter().all(|time| *time > 0.0 && *time < 1.0));

        let mean = |times: &[f32]| times.iter().sum::<f32>() / times.len() as f32;
        // Logit-normal is symmetric about the middle; the shift bends it up.
        assert!((mean(&plain) - 0.5).abs() < 0.03, "{}", mean(&plain));
        assert!(mean(&shifted) > mean(&plain) + 0.1);
    }

    #[test]
    fn dropping_the_condition_swaps_in_the_null_embedding() {
        let mut rng = StdRng::seed_from_u64(7);
        let config = tiny();
        let model = FlowTransformer::new(config, &mut rng).unwrap();
        let tokens = rows(4 * config.cond_tokens, config.cond_dim, 17);

        let (kept, hits) = model.drop_condition(&tokens, 0.0, &mut rng).unwrap();
        assert_eq!(kept.data, tokens.data);
        assert_eq!(hits, vec![false; 4]);

        let (dropped, hits) = model.drop_condition(&tokens, 1.0, &mut rng).unwrap();
        assert_eq!(hits, vec![true; 4]);
        for sequence in 0..4 {
            for token in 0..config.cond_tokens {
                let row = dropped.row(sequence * config.cond_tokens + token);
                assert_eq!(row, model.null_tokens().row(token));
            }
        }
        assert!(model.drop_condition(&tokens, 1.5, &mut rng).is_err());
    }

    #[test]
    fn a_training_step_reaches_every_parameter() {
        let mut rng = StdRng::seed_from_u64(8);
        let config = tiny();
        let mut model = FlowTransformer::new(config, &mut rng).unwrap();
        awaken(&mut model, &mut rng);

        let clean = rows(config.latents, config.latent_dim, 18);
        let tokens = rows(config.cond_tokens, config.cond_dim, 19);
        let before: Vec<Vec<f32>> = model
            .params_mut()
            .iter()
            .map(|param| param.value.data.clone())
            .collect();

        model.zero_grad();
        // One step of each, so both the conditioned path and the null
        // embedding are reached without leaving it to a coin flip.
        model.train_step(&clean, &tokens, 0.0, &mut rng).unwrap();
        model.train_step(&clean, &tokens, 1.0, &mut rng).unwrap();
        let norm = model.step_clipped(1.0, 1.0).unwrap();
        assert!(norm > 0.0 && norm.is_finite(), "gradient norm {norm}");

        for (index, (param, before)) in model.params_mut().iter().zip(&before).enumerate() {
            let moved = param
                .value
                .data
                .iter()
                .zip(before)
                .any(|(now, before)| now != before);
            assert!(
                moved,
                "parameter {index} ({}x{}) never received a gradient",
                param.value.rows, param.value.cols
            );
        }
    }

    /// Clipping has to be the same clipping the language model uses, since the
    /// point of factoring it out was that there is only one copy.
    #[test]
    fn a_large_gradient_is_clipped_to_the_ceiling() {
        let mut rng = StdRng::seed_from_u64(9);
        let mut model = FlowTransformer::new(tiny(), &mut rng).unwrap();
        for param in model.params_mut() {
            param.grad.data.iter_mut().for_each(|slot| *slot = 10.0);
        }
        let norm = model.step_clipped(1.0, 0.5).unwrap();
        assert!(norm > 0.5, "{norm}");
    }

    /// End to end: train one pair until the model can name the velocity that
    /// joins them, then integrate that velocity from noise and land near the
    /// latent it was taught.
    #[test]
    fn one_pair_can_be_overfitted_and_then_sampled_back() {
        let mut rng = StdRng::seed_from_u64(10);
        let config = tiny();
        let mut model = FlowTransformer::new(config, &mut rng).unwrap();
        model.set_optimizer(Optimizer::adam(3e-3));

        let clean = rows(config.latents, config.latent_dim, 20);
        let tokens = rows(config.cond_tokens, config.cond_dim, 21);

        let mut losses = Vec::new();
        for _ in 0..400 {
            model.zero_grad();
            losses.push(model.train_step(&clean, &tokens, 0.0, &mut rng).unwrap());
            model.step_clipped(1.0, 1.0).unwrap();
        }

        let mean = |window: &[f32]| window.iter().sum::<f32>() / window.len() as f32;
        let first = mean(&losses[..50]);
        let last = mean(&losses[350..]);
        assert!(last < first * 0.25, "{first} to {last}");

        // The learned field, integrated. The model saw this condition every
        // step, so the run should land on the latent that went with it.
        let mut denoiser = FlowDenoiser {
            model: &model,
            tokens: &tokens,
        };
        let start = crate::diffusion::noise(config.latents, config.latent_dim, Some(4));
        let landed = sample(
            &mut denoiser,
            scheduler(1.0),
            start,
            &SamplingConfig {
                steps: 32,
                guidance: 1.0,
                ..SamplingConfig::default()
            },
            |_, _| true,
        )
        .unwrap();

        let error: f32 = landed
            .data
            .iter()
            .zip(&clean.data)
            .map(|(landed, clean)| (landed - clean).abs())
            .sum::<f32>()
            / clean.data.len() as f32;
        assert!(error < 0.1, "sampling landed {error} away");
    }

    #[test]
    fn a_checkpoint_round_trip_restores_the_model_and_its_metadata() {
        let path = std::env::temp_dir().join(format!(
            "rusting-brain-flow-{}.safetensors",
            std::process::id()
        ));
        let mut rng = StdRng::seed_from_u64(4);
        let mut model = FlowTransformer::new(tiny(), &mut rng).unwrap();
        awaken(&mut model, &mut rng);

        // Steps first, so the Adam counter is something other than zero and a
        // resume that dropped it would be visible.
        model.zero_grad();
        model
            .train_step(
                &rows(tiny().latents, tiny().latent_dim, 5),
                &rows(tiny().cond_tokens, tiny().cond_dim, 6),
                0.0,
                &mut rng,
            )
            .unwrap();
        model.step_clipped(1.0, 1.0).unwrap();

        let metadata = BTreeMap::from([("latent_scale".to_string(), "0.75".to_string())]);
        model.save(&path, &metadata).unwrap();
        let (restored, read) =
            FlowTransformer::load(&path, &mut StdRng::seed_from_u64(99)).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(restored, model, "the restored model is not the saved one");
        assert_eq!(read, metadata, "the caller's metadata did not survive");
    }
}
