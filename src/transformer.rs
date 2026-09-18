//! Decoder-only transformer assembly: configuration, builder, and the model.

use crate::attention::KvCache;
use crate::batch::TokenBatch;
use crate::causal_lm_loss::{TotalLoss, causal_lm_loss_batch};
use crate::embedding::Embedding;
use crate::matrix::Matrix;
use crate::moe::{DEFAULT_AUX_LOSS_WEIGHT, DEFAULT_ROUTER_Z_LOSS_WEIGHT, MoeConfig};
use crate::network::NetworkError;
use crate::norm::RmsNorm;
use crate::optimizers::Optimizer;
use crate::param::{Linear, Param};
use crate::rope::Rope;
use crate::transformer_block::{FeedForward, TransformerBlock, TransformerBlockCache};
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::Path;

/// Everything needed to lay out a decoder-only model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    /// `n_heads` for plain multi-head attention, fewer for grouped-query.
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Hidden width of the dense feed-forward layers.
    pub d_ff: usize,
    /// Hidden width of one expert, normally a fraction of `d_ff`.
    pub moe_d_ff: usize,
    pub num_experts: usize,
    pub experts_per_token: usize,
    /// Layer indices that use a MoE feed-forward. Everything else is dense,
    /// which is how real MoE models keep their first layers.
    pub moe_layers: Vec<usize>,
    pub shared_expert: bool,
    pub max_seq_len: usize,
    pub rope_base: f32,
    pub rmsnorm_eps: f32,
    /// Reuses the embedding matrix as the output projection.
    pub tie_embeddings: bool,
    pub aux_loss_weight: f32,
    pub router_z_loss_weight: f32,
}

impl Default for TransformerConfig {
    /// A roughly 55M-total / 36M-active model: small enough to train on a CPU
    /// for experiments, large enough that the sparsity is visible.
    fn default() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 512,
            n_layers: 8,
            n_heads: 8,
            n_kv_heads: 2,
            head_dim: 64,
            d_ff: 1408,
            moe_d_ff: 352,
            num_experts: 8,
            experts_per_token: 2,
            moe_layers: (2..8).collect(),
            shared_expert: true,
            max_seq_len: 2048,
            rope_base: 10_000.0,
            rmsnorm_eps: 1e-6,
            tie_embeddings: true,
            aux_loss_weight: DEFAULT_AUX_LOSS_WEIGHT,
            router_z_loss_weight: DEFAULT_ROUTER_Z_LOSS_WEIGHT,
        }
    }
}

impl TransformerConfig {
    pub fn validate(&self) -> Result<(), NetworkError> {
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("d_model", self.d_model),
            ("n_layers", self.n_layers),
            ("n_heads", self.n_heads),
            ("n_kv_heads", self.n_kv_heads),
            ("head_dim", self.head_dim),
            ("d_ff", self.d_ff),
            ("max_seq_len", self.max_seq_len),
        ] {
            if value == 0 {
                return Err(NetworkError::InvalidConfig(format!(
                    "{name} must be non-zero"
                )));
            }
        }

        if self.n_heads % self.n_kv_heads != 0 || self.n_kv_heads > self.n_heads {
            return Err(NetworkError::InvalidConfig(format!(
                "n_heads ({}) must be a multiple of n_kv_heads ({})",
                self.n_heads, self.n_kv_heads
            )));
        }
        if self.head_dim % 2 != 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "head_dim must be even for rotary embeddings, got {}",
                self.head_dim
            )));
        }
        if let Some(&layer) = self.moe_layers.iter().find(|&&l| l >= self.n_layers) {
            return Err(NetworkError::InvalidConfig(format!(
                "moe_layers names layer {layer}, but the model has {} layers",
                self.n_layers
            )));
        }
        if !self.moe_layers.is_empty() {
            self.moe_config().validate()?;
        }

        Ok(())
    }

    pub fn moe_config(&self) -> MoeConfig {
        MoeConfig {
            num_experts: self.num_experts,
            experts_per_token: self.experts_per_token,
            d_ff: self.moe_d_ff,
            shared_expert: self.shared_expert,
            aux_loss_weight: self.aux_loss_weight,
            router_z_loss_weight: self.router_z_loss_weight,
        }
    }

    pub fn is_moe_layer(&self, layer: usize) -> bool {
        self.moe_layers.contains(&layer)
    }

    /// Parameter counts derived from the configuration alone, so a caller can
    /// size a model before paying to build one.
    pub fn parameter_counts(&self) -> ParameterCounts {
        let query = self.d_model * self.n_heads * self.head_dim;
        let key_value = 2 * self.d_model * self.n_kv_heads * self.head_dim;
        let attention = 2 * query + key_value;
        let norms = 2 * self.d_model;

        let dense_ffn = 3 * self.d_model * self.d_ff;
        let expert = 3 * self.d_model * self.moe_d_ff;
        let shared = if self.shared_expert { expert } else { 0 };
        let router = self.num_experts * self.d_model;

        let moe_total = router + self.num_experts * expert + shared;
        let moe_active = router + self.experts_per_token * expert + shared;

        let mut total = self.vocab_size * self.d_model + self.d_model;
        if !self.tie_embeddings {
            total += self.vocab_size * self.d_model;
        }
        let mut active = total;

        for layer in 0..self.n_layers {
            total += attention + norms;
            active += attention + norms;
            if self.is_moe_layer(layer) {
                total += moe_total;
                active += moe_active;
            } else {
                total += dense_ffn;
                active += dense_ffn;
            }
        }

        ParameterCounts { total, active }
    }
}

/// Total and per-token-active weight counts.
///
/// For a MoE model these differ by a lot, and both matter: total sets the
/// memory the model occupies, active sets what one forward pass costs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParameterCounts {
    pub total: usize,
    pub active: usize,
}

impl ParameterCounts {
    /// Total divided by active. One for a dense model, higher the sparser the
    /// routing.
    pub fn sparsity_ratio(&self) -> f32 {
        if self.active == 0 {
            return 0.0;
        }
        self.total as f32 / self.active as f32
    }
}

impl std::fmt::Display for ParameterCounts {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{:.1}M total / {:.1}M active ({:.2}x)",
            self.total as f64 / 1e6,
            self.active as f64 / 1e6,
            self.sparsity_ratio()
        )
    }
}

/// Fluent configuration for [`TransformerLm`], in the style of
/// [`NetworkBuilder`](crate::network::NetworkBuilder).
///
/// It is a separate builder rather than an extension of `NetworkBuilder`
/// because the two describe different things: one is a list of dense layer
/// widths, the other a fixed block structure parameterized by head and expert
/// counts.
#[derive(Clone, Debug)]
pub struct TransformerBuilder {
    config: TransformerConfig,
    optimizer: Optimizer,
    seed: Option<u64>,
    mixed_precision: bool,
}

impl TransformerBuilder {
    pub fn new() -> Self {
        Self {
            config: TransformerConfig::default(),
            optimizer: Optimizer::adam(3e-4),
            seed: None,
            mixed_precision: true,
        }
    }

    pub fn config(mut self, config: TransformerConfig) -> Self {
        self.config = config;
        self
    }

    pub fn vocab_size(mut self, vocab_size: usize) -> Self {
        self.config.vocab_size = vocab_size;
        self
    }

    pub fn d_model(mut self, d_model: usize) -> Self {
        self.config.d_model = d_model;
        self
    }

    pub fn n_layers(mut self, n_layers: usize) -> Self {
        self.config.n_layers = n_layers;
        self
    }

    pub fn heads(mut self, n_heads: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        self.config.n_heads = n_heads;
        self.config.n_kv_heads = n_kv_heads;
        self.config.head_dim = head_dim;
        self
    }

    pub fn d_ff(mut self, d_ff: usize) -> Self {
        self.config.d_ff = d_ff;
        self
    }

    pub fn moe_d_ff(mut self, moe_d_ff: usize) -> Self {
        self.config.moe_d_ff = moe_d_ff;
        self
    }

    pub fn experts(mut self, num_experts: usize, experts_per_token: usize) -> Self {
        self.config.num_experts = num_experts;
        self.config.experts_per_token = experts_per_token;
        self
    }

    /// Layer indices that use a MoE feed-forward.
    pub fn moe_layers(mut self, layers: impl IntoIterator<Item = usize>) -> Self {
        self.config.moe_layers = layers.into_iter().collect();
        self
    }

    pub fn shared_expert(mut self, shared_expert: bool) -> Self {
        self.config.shared_expert = shared_expert;
        self
    }

    pub fn max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.config.max_seq_len = max_seq_len;
        self
    }

    pub fn rope_base(mut self, rope_base: f32) -> Self {
        self.config.rope_base = rope_base;
        self
    }

    pub fn rmsnorm_eps(mut self, rmsnorm_eps: f32) -> Self {
        self.config.rmsnorm_eps = rmsnorm_eps;
        self
    }

    pub fn tie_embeddings(mut self, tie_embeddings: bool) -> Self {
        self.config.tie_embeddings = tie_embeddings;
        self
    }

    pub fn aux_loss_weight(mut self, weight: f32) -> Self {
        self.config.aux_loss_weight = weight;
        self
    }

    pub fn router_z_loss_weight(mut self, weight: f32) -> Self {
        self.config.router_z_loss_weight = weight;
        self
    }

    pub fn optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Lets the device path compute in reduced precision. **On by default**,
    /// and ignored by the CPU path, which is always FP32. Pass `false` to get
    /// a bit-comparable FP32 device run.
    ///
    /// Two things turn on together, and neither touches the parameters, the
    /// optimizer state or the gradients the optimizer consumes, all of which
    /// stay FP32:
    ///
    /// * Every cuBLAS GEMM runs on the tensor cores in TF32: storage stays
    ///   FP32 and the products still accumulate in FP32, but the two
    ///   multiplier inputs are rounded to a 10-bit mantissa first.
    /// * The language-model head — its input, its weight, its logits and both
    ///   of its gradient products — is computed in BF16, with the cross-entropy
    ///   reading and writing BF16 in place. The gradient that leaves the head
    ///   for the rest of the network is widened back to FP32, and the weight
    ///   gradient is accumulated into the FP32 gradient after every chunk
    ///   rather than summed in BF16.
    ///
    /// No loss scaling, and none is needed: both formats keep FP32's exponent
    /// range, so nothing underflows here that FP32 would have kept. That is
    /// the reason for BF16 over FP16.
    ///
    /// The cost is accuracy. A TF32 dot product carries roughly `2^-11`
    /// relative error per term against FP32's `2^-24`, and BF16 storage
    /// carries `2^-8`, so agreement with the FP32 path is around `1e-2`
    /// relative on a loss rather than `1e-4`. Over twenty steps of the 5.2M
    /// dense configuration at a 32k vocabulary the two loss curves still track
    /// each other to four decimal places, because the head's error is
    /// stochastic across 32000 logits rather than a systematic bias.
    ///
    /// The head is worth the casts because it is the only part of the step
    /// whose GEMMs are `vocab`-wide; it is about three quarters of all GEMM
    /// time. Measured on an RTX 3060 at batch 128, sequence 128, the flag is
    /// worth about 1.7x end to end (61k vs 104k tokens/s), which is why it is
    /// the default.
    pub fn mixed_precision(mut self, mixed_precision: bool) -> Self {
        self.mixed_precision = mixed_precision;
        self
    }

    /// Parameter counts for what this builder would produce.
    pub fn parameter_counts(&self) -> ParameterCounts {
        self.config.parameter_counts()
    }

    pub fn build(self) -> Result<TransformerLm, NetworkError> {
        TransformerLm::from_builder(self)
    }
}

impl Default for TransformerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// What [`TransformerLm::backward`] needs, and the auxiliary losses the forward
/// pass produced.
#[derive(Clone, Debug)]
pub struct TransformerCache {
    batch: TokenBatch,
    blocks: Vec<TransformerBlockCache>,
    final_input: Matrix,
    final_output: Matrix,
    auxiliary_loss: f32,
    /// Set when the forward pass ran device-side, in which case the host fields
    /// above are empty and every activation lives in this cache instead. Shared
    /// rather than owned so that a `TransformerCache` stays `Clone`: device
    /// buffers cannot be duplicated by a derive.
    #[cfg(feature = "cuda")]
    device: Option<std::sync::Arc<crate::gpu_model::GpuCache>>,
}

impl TransformerCache {
    /// Summed load-balancing and z-losses from every MoE layer, already
    /// weighted.
    pub fn auxiliary_loss(&self) -> f32 {
        self.auxiliary_loss
    }

    /// The batch this cache was produced from, which the loss needs in order
    /// to find the sequence boundaries.
    pub fn batch(&self) -> &TokenBatch {
        &self.batch
    }
}

/// Magic and version for the optimizer-state sidecar, so an unrelated or
/// outdated file is rejected instead of being read as moments.
const OPTIMIZER_STATE_MAGIC: &[u8; 8] = b"RBOPT001";
const WEIGHTS_MAGIC: &[u8; 8] = b"RBWTS001";

/// How [`TransformerLm::save_bin`] stores each weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Precision {
    /// Raw `f32`, 4 bytes per weight. Lossless, and the only choice that a
    /// training run can resume from without a visible jump in loss.
    F32,
    /// Symmetric int8 with one `f32` scale per row: 1 byte per weight plus 4
    /// bytes per row. Lossy, for inference and for shipping a model.
    Q8,
}

/// The JSON preamble of a binary snapshot. Small: it holds no weights.
#[derive(Serialize, Deserialize)]
struct BinHeader {
    config: TransformerConfig,
    optimizer: Optimizer,
    optimizer_step: u64,
    precision: Precision,
}

/// Rounds to the 255 levels int8 has, leaving -128 unused so the range stays
/// symmetric around zero.
fn quantize(value: f32, scale: f32) -> i8 {
    if scale == 0.0 {
        return 0;
    }
    (value / scale).round().clamp(-127.0, 127.0) as i8
}

/// A decoder-only language model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransformerLm {
    pub config: TransformerConfig,
    pub embedding: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub final_norm: RmsNorm,
    /// `None` when embeddings are tied, in which case the embedding matrix is
    /// the output projection.
    pub lm_head: Option<Linear>,
    pub optimizer: Optimizer,
    optimizer_step: usize,
    /// Set by [`TransformerLm::to_cuda`]. Never serialized: a snapshot is host
    /// data, and a restored model starts on the CPU.
    #[cfg(feature = "cuda")]
    #[serde(skip)]
    device: Option<std::sync::Arc<crate::gpu_transformer::GpuContext>>,
    /// Set by [`TransformerBuilder::mixed_precision`] and read by
    /// [`TransformerLm::to_cuda`]. Not serialized: it describes how to run a
    /// model, not what the model is, and a restored model starts on the CPU.
    #[serde(skip)]
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    mixed_precision: bool,
}

/// Weights and configuration, not transient training or device state: two
/// models that predict identically are equal, wherever their buffers live.
impl PartialEq for TransformerLm {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config
            && self.embedding == other.embedding
            && self.blocks == other.blocks
            && self.final_norm == other.final_norm
            && self.lm_head == other.lm_head
            && self.optimizer == other.optimizer
    }
}

impl TransformerLm {
    pub fn builder() -> TransformerBuilder {
        TransformerBuilder::new()
    }

    fn from_builder(builder: TransformerBuilder) -> Result<Self, NetworkError> {
        let config = builder.config;
        config.validate()?;

        let mut rng = builder
            .seed
            .map_or_else(StdRng::from_entropy, StdRng::seed_from_u64);
        let rope = Rope::new(config.head_dim, config.max_seq_len, config.rope_base)?;

        let embedding = Embedding::new(config.vocab_size, config.d_model, &mut rng);
        let mut blocks = Vec::with_capacity(config.n_layers);

        for layer in 0..config.n_layers {
            let feed_forward = if config.is_moe_layer(layer) {
                FeedForward::moe(config.d_model, config.moe_config(), &mut rng)?
            } else {
                FeedForward::swiglu(config.d_model, config.d_ff, &mut rng)
            };

            blocks.push(TransformerBlock::new(
                config.d_model,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                rope.clone(),
                feed_forward,
                config.rmsnorm_eps,
                &mut rng,
            )?);
        }

        let lm_head = (!config.tie_embeddings)
            .then(|| Linear::new(config.d_model, config.vocab_size, &mut rng));

        Ok(Self {
            final_norm: RmsNorm::new(config.d_model, config.rmsnorm_eps),
            config,
            embedding,
            blocks,
            lm_head,
            optimizer: builder.optimizer,
            optimizer_step: 0,
            #[cfg(feature = "cuda")]
            device: None,
            mixed_precision: builder.mixed_precision,
        })
    }

    /// Counts taken from the built modules. Agrees with
    /// [`TransformerConfig::parameter_counts`].
    pub fn parameter_counts(&self) -> ParameterCounts {
        let head = self.lm_head.as_ref().map_or(0, |head| head.weight.len());
        let base = self.embedding.weight.len() + self.final_norm.weight.len() + head;

        ParameterCounts {
            total: base
                + self
                    .blocks
                    .iter()
                    .map(TransformerBlock::num_parameters)
                    .sum::<usize>(),
            active: base
                + self
                    .blocks
                    .iter()
                    .map(TransformerBlock::active_parameters)
                    .sum::<usize>(),
        }
    }

    /// One empty key/value cache per layer, sized for `max_seq_len`.
    pub fn new_kv_caches(&self) -> Vec<KvCache> {
        (0..self.config.n_layers)
            .map(|_| {
                KvCache::with_capacity(
                    self.config.n_kv_heads,
                    self.config.head_dim,
                    self.config.max_seq_len,
                )
            })
            .collect()
    }

    /// Forward pass over a batch of sequences, returning
    /// `[batch * seq_len, vocab_size]` logits.
    ///
    /// Sequences are packed into one matrix, so every matmul in the model is
    /// `batch` times taller than it would be for a single sequence. Short
    /// sequences are right-padded; their rows produce logits that the loss
    /// ignores.
    pub fn forward_train<S: AsRef<[u32]>>(
        &self,
        sequences: &[S],
    ) -> Result<(Matrix, TransformerCache), NetworkError> {
        self.forward_batch(&TokenBatch::new(sequences)?)
    }

    /// [`TransformerLm::forward_train`] over an already-packed batch.
    pub fn forward_batch(
        &self,
        batch: &TokenBatch,
    ) -> Result<(Matrix, TransformerCache), NetworkError> {
        self.check_length(batch.seq_len(), 0)?;

        #[cfg(feature = "cuda")]
        if let Some(context) = &self.device {
            let (logits, cache) = crate::gpu_model::forward(self, context, batch)?;
            self.check_device()?;
            return Ok((
                logits,
                TransformerCache {
                    batch: batch.clone(),
                    blocks: Vec::new(),
                    final_input: Matrix::new(0, 0),
                    final_output: Matrix::new(0, 0),
                    auxiliary_loss: cache.auxiliary_loss(),
                    device: Some(std::sync::Arc::new(cache)),
                },
            ));
        }

        let layout = batch.layout();
        let mut hidden = self.embedding.forward(batch.ids())?;
        let mut blocks = Vec::with_capacity(self.blocks.len());
        let mut auxiliary_loss = 0.0;

        for block in &self.blocks {
            let (output, cache) = block.forward_train(&hidden, layout)?;
            auxiliary_loss += cache.auxiliary_loss();
            blocks.push(cache);
            hidden = output;
        }

        let final_output = self.final_norm.forward(&hidden);
        let logits = match &self.lm_head {
            Some(head) => head.forward(&final_output),
            None => self.embedding.unembed(&final_output),
        };

        self.check_device()?;

        Ok((
            logits,
            TransformerCache {
                batch: batch.clone(),
                blocks,
                final_input: hidden,
                final_output,
                auxiliary_loss,
                #[cfg(feature = "cuda")]
                device: None,
            },
        ))
    }

    /// Inference-only forward pass, appending to the caches.
    ///
    /// Pass the whole prompt with empty caches to prefill, then one token at a
    /// time; the caches carry the position, so the caller does not track it.
    pub fn forward_cached(
        &self,
        ids: &[u32],
        caches: &mut [KvCache],
    ) -> Result<Matrix, NetworkError> {
        if caches.len() != self.blocks.len() {
            return Err(NetworkError::InvalidConfig(format!(
                "expected one kv cache per layer ({}), got {}",
                self.blocks.len(),
                caches.len()
            )));
        }
        self.check_length(ids.len(), caches.first().map_or(0, KvCache::len))?;

        let mut hidden = self.embedding.forward(ids)?;
        for (block, cache) in self.blocks.iter().zip(caches.iter_mut()) {
            hidden = block.forward_cached(&hidden, cache)?;
        }

        let hidden = self.final_norm.forward(&hidden);
        let logits = match &self.lm_head {
            Some(head) => head.forward(&hidden),
            None => self.embedding.unembed(&hidden),
        };
        self.check_device()?;
        Ok(logits)
    }

    /// Accumulates gradients for every parameter from `dL/dlogits`.
    pub fn backward(
        &mut self,
        cache: &TransformerCache,
        grad_logits: &Matrix,
    ) -> Result<(), NetworkError> {
        #[cfg(feature = "cuda")]
        if let Some(device) = &cache.device {
            let context = self
                .device
                .clone()
                .ok_or_else(|| NetworkError::Cuda("the model left the device mid-step".into()))?;
            crate::gpu_model::backward(self, &context, device, grad_logits)?;
            return self.check_device();
        }

        let mut grad_hidden = match &mut self.lm_head {
            Some(head) => head.backward(&cache.final_output, grad_logits),
            // Tied weights: this accumulates into the same gradient buffer the
            // embedding gather writes to, which is the point of tying.
            None => self
                .embedding
                .unembed_backward(&cache.final_output, grad_logits),
        };

        grad_hidden = self.final_norm.backward(&cache.final_input, &grad_hidden);

        for (block, block_cache) in self.blocks.iter_mut().zip(&cache.blocks).rev() {
            grad_hidden = block.backward(block_cache, &grad_hidden)?;
        }

        self.embedding.backward(cache.batch.ids(), &grad_hidden)?;
        self.check_device()
    }

    /// Moves every matmul-bound parameter onto a CUDA device and keeps it
    /// there until [`TransformerLm::to_cpu`].
    ///
    /// Training then runs entirely on the device:
    /// [`forward_batch`](TransformerLm::forward_batch) and
    /// [`backward`](TransformerLm::backward) dispatch to
    /// [`crate::gpu_model`], which keeps activations in device buffers across
    /// a whole layer and rounds trip to the host only for the token ids, the
    /// logits, `dL/dlogits`, the RMSNorm scales and one routing table per MoE
    /// layer. The norm weights stay host-resident, so they keep using the CPU
    /// optimizer step and cached decode keeps working. Fails closed: no device
    /// means an error, never a silent CPU fallback. A caller who wants a fallback asks
    /// [`accelerator_doctor`](crate::accelerator::accelerator_doctor) first.
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&mut self, device: usize, memory_budget_mib: usize) -> Result<(), NetworkError> {
        let counts = self.config.parameter_counts();
        // Value, gradient and the two Adam moments, all FP32.
        let estimated_mib = (counts.total * 4 * 4).div_ceil(1024 * 1024);
        if memory_budget_mib > 0 && estimated_mib > memory_budget_mib {
            return Err(NetworkError::CudaMemoryBudget {
                estimated_mib,
                budget_mib: memory_budget_mib,
            });
        }

        let context =
            crate::gpu_transformer::GpuContext::with_precision(device, self.mixed_precision)?;
        self.embedding.weight.move_to_cuda(&context)?;
        for block in &mut self.blocks {
            for linear in block.linears_mut() {
                linear.weight.move_to_cuda(&context)?;
            }
        }
        if let Some(head) = &mut self.lm_head {
            head.weight.move_to_cuda(&context)?;
        }
        self.device = Some(context);
        Ok(())
    }

    /// Copies every device parameter back and releases the device buffers.
    #[cfg(feature = "cuda")]
    pub fn to_cpu(&mut self) -> Result<(), NetworkError> {
        for param in self.params_mut() {
            param.move_to_cpu()?;
        }
        if let Some(context) = self.device.take() {
            context.check()?;
        }
        Ok(())
    }

    /// Refreshes the host copies of the device parameters, keeping residency.
    ///
    /// Call this before [`TransformerLm::save_json`] or before reading weights
    /// while training on a device: the host values go stale at the first device
    /// optimizer step.
    #[cfg(feature = "cuda")]
    pub fn sync_from_device(&mut self) -> Result<(), NetworkError> {
        for param in self.params_mut() {
            param.sync_from_device()?;
        }
        self.check_device()
    }

    /// Waits for every queued device operation to finish.
    ///
    /// A no-op on the host path. Timing code needs it because the optimizer
    /// step only enqueues launches, so without a barrier its cost lands in
    /// whatever phase is measured next.
    pub fn synchronize(&self) -> Result<(), NetworkError> {
        #[cfg(feature = "cuda")]
        if let Some(context) = &self.device {
            return context.synchronize();
        }
        Ok(())
    }

    /// Whether the matmul-bound parameters currently live on a device.
    pub fn on_device(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.device.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Reports the first device failure since the last check.
    ///
    /// Device operations sit inside infallible signatures (`Linear::forward`
    /// returns a `Matrix`), so a failure is recorded and surfaced here, at the
    /// next fallible boundary.
    fn check_device(&self) -> Result<(), NetworkError> {
        #[cfg(feature = "cuda")]
        if let Some(context) = &self.device {
            context.check()?;
        }
        Ok(())
    }

    /// Forward, loss, backward and one optimizer update over a batch of
    /// sequences.
    ///
    /// The loss is the mean over every predicted position in the batch, so the
    /// gradient is already a batch mean and the optimizer step needs no extra
    /// scaling.
    pub fn train_step<S: AsRef<[u32]>>(
        &mut self,
        sequences: &[S],
    ) -> Result<TotalLoss, NetworkError> {
        self.train_step_batch(&TokenBatch::new(sequences)?)
    }

    /// [`TransformerLm::train_step`] over an already-packed batch, which a
    /// training loop that reuses one batch shape can build once.
    pub fn train_step_batch(&mut self, batch: &TokenBatch) -> Result<TotalLoss, NetworkError> {
        // On a device the loss is fused into the step, so the `[rows, vocab]`
        // logits stay in device memory and are never materialized in full.
        // See `gpu_model::train_step`.
        #[cfg(feature = "cuda")]
        if let Some(context) = self.device.clone() {
            self.check_length(batch.seq_len(), 0)?;
            self.zero_grad();
            let (lm_loss, auxiliary_loss) = crate::gpu_model::train_step(self, &context, batch)?;
            self.step(1.0);
            self.check_device()?;

            return Ok(TotalLoss {
                lm_loss,
                auxiliary_loss,
            });
        }

        let (logits, cache) = self.forward_batch(batch)?;
        let loss = causal_lm_loss_batch(&logits, batch)?;

        self.zero_grad();
        self.backward(&cache, &loss.grad_logits)?;
        self.step(1.0);
        self.check_device()?;

        Ok(TotalLoss {
            lm_loss: loss.loss,
            auxiliary_loss: cache.auxiliary_loss(),
        })
    }

    /// Forward and backward over one batch, adding into whatever gradients are
    /// already there.
    ///
    /// Unlike [`TransformerLm::train_step_batch`] this neither zeroes the
    /// gradients first nor steps the optimizer after, which is what lets a
    /// caller build an effective batch larger than the device holds:
    ///
    /// ```ignore
    /// model.zero_grad();
    /// for part in parts {
    ///     model.accumulate_step(part)?;
    /// }
    /// model.step(1.0 / parts.len() as f32);
    /// ```
    ///
    /// The averaging belongs on the step rather than on each backward pass:
    /// Adam normalizes by the gradient's own second moment, so scaling every
    /// accumulation identically would cancel out and change nothing.
    ///
    /// For a dense model this reproduces the gradient of the whole batch
    /// exactly. A mixture-of-experts model's load-balancing loss does not: it
    /// is computed from routing fractions over whatever batch it sees, so four
    /// sequences in two accumulations balance the experts against two
    /// different halves rather than against the whole. The language-modelling
    /// gradient is unaffected; only the auxiliary term shifts.
    pub fn accumulate_step(&mut self, batch: &TokenBatch) -> Result<TotalLoss, NetworkError> {
        #[cfg(feature = "cuda")]
        if let Some(context) = self.device.clone() {
            self.check_length(batch.seq_len(), 0)?;
            let (lm_loss, auxiliary_loss) = crate::gpu_model::train_step(self, &context, batch)?;
            self.check_device()?;

            return Ok(TotalLoss {
                lm_loss,
                auxiliary_loss,
            });
        }

        let (logits, cache) = self.forward_batch(batch)?;
        let loss = causal_lm_loss_batch(&logits, batch)?;
        self.backward(&cache, &loss.grad_logits)?;

        Ok(TotalLoss {
            lm_loss: loss.loss,
            auxiliary_loss: cache.auxiliary_loss(),
        })
    }

    /// [`TransformerLm::accumulate_step`] with a per-token loss mask.
    ///
    /// `mask` holds one flag per token position of `batch`, in the same order
    /// and of the same length as [`TokenBatch::ids`] - `batch.rows()` flags,
    /// padding included. A flag marks its token as a *target*: `false` makes
    /// the position that predicts that token contribute exactly zero to the
    /// loss and zero to the gradient, and the mean is taken over the flagged
    /// positions alone, so no scaling is needed on the caller's side. `None`
    /// is [`TransformerLm::accumulate_step`] exactly.
    ///
    /// Masked tokens still run the forward pass and still condition the
    /// positions that count, which is what supervised fine-tuning needs: for
    /// `<|user|>{instruction}<|assistant|>{response}<|end|>`, flag the
    /// `{response}<|end|>` span and leave the instruction span unflagged, and
    /// the instruction is read as context without being learned.
    ///
    /// A sequence's first token is never a target - nothing precedes it - so
    /// its flag is ignored.
    pub fn accumulate_step_masked(
        &mut self,
        batch: &TokenBatch,
        mask: Option<&[bool]>,
    ) -> Result<TotalLoss, NetworkError> {
        match mask {
            None => self.accumulate_step(batch),
            Some(mask) => self.accumulate_step(&batch.clone().with_loss_mask(mask)?),
        }
    }

    /// Applies the optimizer to every parameter and clears the gradients.
    ///
    /// `scale` divides the accumulated gradient, so a caller that ran several
    /// sequences before stepping passes `1.0 / sequences`.
    pub fn step(&mut self, scale: f32) {
        self.optimizer_step += 1;
        let step = self.optimizer_step;
        let optimizer = self.optimizer.clone();

        if self.on_device() {
            // One stream, one cuBLAS handle: the launches would serialize
            // anyway, and a rayon pool around them only adds contention.
            for param in self.params_mut() {
                param.step(&optimizer, step, scale);
            }
            return;
        }

        self.params_mut()
            .par_iter_mut()
            .for_each(|param| param.step(&optimizer, step, scale));
    }

    pub fn zero_grad(&mut self) {
        if self.on_device() {
            for param in self.params_mut() {
                param.zero_grad();
            }
            return;
        }

        self.params_mut()
            .par_iter_mut()
            .for_each(|param| param.zero_grad());
    }

    pub fn optimizer_step(&self) -> usize {
        self.optimizer_step
    }

    /// Sets the Adam step counter, for a resume that has weights but no saved
    /// moments: bias correction only matches moments that were accumulated
    /// over the same number of steps, and correcting zero moments as if they
    /// were warmed up makes the first updates several times too large.
    pub fn set_optimizer_step(&mut self, step: usize) {
        self.optimizer_step = step;
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = self.embedding.params_mut();
        for block in &mut self.blocks {
            params.extend(block.params_mut());
        }
        params.extend(self.final_norm.params_mut());
        if let Some(head) = &mut self.lm_head {
            params.extend(head.params_mut());
        }
        params
    }

    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), NetworkError> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        serde_json::to_writer(&mut writer, self)?;
        std::io::Write::flush(&mut writer)?;
        Ok(())
    }

    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, NetworkError> {
        let file = std::fs::File::open(path)?;
        let model: Self = serde_json::from_reader(std::io::BufReader::new(file))?;
        model.config.validate()?;
        Ok(model)
    }

    /// Overrides the precision chosen at build time.
    ///
    /// A snapshot records what a model is, not how to run it, so a model
    /// restored by [`TransformerLm::load_json`] always comes back in full
    /// precision. A resumed training run that wants reduced precision has to
    /// say so again, before [`TransformerLm::to_cuda`].
    pub fn set_mixed_precision(&mut self, mixed_precision: bool) {
        self.mixed_precision = mixed_precision;
    }

    /// Writes the Adam moments to `path`, in the order
    /// [`TransformerLm::params_mut`] yields them.
    ///
    /// The JSON snapshot holds weights alone: the moments triple its size and
    /// nothing that only runs the model ever reads them. A run that stops and
    /// resumes does need them. Without them Adam restarts from zero while the
    /// step counter carries on, so bias correction no longer compensates and
    /// the first updates after the resume are several times larger than the
    /// ones the run was taking before it stopped.
    ///
    /// Call this after [`TransformerLm::to_cuda`] on a device-resident model:
    /// the device owns the moments, and they are read back here.
    pub fn save_optimizer_state<P: AsRef<Path>>(&mut self, path: P) -> Result<(), NetworkError> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        let step = self.optimizer_step as u64;
        let params = self.params_mut();
        writer.write_all(OPTIMIZER_STATE_MAGIC)?;
        writer.write_all(&(params.len() as u64).to_le_bytes())?;
        writer.write_all(&step.to_le_bytes())?;
        for param in params {
            let (first, second) = param.moments()?;
            writer.write_all(&(first.rows as u64).to_le_bytes())?;
            writer.write_all(&(first.cols as u64).to_le_bytes())?;
            for matrix in [first, second] {
                let bytes: Vec<u8> = matrix.data.iter().flat_map(|v| v.to_le_bytes()).collect();
                writer.write_all(&bytes)?;
            }
        }
        writer.flush()?;
        Ok(())
    }

    /// Restores the moments and step counter written by
    /// [`TransformerLm::save_optimizer_state`].
    ///
    /// Order matters on a device-resident model: uploading a parameter zeroes
    /// its moments, so call this after [`TransformerLm::to_cuda`], not before.
    pub fn load_optimizer_state<P: AsRef<Path>>(&mut self, path: P) -> Result<(), NetworkError> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != OPTIMIZER_STATE_MAGIC {
            return Err(NetworkError::InvalidSnapshot(
                "not a RustingBrain optimizer state file".into(),
            ));
        }
        let mut word = [0u8; 8];
        reader.read_exact(&mut word)?;
        let count = u64::from_le_bytes(word) as usize;
        reader.read_exact(&mut word)?;
        let step = u64::from_le_bytes(word) as usize;
        let params = self.params_mut();
        if count != params.len() {
            return Err(NetworkError::InvalidSnapshot(format!(
                "optimizer state holds {count} parameters, this model has {}",
                params.len()
            )));
        }
        for param in params {
            reader.read_exact(&mut word)?;
            let rows = u64::from_le_bytes(word) as usize;
            reader.read_exact(&mut word)?;
            let cols = u64::from_le_bytes(word) as usize;
            if rows != param.value.rows || cols != param.value.cols {
                return Err(NetworkError::InvalidSnapshot(format!(
                    "optimizer state has a {rows}x{cols} parameter where the model has {}x{}",
                    param.value.rows, param.value.cols
                )));
            }
            let mut bytes = vec![0u8; rows * cols * 4];
            let mut moments = [Matrix::new(rows, cols), Matrix::new(rows, cols)];
            for matrix in &mut moments {
                reader.read_exact(&mut bytes)?;
                for (slot, chunk) in matrix.data.iter_mut().zip(bytes.chunks_exact(4)) {
                    *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
            }
            let [first, second] = moments;
            param.set_moments(first, second)?;
        }
        self.optimizer_step = step;
        Ok(())
    }

    /// Writes weights in a binary snapshot: raw f32, or int8 with a per-row
    /// scale.
    ///
    /// JSON spends about 13 bytes on every weight, because a shortest
    /// round-trip `f32` prints as roughly a dozen characters plus a comma, so a
    /// 50M-parameter model lands near 670 MB. The same weights are 4 bytes each
    /// as raw `f32` and 1 byte each as [`Precision::Q8`], which is 200 MB and
    /// 50 MB for that model.
    ///
    /// The header holds the configuration and optimizer as JSON, so
    /// [`TransformerLm::load_bin`] rebuilds the module tree before it reads any
    /// weights and does not need a matching model to load into.
    pub fn save_bin<P: AsRef<Path>>(
        &mut self,
        path: P,
        precision: Precision,
    ) -> Result<(), NetworkError> {
        let header = serde_json::to_vec(&BinHeader {
            config: self.config.clone(),
            optimizer: self.optimizer.clone(),
            optimizer_step: self.optimizer_step as u64,
            precision,
        })?;

        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        writer.write_all(WEIGHTS_MAGIC)?;
        writer.write_all(&(header.len() as u64).to_le_bytes())?;
        writer.write_all(&header)?;

        let params = self.params_mut();
        writer.write_all(&(params.len() as u64).to_le_bytes())?;
        for param in params {
            let value = &param.value;
            writer.write_all(&(value.rows as u64).to_le_bytes())?;
            writer.write_all(&(value.cols as u64).to_le_bytes())?;
            match precision {
                Precision::F32 => {
                    let bytes: Vec<u8> = value.data.iter().flat_map(|v| v.to_le_bytes()).collect();
                    writer.write_all(&bytes)?;
                }
                Precision::Q8 => {
                    for row in value.data.chunks(value.cols.max(1)) {
                        let absmax = row.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
                        let scale = absmax / 127.0;
                        writer.write_all(&scale.to_le_bytes())?;
                        let quantized: Vec<u8> =
                            row.iter().map(|v| quantize(*v, scale) as u8).collect();
                        writer.write_all(&quantized)?;
                    }
                }
            }
        }
        writer.flush()?;
        Ok(())
    }

    /// Rebuilds the model written by [`TransformerLm::save_bin`].
    ///
    /// A [`Precision::Q8`] file restores dequantized `f32` weights: the model
    /// runs in full precision, it is only the file that is small. Rounding to
    /// 255 levels per row costs about 0.4% relative error on each weight, which
    /// inference absorbs and a resumed training run does not, so keep
    /// [`Precision::F32`] for checkpoints you intend to train from.
    pub fn load_bin<P: AsRef<Path>>(path: P) -> Result<Self, NetworkError> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);

        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != WEIGHTS_MAGIC {
            return Err(NetworkError::InvalidSnapshot(
                "not a RustingBrain weight file".into(),
            ));
        }
        let mut word = [0u8; 8];
        reader.read_exact(&mut word)?;
        let mut header = vec![0u8; u64::from_le_bytes(word) as usize];
        reader.read_exact(&mut header)?;
        let header: BinHeader = serde_json::from_slice(&header)?;
        header.config.validate()?;

        let mut model = Self::from_builder(TransformerBuilder {
            config: header.config,
            optimizer: header.optimizer,
            seed: Some(0),
            mixed_precision: TransformerBuilder::new().mixed_precision,
        })?;
        model.optimizer_step = header.optimizer_step as usize;

        reader.read_exact(&mut word)?;
        let count = u64::from_le_bytes(word) as usize;
        let params = model.params_mut();
        if count != params.len() {
            return Err(NetworkError::InvalidSnapshot(format!(
                "weight file holds {count} parameters, this configuration builds {}",
                params.len()
            )));
        }

        for param in params {
            reader.read_exact(&mut word)?;
            let rows = u64::from_le_bytes(word) as usize;
            reader.read_exact(&mut word)?;
            let cols = u64::from_le_bytes(word) as usize;
            if rows != param.value.rows || cols != param.value.cols {
                return Err(NetworkError::InvalidSnapshot(format!(
                    "weight file has a {rows}x{cols} parameter where the model has {}x{}",
                    param.value.rows, param.value.cols
                )));
            }
            match header.precision {
                Precision::F32 => {
                    let mut bytes = vec![0u8; rows * cols * 4];
                    reader.read_exact(&mut bytes)?;
                    for (slot, chunk) in param.value.data.iter_mut().zip(bytes.chunks_exact(4)) {
                        *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    }
                }
                Precision::Q8 => {
                    let mut scale_bytes = [0u8; 4];
                    let mut row_bytes = vec![0u8; cols];
                    for row in param.value.data.chunks_mut(cols.max(1)) {
                        reader.read_exact(&mut scale_bytes)?;
                        let scale = f32::from_le_bytes(scale_bytes);
                        reader.read_exact(&mut row_bytes)?;
                        for (slot, byte) in row.iter_mut().zip(&row_bytes) {
                            *slot = *byte as i8 as f32 * scale;
                        }
                    }
                }
            }
        }
        Ok(model)
    }

    fn check_length(&self, new_tokens: usize, already_cached: usize) -> Result<(), NetworkError> {
        let length = new_tokens + already_cached;
        if length > self.config.max_seq_len {
            return Err(NetworkError::SequenceTooLong {
                length,
                max_seq_len: self.config.max_seq_len,
            });
        }
        if new_tokens == 0 {
            return Err(NetworkError::EmptyDataset);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal_lm_loss::causal_lm_loss;

    #[test]
    fn accumulating_two_half_batches_sums_the_half_gradients() {
        let ids: [&[u32]; 4] = [
            &[1, 2, 3, 4],
            &[5, 6, 7, 8],
            &[9, 10, 11, 12],
            &[13, 14, 15, 16],
        ];

        // Dense: a mixture-of-experts model's load-balancing loss is computed
        // from routing fractions over whatever batch it sees, so it is not
        // linear in the batch and splitting one changes it. See
        // `accumulate_step`'s note.
        let dense = || tiny().moe_layers(0..0).seed(3);

        // One backward over all four sequences: a mean over four.
        let mut whole = dense().build().unwrap();
        whole.zero_grad();
        whole
            .accumulate_step(&TokenBatch::new(&ids).unwrap())
            .unwrap();
        let whole_grads: Vec<Vec<f32>> = whole
            .params_mut()
            .iter()
            .map(|param| param.grad.data.clone())
            .collect();

        // Two backwards over two sequences each, nothing cleared between them.
        // Each contributes a mean over two, so the sum is twice the mean over
        // four -- which is exactly what `step(1.0 / parts)` divides back out.
        let mut split = dense().build().unwrap();
        split.zero_grad();
        split
            .accumulate_step(&TokenBatch::new(&ids[..2]).unwrap())
            .unwrap();
        split
            .accumulate_step(&TokenBatch::new(&ids[2..]).unwrap())
            .unwrap();
        let split_grads: Vec<Vec<f32>> = split
            .params_mut()
            .iter()
            .map(|param| param.grad.data.clone())
            .collect();

        assert_eq!(whole_grads.len(), split_grads.len());
        let mut compared = 0usize;
        for (whole_param, split_param) in whole_grads.iter().zip(&split_grads) {
            for (whole_grad, split_grad) in whole_param.iter().zip(split_param) {
                let tolerance = 1e-5 + 1e-3 * whole_grad.abs();
                assert!(
                    (split_grad - 2.0 * whole_grad).abs() < tolerance,
                    "accumulated {split_grad} against twice the whole-batch {whole_grad}"
                );
                compared += 1;
            }
        }
        assert!(compared > 1000, "only {compared} gradients compared");
    }

    #[test]
    fn a_loss_mask_zeroes_the_masked_positions_gradient() {
        let ids: [&[u32]; 2] = [&[1, 2, 3, 4], &[5, 6, 7, 8]];
        let batch = TokenBatch::new(&ids).unwrap();
        let grads = |model: &mut TransformerLm| -> Vec<Vec<f32>> {
            model
                .params_mut()
                .iter()
                .map(|param| param.grad.data.clone())
                .collect()
        };

        // A mask that flags every token is the unmasked step exactly.
        let mut plain = tiny().seed(3).build().unwrap();
        plain.zero_grad();
        plain.accumulate_step(&batch).unwrap();
        let mut flagged = tiny().seed(3).build().unwrap();
        flagged.zero_grad();
        flagged
            .accumulate_step_masked(&batch, Some(&[true; 8]))
            .unwrap();
        assert_eq!(grads(&mut plain), grads(&mut flagged));

        // Masking the first half of every sequence leaves a different
        // gradient, but one that is still finite and non-zero: the masked
        // tokens went through the forward pass as context.
        let mut masked = tiny().seed(3).build().unwrap();
        masked.zero_grad();
        let mask = [false, false, true, true, false, false, true, true];
        masked.accumulate_step_masked(&batch, Some(&mask)).unwrap();
        let masked_grads = grads(&mut masked);
        let plain_grads = grads(&mut plain);

        assert_ne!(masked_grads, plain_grads);
        let total: f32 = masked_grads.iter().flatten().map(|g| g.abs()).sum();
        assert!(total.is_finite() && total > 0.0, "gradient was {total}");
    }

    #[test]
    fn a_loss_mask_that_covers_the_wrong_number_of_tokens_is_rejected() {
        let batch = TokenBatch::new(&[[1u32, 2, 3, 4]]).unwrap();
        let mut model = tiny().seed(3).build().unwrap();

        assert!(
            model
                .accumulate_step_masked(&batch, Some(&[true; 3]))
                .is_err()
        );
    }

    #[test]
    fn accumulating_without_zeroing_keeps_adding() {
        let ids: [&[u32]; 2] = [&[1, 2, 3, 4], &[5, 6, 7, 8]];
        let batch = TokenBatch::new(&ids).unwrap();

        let mut model = tiny().seed(3).build().unwrap();
        model.zero_grad();
        model.accumulate_step(&batch).unwrap();
        let once = model.params_mut()[0].grad.data.clone();

        model.accumulate_step(&batch).unwrap();
        let twice = model.params_mut()[0].grad.data.clone();

        for (single, double) in once.iter().zip(&twice) {
            assert!(
                (double - 2.0 * single).abs() < 1e-5 + 1e-3 * single.abs(),
                "second accumulation gave {double}, not twice {single}"
            );
        }
    }

    fn tiny() -> TransformerBuilder {
        TransformerLm::builder()
            .vocab_size(24)
            .d_model(16)
            .n_layers(3)
            .heads(4, 2, 4)
            .d_ff(32)
            .moe_d_ff(12)
            .experts(4, 2)
            .moe_layers([1, 2])
            .shared_expert(true)
            .max_seq_len(32)
            .seed(1234)
    }

    #[test]
    fn forward_produces_one_logit_row_per_token() {
        let model = tiny().build().unwrap();
        let (logits, _) = model.forward_train(&[[1, 2, 3, 4]]).unwrap();

        assert_eq!((logits.rows, logits.cols), (4, 24));
    }

    #[test]
    fn the_configured_layers_are_the_moe_layers() {
        let model = tiny().build().unwrap();

        assert!(!model.blocks[0].feed_forward.is_moe());
        assert!(model.blocks[1].feed_forward.is_moe());
        assert!(model.blocks[2].feed_forward.is_moe());
    }

    #[test]
    fn module_and_config_parameter_counts_agree() {
        for tied in [true, false] {
            let builder = tiny().tie_embeddings(tied);
            let predicted = builder.parameter_counts();
            let model = builder.build().unwrap();

            assert_eq!(model.parameter_counts(), predicted, "tied: {tied}");
        }
    }

    #[test]
    fn a_sparse_model_has_more_total_than_active_parameters() {
        let counts = tiny().experts(16, 2).parameter_counts();

        assert!(counts.total > counts.active);
        assert!(counts.sparsity_ratio() > 1.0);
    }

    #[test]
    fn a_fully_dense_model_has_no_inactive_parameters() {
        let counts = tiny().moe_layers([]).parameter_counts();

        assert_eq!(counts.total, counts.active);
        assert!((counts.sparsity_ratio() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn the_default_configuration_lands_in_the_fifty_million_range() {
        let counts = TransformerConfig::default().parameter_counts();

        assert!((40e6..80e6).contains(&(counts.total as f64)), "{counts}");
        assert!((20e6..60e6).contains(&(counts.active as f64)), "{counts}");
    }

    #[test]
    fn tying_embeddings_removes_the_output_projection() {
        let tied = tiny().tie_embeddings(true).build().unwrap();
        let untied = tiny().tie_embeddings(false).build().unwrap();

        assert!(tied.lm_head.is_none());
        assert!(untied.lm_head.is_some());
        assert_eq!(
            untied.parameter_counts().total - tied.parameter_counts().total,
            24 * 16
        );
    }

    #[test]
    fn a_cached_decode_reproduces_the_full_sequence_forward() {
        let model = tiny().build().unwrap();
        let ids = [5u32, 9, 2, 17, 3];
        let (expected, _) = model.forward_train(&[&ids[..]]).unwrap();

        let mut caches = model.new_kv_caches();
        for (position, &id) in ids.iter().enumerate() {
            let logits = model.forward_cached(&[id], &mut caches).unwrap();
            for (actual, expected) in logits.data.iter().zip(expected.row(position)) {
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "position {position}: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn a_prefill_then_decode_reproduces_the_full_sequence_forward() {
        let model = tiny().build().unwrap();
        let ids = [7u32, 1, 12, 4];
        let (expected, _) = model.forward_train(&[&ids[..]]).unwrap();

        let mut caches = model.new_kv_caches();
        model.forward_cached(&ids[..2], &mut caches).unwrap();
        let logits = model.forward_cached(&ids[2..3], &mut caches).unwrap();

        for (actual, expected) in logits.data.iter().zip(expected.row(2)) {
            assert!((actual - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn end_to_end_backward_matches_finite_differences_on_the_embedding() {
        let mut model = tiny()
            .moe_layers([1])
            .aux_loss_weight(0.0)
            .router_z_loss_weight(0.0)
            .build()
            .unwrap();
        let ids = [3u32, 8, 1, 5];

        let (logits, cache) = model.forward_train(&[&ids[..]]).unwrap();
        let loss = causal_lm_loss(&logits, &ids).unwrap();
        model.zero_grad();
        model.backward(&cache, &loss.grad_logits).unwrap();
        let analytic = model.embedding.weight.grad.data.clone();

        // Only the rows the sequence actually used can have a gradient, so the
        // probe walks one of those. The step stays small because top-k routing
        // is a step function: a larger nudge flips a token to another expert and
        // the numeric slope then measures a jump, not a derivative.
        let epsilon = 1e-4;
        let row = 8usize;
        for column in 0..model.config.d_model {
            let index = row * model.config.d_model + column;
            let mut probe = model.clone();
            probe.embedding.weight.value.data[index] += epsilon;
            let high = causal_lm_loss(&probe.forward_train(&[&ids[..]]).unwrap().0, &ids)
                .unwrap()
                .loss;
            probe.embedding.weight.value.data[index] -= 2.0 * epsilon;
            let low = causal_lm_loss(&probe.forward_train(&[&ids[..]]).unwrap().0, &ids)
                .unwrap()
                .loss;
            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (analytic[index] - numeric).abs() < 1e-2,
                "column {column}: {} vs {numeric}",
                analytic[index]
            );
        }
    }

    #[test]
    fn an_unused_embedding_row_gets_no_gradient_when_untied() {
        let mut model = tiny().tie_embeddings(false).build().unwrap();
        let ids = [2u32, 4, 6];

        let (logits, cache) = model.forward_train(&[&ids[..]]).unwrap();
        let loss = causal_lm_loss(&logits, &ids).unwrap();
        model.zero_grad();
        model.backward(&cache, &loss.grad_logits).unwrap();

        // Row 23 is never gathered, and an untied head is what writes the rest
        // of the matrix, so this row must be untouched.
        assert!(
            model
                .embedding
                .weight
                .grad
                .row(23)
                .iter()
                .all(|&g| g == 0.0)
        );
        assert!(model.embedding.weight.grad.row(2).iter().any(|&g| g != 0.0));
    }

    #[test]
    fn tied_embeddings_collect_gradients_from_both_ends() {
        let mut model = tiny().tie_embeddings(true).build().unwrap();
        let ids = [2u32, 4, 6];

        let (logits, cache) = model.forward_train(&[&ids[..]]).unwrap();
        let loss = causal_lm_loss(&logits, &ids).unwrap();
        model.zero_grad();
        model.backward(&cache, &loss.grad_logits).unwrap();

        // With tying, the unembedding touches every vocabulary row, including
        // ones the input never used.
        assert!(
            model
                .embedding
                .weight
                .grad
                .row(23)
                .iter()
                .any(|&g| g != 0.0)
        );
    }

    #[test]
    fn a_batched_forward_matches_one_forward_per_sequence() {
        let model = tiny().build().unwrap();
        let sequences: [&[u32]; 2] = [&[3, 8, 1, 5], &[7, 2]];

        let (batched, _) = model.forward_train(&sequences).unwrap();
        let seq_len = 4;

        for (sequence, ids) in sequences.iter().enumerate() {
            let (alone, _) = model.forward_train(&[*ids]).unwrap();
            for position in 0..ids.len() {
                let packed = batched.row((sequence * seq_len) + position);
                for (left, right) in packed.iter().zip(alone.row(position)) {
                    assert!((left - right).abs() < 1e-4, "{left} vs {right}");
                }
            }
        }
    }

    #[test]
    fn a_padded_batch_loss_is_the_mean_over_every_predicted_position() {
        let model = tiny()
            .aux_loss_weight(0.0)
            .router_z_loss_weight(0.0)
            .build()
            .unwrap();
        let sequences: [&[u32]; 2] = [&[3, 8, 1, 5], &[7, 2]];

        let batch = TokenBatch::new(&sequences).unwrap();
        let (logits, _) = model.forward_batch(&batch).unwrap();
        let batched = causal_lm_loss_batch(&logits, &batch).unwrap();

        let mut total = 0.0;
        for ids in sequences {
            let (alone, _) = model.forward_train(&[ids]).unwrap();
            total += causal_lm_loss(&alone, ids).unwrap().loss * (ids.len() - 1) as f32;
        }

        let expected = total / batch.predicted() as f32;
        assert!(
            (batched.loss - expected).abs() < 1e-4,
            "{} vs {expected}",
            batched.loss
        );
    }

    #[test]
    fn a_batched_gradient_is_the_mean_of_the_per_sequence_gradients() {
        let mut model = tiny()
            .aux_loss_weight(0.0)
            .router_z_loss_weight(0.0)
            .build()
            .unwrap();
        let sequences: [&[u32]; 2] = [&[3, 8, 1, 5], &[7, 2, 9, 4]];

        let batch = TokenBatch::new(&sequences).unwrap();
        let (logits, cache) = model.forward_batch(&batch).unwrap();
        let loss = causal_lm_loss_batch(&logits, &batch).unwrap();
        model.zero_grad();
        model.backward(&cache, &loss.grad_logits).unwrap();
        let batched = model.embedding.weight.grad.data.clone();

        let mut summed = vec![0.0f32; batched.len()];
        for ids in sequences {
            let (alone, cache) = model.forward_train(&[ids]).unwrap();
            let loss = causal_lm_loss(&alone, ids).unwrap();
            model.zero_grad();
            model.backward(&cache, &loss.grad_logits).unwrap();
            for (total, one) in summed.iter_mut().zip(&model.embedding.weight.grad.data) {
                *total += one / sequences.len() as f32;
            }
        }

        for (left, right) in batched.iter().zip(&summed) {
            assert!((left - right).abs() < 1e-5, "{left} vs {right}");
        }
    }

    #[test]
    fn a_training_step_lowers_the_loss_on_a_repeated_sequence() {
        let mut model = tiny()
            .optimizer(crate::optimizers::Optimizer::adam(1e-2))
            .build()
            .unwrap();
        let ids = [1u32, 2, 3, 4, 5, 6];

        let first = model.train_step(&[&ids[..]]).unwrap();
        let mut last = first;
        for _ in 0..20 {
            last = model.train_step(&[&ids[..]]).unwrap();
        }

        assert!(
            last.lm_loss < first.lm_loss,
            "loss went from {} to {}",
            first.lm_loss,
            last.lm_loss
        );
        assert!(last.total() >= last.lm_loss);
    }

    #[test]
    fn identical_seeds_produce_identical_models() {
        let left = tiny().seed(77).build().unwrap();
        let right = tiny().seed(77).build().unwrap();

        assert_eq!(left, right);
        assert_eq!(
            left.forward_train(&[[1, 2, 3]]).unwrap().0,
            right.forward_train(&[[1, 2, 3]]).unwrap().0
        );
    }

    #[test]
    fn a_binary_snapshot_round_trips_and_is_far_smaller_than_json() {
        let mut model = tiny().seed(5).build().unwrap();
        let dir = std::env::temp_dir();
        let json = dir.join("rusting_brain_size_check.json");
        let f32_path = dir.join("rusting_brain_size_check.f32.rbw");
        let q8_path = dir.join("rusting_brain_size_check.q8.rbw");

        model.save_json(&json).unwrap();
        model.save_bin(&f32_path, Precision::F32).unwrap();
        model.save_bin(&q8_path, Precision::Q8).unwrap();

        let size = |path: &std::path::Path| std::fs::metadata(path).unwrap().len() as f64;
        let (json_size, f32_size, q8_size) = (size(&json), size(&f32_path), size(&q8_path));

        let restored = TransformerLm::load_bin(&f32_path).unwrap();
        assert_eq!(restored, model, "f32 is lossless");
        assert_eq!(
            restored.forward_train(&[[4, 11, 2]]).unwrap().0,
            model.forward_train(&[[4, 11, 2]]).unwrap().0
        );

        let quantized = TransformerLm::load_bin(&q8_path).unwrap();
        assert_eq!(quantized.config, model.config);
        let (left, right) = (
            &quantized.embedding.weight.value,
            &model.embedding.weight.value,
        );
        let scale = right.data.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        for (a, b) in left.data.iter().zip(&right.data) {
            assert!(
                (a - b).abs() <= scale / 127.0,
                "q8 weight {a} is further than one quantization step from {b}"
            );
        }

        for path in [&json, &f32_path, &q8_path] {
            std::fs::remove_file(path).ok();
        }

        assert!(
            f32_size < json_size / 2.5,
            "f32 {f32_size} should be far under json {json_size}"
        );
        assert!(
            q8_size < json_size / 8.0,
            "q8 {q8_size} should be far under json {json_size}"
        );
    }

    #[test]
    fn a_snapshot_round_trips_and_predicts_identically() {
        let model = tiny().build().unwrap();
        let path = std::env::temp_dir().join("rusting_brain_transformer_round_trip.json");
        model.save_json(&path).unwrap();

        let restored = TransformerLm::load_json(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(restored.config, model.config);
        assert_eq!(restored, model);
        assert_eq!(
            restored.forward_train(&[[4, 11, 2]]).unwrap().0,
            model.forward_train(&[[4, 11, 2]]).unwrap().0
        );
    }

    #[test]
    fn optimizer_state_round_trips_through_a_file() {
        let mut model = tiny().build().unwrap();
        model.train_step(&[[1, 2, 3, 4]]).unwrap();
        let path = std::env::temp_dir().join("rusting_brain_optimizer_state.bin");
        model.save_optimizer_state(&path).unwrap();

        // A fresh model has the same shapes and zero moments, so anything that
        // comes back non-zero came out of the file.
        let mut restored = tiny().build().unwrap();
        restored.load_optimizer_state(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(restored.optimizer_step(), model.optimizer_step());
        let mut saved = model.params_mut();
        let mut loaded = restored.params_mut();
        assert_eq!(saved.len(), loaded.len());
        let mut moved = 0.0f32;
        for (from, to) in saved.iter_mut().zip(loaded.iter_mut()) {
            let (first, second) = from.moments().unwrap();
            let (first, second) = (first.clone(), second.clone());
            let (restored_first, restored_second) = to.moments().unwrap();
            assert_eq!(&first, restored_first);
            assert_eq!(&second, restored_second);
            moved += first.data.iter().map(|v| v.abs()).sum::<f32>();
        }
        assert!(moved > 0.0, "a training step should leave non-zero moments");
    }

    #[test]
    fn a_snapshot_survives_a_training_step() {
        let mut model = tiny().build().unwrap();
        model.train_step(&[[1, 2, 3, 4]]).unwrap();

        let json = serde_json::to_string(&model).unwrap();
        let restored: TransformerLm = serde_json::from_str(&json).unwrap();

        assert_eq!(restored, model);
    }

    #[test]
    fn a_sequence_past_the_maximum_is_rejected() {
        let model = tiny().max_seq_len(4).build().unwrap();

        assert!(matches!(
            model.forward_train(&[[1, 2, 3, 4, 5]]),
            Err(NetworkError::SequenceTooLong { length: 5, .. })
        ));
    }

    #[test]
    fn a_cached_decode_past_the_maximum_is_rejected() {
        let model = tiny().max_seq_len(3).build().unwrap();
        let mut caches = model.new_kv_caches();
        model.forward_cached(&[1, 2, 3], &mut caches).unwrap();

        assert!(matches!(
            model.forward_cached(&[4], &mut caches),
            Err(NetworkError::SequenceTooLong { length: 4, .. })
        ));
    }

    #[test]
    fn an_invalid_configuration_is_rejected_before_allocation() {
        assert!(tiny().heads(4, 3, 4).build().is_err());
        assert!(tiny().heads(4, 2, 5).build().is_err());
        assert!(tiny().moe_layers([9]).build().is_err());
        assert!(tiny().experts(2, 4).build().is_err());
    }
}
