//! Decoder-only transformer assembly: configuration, builder, and the model.

use crate::attention::{KvCache, causal_by_default};
use crate::batch::TokenBatch;
use crate::causal_lm_loss::{TotalLoss, causal_lm_loss_batch};
use crate::embedding::Embedding;
use crate::matrix::Matrix;
use crate::moe::{DEFAULT_AUX_LOSS_WEIGHT, DEFAULT_ROUTER_Z_LOSS_WEIGHT, MoeConfig};
use crate::network::NetworkError;
use crate::norm::RmsNorm;
use crate::optimizers::Optimizer;
use crate::param::{Linear, Param};
use crate::rope::{Rope, RopeScaling};
use crate::transformer_block::{FeedForward, TransformerBlock, TransformerBlockCache};
use rand::{SeedableRng, rngs::StdRng};
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
    /// Stretches the rotary angles past the length the weights were trained
    /// at. Absent in checkpoints written before it existed.
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    pub rmsnorm_eps: f32,
    /// Reuses the embedding matrix as the output projection.
    pub tie_embeddings: bool,
    /// Whether attention is causal. False is the encoder shape: every position
    /// reads the whole sequence, which is what a masked-language-model or a
    /// vision transformer wants and what generation cannot use.
    ///
    /// Defaults to true on load, so a checkpoint written before this existed
    /// restores as the decoder it was.
    #[serde(default = "causal_by_default")]
    pub causal: bool,
    pub aux_loss_weight: f32,
    pub router_z_loss_weight: f32,
    /// Set by [`TransformerLm::add_lora`]. Recorded here because it is what
    /// tells [`TransformerLm::load_bin`] to rebuild the adapters and re-freeze
    /// the base weights before it reads a file that holds both.
    #[serde(default)]
    pub lora: Option<LoraConfig>,
}

/// The shape of the low-rank adapters [`TransformerLm::add_lora`] attaches.
///
/// `rank` is the bottleneck width; 8 to 32 is the usual range, and the cost of
/// the adapters is `rank * (in + out)` per projection against the `in * out`
/// the projection itself costs. `alpha / rank` scales the adapter's output, so
/// raising the rank does not by itself raise how much the adapter moves the
/// model; the convention is `alpha == rank` or `alpha == 2 * rank`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f32,
    /// Seeds the initialization of the down-projections, so two runs that
    /// adapt the same base model the same way start from the same adapters.
    pub seed: u64,
}

impl LoraConfig {
    /// `alpha` defaults to `rank`, which is scale one.
    pub fn new(rank: usize) -> Self {
        Self {
            rank,
            alpha: rank as f32,
            seed: 0,
        }
    }

    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
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
            rope_scaling: None,
            rmsnorm_eps: 1e-6,
            tie_embeddings: true,
            causal: true,
            aux_loss_weight: DEFAULT_AUX_LOSS_WEIGHT,
            router_z_loss_weight: DEFAULT_ROUTER_Z_LOSS_WEIGHT,
            lora: None,
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
        if self.lora.is_some_and(|lora| lora.rank == 0) {
            return Err(NetworkError::InvalidConfig(
                "lora rank must be non-zero".into(),
            ));
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

    pub fn rope_scaling(mut self, rope_scaling: Option<RopeScaling>) -> Self {
        self.config.rope_scaling = rope_scaling;
        self
    }

    pub fn rmsnorm_eps(mut self, rmsnorm_eps: f32) -> Self {
        self.config.rmsnorm_eps = rmsnorm_eps;
        self
    }

    /// Drops the causal mask, so every position reads the whole sequence.
    ///
    /// This is the encoder shape — BERT-style masked language modelling, or a
    /// vision transformer over patches. A bidirectional model cannot generate:
    /// appending a token changes the tokens before it, so the KV cache has
    /// nothing to cache and [`TransformerLm::generate`] reports that. Training
    /// it on the next-token loss would also be meaningless, because every
    /// position can see the token it is asked to predict.
    ///
    /// CUDA is decoder-only: the flash-attention kernel is causal, so
    /// [`TransformerLm::to_cuda`] reports that rather than training the wrong
    /// mask on a device.
    pub fn bidirectional(mut self, bidirectional: bool) -> Self {
        self.config.causal = !bidirectional;
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
const LORA_MAGIC: &[u8; 8] = b"RBLOR001";

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

/// Writes `data` as little-endian bytes through a 16 KiB scratch buffer, so a
/// checkpoint never holds a second full copy of its largest tensor.
fn write_f32s(writer: &mut impl Write, data: &[f32]) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(4 * 4096);
    for chunk in data.chunks(4096) {
        bytes.clear();
        bytes.extend(chunk.iter().flat_map(|v| v.to_le_bytes()));
        writer.write_all(&bytes)?;
    }
    Ok(())
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
        let rope = Rope::scaled(
            config.head_dim,
            config.max_seq_len,
            config.rope_base,
            config.rope_scaling,
        )?;

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
            if !config.causal {
                // Set after construction rather than threaded through a
                // ninth argument: the block owns the attention and nothing
                // else in it changes.
                blocks
                    .last_mut()
                    .expect("a block was just pushed")
                    .attention
                    .set_causal(false);
            }
        }

        let lm_head = (!config.tie_embeddings)
            .then(|| Linear::new(config.d_model, config.vocab_size, &mut rng));

        let mut model = Self {
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
        };
        if let Some(lora) = model.config.lora {
            model.attach_lora(lora);
        }
        Ok(model)
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

    /// Continues `prompt` by `max_new_tokens` ids, returning only the new ones.
    ///
    /// The prompt is prefilled into a fresh set of caches in one pass and each
    /// token after it costs one row of attention. A caller that wants the
    /// tokens as they arrive, or wants to stop on one, uses
    /// [`generate_with`](TransformerLm::generate_with); a caller that wants to
    /// keep the caches for a second turn runs the loop itself.
    ///
    /// ```no_run
    /// # use rusting_brain::{Sampler, TransformerLm};
    /// # let mut model = TransformerLm::load_bin("model.rbw")?;
    /// let mut sampler = Sampler::temperature(0.8, None).top_k(40);
    /// let continuation = model.generate(&[1, 2, 3], 128, &mut sampler)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn generate(
        &self,
        prompt: &[u32],
        max_new_tokens: usize,
        sampler: &mut crate::sampling::Sampler,
    ) -> Result<Vec<u32>, NetworkError> {
        self.generate_with(prompt, max_new_tokens, sampler, |_| true)
    }

    /// [`generate`](TransformerLm::generate), calling `on_token` with each id
    /// as it is decoded and stopping early when it returns `false`.
    ///
    /// Decoding a hundred tokens takes as long as a hundred forward passes, so
    /// a caller that prints the result at the end prints nothing for several
    /// seconds. This hands over each token at the point it exists, which is
    /// also where an end-of-text id or a stop sequence is noticed:
    ///
    /// ```no_run
    /// # use rusting_brain::{Sampler, TransformerLm};
    /// # let mut model = TransformerLm::load_bin("model.rbw")?;
    /// # let mut sampler = Sampler::greedy();
    /// # let end_of_text = 0;
    /// let continuation = model.generate_with(&[1, 2, 3], 128, &mut sampler, |id| {
    ///     print!("{id} ");
    ///     id != end_of_text
    /// })?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// The id that stopped the generation is in the returned continuation: it
    /// was sampled, and hiding it would make the two functions disagree about
    /// what the model produced.
    pub fn generate_with(
        &self,
        prompt: &[u32],
        max_new_tokens: usize,
        sampler: &mut crate::sampling::Sampler,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>, NetworkError> {
        if prompt.is_empty() {
            return Err(NetworkError::InvalidConfig(
                "generation needs at least one prompt token".into(),
            ));
        }

        let mut caches = self.new_kv_caches();
        let mut logits = self.forward_cached(prompt, &mut caches)?;
        let mut history = prompt.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);

        for remaining in (1..=max_new_tokens).rev() {
            let next = sampler.pick(logits.row(logits.rows - 1), &history);
            generated.push(next);
            if !on_token(next) {
                break;
            }
            // The last token needs no forward pass, and running one anyway
            // would fail a generation that ends exactly on `max_seq_len`.
            if remaining > 1 {
                history.push(next);
                logits = self.forward_cached(&[next], &mut caches)?;
            }
        }

        Ok(generated)
    }

    /// A decoding session that keeps its caches between turns.
    ///
    /// [`generate`](TransformerLm::generate) throws the caches away when it
    /// returns, so a chat loop re-reads the whole conversation on every turn —
    /// quadratic work in the number of turns, and the second turn of a long
    /// conversation is the expensive one. A [`Decoder`] holds them:
    ///
    /// ```no_run
    /// # use rusting_brain::{Sampler, TransformerLm};
    /// # let model = TransformerLm::load_bin("model.rbw")?;
    /// # let mut sampler = Sampler::greedy();
    /// let mut decoder = model.decoder();
    /// decoder.feed(&[1, 2, 3])?;                  // prompt
    /// let reply: Vec<u32> = (0..20).map(|_| decoder.next(&mut sampler)).collect::<Result<_, _>>()?;
    /// decoder.feed(&[4, 5])?;                     // the user's next turn
    /// let second = decoder.next(&mut sampler)?;   // no re-read of anything above
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn decoder(&self) -> Decoder<'_> {
        Decoder {
            caches: self.new_kv_caches(),
            model: self,
            history: Vec::new(),
            logits: None,
            pending: None,
        }
    }

    /// Replaces every projection weight with one `i8` per value and a scale per
    /// row, and drops the gradients and Adam moments. Returns the bytes the
    /// weights now occupy.
    ///
    /// A CPU decode step reads every active weight once, so it is bound by
    /// memory bandwidth: a quarter of the bytes is most of the way to a quarter
    /// of the time. Rounding to 255 levels per row costs about 0.4% relative
    /// error per weight, which generation absorbs.
    ///
    /// One-way. The `f32` weights are gone afterwards, so
    /// [`train_step`](TransformerLm::train_step),
    /// [`backward`](TransformerLm::backward), `save_bin` and `to_cuda` all
    /// refuse rather than write out or train on rounded weights.
    ///
    /// ```no_run
    /// # use rusting_brain::{Sampler, TransformerLm};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut model = TransformerLm::load_bin("model.rbw")?;
    /// let bytes = model.quantize();
    /// println!("{} MiB of weights", bytes / (1024 * 1024));
    /// let continuation = model.generate(&[1, 2, 3], 128, &mut Sampler::greedy())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn quantize(&mut self) -> usize {
        let mut bytes = self.embedding.weight.quantize();
        for block in &mut self.blocks {
            for linear in block.linears_mut() {
                bytes += linear.weight.quantize();
            }
        }
        if let Some(head) = &mut self.lm_head {
            bytes += head.weight.quantize();
        }
        bytes
    }

    /// Rounds every weight [`quantize`](TransformerLm::quantize) would round
    /// through the int8 grid in the forward pass, while the stored weights and
    /// the optimizer stay in full precision.
    ///
    /// This is quantization-aware training: what backpropagates is the loss of
    /// the rounded model, so the weights settle where rounding costs least and
    /// the accuracy `quantize` gives up afterwards shrinks. The backward pass
    /// is a straight-through estimator: it differentiates as though the
    /// rounding were the identity, rounding having a derivative of zero almost
    /// everywhere.
    ///
    /// It covers the embedding table, every projection in every block and an
    /// untied output head, which is the set `quantize` replaces. A forward pass
    /// costs one rounding pass over each of those weights, so this is usually
    /// worth turning on for the last part of a run rather than all of it.
    ///
    /// CPU only: device training never reads these weights, so a model on a
    /// device reports that rather than training as though the flag had taken.
    ///
    /// ```no_run
    /// # use rusting_brain::{Precision, TransformerLm};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut model = TransformerLm::load_bin("model.rbw")?;
    /// # let sequences: Vec<Vec<u32>> = Vec::new();
    /// // The last part of a run, once the loss has mostly settled.
    /// model.quantization_aware(true)?;
    /// for batch in &sequences {
    ///     model.train_step(&[batch])?;
    /// }
    /// model.quantization_aware(false)?;
    /// model.save_bin("model.q8.rbw", Precision::Q8)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn quantization_aware(&mut self, enabled: bool) -> Result<(), NetworkError> {
        self.check_not_quantized("train quantization-aware")?;
        self.check_not_on_device("train quantization-aware")?;
        self.embedding.weight.set_fake_quantize(enabled);
        for block in &mut self.blocks {
            for linear in block.linears_mut() {
                linear.weight.set_fake_quantize(enabled);
            }
        }
        if let Some(head) = &mut self.lm_head {
            head.weight.set_fake_quantize(enabled);
        }
        Ok(())
    }

    /// Whether [`quantization_aware`](TransformerLm::quantization_aware) is on.
    pub fn is_quantization_aware(&self) -> bool {
        self.embedding.weight.is_fake_quantized()
    }

    /// Whether [`quantize`](TransformerLm::quantize) has been called.
    pub fn is_quantized(&self) -> bool {
        self.embedding.weight.quantized.is_some()
    }

    fn check_not_quantized(&self, action: &str) -> Result<(), NetworkError> {
        if self.is_quantized() {
            return Err(NetworkError::InvalidConfig(format!(
                "this model was quantized for inference, so it cannot {action}"
            )));
        }
        Ok(())
    }

    /// Attaches a low-rank adapter to every attention and feed-forward
    /// projection and freezes everything else.
    ///
    /// This is the fine-tuning shape: the base weights keep their values but
    /// lose their gradient and Adam moments, so a model that needed four
    /// weight-sized buffers to train now needs one plus the adapters. The
    /// adapters start at zero, so the first forward pass after this call
    /// returns exactly what the base model returned.
    ///
    /// The MoE router is left alone. Everything else the training loop already
    /// does — [`train_step_batch`](TransformerLm::train_step_batch),
    /// [`step`](TransformerLm::step), [`save_bin`](TransformerLm::save_bin),
    /// [`to_cuda`](TransformerLm::to_cuda) — keeps working unchanged; the
    /// frozen parameters simply ignore the step.
    ///
    /// Call this while the model is on the host. A device-resident model has
    /// already allocated the gradients and moments freezing would release, so
    /// attaching there would cost the memory the adapter exists to save.
    ///
    /// ```no_run
    /// # use rusting_brain::{LoraConfig, TransformerLm};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut model = TransformerLm::load_bin("base.rbw")?;
    /// model.add_lora(LoraConfig::new(16).alpha(32.0))?;
    /// // ... train ...
    /// model.save_lora("adapter.rbl")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_lora(&mut self, lora: LoraConfig) -> Result<(), NetworkError> {
        self.check_not_quantized("take a LoRA adapter")?;
        if lora.rank == 0 {
            return Err(NetworkError::InvalidConfig(
                "lora rank must be non-zero".into(),
            ));
        }
        if self.config.lora.is_some() {
            return Err(NetworkError::InvalidConfig(
                "this model already has a LoRA adapter; merge it first".into(),
            ));
        }
        self.check_not_on_device("take a LoRA adapter")?;
        self.config.lora = Some(lora);
        self.attach_lora(lora);
        Ok(())
    }

    /// Puts the frozen flags back after a snapshot that carried the adapters
    /// but not the flags. Freezing everything and then thawing the adapters
    /// avoids a second traversal that could disagree with `params_mut`.
    fn refreeze_for_lora(&mut self) {
        for param in self.params_mut() {
            param.freeze();
        }
        for block in &mut self.blocks {
            for linear in block.lora_linears_mut() {
                if let Some(lora) = &mut linear.lora {
                    for param in lora.params_mut() {
                        param.unfreeze();
                    }
                }
            }
        }
    }

    fn attach_lora(&mut self, lora: LoraConfig) {
        for param in self.params_mut() {
            param.freeze();
        }
        let mut rng = StdRng::seed_from_u64(lora.seed);
        for block in &mut self.blocks {
            for linear in block.lora_linears_mut() {
                linear.attach_lora(lora.rank, lora.alpha, &mut rng);
            }
        }
    }

    /// Folds every adapter into the weight it adapts and unfreezes the model.
    ///
    /// What comes out is an ordinary model that predicts what the adapted one
    /// predicted, with no adapters left to carry and no inference cost over the
    /// base. It is the only way to get a trained adapter through
    /// [`quantize`](TransformerLm::quantize), and, like
    /// [`add_lora`](TransformerLm::add_lora), it runs on the host: call
    /// [`to_cpu`](TransformerLm::to_cpu) first.
    pub fn merge_lora(&mut self) -> Result<(), NetworkError> {
        self.check_not_quantized("merge a LoRA adapter")?;
        self.check_not_on_device("merge a LoRA adapter")?;
        for block in &mut self.blocks {
            for linear in block.lora_linears_mut() {
                linear.merge_lora();
            }
        }
        for param in self.params_mut() {
            param.unfreeze();
        }
        self.config.lora = None;
        Ok(())
    }

    /// Whether [`add_lora`](TransformerLm::add_lora) is in effect.
    pub fn has_lora(&self) -> bool {
        self.config.lora.is_some()
    }

    /// Every parameter the optimizer will actually move: the adapters alone
    /// once [`add_lora`](TransformerLm::add_lora) has run, and the whole model
    /// otherwise.
    pub fn trainable_params_mut(&mut self) -> Vec<&mut Param> {
        self.params_mut()
            .into_iter()
            .filter(|param| !param.is_frozen())
            .collect()
    }

    /// How many weights a training step updates.
    pub fn trainable_parameters(&mut self) -> usize {
        self.trainable_params_mut()
            .iter()
            .map(|param| param.len())
            .sum()
    }

    /// Writes the adapter weights alone, which is the point of training one:
    /// a rank-16 adapter over a 50M-parameter model is a few megabytes, and
    /// several of them share one base checkpoint.
    ///
    /// Read back by [`load_lora`](TransformerLm::load_lora) onto a base model
    /// that [`add_lora`](TransformerLm::add_lora) has prepared the same way.
    pub fn save_lora<P: AsRef<Path>>(&mut self, path: P) -> Result<(), NetworkError> {
        let lora = self.config.lora.ok_or_else(|| {
            NetworkError::InvalidConfig("this model has no LoRA adapter to save".into())
        })?;
        let header = serde_json::to_vec(&lora)?;

        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        writer.write_all(LORA_MAGIC)?;
        writer.write_all(&(header.len() as u64).to_le_bytes())?;
        writer.write_all(&header)?;

        let params = self.trainable_params_mut();
        writer.write_all(&(params.len() as u64).to_le_bytes())?;
        for param in params {
            writer.write_all(&(param.value.rows as u64).to_le_bytes())?;
            writer.write_all(&(param.value.cols as u64).to_le_bytes())?;
            let bytes: Vec<u8> = param
                .value
                .data
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            writer.write_all(&bytes)?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Restores adapter weights written by [`save_lora`](TransformerLm::save_lora).
    ///
    /// The model must already carry adapters of the same shape, so the usual
    /// sequence is `load_bin` the base, `add_lora` with the same
    /// [`LoraConfig`], then this. Fails closed on any mismatch rather than
    /// loading part of an adapter.
    pub fn load_lora<P: AsRef<Path>>(&mut self, path: P) -> Result<(), NetworkError> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);

        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != LORA_MAGIC {
            return Err(NetworkError::InvalidSnapshot(
                "not a RustingBrain LoRA file".into(),
            ));
        }
        let mut word = [0u8; 8];
        reader.read_exact(&mut word)?;
        let mut header = vec![0u8; u64::from_le_bytes(word) as usize];
        reader.read_exact(&mut header)?;
        let header: LoraConfig = serde_json::from_slice(&header)?;
        match self.config.lora {
            Some(lora) if lora == header => {}
            Some(lora) => {
                return Err(NetworkError::InvalidSnapshot(format!(
                    "adapter file holds rank {} alpha {}, this model has rank {} alpha {}",
                    header.rank, header.alpha, lora.rank, lora.alpha
                )));
            }
            None => {
                return Err(NetworkError::InvalidSnapshot(
                    "this model has no LoRA adapter to load into; call add_lora first".into(),
                ));
            }
        }

        reader.read_exact(&mut word)?;
        let count = u64::from_le_bytes(word) as usize;
        let params = self.trainable_params_mut();
        if count != params.len() {
            return Err(NetworkError::InvalidSnapshot(format!(
                "adapter file holds {count} parameters, this model has {}",
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
                    "adapter file has a {rows}x{cols} parameter where the model has {}x{}",
                    param.value.rows, param.value.cols
                )));
            }
            let mut bytes = vec![0u8; rows * cols * 4];
            reader.read_exact(&mut bytes)?;
            for (slot, chunk) in param.value.data.iter_mut().zip(bytes.chunks_exact(4)) {
                *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        Ok(())
    }

    fn check_not_on_device(&self, action: &str) -> Result<(), NetworkError> {
        if self.on_device() {
            return Err(NetworkError::InvalidConfig(format!(
                "a device-resident model cannot {action}; call to_cpu first"
            )));
        }
        Ok(())
    }

    /// Accumulates gradients for every parameter from `dL/dlogits`.
    pub fn backward(
        &mut self,
        cache: &TransformerCache,
        grad_logits: &Matrix,
    ) -> Result<(), NetworkError> {
        self.check_not_quantized("be trained")?;
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
        self.check_not_quantized("move to a device")?;
        let counts = self.config.parameter_counts();
        // Value, gradient and the two Adam moments, all FP32 - except under a
        // LoRA adapter, where the frozen base carries its value alone and the
        // adapters, which the counts above do not include, are the only things
        // holding the other three.
        let buffers_per_weight = if self.has_lora() { 1 } else { 4 };
        let estimated_mib = (counts.total * 4 * buffers_per_weight).div_ceil(1024 * 1024);
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
                // `params_mut` rather than `weight`, so a LoRA adapter rides
                // onto the device with the projection it adapts.
                for param in linear.params_mut() {
                    param.move_to_cuda(&context)?;
                }
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

    /// The loss over a batch, without gradients and without a step.
    ///
    /// This is what a held-out split is for: training loss falls whether or not
    /// the model is learning anything general, and the gap between the two is
    /// the only thing that says which.
    ///
    /// ponytail: on a device this materializes the full `[rows, vocab]` logits
    /// on the host, where `train_step_batch` fuses the loss into the head and
    /// never does. Evaluate in batches the size of a training batch, or add a
    /// device loss-only path if that stops being enough.
    pub fn evaluate(&self, batch: &TokenBatch) -> Result<TotalLoss, NetworkError> {
        let (logits, cache) = self.forward_batch(batch)?;
        Ok(TotalLoss {
            lm_loss: causal_lm_loss_batch(&logits, batch)?.loss,
            auxiliary_loss: cache.auxiliary_loss(),
        })
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

    /// Forward, masked-language-model loss, backward and one optimizer update.
    ///
    /// This is the encoder objective: the batch carries corrupted ids, and the
    /// loss scores the model on the tokens that were corrupted, each from both
    /// sides at once. Build the batch with
    /// [`MaskedBatch::corrupt`](crate::masked_lm::MaskedBatch::corrupt).
    ///
    /// The model must be bidirectional, which is what
    /// [`TransformerBuilder::bidirectional`] makes it: with a causal mask a
    /// masked position reads nothing after itself, so most of the objective's
    /// signal is not there to learn from.
    ///
    /// ```no_run
    /// # use rusting_brain::{MaskedBatch, TransformerLm};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut model = TransformerLm::builder()
    ///     .vocab_size(32_000)
    ///     .bidirectional(true)
    ///     .build()?;
    /// let batch = MaskedBatch::corrupt(&[vec![5u32, 6, 7, 8]], 32_000, 4, 0.15, None)?;
    /// let loss = model.train_step_masked(&batch)?;
    /// println!("{}", loss.lm_loss);
    /// # Ok(())
    /// # }
    /// ```
    pub fn train_step_masked(
        &mut self,
        batch: &crate::masked_lm::MaskedBatch,
    ) -> Result<TotalLoss, NetworkError> {
        self.check_bidirectional()?;
        let (logits, cache) = self.forward_batch(batch.inputs())?;
        let loss = crate::masked_lm::masked_lm_loss(&logits, batch)?;

        self.zero_grad();
        self.backward(&cache, &loss.grad_logits)?;
        self.step(1.0);

        Ok(TotalLoss {
            lm_loss: loss.loss,
            auxiliary_loss: cache.auxiliary_loss(),
        })
    }

    /// The masked loss over a batch, without gradients and without a step.
    pub fn evaluate_masked(
        &self,
        batch: &crate::masked_lm::MaskedBatch,
    ) -> Result<TotalLoss, NetworkError> {
        self.check_bidirectional()?;
        let (logits, cache) = self.forward_batch(batch.inputs())?;
        Ok(TotalLoss {
            lm_loss: crate::masked_lm::masked_lm_loss(&logits, batch)?.loss,
            auxiliary_loss: cache.auxiliary_loss(),
        })
    }

    fn check_bidirectional(&self) -> Result<(), NetworkError> {
        if self.config.causal {
            return Err(NetworkError::InvalidConfig(
                "a masked language-modelling loss needs a bidirectional model: under a causal \
                 mask a masked position reads only the tokens before it, which is the next-token \
                 objective with holes in it. Build the model with `bidirectional(true)`"
                    .into(),
            ));
        }
        Ok(())
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
        self.check_not_quantized("be trained")?;
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
        crate::optimizers::apply_step(&mut self.params_mut(), &optimizer, step, scale);
    }

    /// The L2 norm of the accumulated gradients, over every parameter at once.
    ///
    /// Worth logging on its own: a run that is about to diverge shows it in
    /// this number one or two steps before the loss moves.
    pub fn grad_norm(&mut self) -> Result<f32, NetworkError> {
        crate::optimizers::grad_norm(&mut self.params_mut())
    }

    /// [`TransformerLm::step`] with the gradients clipped to a global norm of
    /// `max_norm` first, and the pre-clip norm returned for logging.
    ///
    /// Clipping is a uniform rescale of every gradient, and `scale` already
    /// multiplies every gradient uniformly, so this folds the clip into that
    /// factor rather than rewriting the gradient buffers.
    pub fn step_clipped(&mut self, scale: f32, max_norm: f32) -> Result<f32, NetworkError> {
        self.optimizer_step += 1;
        let step = self.optimizer_step;
        let optimizer = self.optimizer.clone();
        crate::optimizers::step_clipped(&mut self.params_mut(), &optimizer, step, scale, max_norm)
    }

    pub fn zero_grad(&mut self) {
        crate::optimizers::zero_grad(&mut self.params_mut());
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
        let mut model: Self = serde_json::from_reader(std::io::BufReader::new(file))?;
        model.config.validate()?;
        // What is frozen is not in the snapshot; the adapters are, and the
        // configuration says they are there, which is enough to put the base
        // weights back the way `add_lora` left them.
        if model.has_lora() {
            model.refreeze_for_lora();
        }
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
                write_f32s(&mut writer, &matrix.data)?;
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
            // A frozen parameter released its moments, so it wrote a 0x0
            // placeholder and expects one back.
            let (expected_rows, expected_cols) = if param.is_frozen() {
                (0, 0)
            } else {
                (param.value.rows, param.value.cols)
            };
            if rows != expected_rows || cols != expected_cols {
                return Err(NetworkError::InvalidSnapshot(format!(
                    "optimizer state has a {rows}x{cols} parameter where the model has {expected_rows}x{expected_cols}"
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
    /// Writes the model out as an ONNX graph for another runtime to serve.
    ///
    /// The graph takes `seq_len` token ids as `int64` under the name `ids` and
    /// returns `[seq_len, vocab_size]` logits. Both the rotary tables and the
    /// causal mask are baked in at that length, so a model that has to serve
    /// several lengths is exported once per length. Weights are written as
    /// `f32` whatever [`Precision`] a checkpoint would use.
    ///
    /// What it does not cover:
    ///
    /// - Mixture-of-experts layers. Routing is data-dependent control flow
    ///   rather than a graph of tensor ops, and exporting one silently as a
    ///   dense layer would be a different model.
    /// - An attached LoRA adapter. Call
    ///   [`merge_lora`](TransformerLm::merge_lora) first, which folds it into
    ///   the weights the export writes.
    /// - A quantized or device-resident model, as everywhere else.
    ///
    /// There is no KV cache in the graph: it is a prefill, so a runtime
    /// generating from it re-reads the whole prefix per token. Export is a
    /// serving seam for scoring and short prompts, not a fast decode loop.
    ///
    /// ```no_run
    /// # use rusting_brain::TransformerLm;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = TransformerLm::load_bin("model.rbw")?;
    /// model.save_onnx("model.onnx", 128)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn save_onnx<P: AsRef<Path>>(&self, path: P, seq_len: usize) -> Result<(), NetworkError> {
        self.check_not_quantized("be exported to ONNX")?;
        self.check_not_on_device("be exported to ONNX")?;
        if self.has_lora() {
            return Err(NetworkError::InvalidConfig(
                "an adapted model cannot be exported to ONNX; call merge_lora first".into(),
            ));
        }
        self.check_length(seq_len, 0)?;
        crate::onnx_export::export_transformer(self, seq_len, path)
    }

    pub fn save_bin<P: AsRef<Path>>(
        &mut self,
        path: P,
        precision: Precision,
    ) -> Result<(), NetworkError> {
        self.check_not_quantized("be saved")?;
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
                Precision::F32 => write_f32s(&mut writer, &value.data)?,
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

/// A decoding session: the KV caches, the ids that filled them, and the logits
/// for whatever comes next. Built by [`TransformerLm::decoder`].
pub struct Decoder<'a> {
    model: &'a TransformerLm,
    caches: Vec<KvCache>,
    history: Vec<u32>,
    logits: Option<Matrix>,
    /// Sampled but not yet run through the model. The forward pass for a token
    /// is only needed to produce the token after it, so deferring it means a
    /// session that stops decoding never pays for one, and a session that fills
    /// `max_seq_len` exactly does not overflow its caches on the way out.
    pending: Option<u32>,
}

impl Decoder<'_> {
    /// Appends `ids` to the session: a prompt, or the next turn of a
    /// conversation. Only these tokens are read, not the ones before them.
    pub fn feed(&mut self, ids: &[u32]) -> Result<(), NetworkError> {
        self.catch_up()?;
        if ids.is_empty() {
            return Ok(());
        }
        self.logits = Some(self.model.forward_cached(ids, &mut self.caches)?);
        self.history.extend_from_slice(ids);
        Ok(())
    }

    /// Samples the next token and appends it to the session.
    pub fn next(&mut self, sampler: &mut crate::sampling::Sampler) -> Result<u32, NetworkError> {
        self.catch_up()?;
        let logits = self.logits.as_ref().ok_or_else(|| {
            NetworkError::InvalidConfig("decoding needs a prompt fed in first".into())
        })?;

        let next = sampler.pick(logits.row(logits.rows - 1), &self.history);
        self.history.push(next);
        self.pending = Some(next);
        Ok(next)
    }

    /// Every id the session has seen, prompts and generated tokens alike.
    pub fn history(&self) -> &[u32] {
        &self.history
    }

    fn catch_up(&mut self) -> Result<(), NetworkError> {
        if let Some(id) = self.pending.take() {
            self.logits = Some(self.model.forward_cached(&[id], &mut self.caches)?);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal_lm_loss::causal_lm_loss;
    use crate::sampling::Sampler;

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

    #[test]
    fn grad_norm_matches_the_flattened_gradient() {
        let mut model = tiny().build().unwrap();
        model.zero_grad();
        model
            .accumulate_step(&TokenBatch::new(&[[1u32, 2, 3, 4]]).unwrap())
            .unwrap();

        let expected: f64 = model
            .params_mut()
            .iter()
            .flat_map(|param| param.grad.data.iter())
            .map(|&g| f64::from(g) * f64::from(g))
            .sum();

        let norm = model.grad_norm().unwrap();
        assert!(norm > 0.0);
        assert!((norm - expected.sqrt() as f32).abs() < 1e-5, "{norm}");
    }

    /// Clipping to a norm the gradients already sit under must leave the step
    /// byte for byte identical to an unclipped one, and clipping to half the
    /// norm must move the weights exactly half as far as clipping to the full
    /// norm does.
    #[test]
    fn clipping_rescales_the_step_it_takes() {
        let batch = TokenBatch::new(&[[1u32, 2, 3, 4]]).unwrap();
        let stepped = |max_norm: Option<f32>| {
            let mut model = tiny().optimizer(Optimizer::sgd(0.1)).build().unwrap();
            let before = model.blocks[0].attention.query.weight.value.clone();
            model.zero_grad();
            model.accumulate_step(&batch).unwrap();
            let norm = match max_norm {
                Some(max_norm) => model.step_clipped(1.0, max_norm).unwrap(),
                None => {
                    let norm = model.grad_norm().unwrap();
                    model.step(1.0);
                    norm
                }
            };
            let after = &model.blocks[0].attention.query.weight.value;
            let delta: Vec<f32> = after
                .data
                .iter()
                .zip(&before.data)
                .map(|(after, before)| after - before)
                .collect();
            (norm, delta)
        };

        let (norm, unclipped) = stepped(None);
        assert!(norm > 0.0);
        assert_eq!(stepped(Some(norm * 2.0)).1, unclipped);

        let (_, half) = stepped(Some(norm / 2.0));
        for (half, unclipped) in half.iter().zip(&unclipped) {
            assert!((half - unclipped / 2.0).abs() < 1e-6, "{half} {unclipped}");
        }
    }

    /// Fake quantization has to be the *same* rounding `quantize` performs,
    /// or a run trained under it optimizes for arithmetic the checkpoint will
    /// never do. Both models see the same batch, so the two losses agree to
    /// within summation order.
    #[test]
    fn a_fake_quantized_forward_matches_the_quantized_model() {
        let batch = TokenBatch::new(&[[1u32, 2, 3, 4, 5, 6]]).unwrap();
        let mut aware = tiny().build().unwrap();
        let mut rounded = aware.clone();

        aware.quantization_aware(true).unwrap();
        assert!(aware.is_quantization_aware());
        rounded.quantize();

        let aware_loss = aware.evaluate(&batch).unwrap().total();
        let rounded_loss = rounded.evaluate(&batch).unwrap().total();
        assert!(
            (aware_loss - rounded_loss).abs() < 1e-4,
            "{aware_loss} against {rounded_loss}"
        );

        // And it is reversible, unlike `quantize`.
        aware.quantization_aware(false).unwrap();
        assert!(!aware.is_quantization_aware());
        assert!((aware.evaluate(&batch).unwrap().total() - aware_loss).abs() > 1e-6);
    }

    /// Rounding has a derivative of zero almost everywhere, so a backward
    /// pass that differentiated it would learn nothing. The straight-through
    /// estimator is what keeps the gradients flowing, and the weights that
    /// move are the full precision ones, not the grid.
    #[test]
    fn a_fake_quantized_weight_still_trains_in_full_precision() {
        let batch = TokenBatch::new(&[[1u32, 2, 3, 4, 5, 6]]).unwrap();
        let mut model = tiny().optimizer(Optimizer::sgd(0.05)).build().unwrap();
        model.quantization_aware(true).unwrap();

        let before = model.blocks[0].attention.query.weight.value.clone();
        model.zero_grad();
        model.accumulate_step(&batch).unwrap();
        assert!(model.grad_norm().unwrap() > 0.0);
        model.step(1.0);

        let after = &model.blocks[0].attention.query.weight.value;
        assert_ne!(after.data, before.data);
        // A step smaller than the grid it is rounded onto still lands
        // somewhere new: the stored weight never touches the grid.
        let step = after.data.iter().fold(0.0f32, |acc, v| acc.max(v.abs())) / 127.0;
        let moved = after
            .data
            .iter()
            .zip(&before.data)
            .map(|(after, before)| (after - before).abs())
            .fold(0.0f32, f32::max);
        assert!(
            moved > 0.0 && moved < step,
            "moved {moved}, grid step {step}"
        );
    }

    #[test]
    fn a_quantized_model_still_generates_and_refuses_to_train() {
        let mut model = tiny().build().unwrap();
        let before = model
            .generate(&[1, 2, 3], 4, &mut Sampler::greedy())
            .unwrap();

        assert!(model.quantize() > 0);
        assert!(model.is_quantized());

        // Rounded weights are close enough that a three-layer toy model picks
        // the same greedy tokens; what matters here is that it runs at all.
        let after = model
            .generate(&[1, 2, 3], 4, &mut Sampler::greedy())
            .unwrap();
        assert_eq!(before.len(), after.len());

        // One-way: the `f32` weights are gone, so anything that would write
        // them out or update them has to say so rather than produce rounded
        // nonsense.
        assert!(model.train_step(&[[1u32, 2, 3]]).is_err());
        assert!(model.save_bin("/dev/null", Precision::F32).is_err());
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
    fn masked_training_needs_a_bidirectional_model_and_drives_the_loss_down() {
        use crate::masked_lm::MaskedBatch;

        // Every sequence is the same, so the model can learn to reconstruct a
        // hole in it from the tokens on both sides.
        let sequences = [[3u32, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8]];
        let batch = MaskedBatch::corrupt(&sequences, 24, 1, 0.3, Some(5)).unwrap();

        let mut causal = tiny().optimizer(Optimizer::adam(3e-3)).build().unwrap();
        let error = causal.train_step_masked(&batch).unwrap_err().to_string();
        assert!(error.contains("bidirectional"), "{error}");

        let mut model = tiny()
            .bidirectional(true)
            .optimizer(Optimizer::adam(3e-3))
            .build()
            .unwrap();
        let first = model.train_step_masked(&batch).unwrap().lm_loss;
        for _ in 0..40 {
            model.train_step_masked(&batch).unwrap();
        }
        let last = model.evaluate_masked(&batch).unwrap().lm_loss;

        assert!(last < first * 0.5, "{first} -> {last}");
    }

    #[test]
    fn a_bidirectional_model_carries_the_flag_to_every_block_and_back_from_disk() {
        let mut model = tiny().bidirectional(true).build().unwrap();
        assert!(
            model
                .blocks
                .iter()
                .all(|block| !block.attention.is_causal())
        );

        let path = std::env::temp_dir().join("rb_bidirectional.rbw");
        model.save_bin(&path, Precision::F32).unwrap();
        let loaded = TransformerLm::load_bin(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert!(!loaded.config.causal);
        assert!(
            loaded
                .blocks
                .iter()
                .all(|block| !block.attention.is_causal())
        );
        // Generation needs a cache, and a cache needs causality.
        assert!(
            loaded
                .generate(&[1, 2, 3], 4, &mut crate::sampling::Sampler::greedy())
                .is_err()
        );
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
    fn evaluate_reports_the_loss_a_step_would_and_changes_nothing() {
        let mut model = tiny().build().unwrap();
        let batch = TokenBatch::new(&[[1u32, 2, 3, 4], [5, 6, 7, 8]]).unwrap();

        let before = model.evaluate(&batch).unwrap();
        // Same value twice: evaluating must not have moved the weights.
        assert_eq!(before, model.evaluate(&batch).unwrap());

        // `train_step_batch` reports the loss it measured before updating, so
        // the two have to agree.
        let stepped = model.train_step_batch(&batch).unwrap();
        assert!((before.lm_loss - stepped.lm_loss).abs() < 1e-6);
        assert!(model.evaluate(&batch).unwrap().lm_loss < before.lm_loss);
        assert!((before.perplexity() - before.lm_loss.exp()).abs() < 1e-4);
    }

    #[test]
    fn greedy_generation_matches_a_hand_written_decode_loop() {
        let model = tiny().build().unwrap();
        let prompt = [7u32, 1, 12];

        let generated = model
            .generate(&prompt, 5, &mut crate::Sampler::greedy())
            .unwrap();

        let mut caches = model.new_kv_caches();
        let mut logits = model.forward_cached(&prompt, &mut caches).unwrap();
        let mut expected = Vec::new();
        for _ in 0..5 {
            let row = logits.row(logits.rows - 1);
            let next = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            expected.push(next);
            logits = model.forward_cached(&[next], &mut caches).unwrap();
        }
        assert_eq!(generated, expected);
    }

    #[test]
    fn a_callback_sees_every_token_and_can_stop_the_generation() {
        let model = tiny().build().unwrap();
        let prompt = [7u32, 1, 12];

        let mut seen = Vec::new();
        let all = model
            .generate_with(&prompt, 5, &mut crate::Sampler::greedy(), |id| {
                seen.push(id);
                true
            })
            .unwrap();
        assert_eq!(seen, all);
        assert_eq!(all.len(), 5);

        // Stopping on the second token keeps it: it was sampled.
        let mut count = 0;
        let stopped = model
            .generate_with(&prompt, 5, &mut crate::Sampler::greedy(), |_| {
                count += 1;
                count < 2
            })
            .unwrap();
        assert_eq!(stopped, all[..2]);
    }

    #[test]
    fn a_decoder_continues_across_turns_and_matches_generate() {
        let model = tiny().build().unwrap();
        let prompt = [7u32, 1, 12];

        let expected = model
            .generate(&prompt, 5, &mut crate::Sampler::greedy())
            .unwrap();

        let mut decoder = model.decoder();
        // Sampling before a prompt has nothing to sample from.
        assert!(decoder.next(&mut crate::Sampler::greedy()).is_err());

        decoder.feed(&prompt).unwrap();
        let mut sampler = crate::Sampler::greedy();
        let generated: Vec<u32> = (0..5)
            .map(|_| decoder.next(&mut sampler).unwrap())
            .collect();
        assert_eq!(generated, expected);
        assert_eq!(decoder.history().len(), prompt.len() + 5);

        // A second turn continues from the same caches, and the result is what
        // a fresh generation over the whole history would have produced.
        let turn = [3u32, 4];
        decoder.feed(&turn).unwrap();
        let after = decoder.next(&mut sampler).unwrap();

        let mut whole = prompt.to_vec();
        whole.extend(&generated);
        whole.extend(&turn);
        assert_eq!(
            after,
            model
                .generate(&whole, 1, &mut crate::Sampler::greedy())
                .unwrap()[0]
        );

        // Deferring the forward pass means a session can fill the context
        // exactly, as `generate` can.
        let mut decoder = model.decoder();
        decoder.feed(&[3]).unwrap();
        // 32 tokens from a one-token prompt, the same count `generate` manages,
        // because the last one is never run back through the model.
        for _ in 0..32 {
            decoder.next(&mut sampler).unwrap();
        }
        assert_eq!(decoder.history().len(), 33);
        assert!(decoder.next(&mut sampler).is_err());
    }

    #[test]
    fn generation_can_fill_the_context_exactly() {
        let model = tiny().build().unwrap();
        // The final token needs no forward pass, so a run that ends on
        // `max_seq_len` must not overflow the cache.
        let generated = model
            .generate(&[3], 32, &mut crate::Sampler::greedy())
            .unwrap();
        assert_eq!(generated.len(), 32);
        assert!(
            model
                .generate(&[3], 33, &mut crate::Sampler::greedy())
                .is_err()
        );
    }

    #[test]
    fn generation_rejects_an_empty_prompt() {
        let model = tiny().build().unwrap();
        assert!(
            model
                .generate(&[], 4, &mut crate::Sampler::greedy())
                .is_err()
        );
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

    /// The whole adapter workflow on a model with both dense and routed
    /// layers: attaching changes nothing, training moves only the adapters,
    /// the adapter file round-trips, and merging keeps the predictions.
    #[test]
    fn a_lora_adapter_trains_saves_and_merges() {
        let batch = TokenBatch::new(&[[1u32, 2, 3, 4]]).unwrap();
        let mut model = tiny().build().unwrap();
        let base_logits = model.forward_batch(&batch).unwrap().0;
        let base_params = model.parameter_counts().total;

        model.add_lora(LoraConfig::new(4).alpha(8.0)).unwrap();
        assert_eq!(model.forward_batch(&batch).unwrap().0, base_logits);
        // A rank-4 adapter over a 16-wide toy model is not the fraction it
        // would be over a real one, but it is still strictly less than the
        // model, and it is all the optimizer touches.
        let trainable = model.trainable_parameters();
        assert!(trainable > 0 && trainable < base_params, "{trainable}");

        let base_embedding = model.embedding.weight.value.clone();
        model.zero_grad();
        model.train_step_batch(&batch).unwrap();
        model.step(1.0);

        // The frozen half did not move; the adapted half did.
        assert_eq!(model.embedding.weight.value, base_embedding);
        let adapted = model.forward_batch(&batch).unwrap().0;
        assert_ne!(adapted, base_logits);

        let directory = std::env::temp_dir().join("rusting_brain_lora_test");
        std::fs::create_dir_all(&directory).unwrap();
        let adapter = directory.join("adapter.rbl");
        model.save_lora(&adapter).unwrap();

        // A fresh base model plus the adapter file predicts what the trained
        // model predicts.
        let mut restored = tiny().build().unwrap();
        restored.add_lora(LoraConfig::new(4).alpha(8.0)).unwrap();
        restored.load_lora(&adapter).unwrap();
        assert_eq!(restored.forward_batch(&batch).unwrap().0, adapted);

        // A rank the file does not hold is refused rather than half-loaded.
        let mut wrong = tiny().build().unwrap();
        wrong.add_lora(LoraConfig::new(2)).unwrap();
        assert!(wrong.load_lora(&adapter).is_err());

        // A full snapshot carries base and adapters, and comes back adapted.
        let snapshot = directory.join("adapted.rbw");
        model.save_bin(&snapshot, Precision::F32).unwrap();
        let reloaded = TransformerLm::load_bin(&snapshot).unwrap();
        assert!(reloaded.has_lora());
        assert_eq!(reloaded.forward_batch(&batch).unwrap().0, adapted);

        model.merge_lora().unwrap();
        assert!(!model.has_lora());
        assert_eq!(model.trainable_parameters(), base_params);
        for (merged, expected) in model
            .forward_batch(&batch)
            .unwrap()
            .0
            .data
            .iter()
            .zip(&adapted.data)
        {
            assert!((merged - expected).abs() < 1e-4, "{merged} vs {expected}");
        }

        std::fs::remove_dir_all(&directory).ok();
    }

    /// A JSON snapshot carries the adapters and the configuration, but not the
    /// frozen flags, so loading has to put them back.
    #[test]
    fn a_json_snapshot_restores_the_frozen_base() {
        let directory = std::env::temp_dir().join("rusting_brain_lora_json_test");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("adapted.json");

        let mut model = tiny().build().unwrap();
        model.add_lora(LoraConfig::new(4)).unwrap();
        model.save_json(&path).unwrap();

        let mut restored = TransformerLm::load_json(&path).unwrap();
        assert!(restored.has_lora());
        assert!(restored.embedding.weight.is_frozen());
        assert_eq!(
            restored.trainable_parameters(),
            model.trainable_parameters()
        );

        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn an_invalid_configuration_is_rejected_before_allocation() {
        assert!(tiny().heads(4, 3, 4).build().is_err());
        assert!(tiny().heads(4, 2, 5).build().is_err());
        assert!(tiny().moe_layers([9]).build().is_err());
        assert!(tiny().experts(2, 4).build().is_err());
    }
}
