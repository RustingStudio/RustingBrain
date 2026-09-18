//! Sparse mixture-of-experts feed-forward layers.
//!
//! Each token is routed to the `experts_per_token` experts whose router
//! probability is highest, and only those experts run for it. Total parameter
//! count therefore grows with `num_experts` while the work per token does not,
//! which is the whole point of the layer.

use crate::activations::softmax;
use crate::batch::Layout;
use crate::ffn::{SwiGlu, SwiGluCache};
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::param::{Linear, Param};
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Default weight for the load-balancing loss, as in the Switch Transformer
/// paper.
pub const DEFAULT_AUX_LOSS_WEIGHT: f32 = 0.01;

/// Default weight for the router z-loss, as in ST-MoE.
pub const DEFAULT_ROUTER_Z_LOSS_WEIGHT: f32 = 1e-3;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct MoeConfig {
    pub num_experts: usize,
    /// Top-k. `1` is switch-style routing.
    pub experts_per_token: usize,
    /// Hidden width of one expert, normally much smaller than a dense layer's
    /// `d_ff`, because capacity comes from the expert count instead.
    pub d_ff: usize,
    /// Adds one always-active expert on top of the routed ones, as Qwen-MoE and
    /// DeepSeek-MoE do, to hold knowledge every token needs.
    pub shared_expert: bool,
    pub aux_loss_weight: f32,
    /// `0.0` disables the z-loss.
    pub router_z_loss_weight: f32,
}

impl MoeConfig {
    pub fn new(num_experts: usize, experts_per_token: usize, d_ff: usize) -> Self {
        Self {
            num_experts,
            experts_per_token,
            d_ff,
            shared_expert: false,
            aux_loss_weight: DEFAULT_AUX_LOSS_WEIGHT,
            router_z_loss_weight: DEFAULT_ROUTER_Z_LOSS_WEIGHT,
        }
    }

    pub fn with_shared_expert(mut self, shared_expert: bool) -> Self {
        self.shared_expert = shared_expert;
        self
    }

    pub fn with_aux_loss_weight(mut self, weight: f32) -> Self {
        self.aux_loss_weight = weight;
        self
    }

    pub fn with_router_z_loss_weight(mut self, weight: f32) -> Self {
        self.router_z_loss_weight = weight;
        self
    }

    pub fn validate(&self) -> Result<(), NetworkError> {
        if self.num_experts == 0 {
            return Err(NetworkError::InvalidConfig(
                "a MoE layer needs at least one expert".into(),
            ));
        }
        if self.experts_per_token == 0 || self.experts_per_token > self.num_experts {
            return Err(NetworkError::InvalidConfig(format!(
                "experts_per_token must be between 1 and num_experts ({}), got {}",
                self.num_experts, self.experts_per_token
            )));
        }
        if self.d_ff == 0 {
            return Err(NetworkError::InvalidConfig(
                "expert d_ff must be non-zero".into(),
            ));
        }
        Ok(())
    }
}

/// A bias-free `d_model -> num_experts` projection producing one logit per
/// expert per token.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Router {
    pub projection: Linear,
}

impl Router {
    pub fn new(d_model: usize, num_experts: usize, rng: &mut StdRng) -> Self {
        Self {
            projection: Linear::new(d_model, num_experts, rng),
        }
    }

    pub fn num_experts(&self) -> usize {
        self.projection.out_features()
    }

    /// `[tokens, d_model] -> [tokens, num_experts]` logits.
    pub fn forward(&self, input: &Matrix) -> Matrix {
        self.projection.forward(input)
    }

    pub fn backward(&mut self, input: &Matrix, grad_logits: &Matrix) -> Matrix {
        self.projection.backward(input, grad_logits)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn linears_mut(&mut self) -> Vec<&mut Linear> {
        vec![&mut self.projection]
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        self.projection.params_mut()
    }
}

/// One routed expert. Experts have the same shape as a dense feed-forward
/// block, just a narrower one.
pub type Expert = SwiGlu;

/// What the backward pass needs, plus the two auxiliary losses the training
/// loop has to add to the language-modelling loss.
#[derive(Clone, Debug)]
pub struct MoeCache {
    input: Matrix,
    /// Full softmax over the router logits, `[tokens, num_experts]`. Needed
    /// whole (not just the top-k) because the load-balancing loss measures the
    /// probability mass on *every* expert.
    probabilities: Matrix,
    /// `log(sum(exp(logits)))` per token, for the z-loss.
    log_sum_exp: Vec<f32>,
    /// `[token * experts_per_token + rank] -> (expert, gate)`, gates already
    /// renormalized over the selected experts. A padding row is routed
    /// nowhere: its slots hold [`UNROUTED`] and a zero gate.
    assignments: Vec<(usize, f32)>,
    /// Rows that held a real token. Padding rows are excluded from the
    /// auxiliary losses, which would otherwise measure the padding.
    valid_tokens: usize,
    /// Token indices routed to each expert, and the flat assignment slot each
    /// one came from.
    expert_tokens: Vec<Vec<usize>>,
    expert_slots: Vec<Vec<usize>>,
    expert_outputs: Vec<Matrix>,
    expert_caches: Vec<Option<SwiGluCache>>,
    shared_cache: Option<SwiGluCache>,
    aux_loss: f32,
    z_loss: f32,
}

impl MoeCache {
    /// Load-balancing loss, already multiplied by `aux_loss_weight`.
    pub fn aux_loss(&self) -> f32 {
        self.aux_loss
    }

    /// Router z-loss, already multiplied by `router_z_loss_weight`. Zero when
    /// the z-loss is disabled.
    pub fn z_loss(&self) -> f32 {
        self.z_loss
    }

    /// `aux_loss + z_loss`, which is what the total loss needs.
    pub fn auxiliary_loss(&self) -> f32 {
        self.aux_loss + self.z_loss
    }

    /// Fraction of routed token slots that went to each expert. Sums to one.
    pub fn load_fractions(&self) -> Vec<f32> {
        let experts = self.expert_tokens.len();
        let slots = self.routed_slots().max(1) as f32;
        (0..experts)
            .map(|expert| self.expert_tokens[expert].len() as f32 / slots)
            .collect()
    }

    /// Assignment slots that actually carry a token.
    fn routed_slots(&self) -> usize {
        self.expert_tokens.iter().map(Vec::len).sum()
    }
}

/// Expert index stored for a padding row's assignment slots.
pub const UNROUTED: usize = usize::MAX;

/// `num_experts` experts, a router, and an optional always-active shared
/// expert.
///
/// This is the dropless formulation: every token is processed by all of its
/// chosen experts, and no token is ever dropped.
///
/// The alternative, not implemented here, is a fixed-capacity layer: each
/// expert accepts at most `capacity_factor * tokens * k / num_experts` tokens
/// and the overflow passes through the residual unchanged. That trades a
/// little quality for statically shaped per-expert buffers, which is what
/// makes batched GPU dispatch and expert parallelism practical. Only
/// [`MoeLayer::route`] and a drop mask in the scatter would have to change;
/// nothing else here assumes the groups are unbounded.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MoeLayer {
    pub router: Router,
    pub experts: Vec<Expert>,
    pub shared: Option<Expert>,
    pub config: MoeConfig,
}

impl MoeLayer {
    pub fn new(d_model: usize, config: MoeConfig, rng: &mut StdRng) -> Result<Self, NetworkError> {
        config.validate()?;

        let experts = (0..config.num_experts)
            .map(|_| SwiGlu::new(d_model, config.d_ff, rng))
            .collect();
        let shared = config
            .shared_expert
            .then(|| SwiGlu::new(d_model, config.d_ff, rng));

        Ok(Self {
            router: Router::new(d_model, config.num_experts, rng),
            experts,
            shared,
            config,
        })
    }

    pub fn d_model(&self) -> usize {
        self.router.projection.in_features()
    }

    pub fn forward(&self, input: &Matrix) -> Result<Matrix, NetworkError> {
        Ok(self.forward_train(input, Layout::default())?.0)
    }

    /// `layout.valid` marks padding rows, which are routed nowhere and left out
    /// of the load-balancing statistics.
    pub fn forward_train(
        &self,
        input: &Matrix,
        layout: Layout<'_>,
    ) -> Result<(Matrix, MoeCache), NetworkError> {
        if input.cols != self.d_model() {
            return Err(NetworkError::InvalidInput {
                expected: self.d_model(),
                actual: input.cols,
            });
        }

        let tokens = input.rows;
        let experts = self.config.num_experts;
        let top_k = self.config.experts_per_token;

        layout.check(tokens)?;
        let valid_tokens = (0..tokens).filter(|&token| layout.is_valid(token)).count();

        let logits = self.router.forward(input);
        let (probabilities, log_sum_exp) = softmax_rows(&logits);
        let assignments = self.route(&probabilities, layout);

        let mut expert_tokens = vec![Vec::new(); experts];
        let mut expert_slots = vec![Vec::new(); experts];
        for (slot, &(expert, _)) in assignments.iter().enumerate() {
            if expert == UNROUTED {
                continue;
            }
            expert_tokens[expert].push(slot / top_k);
            expert_slots[expert].push(slot);
        }

        let expert_inputs: Vec<Matrix> = expert_tokens
            .iter()
            .map(|tokens| gather_rows(input, tokens))
            .collect();

        // Experts are independent, and an expert only sees the tokens routed to
        // it, so this is the natural place for rayon in the layer.
        let evaluated: Vec<Option<(Matrix, SwiGluCache)>> = self
            .experts
            .par_iter()
            .zip(expert_inputs.par_iter())
            .map(|(expert, tokens)| (tokens.rows > 0).then(|| expert.forward_train(tokens)))
            .collect();

        let mut output = Matrix::new(tokens, input.cols);
        let mut expert_outputs = Vec::with_capacity(experts);
        let mut expert_caches = Vec::with_capacity(experts);

        for (expert, evaluated) in evaluated.into_iter().enumerate() {
            let (rows, cache) = match evaluated {
                Some((rows, cache)) => (rows, Some(cache)),
                None => (Matrix::new(0, input.cols), None),
            };

            for (row, &slot) in expert_slots[expert].iter().enumerate() {
                let gate = assignments[slot].1;
                let token = expert_tokens[expert][row];
                let source = rows.row(row);
                for (target, &value) in output.row_mut(token).iter_mut().zip(source) {
                    *target += gate * value;
                }
            }

            expert_outputs.push(rows);
            expert_caches.push(cache);
        }

        let shared_cache = self.shared.as_ref().map(|shared| {
            let (rows, cache) = shared.forward_train(input);
            for (target, &value) in output.data.iter_mut().zip(&rows.data) {
                *target += value;
            }
            cache
        });

        let (aux_loss, z_loss) =
            self.auxiliary_losses(&probabilities, &expert_tokens, &log_sum_exp, layout);

        Ok((
            output,
            MoeCache {
                input: input.clone(),
                probabilities,
                log_sum_exp,
                assignments,
                valid_tokens,
                expert_tokens,
                expert_slots,
                expert_outputs,
                expert_caches,
                shared_cache,
                aux_loss,
                z_loss,
            },
        ))
    }

    /// Accumulates expert and router gradients and returns `dL/dinput`.
    ///
    /// The auxiliary losses are differentiated here too, so a caller only has
    /// to add their *values* to the reported loss; their gradients are already
    /// in the router.
    pub fn backward(&mut self, cache: &MoeCache, grad_output: &Matrix) -> Matrix {
        let tokens = cache.input.rows;
        let width = cache.input.cols;
        let experts = self.config.num_experts;
        let top_k = self.config.experts_per_token;

        let mut grad_input = Matrix::new(tokens, width);

        if let (Some(shared), Some(shared_cache)) = (self.shared.as_mut(), &cache.shared_cache) {
            let contribution = shared.backward(shared_cache, grad_output);
            for (slot, value) in grad_input.data.iter_mut().zip(&contribution.data) {
                *slot += value;
            }
        }

        // Each expert sees only the rows routed to it, scaled by that token's
        // gate. An expert with no tokens is skipped entirely, so no gradient
        // reaches its weights.
        let mut grad_gates = vec![0.0f32; cache.assignments.len()];
        let per_expert: Vec<Option<Matrix>> = self
            .experts
            .par_iter_mut()
            .enumerate()
            .map(|(expert, module)| {
                let expert_cache = cache.expert_caches[expert].as_ref()?;
                let rows = &cache.expert_tokens[expert];

                let mut scaled = Matrix::new(rows.len(), width);
                for (row, &token) in rows.iter().enumerate() {
                    let slot = cache.expert_slots[expert][row];
                    let gate = cache.assignments[slot].1;
                    let upstream = grad_output.row(token);
                    for (target, &value) in scaled.row_mut(row).iter_mut().zip(upstream) {
                        *target = gate * value;
                    }
                }

                Some(module.backward(expert_cache, &scaled))
            })
            .collect();

        for (expert, contribution) in per_expert.into_iter().enumerate() {
            let Some(contribution) = contribution else {
                continue;
            };
            for (row, &token) in cache.expert_tokens[expert].iter().enumerate() {
                let source = contribution.row(row);
                for (target, &value) in grad_input.row_mut(token).iter_mut().zip(source) {
                    *target += value;
                }
            }
        }

        // Gate gradients are one dot product per routed token, and they write
        // into a single flat buffer, so they stay on this thread rather than
        // being shared across the expert tasks above.
        for (expert, rows) in cache.expert_tokens.iter().enumerate() {
            let outputs = &cache.expert_outputs[expert];
            for (row, &token) in rows.iter().enumerate() {
                let slot = cache.expert_slots[expert][row];
                grad_gates[slot] = grad_output
                    .row(token)
                    .iter()
                    .zip(outputs.row(row))
                    .map(|(g, o)| g * o)
                    .sum::<f32>();
            }
        }

        // Gate gradients flow back through the top-k renormalization, then
        // through the softmax. The load-balancing loss adds a term on the
        // probabilities; the z-loss acts on the logits directly.
        let mut grad_probabilities = Matrix::new(tokens, experts);
        for token in 0..tokens {
            let selected = &cache.assignments[token * top_k..(token + 1) * top_k];
            if selected[0].0 == UNROUTED {
                continue;
            }
            let probabilities = cache.probabilities.row(token);

            let total: f32 = selected.iter().map(|&(e, _)| probabilities[e]).sum();
            if total <= 0.0 {
                continue;
            }
            let weighted: f32 = selected
                .iter()
                .enumerate()
                .map(|(rank, &(e, _))| grad_gates[token * top_k + rank] * probabilities[e])
                .sum();

            let row = grad_probabilities.row_mut(token);
            for (rank, &(expert, _)) in selected.iter().enumerate() {
                row[expert] +=
                    grad_gates[token * top_k + rank] / total - weighted / (total * total);
            }
        }

        if self.config.aux_loss_weight != 0.0 && cache.valid_tokens > 0 {
            let scale = self.config.aux_loss_weight * experts as f32 / cache.valid_tokens as f32;
            let slots = (cache.valid_tokens * top_k) as f32;
            for expert in 0..experts {
                let load = cache.expert_tokens[expert].len() as f32 / slots;
                for token in 0..tokens {
                    if cache.assignments[token * top_k].0 == UNROUTED {
                        continue;
                    }
                    grad_probabilities.row_mut(token)[expert] += scale * load;
                }
            }
        }

        let mut grad_logits = Matrix::new(tokens, experts);
        for token in 0..tokens {
            if cache.assignments[token * top_k].0 == UNROUTED {
                continue;
            }
            let probabilities = cache.probabilities.row(token);
            let upstream = grad_probabilities.row(token);
            let dot: f32 = probabilities.iter().zip(upstream).map(|(p, g)| p * g).sum();

            let row = grad_logits.row_mut(token);
            for expert in 0..experts {
                row[expert] = probabilities[expert] * (upstream[expert] - dot);
            }

            if self.config.router_z_loss_weight != 0.0 {
                // d/dlogit of mean(logsumexp^2) is 2 * logsumexp * softmax.
                let factor = 2.0 * self.config.router_z_loss_weight * cache.log_sum_exp[token]
                    / cache.valid_tokens.max(1) as f32;
                for expert in 0..experts {
                    row[expert] += factor * probabilities[expert];
                }
            }
        }

        let from_router = self.router.backward(&cache.input, &grad_logits);
        for (slot, value) in grad_input.data.iter_mut().zip(&from_router.data) {
            *slot += value;
        }

        grad_input
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn linears_mut(&mut self) -> Vec<&mut Linear> {
        let mut linears = self.router.linears_mut();
        for expert in &mut self.experts {
            linears.extend(expert.linears_mut());
        }
        if let Some(shared) = &mut self.shared {
            linears.extend(shared.linears_mut());
        }
        linears
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = self.router.params_mut();
        for expert in &mut self.experts {
            params.extend(expert.params_mut());
        }
        if let Some(shared) = &mut self.shared {
            params.extend(shared.params_mut());
        }
        params
    }

    /// Every weight in the layer, routed experts included.
    pub fn num_parameters(&self) -> usize {
        self.router.projection.weight.len()
            + self
                .experts
                .iter()
                .map(SwiGlu::num_parameters)
                .sum::<usize>()
            + self.shared.as_ref().map_or(0, SwiGlu::num_parameters)
    }

    /// Weights a single token actually multiplies against: the router, its
    /// `experts_per_token` experts, and the shared expert if there is one.
    pub fn active_parameters(&self) -> usize {
        let per_expert = self.experts.first().map_or(0, SwiGlu::num_parameters);
        self.router.projection.weight.len()
            + per_expert * self.config.experts_per_token
            + self.shared.as_ref().map_or(0, SwiGlu::num_parameters)
    }

    /// Top-k selection with the selected probabilities renormalized to sum to
    /// one.
    ///
    /// Taking a softmax over the selected logits alone gives exactly this, so
    /// the full softmax computed for the balancing loss is reused instead of
    /// running a second one.
    fn route(&self, probabilities: &Matrix, layout: Layout<'_>) -> Vec<(usize, f32)> {
        let top_k = self.config.experts_per_token;
        let experts = self.config.num_experts;
        let mut assignments = Vec::with_capacity(probabilities.rows * top_k);

        for token in 0..probabilities.rows {
            if !layout.is_valid(token) {
                assignments.extend(std::iter::repeat_n((UNROUTED, 0.0), top_k));
                continue;
            }
            let row = probabilities.row(token);
            let start = assignments.len();

            for _ in 0..top_k {
                let mut best = usize::MAX;
                for expert in 0..experts {
                    let already_taken = assignments[start..]
                        .iter()
                        .any(|&(taken, _)| taken == expert);
                    if already_taken {
                        continue;
                    }
                    // Strictly greater keeps ties resolving to the lower index,
                    // so routing is reproducible run to run.
                    if best == usize::MAX || row[expert] > row[best] {
                        best = expert;
                    }
                }
                assignments.push((best, row[best]));
            }

            let total: f32 = assignments[start..].iter().map(|&(_, p)| p).sum();
            if total > 0.0 {
                for slot in &mut assignments[start..] {
                    slot.1 /= total;
                }
            }
        }

        assignments
    }

    /// The Switch Transformer load-balancing loss and the ST-MoE z-loss.
    ///
    /// `aux = num_experts * sum_i f_i * P_i * weight`, where `f_i` is the
    /// fraction of routed token slots that went to expert `i` and `P_i` is the
    /// mean router probability on it. Note that its *minimum* is
    /// `weight * 1.0`, reached when routing is perfectly uniform, not zero: the
    /// loss is a scaled dot product of two distributions that each sum to one.
    fn auxiliary_losses(
        &self,
        probabilities: &Matrix,
        expert_tokens: &[Vec<usize>],
        log_sum_exp: &[f32],
        layout: Layout<'_>,
    ) -> (f32, f32) {
        let rows = probabilities.rows;
        let valid: Vec<usize> = (0..rows).filter(|&token| layout.is_valid(token)).collect();
        if valid.is_empty() {
            return (0.0, 0.0);
        }

        let experts = self.config.num_experts;
        let tokens = valid.len() as f32;
        let slots = valid.len() as f32 * self.config.experts_per_token as f32;

        let mut aux = 0.0;
        for (expert, routed) in expert_tokens.iter().enumerate().take(experts) {
            let load = routed.len() as f32 / slots;
            let mass = valid
                .iter()
                .map(|&token| probabilities.row(token)[expert])
                .sum::<f32>()
                / tokens;
            aux += load * mass;
        }
        aux *= experts as f32 * self.config.aux_loss_weight;

        let z = self.config.router_z_loss_weight
            * valid
                .iter()
                .map(|&token| log_sum_exp[token] * log_sum_exp[token])
                .sum::<f32>()
            / tokens;

        (aux, z)
    }
}

/// Row-wise softmax, also returning `log(sum(exp(logits)))` per row.
fn softmax_rows(logits: &Matrix) -> (Matrix, Vec<f32>) {
    let mut probabilities = logits.clone();
    let mut log_sum_exp = Vec::with_capacity(logits.rows);

    for row in 0..logits.rows {
        let source = logits.row(row);
        let max = source.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = source.iter().map(|&v| (v - max).exp()).sum();
        log_sum_exp.push(max + sum.ln());
        softmax(probabilities.row_mut(row));
    }

    (probabilities, log_sum_exp)
}

fn gather_rows(source: &Matrix, rows: &[usize]) -> Matrix {
    let mut gathered = Matrix::new(rows.len(), source.cols);
    for (target, &row) in rows.iter().enumerate() {
        gathered.row_mut(target).copy_from_slice(source.row(row));
    }
    gathered
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn layer(config: MoeConfig, d_model: usize, seed: u64) -> MoeLayer {
        MoeLayer::new(d_model, config, &mut StdRng::seed_from_u64(seed)).unwrap()
    }

    /// A router that sends token `t` to expert `t % num_experts` with a
    /// near-one-hot probability, by making the weight a scaled identity and
    /// feeding it one-hot rows.
    fn one_hot_router(layer: &mut MoeLayer, strength: f32) {
        let experts = layer.config.num_experts;
        let width = layer.d_model();
        let weight = &mut layer.router.projection.weight.value;
        weight.data.fill(0.0);
        for expert in 0..experts {
            weight.row_mut(expert)[expert % width] = strength;
        }
    }

    fn one_hot_rows(rows: &[usize], width: usize) -> Matrix {
        let mut input = Matrix::new(rows.len(), width);
        for (row, &column) in rows.iter().enumerate() {
            input.row_mut(row)[column] = 1.0;
        }
        input
    }

    #[test]
    fn output_keeps_the_model_dimension() {
        let moe = layer(MoeConfig::new(4, 2, 6), 8, 1);
        let output = moe.forward(&Matrix::random(5, 8)).unwrap();

        assert_eq!(output.rows, 5);
        assert_eq!(output.cols, 8);
    }

    #[test]
    fn invalid_top_k_is_rejected() {
        assert!(MoeConfig::new(4, 0, 6).validate().is_err());
        assert!(MoeConfig::new(4, 5, 6).validate().is_err());
        assert!(MoeConfig::new(0, 1, 6).validate().is_err());
    }

    #[test]
    fn routing_picks_the_highest_probability_experts() {
        let moe = layer(MoeConfig::new(5, 2, 4), 4, 2);
        // Hand-built probabilities: expert 3 then expert 1 are the top two.
        let probabilities = Matrix::from_vec(1, 5, vec![0.1, 0.3, 0.05, 0.5, 0.05]);

        let assignments = moe.route(&probabilities, Layout::default());

        assert_eq!(assignments.len(), 2);
        assert_eq!(assignments[0].0, 3);
        assert_eq!(assignments[1].0, 1);
    }

    #[test]
    fn gate_weights_sum_to_one_per_token() {
        for top_k in 1..=4 {
            let moe = layer(MoeConfig::new(4, top_k, 6), 8, 3);
            let (_, cache) = moe
                .forward_train(&Matrix::random(7, 8), Layout::default())
                .unwrap();

            for token in 0..7 {
                let total: f32 = cache.assignments[token * top_k..(token + 1) * top_k]
                    .iter()
                    .map(|&(_, gate)| gate)
                    .sum();
                assert!((total - 1.0).abs() < 1e-5, "top_k {top_k}: {total}");
            }
        }
    }

    #[test]
    fn top_one_routing_uses_a_single_expert_per_token() {
        let moe = layer(MoeConfig::new(4, 1, 6), 8, 4);
        let (_, cache) = moe
            .forward_train(&Matrix::random(6, 8), Layout::default())
            .unwrap();

        assert_eq!(cache.assignments.len(), 6);
        for &(_, gate) in &cache.assignments {
            assert!((gate - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn every_token_reaches_exactly_top_k_experts() {
        let moe = layer(MoeConfig::new(6, 3, 4), 8, 5);
        let (_, cache) = moe
            .forward_train(&Matrix::random(9, 8), Layout::default())
            .unwrap();

        let routed: usize = cache.expert_tokens.iter().map(Vec::len).sum();
        assert_eq!(routed, 9 * 3);
    }

    #[test]
    fn balanced_routing_reaches_the_minimum_of_the_load_balancing_loss() {
        // The Switch formulation is `num_experts * sum_i f_i * P_i`, a scaled
        // dot product of two distributions that each sum to one. Its minimum is
        // 1.0 (uniform routing), not 0.0, and its maximum is `num_experts`.
        let mut balanced = layer(MoeConfig::new(4, 1, 4).with_router_z_loss_weight(0.0), 4, 6);
        one_hot_router(&mut balanced, 30.0);
        let (_, cache) = balanced
            .forward_train(&one_hot_rows(&[0, 1, 2, 3], 4), Layout::default())
            .unwrap();

        let weight = balanced.config.aux_loss_weight;
        assert!(
            (cache.aux_loss() - weight).abs() < 1e-3,
            "balanced aux loss was {}",
            cache.aux_loss()
        );
        for fraction in cache.load_fractions() {
            assert!((fraction - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn imbalanced_routing_costs_more_than_balanced_routing() {
        let mut collapsed = layer(MoeConfig::new(4, 1, 4).with_router_z_loss_weight(0.0), 4, 6);
        one_hot_router(&mut collapsed, 30.0);
        // Every token now looks the same, so every token picks expert 0.
        let (_, cache) = collapsed
            .forward_train(&one_hot_rows(&[0, 0, 0, 0], 4), Layout::default())
            .unwrap();

        let weight = collapsed.config.aux_loss_weight;
        assert!((cache.aux_loss() - 4.0 * weight).abs() < 1e-3);
        assert!(cache.aux_loss() > weight);
        assert_eq!(cache.load_fractions(), vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn the_z_loss_grows_with_the_router_logits() {
        let mut small = layer(MoeConfig::new(4, 1, 4), 4, 7);
        one_hot_router(&mut small, 1.0);
        let mut large = small.clone();
        one_hot_router(&mut large, 20.0);

        let input = one_hot_rows(&[0, 1, 2, 3], 4);
        let quiet = small
            .forward_train(&input, Layout::default())
            .unwrap()
            .1
            .z_loss();
        let loud = large
            .forward_train(&input, Layout::default())
            .unwrap()
            .1
            .z_loss();

        assert!(loud > quiet);
        assert!(quiet > 0.0);
    }

    #[test]
    fn a_disabled_z_loss_is_zero() {
        let moe = layer(MoeConfig::new(4, 2, 6).with_router_z_loss_weight(0.0), 8, 8);
        let (_, cache) = moe
            .forward_train(&Matrix::random(4, 8), Layout::default())
            .unwrap();

        assert_eq!(cache.z_loss(), 0.0);
        assert_eq!(cache.auxiliary_loss(), cache.aux_loss());
    }

    #[test]
    fn the_shared_expert_contributes_to_every_token() {
        let plain = layer(MoeConfig::new(4, 1, 6), 8, 9);
        let mut shared = plain.clone();
        shared.config.shared_expert = true;
        shared.shared = Some(SwiGlu::new(8, 6, &mut StdRng::seed_from_u64(99)));

        let input = Matrix::random(4, 8);
        let without = plain.forward(&input).unwrap();
        let with = shared.forward(&input).unwrap();

        for token in 0..4 {
            assert!(
                without
                    .row(token)
                    .iter()
                    .zip(with.row(token))
                    .any(|(a, b)| (a - b).abs() > 1e-6),
                "token {token} was unchanged by the shared expert"
            );
        }
    }

    #[test]
    fn gradients_reach_only_the_experts_a_token_was_routed_to() {
        let mut moe = layer(
            MoeConfig::new(4, 1, 4)
                .with_aux_loss_weight(0.0)
                .with_router_z_loss_weight(0.0),
            4,
            10,
        );
        one_hot_router(&mut moe, 30.0);

        // Every token routes to expert 0, so experts 1..4 must stay untouched.
        let input = one_hot_rows(&[0, 0, 0], 4);
        let (output, cache) = moe.forward_train(&input, Layout::default()).unwrap();
        let grad_output = Matrix::from_vec(output.rows, output.cols, vec![1.0; output.data.len()]);
        moe.backward(&cache, &grad_output);

        assert!(
            moe.experts[0]
                .gate
                .weight
                .grad
                .data
                .iter()
                .any(|&g| g != 0.0)
        );
        for expert in &moe.experts[1..] {
            assert!(
                expert.gate.weight.grad.data.iter().all(|&g| g == 0.0),
                "an unrouted expert received a gradient"
            );
        }
    }

    #[test]
    fn backward_matches_finite_differences() {
        let mut moe = layer(
            MoeConfig::new(4, 2, 5)
                .with_shared_expert(true)
                .with_aux_loss_weight(0.0)
                .with_router_z_loss_weight(0.0),
            6,
            11,
        );
        let input = Matrix::from_vec(
            3,
            6,
            (0..18)
                .map(|i| ((i * 29) % 17) as f32 / 8.0 - 1.0)
                .collect(),
        );

        let (output, cache) = moe.forward_train(&input, Layout::default()).unwrap();
        let grad_output = Matrix::from_vec(output.rows, output.cols, vec![1.0; output.data.len()]);
        let grad_input = moe.backward(&cache, &grad_output);

        let epsilon = 1e-3;
        for index in 0..input.data.len() {
            let mut bumped = input.clone();
            bumped.data[index] += epsilon;
            let high: f32 = moe.forward(&bumped).unwrap().data.iter().sum();
            bumped.data[index] -= 2.0 * epsilon;
            let low: f32 = moe.forward(&bumped).unwrap().data.iter().sum();
            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (grad_input.data[index] - numeric).abs() < 2e-2,
                "index {index}: {} vs {numeric}",
                grad_input.data[index]
            );
        }
    }

    #[test]
    fn router_weight_gradient_matches_finite_differences() {
        let mut moe = layer(
            MoeConfig::new(3, 2, 4)
                .with_aux_loss_weight(0.0)
                .with_router_z_loss_weight(0.0),
            4,
            12,
        );
        let input = Matrix::from_vec(
            3,
            4,
            (0..12)
                .map(|i| ((i * 13) % 11) as f32 / 5.0 - 1.0)
                .collect(),
        );

        let (output, cache) = moe.forward_train(&input, Layout::default()).unwrap();
        let grad_output = Matrix::from_vec(output.rows, output.cols, vec![1.0; output.data.len()]);
        moe.backward(&cache, &grad_output);
        let analytic = moe.router.projection.weight.grad.data.clone();

        // Small enough not to move a token across a top-k boundary, where the
        // objective is genuinely non-differentiable.
        let epsilon = 1e-3;
        for (index, &expected) in analytic.iter().enumerate() {
            let mut probe = moe.clone();
            probe.router.projection.weight.value.data[index] += epsilon;
            let high: f32 = probe.forward(&input).unwrap().data.iter().sum();
            probe.router.projection.weight.value.data[index] -= 2.0 * epsilon;
            let low: f32 = probe.forward(&input).unwrap().data.iter().sum();
            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (expected - numeric).abs() < 2e-2,
                "index {index}: {} vs {numeric}",
                expected
            );
        }
    }

    #[test]
    fn auxiliary_loss_gradient_matches_finite_differences() {
        let mut moe = layer(MoeConfig::new(3, 2, 4), 4, 13);
        let input = Matrix::from_vec(
            4,
            4,
            (0..16)
                .map(|i| ((i * 19) % 13) as f32 / 6.0 - 1.0)
                .collect(),
        );

        // Zero upstream gradient isolates the auxiliary losses: the only thing
        // reaching the router is the balancing and z terms.
        let (output, cache) = moe.forward_train(&input, Layout::default()).unwrap();
        let grad_output = Matrix::new(output.rows, output.cols);
        moe.backward(&cache, &grad_output);
        let analytic = moe.router.projection.weight.grad.data.clone();

        let epsilon = 1e-3;
        for (index, &expected) in analytic.iter().enumerate() {
            let mut probe = moe.clone();
            probe.router.projection.weight.value.data[index] += epsilon;
            let high = probe
                .forward_train(&input, Layout::default())
                .unwrap()
                .1
                .auxiliary_loss();
            probe.router.projection.weight.value.data[index] -= 2.0 * epsilon;
            let low = probe
                .forward_train(&input, Layout::default())
                .unwrap()
                .1
                .auxiliary_loss();
            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (expected - numeric).abs() < 1e-3,
                "index {index}: {} vs {numeric}",
                expected
            );
        }
    }
}
