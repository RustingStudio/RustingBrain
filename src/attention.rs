//! Grouped-query self-attention with rotary positions and a causal mask.

use crate::activations::softmax;
use crate::batch::Layout;
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::param::{Linear, Param};
use crate::rope::Rope;
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Growable key/value history for one attention layer.
///
/// Keys and values are stored post-RoPE and post-projection, laid out
/// `[position, kv_heads * head_dim]`, so a decode step appends one row and the
/// attention math reads the whole buffer unchanged.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    width: usize,
    length: usize,
}

impl KvCache {
    pub fn new(kv_heads: usize, head_dim: usize) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            width: kv_heads * head_dim,
            length: 0,
        }
    }

    /// Reserves room for `positions` tokens, so a generation loop does not
    /// reallocate on every step.
    pub fn with_capacity(kv_heads: usize, head_dim: usize, positions: usize) -> Self {
        let width = kv_heads * head_dim;
        Self {
            keys: Vec::with_capacity(width * positions),
            values: Vec::with_capacity(width * positions),
            width,
            length: 0,
        }
    }

    /// Number of cached positions, which is also the absolute position of the
    /// next token to be appended.
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.length = 0;
    }

    /// Appends one or more positions worth of keys and values.
    pub fn append(&mut self, keys: &Matrix, values: &Matrix) -> Result<(), NetworkError> {
        if keys.cols != self.width || values.cols != self.width {
            return Err(NetworkError::InvalidConfig(format!(
                "kv cache expected {} columns, got {} keys and {} values",
                self.width, keys.cols, values.cols
            )));
        }

        self.keys.extend_from_slice(&keys.data);
        self.values.extend_from_slice(&values.data);
        self.length += keys.rows;
        Ok(())
    }

    pub fn keys(&self) -> &[f32] {
        &self.keys
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Everything the backward pass needs from a training forward pass.
///
/// Attention backward needs the post-RoPE projections and the attention
/// probabilities, none of which can be recovered cheaply from the output. The
/// cached-decode path builds no cache at all, because generation never runs
/// backward.
#[derive(Clone, Debug)]
pub struct AttentionCache {
    input: Matrix,
    queries: Matrix,
    keys: Matrix,
    values: Matrix,
    /// `[head][row, key]`, one softmax row per query. A key column is an
    /// offset *within the query's own sequence*, so the matrix is
    /// `[batch * seq_len, seq_len]` however many sequences are packed in.
    probabilities: Vec<Matrix>,
    merged: Matrix,
    seq_len: usize,
}

/// Multi-head attention with `num_kv_heads <= num_heads`.
///
/// Plain multi-head attention is the `num_kv_heads == num_heads` case, so there
/// is one implementation rather than two.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MultiHeadAttention {
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub output: Linear,
    pub rope: Rope,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Whether a query may only read keys at or before its own position.
    ///
    /// False is the encoder shape: every position reads the whole sequence.
    /// Defaults to true on load, so a snapshot written before this existed
    /// restores as the decoder it was.
    #[serde(default = "causal_by_default")]
    causal: bool,
}

pub(crate) fn causal_by_default() -> bool {
    true
}

impl MultiHeadAttention {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: Rope,
        rng: &mut StdRng,
    ) -> Result<Self, NetworkError> {
        if num_heads == 0 || num_kv_heads == 0 {
            return Err(NetworkError::InvalidConfig(
                "attention needs at least one query head and one key/value head".into(),
            ));
        }
        if num_kv_heads > num_heads || num_heads % num_kv_heads != 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
            )));
        }
        if rope.head_dim() != head_dim {
            return Err(NetworkError::InvalidConfig(format!(
                "rope was built for head_dim {} but attention uses {head_dim}",
                rope.head_dim()
            )));
        }

        Ok(Self {
            query: Linear::new(d_model, num_heads * head_dim, rng),
            key: Linear::new(d_model, num_kv_heads * head_dim, rng),
            value: Linear::new(d_model, num_kv_heads * head_dim, rng),
            output: Linear::new(num_heads * head_dim, d_model, rng),
            rope,
            num_heads,
            num_kv_heads,
            head_dim,
            causal: true,
        })
    }

    /// Drops the causal mask, so every position reads the whole sequence.
    ///
    /// This is what separates an encoder from a decoder. Nothing else about
    /// the layer changes: the backward pass already treats a masked weight as
    /// a zero probability rather than as a special case, so it needs no
    /// knowledge of which shape it is differentiating.
    pub fn set_causal(&mut self, causal: bool) {
        self.causal = causal;
    }

    pub fn is_causal(&self) -> bool {
        self.causal
    }

    pub fn d_model(&self) -> usize {
        self.query.in_features()
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// How many query heads share one key/value head.
    fn group_size(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    fn scale(&self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    /// Full-sequence forward pass, keeping what the backward pass needs.
    ///
    /// `layout` says how the rows split into sequences; the default is one
    /// sequence covering every row. Attention is the only sub-layer where the
    /// split matters, because a token must not attend across a sequence
    /// boundary: a query at position `t` of its sequence sees keys `0..=t` of
    /// *that* sequence and nothing else.
    pub fn forward_train(
        &self,
        input: &Matrix,
        layout: Layout<'_>,
    ) -> Result<(Matrix, AttentionCache), NetworkError> {
        layout.check(input.rows)?;
        let seq_len = layout.seq_len(input.rows);
        let (queries, keys, values) = self.project_batched(input, seq_len)?;

        let mut probabilities = Vec::with_capacity(self.num_heads);
        let mut merged = Matrix::new(input.rows, self.num_heads * self.head_dim);

        for head in 0..self.num_heads {
            let scores = self.head_scores_batched(&queries, &keys, head, seq_len);
            self.accumulate_head_output_batched(&scores, &values, head, seq_len, &mut merged);
            probabilities.push(scores);
        }

        let output = self.output.forward(&merged);

        Ok((
            output,
            AttentionCache {
                input: input.clone(),
                queries,
                keys,
                values,
                probabilities,
                merged,
                seq_len,
            },
        ))
    }

    /// Incremental forward pass against a cache.
    ///
    /// `input` holds the new tokens only; the cache supplies the history. The
    /// first call with an empty cache and a full prompt is the prefill, every
    /// later call is normally a single row.
    pub fn forward_cached(
        &self,
        input: &Matrix,
        cache: &mut KvCache,
    ) -> Result<Matrix, NetworkError> {
        if !self.causal {
            return Err(NetworkError::InvalidConfig(
                "a bidirectional layer cannot decode from a cache: every position reads the \
                 whole sequence, so an appended token changes the tokens before it"
                    .into(),
            ));
        }
        let position_offset = cache.len();
        let (queries, keys, values) = self.project(input, position_offset)?;
        cache.append(&keys, &values)?;

        let history = cache.len();
        let width = keys.cols;

        #[cfg(feature = "cuda")]
        if let Some(context) = self.gpu_context() {
            let cached_keys = Matrix::from_vec(history, width, cache.keys().to_vec());
            let cached_values = Matrix::from_vec(history, width, cache.values().to_vec());
            let (_, merged) = crate::gpu_transformer::attention_heads(
                &context,
                &queries,
                &cached_keys,
                &cached_values,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                position_offset,
                false,
            );
            return Ok(self.output.forward(&merged));
        }

        // The cache is read in place. Copying it into a `Matrix` per layer per
        // decoded token is O(history) memory traffic on top of O(history) math,
        // which dominates once the history is a few hundred tokens long.
        let mut merged = Matrix::new(input.rows, self.num_heads * self.head_dim);
        for head in 0..self.num_heads {
            let scores = self.head_scores(
                &queries,
                cache.keys(),
                width,
                head,
                position_offset,
                history,
            );
            self.accumulate_head_output(&scores, cache.values(), width, head, &mut merged);
        }

        Ok(self.output.forward(&merged))
    }

    /// Accumulates weight gradients and returns `dL/dinput`.
    ///
    /// Training always sees the whole sequence at once, so this is the
    /// counterpart of [`MultiHeadAttention::forward_train`] only.
    pub fn backward(
        &mut self,
        cache: &AttentionCache,
        grad_output: &Matrix,
    ) -> Result<Matrix, NetworkError> {
        let rows = cache.input.rows;
        let seq_len = cache.seq_len;
        let head_dim = self.head_dim;
        let scale = self.scale();
        let group_size = self.group_size();

        let grad_merged = self.output.backward(&cache.merged, grad_output);

        let mut grad_queries = Matrix::new(rows, self.num_heads * head_dim);
        let mut grad_keys = Matrix::new(rows, self.num_kv_heads * head_dim);
        let mut grad_values = Matrix::new(rows, self.num_kv_heads * head_dim);

        // One head of one sequence is a strided window of the packed
        // `[rows, heads * head_dim]` projections, and `sgemm` takes a row and a
        // column stride per operand, so each of the five products below reads
        // its head where it lies instead of gathering it into a temporary.
        // `dL/dscore` is the only scratch, and it is reused across heads.
        let sequences = rows / seq_len;
        let mut grad_scores = vec![0.0f32; seq_len * seq_len];

        for head in 0..self.num_heads {
            let probabilities = &cache.probabilities[head];
            let query_base = head * head_dim;
            let kv_base = (head / group_size) * head_dim;

            for sequence in 0..sequences {
                let probability_offset = sequence * seq_len * seq_len;
                let query_offset = sequence * seq_len * cache.queries.cols + query_base;
                let merged_offset = sequence * seq_len * grad_merged.cols + query_base;
                let key_offset = sequence * seq_len * cache.keys.cols + kv_base;
                let value_offset = sequence * seq_len * cache.values.cols + kv_base;

                unsafe {
                    // dL/dprobability = dL/dmerged_head * V_head^T.
                    matrixmultiply::sgemm(
                        seq_len,
                        head_dim,
                        seq_len,
                        1.0,
                        grad_merged.data.as_ptr().add(merged_offset),
                        grad_merged.cols as isize,
                        1,
                        cache.values.data.as_ptr().add(value_offset),
                        1,
                        cache.values.cols as isize,
                        0.0,
                        grad_scores.as_mut_ptr(),
                        seq_len as isize,
                        1,
                    );

                    // dL/dV_head += P^T * dL/dmerged_head. Accumulating rather
                    // than assigning is what makes grouped-query attention
                    // correct: `group_size` query heads share these columns.
                    matrixmultiply::sgemm(
                        seq_len,
                        seq_len,
                        head_dim,
                        1.0,
                        probabilities.data.as_ptr().add(probability_offset),
                        1,
                        seq_len as isize,
                        grad_merged.data.as_ptr().add(merged_offset),
                        grad_merged.cols as isize,
                        1,
                        1.0,
                        grad_values.data.as_mut_ptr().add(value_offset),
                        grad_values.cols as isize,
                        1,
                    );
                }

                // Softmax backward, in place over the whole square. A masked
                // entry has probability zero, so it stays zero here and the
                // causal mask needs no separate handling.
                for position in 0..seq_len {
                    let weights =
                        &probabilities.data[probability_offset + position * seq_len..][..seq_len];
                    let row = &mut grad_scores[position * seq_len..][..seq_len];
                    let dot: f32 = weights.iter().zip(row.iter()).map(|(p, g)| p * g).sum();
                    for (slot, &weight) in row.iter_mut().zip(weights) {
                        *slot = weight * (*slot - dot) * scale;
                    }
                }

                unsafe {
                    // dL/dQ_head += dL/dscore * K_head.
                    matrixmultiply::sgemm(
                        seq_len,
                        seq_len,
                        head_dim,
                        1.0,
                        grad_scores.as_ptr(),
                        seq_len as isize,
                        1,
                        cache.keys.data.as_ptr().add(key_offset),
                        cache.keys.cols as isize,
                        1,
                        1.0,
                        grad_queries.data.as_mut_ptr().add(query_offset),
                        grad_queries.cols as isize,
                        1,
                    );

                    // dL/dK_head += dL/dscore^T * Q_head, accumulating for the
                    // same grouped-query reason as the value gradient.
                    matrixmultiply::sgemm(
                        seq_len,
                        seq_len,
                        head_dim,
                        1.0,
                        grad_scores.as_ptr(),
                        1,
                        seq_len as isize,
                        cache.queries.data.as_ptr().add(query_offset),
                        cache.queries.cols as isize,
                        1,
                        1.0,
                        grad_keys.data.as_mut_ptr().add(key_offset),
                        grad_keys.cols as isize,
                        1,
                    );
                }
            }
        }

        // RoPE is orthogonal, so its backward pass is the same rotation
        // applied with the opposite sign.
        self.rope
            .apply_inverse_batched(&mut grad_queries, self.num_heads, seq_len)?;
        self.rope
            .apply_inverse_batched(&mut grad_keys, self.num_kv_heads, seq_len)?;

        let mut grad_input = self.query.backward(&cache.input, &grad_queries);
        let from_keys = self.key.backward(&cache.input, &grad_keys);
        let from_values = self.value.backward(&cache.input, &grad_values);
        for ((slot, k), v) in grad_input
            .data
            .iter_mut()
            .zip(&from_keys.data)
            .zip(&from_values.data)
        {
            *slot += k + v;
        }

        Ok(grad_input)
    }

    /// Every projection in this layer, for uploading to a device or quantizing.
    pub(crate) fn linears_mut(&mut self) -> Vec<&mut Linear> {
        vec![
            &mut self.query,
            &mut self.key,
            &mut self.value,
            &mut self.output,
        ]
    }

    /// The device the projections live on, if any. Attention borrows it for the
    /// two score matmuls; everything between them stays on the host.
    #[cfg(feature = "cuda")]
    fn gpu_context(&self) -> Option<std::sync::Arc<crate::gpu_transformer::GpuContext>> {
        self.query
            .weight
            .device
            .as_ref()
            .map(|device| device.context().clone())
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = self.query.params_mut();
        params.extend(self.key.params_mut());
        params.extend(self.value.params_mut());
        params.extend(self.output.params_mut());
        params
    }

    /// Q, K and V for the new tokens, with RoPE already applied to Q and K.
    fn project(
        &self,
        input: &Matrix,
        position_offset: usize,
    ) -> Result<(Matrix, Matrix, Matrix), NetworkError> {
        if input.cols != self.d_model() {
            return Err(NetworkError::InvalidInput {
                expected: self.d_model(),
                actual: input.cols,
            });
        }

        let mut queries = self.query.forward(input);
        let mut keys = self.key.forward(input);
        let values = self.value.forward(input);

        self.rope
            .apply(&mut queries, self.num_heads, position_offset)?;
        self.rope
            .apply(&mut keys, self.num_kv_heads, position_offset)?;

        Ok((queries, keys, values))
    }

    /// Q, K and V for a packed batch, with RoPE already applied to Q and K.
    fn project_batched(
        &self,
        input: &Matrix,
        seq_len: usize,
    ) -> Result<(Matrix, Matrix, Matrix), NetworkError> {
        if input.cols != self.d_model() {
            return Err(NetworkError::InvalidInput {
                expected: self.d_model(),
                actual: input.cols,
            });
        }

        let mut queries = self.query.forward(input);
        let mut keys = self.key.forward(input);
        let values = self.value.forward(input);

        self.rope
            .apply_batched(&mut queries, self.num_heads, seq_len)?;
        self.rope
            .apply_batched(&mut keys, self.num_kv_heads, seq_len)?;

        Ok((queries, keys, values))
    }

    /// Softmaxed attention weights for one head of a packed batch,
    /// `[batch * seq_len, seq_len]`.
    ///
    /// Column `k` of row `r` is the weight the query at position `r % seq_len`
    /// puts on key `k` *of its own sequence*. Masked entries are exactly zero
    /// rather than `-inf` exponentiated, so nothing downstream has to carry the
    /// mask around and no numerical accident can soften it.
    fn head_scores_batched(
        &self,
        queries: &Matrix,
        keys: &Matrix,
        head: usize,
        seq_len: usize,
    ) -> Matrix {
        let head_dim = self.head_dim;
        let query_base = head * head_dim;
        let kv_base = (head / self.group_size()) * head_dim;
        let scale = self.scale();
        let sequences = queries.rows / seq_len;

        let mut scores = Matrix::new(queries.rows, seq_len);
        for sequence in 0..sequences {
            let query_offset = sequence * seq_len * queries.cols + query_base;
            let key_offset = sequence * seq_len * keys.cols + kv_base;
            let score_offset = sequence * seq_len * seq_len;

            // Q_head times K_head transposed, both read in place out of the
            // packed projections through `sgemm`'s row and column strides. The
            // masked upper triangle is computed and then discarded, which costs
            // half the multiplies but buys a single blocked, threaded kernel in
            // place of `seq_len` growing dot-product loops.
            unsafe {
                matrixmultiply::sgemm(
                    seq_len,
                    head_dim,
                    seq_len,
                    scale,
                    queries.data.as_ptr().add(query_offset),
                    queries.cols as isize,
                    1,
                    keys.data.as_ptr().add(key_offset),
                    1,
                    keys.cols as isize,
                    0.0,
                    scores.data.as_mut_ptr().add(score_offset),
                    seq_len as isize,
                    1,
                );
            }
        }

        // Normalizing afterwards rather than inside the loop above puts every
        // row of every sequence in one parallel pass. `softmax` is an `exp` per
        // element, and the masked tail is cleared to an exact zero so that
        // nothing downstream has to know where the causal boundary is.
        scores
            .data
            .par_chunks_mut(seq_len)
            .enumerate()
            .for_each(|(row, weights)| {
                let visible = if self.causal {
                    row % seq_len + 1
                } else {
                    seq_len
                };
                softmax(&mut weights[..visible]);
                weights[visible..].fill(0.0);
            });

        scores
    }

    /// `merged[:, head] += probabilities * V_head`, per sequence.
    fn accumulate_head_output_batched(
        &self,
        probabilities: &Matrix,
        values: &Matrix,
        head: usize,
        seq_len: usize,
        merged: &mut Matrix,
    ) {
        let head_dim = self.head_dim;
        let query_base = head * head_dim;
        let kv_base = (head / self.group_size()) * head_dim;
        let sequences = probabilities.rows / seq_len;

        for sequence in 0..sequences {
            // Masked weights are exactly zero, so the whole square takes part
            // in the multiply and the causal mask costs nothing here. `beta` is
            // one because the head writes its own column range of `merged`.
            unsafe {
                matrixmultiply::sgemm(
                    seq_len,
                    seq_len,
                    head_dim,
                    1.0,
                    probabilities
                        .data
                        .as_ptr()
                        .add(sequence * seq_len * seq_len),
                    seq_len as isize,
                    1,
                    values
                        .data
                        .as_ptr()
                        .add(sequence * seq_len * values.cols + kv_base),
                    values.cols as isize,
                    1,
                    1.0,
                    merged
                        .data
                        .as_mut_ptr()
                        .add(sequence * seq_len * merged.cols + query_base),
                    merged.cols as isize,
                    1,
                );
            }
        }
    }

    /// Softmaxed attention weights for one head, `[queries, history]`.
    ///
    /// Row `i` describes the query at absolute position
    /// `position_offset + i`, which may attend to every key up to and including
    /// its own position. Masked entries are left at zero rather than set to
    /// `-inf` and exponentiated, so the causal structure is in the loop bounds
    /// and cannot be softened by a numerical accident.
    fn head_scores(
        &self,
        queries: &Matrix,
        keys: &[f32],
        key_cols: usize,
        head: usize,
        position_offset: usize,
        history: usize,
    ) -> Matrix {
        let head_dim = self.head_dim;
        let query_base = head * head_dim;
        let kv_base = (head / self.group_size()) * head_dim;
        let scale = self.scale();

        let mut scores = Matrix::new(queries.rows, history);
        for query in 0..queries.rows {
            let query_row = &queries.row(query)[query_base..query_base + head_dim];
            let visible = position_offset + query + 1;
            debug_assert!(visible <= history);

            let row = scores.row_mut(query);
            for (key, slot) in row.iter_mut().enumerate().take(visible) {
                let base = key * key_cols + kv_base;
                let key_row = &keys[base..base + head_dim];
                *slot = crate::matrix::dot(query_row, key_row) * scale;
            }
            softmax(&mut row[..visible]);
        }

        scores
    }

    /// `merged[:, head] += probabilities * V_head`.
    fn accumulate_head_output(
        &self,
        probabilities: &Matrix,
        values: &[f32],
        value_cols: usize,
        head: usize,
        merged: &mut Matrix,
    ) {
        let head_dim = self.head_dim;
        let query_base = head * head_dim;
        let kv_base = (head / self.group_size()) * head_dim;

        for query in 0..probabilities.rows {
            let weights = probabilities.row(query);
            let target = &mut merged.row_mut(query)[query_base..query_base + head_dim];

            for (key, &weight) in weights.iter().enumerate() {
                if weight == 0.0 {
                    continue;
                }
                let base = key * value_cols + kv_base;
                let value_row = &values[base..base + head_dim];
                for (slot, &value) in target.iter_mut().zip(value_row) {
                    *slot += weight * value;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn attention(num_heads: usize, num_kv_heads: usize) -> MultiHeadAttention {
        let mut rng = StdRng::seed_from_u64(11);
        let head_dim = 4;
        let rope = Rope::new(head_dim, 32, 10000.0).unwrap();
        MultiHeadAttention::new(8, num_heads, num_kv_heads, head_dim, rope, &mut rng).unwrap()
    }

    fn inputs(tokens: usize, d_model: usize) -> Matrix {
        // Deterministic, spread over both signs, and not symmetric in a way
        // that could hide an indexing mistake.
        Matrix::from_vec(
            tokens,
            d_model,
            (0..tokens * d_model)
                .map(|i| ((i * 37) % 23) as f32 / 11.0 - 1.0)
                .collect(),
        )
    }

    #[test]
    fn output_keeps_the_model_dimension() {
        let layer = attention(2, 2);
        let (output, _) = layer
            .forward_train(&inputs(5, 8), Layout::default())
            .unwrap();

        assert_eq!(output.rows, 5);
        assert_eq!(output.cols, 8);
    }

    #[test]
    fn grouped_query_heads_must_divide_evenly() {
        let mut rng = StdRng::seed_from_u64(1);
        let rope = Rope::new(4, 32, 10000.0).unwrap();

        assert!(MultiHeadAttention::new(8, 6, 4, 4, rope.clone(), &mut rng).is_err());
        assert!(MultiHeadAttention::new(8, 4, 8, 4, rope, &mut rng).is_err());
    }

    #[test]
    fn attention_weights_sum_to_one_over_the_visible_prefix() {
        let layer = attention(2, 1);
        let (_, cache) = layer
            .forward_train(&inputs(4, 8), Layout::default())
            .unwrap();

        for head in &cache.probabilities {
            for query in 0..head.rows {
                let total: f32 = head.row(query).iter().sum();
                assert!((total - 1.0).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn the_causal_mask_zeroes_every_future_key() {
        let layer = attention(2, 2);
        let (_, cache) = layer
            .forward_train(&inputs(5, 8), Layout::default())
            .unwrap();

        for head in &cache.probabilities {
            for query in 0..head.rows {
                for key in (query + 1)..head.cols {
                    assert_eq!(head.row(query)[key], 0.0, "query {query} saw key {key}");
                }
            }
        }
    }

    #[test]
    fn a_later_token_cannot_change_an_earlier_output() {
        let layer = attention(2, 1);
        let original = inputs(4, 8);
        let (before, _) = layer.forward_train(&original, Layout::default()).unwrap();

        // Rewrite the last token entirely. Everything before it must be
        // bit-for-bit unchanged, which is the property the mask exists for.
        let mut edited = original.clone();
        for value in edited.row_mut(3) {
            *value = 9.0;
        }
        let (after, _) = layer.forward_train(&edited, Layout::default()).unwrap();

        assert_eq!(&before.data[..3 * 8], &after.data[..3 * 8]);
        assert_ne!(before.row(3), after.row(3));
    }

    #[test]
    fn a_bidirectional_layer_lets_a_later_token_change_an_earlier_output() {
        let mut layer = attention(2, 1);
        layer.set_causal(false);
        let original = inputs(4, 8);
        let (before, _) = layer.forward_train(&original, Layout::default()).unwrap();

        let mut edited = original.clone();
        for value in edited.row_mut(3) {
            *value = 9.0;
        }
        let (after, _) = layer.forward_train(&edited, Layout::default()).unwrap();

        // The exact opposite of the causal case: every earlier position reads
        // the token that changed, so every output moves.
        for token in 0..4 {
            assert_ne!(before.row(token), after.row(token), "token {token}");
        }
    }

    #[test]
    fn a_bidirectional_layer_refuses_to_decode_from_a_cache() {
        let mut layer = attention(2, 1);
        layer.set_causal(false);
        let mut cache = KvCache::new(1, 4);

        assert!(layer.forward_cached(&inputs(1, 8), &mut cache).is_err());
    }

    #[test]
    fn bidirectional_backward_matches_finite_differences() {
        let mut layer = attention(2, 1);
        layer.set_causal(false);
        let input = inputs(4, 8);

        let (output, cache) = layer.forward_train(&input, Layout::default()).unwrap();
        let grad_output = Matrix::from_vec(output.rows, output.cols, vec![1.0; output.data.len()]);
        let grad_input = layer.backward(&cache, &grad_output).unwrap();

        let epsilon = 1e-3;
        for index in 0..input.data.len() {
            let mut bumped = input.clone();
            bumped.data[index] += epsilon;
            let high: f32 = layer
                .forward_train(&bumped, Layout::default())
                .unwrap()
                .0
                .data
                .iter()
                .sum();
            bumped.data[index] -= 2.0 * epsilon;
            let low: f32 = layer
                .forward_train(&bumped, Layout::default())
                .unwrap()
                .0
                .data
                .iter()
                .sum();
            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (grad_input.data[index] - numeric).abs() < 2e-2,
                "index {index}: {} vs {numeric}",
                grad_input.data[index]
            );
        }
    }

    #[test]
    fn a_cached_decode_reproduces_the_full_sequence_forward() {
        let layer = attention(4, 2);
        let sequence = inputs(6, 8);
        let (expected, _) = layer.forward_train(&sequence, Layout::default()).unwrap();

        let mut cache = KvCache::new(2, 4);
        for token in 0..sequence.rows {
            let step = Matrix::from_vec(1, 8, sequence.row(token).to_vec());
            let output = layer.forward_cached(&step, &mut cache).unwrap();

            assert_eq!(cache.len(), token + 1);
            for (actual, expected) in output.data.iter().zip(expected.row(token)) {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "token {token}: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn a_prefill_then_decode_reproduces_the_full_sequence_forward() {
        let layer = attention(2, 2);
        let sequence = inputs(5, 8);
        let (expected, _) = layer.forward_train(&sequence, Layout::default()).unwrap();

        let mut cache = KvCache::new(2, 4);
        let prompt = Matrix::from_vec(3, 8, sequence.data[..3 * 8].to_vec());
        layer.forward_cached(&prompt, &mut cache).unwrap();

        for token in 3..sequence.rows {
            let step = Matrix::from_vec(1, 8, sequence.row(token).to_vec());
            let output = layer.forward_cached(&step, &mut cache).unwrap();
            for (actual, expected) in output.data.iter().zip(expected.row(token)) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn backward_matches_finite_differences() {
        let mut layer = attention(2, 1);
        let input = inputs(4, 8);

        let (output, cache) = layer.forward_train(&input, Layout::default()).unwrap();
        let grad_output = Matrix::from_vec(output.rows, output.cols, vec![1.0; output.data.len()]);
        let grad_input = layer.backward(&cache, &grad_output).unwrap();

        let epsilon = 1e-3;
        for index in 0..input.data.len() {
            let mut bumped = input.clone();
            bumped.data[index] += epsilon;
            let high: f32 = layer
                .forward_train(&bumped, Layout::default())
                .unwrap()
                .0
                .data
                .iter()
                .sum();
            bumped.data[index] -= 2.0 * epsilon;
            let low: f32 = layer
                .forward_train(&bumped, Layout::default())
                .unwrap()
                .0
                .data
                .iter()
                .sum();
            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (grad_input.data[index] - numeric).abs() < 2e-2,
                "index {index}: {} vs {numeric}",
                grad_input.data[index]
            );
        }
    }

    #[test]
    fn query_weight_gradient_matches_finite_differences() {
        let mut layer = attention(2, 2);
        let input = inputs(3, 8);

        let (output, cache) = layer.forward_train(&input, Layout::default()).unwrap();
        let grad_output = Matrix::from_vec(output.rows, output.cols, vec![1.0; output.data.len()]);
        layer.backward(&cache, &grad_output).unwrap();
        let analytic = layer.query.weight.grad.data.clone();

        let epsilon = 1e-3;
        for (index, &expected) in analytic.iter().enumerate() {
            let mut probe = layer.clone();
            probe.query.weight.value.data[index] += epsilon;
            let high: f32 = probe
                .forward_train(&input, Layout::default())
                .unwrap()
                .0
                .data
                .iter()
                .sum();
            probe.query.weight.value.data[index] -= 2.0 * epsilon;
            let low: f32 = probe
                .forward_train(&input, Layout::default())
                .unwrap()
                .0
                .data
                .iter()
                .sum();
            let numeric = (high - low) / (2.0 * epsilon);
            assert!(
                (expected - numeric).abs() < 2e-2,
                "index {index}: {} vs {numeric}",
                expected
            );
        }
    }

    #[test]
    fn a_cache_rejects_the_wrong_width() {
        let mut cache = KvCache::new(2, 4);
        assert!(
            cache
                .append(&Matrix::new(1, 4), &Matrix::new(1, 4))
                .is_err()
        );
    }
}
