//! Low-rank adapters on the DiT's block projections, and the backward pass
//! that trains them.
//!
//! This is the host reference the device path is checked against, not a
//! training path: a DiT worth adapting is far too slow to train here. The base
//! weights and the conditioning are frozen, so only adapter gradients are kept,
//! but the gradient still runs back through every block, because an adapter
//! in the first block changes what all the later ones see.
//!
//! Adapters are placed as diffusers places them: `to_q`, `to_k` and `to_v`
//! are three adapters on three row ranges of the fused `qkv` projection, which
//! is how a PEFT file trained against diffusers' FLUX lands on this layout.

use crate::conv::Dense;
use crate::ffn::{gelu, gelu_derivative};
use crate::matrix::Matrix;
use crate::mmdit::{
    Conditioning, Dit, HeadNorm, Positions, attention, concatenate, concatenate_columns, gated_add,
    layer_norm, modulate, split,
};
use crate::network::NetworkError;
use crate::param::{Lora, Param};
use std::collections::BTreeMap;
use std::ops::Range;

/// One adapter and the output rows of the projection it adds to.
#[derive(Clone, Debug)]
pub struct Adapter {
    pub lora: Lora,
    pub rows: Range<usize>,
}

/// Adapters keyed by the projection they adapt:
/// `double.{block}.{image|text}.{qkv|projection|mlp_in|mlp_out}` and
/// `single.{block}.{fused_in|fused_out}`.
#[derive(Clone, Debug, Default)]
pub struct DitLora {
    pub adapters: BTreeMap<String, Vec<Adapter>>,
}

/// Where a diffusers FLUX module lands: the projection's key and its rows.
fn peft_target(module: &str, d_model: usize, mlp: usize) -> Option<(String, Range<usize>)> {
    let part = |index: usize| index * d_model..(index + 1) * d_model;
    if let Some(rest) = module.strip_prefix("single_transformer_blocks.") {
        let (block, name) = rest.split_once('.')?;
        let (projection, rows) = match name {
            "attn.to_q" => ("fused_in", part(0)),
            "attn.to_k" => ("fused_in", part(1)),
            "attn.to_v" => ("fused_in", part(2)),
            "proj_mlp" => ("fused_in", 3 * d_model..3 * d_model + mlp),
            "proj_out" => ("fused_out", 0..d_model),
            _ => return None,
        };
        return Some((format!("single.{block}.{projection}"), rows));
    }
    let (block, name) = module
        .strip_prefix("transformer_blocks.")?
        .split_once('.')?;
    let (stream, projection, rows) = match name {
        "attn.to_q" => ("image", "qkv", part(0)),
        "attn.to_k" => ("image", "qkv", part(1)),
        "attn.to_v" => ("image", "qkv", part(2)),
        "attn.to_out.0" => ("image", "projection", part(0)),
        "ff.net.0.proj" => ("image", "mlp_in", 0..mlp),
        "ff.net.2" => ("image", "mlp_out", part(0)),
        "attn.add_q_proj" => ("text", "qkv", part(0)),
        "attn.add_k_proj" => ("text", "qkv", part(1)),
        "attn.add_v_proj" => ("text", "qkv", part(2)),
        "attn.to_add_out" => ("text", "projection", part(0)),
        "ff_context.net.0.proj" => ("text", "mlp_in", 0..mlp),
        "ff_context.net.2" => ("text", "mlp_out", part(0)),
        _ => return None,
    };
    Some((format!("double.{block}.{stream}.{projection}"), rows))
}

impl DitLora {
    /// Reads a PEFT file trained against diffusers' FLUX (`{module}.lora_A`
    /// and `lora_B`), with `scale` the `alpha / rank` it was trained at, which
    /// a PEFT file does not record.
    pub fn read_peft(
        path: impl AsRef<std::path::Path>,
        dit: &Dit,
        scale: f32,
    ) -> Result<Self, NetworkError> {
        let config = dit.config();
        let mut file = crate::safetensors::SafeTensors::open(path)?;
        let names: Vec<String> = file.names().map(str::to_string).collect();
        let mut result = Self::default();
        for name in &names {
            let Some(module) = name.strip_suffix(".lora_A.weight") else {
                continue;
            };
            let module = module.strip_prefix("transformer.").unwrap_or(module);
            let (key, rows) =
                peft_target(module, config.d_model, config.mlp_dim()).ok_or_else(|| {
                    NetworkError::InvalidConfig(format!("{name} adapts no block projection"))
                })?;
            let down = file.matrix(name)?;
            let up = file.matrix(&name.replace(".lora_A.", ".lora_B."))?;
            if up.rows != rows.len() || up.cols != down.rows {
                return Err(NetworkError::InvalidConfig(format!(
                    "{name}: an up factor of {}x{} cannot feed {} rows from rank {}",
                    up.rows,
                    up.cols,
                    rows.len(),
                    down.rows
                )));
            }
            let lora = Lora {
                down: Param::new(down),
                up: Param::new(up),
                scale,
            };
            result
                .adapters
                .entry(key)
                .or_default()
                .push(Adapter { lora, rows });
        }
        Ok(result)
    }

    fn get(&self, key: &str) -> &[Adapter] {
        self.adapters.get(key).map_or(&[], Vec::as_slice)
    }

    /// The prediction with the adapters added, and what the backward pass
    /// needs of it.
    pub fn forward(
        &self,
        dit: &Dit,
        patches: &Matrix,
        timestep: f32,
        conditioning: &Conditioning,
    ) -> Result<(Matrix, Tape), NetworkError> {
        let config = dit.config();
        let (heads, eps) = (config.num_heads, config.eps);
        let vector = dit.vector(timestep, conditioning)?;
        let positions = Positions::new(
            config,
            conditioning.text.rows,
            conditioning.height,
            conditioning.width,
        );
        let mut image = dit.image_in.forward(patches)?;
        let mut text = dit.text_in.forward(&conditioning.text)?;
        let text_len = text.rows;

        let mut double = Vec::new();
        for (index, block) in dit.double.iter().enumerate() {
            let key = |stream: &str, name: &str| format!("double.{index}.{stream}.{name}");
            let image_mod = block.image.modulation.forward(&vector)?;
            let text_mod = block.text.modulation.forward(&vector)?;
            let image_normed =
                modulate(&layer_norm(&image, eps), &image_mod[0][0], &image_mod[0][1]);
            let text_normed = modulate(&layer_norm(&text, eps), &text_mod[0][0], &text_mod[0][1]);
            let image_qkv = project(
                &block.image.qkv,
                self.get(&key("image", "qkv")),
                &image_normed,
            )?;
            let text_qkv = project(&block.text.qkv, self.get(&key("text", "qkv")), &text_normed)?;
            let (text_raw, text_keys_raw, text_values) = thirds(&text_qkv);
            let (image_raw, image_keys_raw, image_values) = thirds(&image_qkv);
            let mut queries = concatenate(
                &normed(&block.text.query_norm, &text_raw, heads),
                &normed(&block.image.query_norm, &image_raw, heads),
            );
            let mut keys = concatenate(
                &normed(&block.text.key_norm, &text_keys_raw, heads),
                &normed(&block.image.key_norm, &image_keys_raw, heads),
            );
            let values = concatenate(&text_values, &image_values);
            positions.apply(&mut queries, heads);
            positions.apply(&mut keys, heads);
            let attended = attention(&queries, &keys, &values, heads, heads, false);
            let (text_attended, image_attended) = split(&attended, text_len);

            let input = (image.clone(), text.clone());
            let projected = project(
                &block.image.projection,
                self.get(&key("image", "projection")),
                &image_attended,
            )?;
            gated_add(&mut image, &projected, &image_mod[0][2]);
            let projected = project(
                &block.text.projection,
                self.get(&key("text", "projection")),
                &text_attended,
            )?;
            gated_add(&mut text, &projected, &text_mod[0][2]);

            let mid = (image.clone(), text.clone());
            let feed = |stream: &mut Matrix,
                        name: &str,
                        block: &crate::mmdit::Stream,
                        modulation: &[[Vec<f32>; 3]]|
             -> Result<Matrix, NetworkError> {
                let normed = modulate(
                    &layer_norm(stream, eps),
                    &modulation[1][0],
                    &modulation[1][1],
                );
                let pre = project(&block.mlp_in, self.get(&key(name, "mlp_in")), &normed)?;
                let activated = map(&pre, gelu);
                let output = project(&block.mlp_out, self.get(&key(name, "mlp_out")), &activated)?;
                gated_add(stream, &output, &modulation[1][2]);
                Ok(pre)
            };
            let image_pre = feed(&mut image, "image", &block.image, &image_mod)?;
            let text_pre = feed(&mut text, "text", &block.text, &text_mod)?;
            double.push(DoubleTape {
                input,
                normed: (image_normed, text_normed),
                raw: [text_raw, image_raw, text_keys_raw, image_keys_raw],
                attention: [queries, keys, values, attended],
                mid,
                pre: (image_pre, text_pre),
            });
        }

        let mut hidden = concatenate(&text, &image);
        let mlp_dim = config.mlp_dim();
        let width = config.d_model;
        let mut single = Vec::new();
        for (index, block) in dit.single.iter().enumerate() {
            let modulation = block.modulation.forward(&vector)?;
            let normed = modulate(
                &layer_norm(&hidden, eps),
                &modulation[0][0],
                &modulation[0][1],
            );
            let fused = project(
                &block.fused_in,
                self.get(&format!("single.{index}.fused_in")),
                &normed,
            )?;
            let raw_queries = columns(&fused, 0..width);
            let raw_keys = columns(&fused, width..2 * width);
            let values = columns(&fused, 2 * width..3 * width);
            let pre = columns(&fused, 3 * width..3 * width + mlp_dim);
            let mut queries = normed_by(&block.query_norm, &raw_queries, heads);
            let mut keys = normed_by(&block.key_norm, &raw_keys, heads);
            positions.apply(&mut queries, heads);
            positions.apply(&mut keys, heads);
            let attended = attention(&queries, &keys, &values, heads, heads, false);
            let output = project(
                &block.fused_out,
                self.get(&format!("single.{index}.fused_out")),
                &concatenate_columns(&attended, &map(&pre, gelu)),
            )?;
            let input = hidden.clone();
            gated_add(&mut hidden, &output, &modulation[0][2]);
            single.push(SingleTape {
                input,
                normed,
                raw: [raw_queries, raw_keys],
                attention: [queries, keys, values, attended],
                pre,
            });
        }

        let (_, image) = split(&hidden, text_len);
        let (shift, scale) = final_modulation(dit, &vector)?;
        let normed = modulate(&layer_norm(&image, eps), &shift, &scale);
        let output = dit.final_out.forward(&normed)?;
        let tape = Tape {
            vector,
            positions,
            text_len,
            double,
            single,
            image,
            normed,
        };
        Ok((output, tape))
    }

    /// Adds `dL/d(adapter)` for every adapter to its gradient, given
    /// `dL/d(prediction)`.
    pub fn backward(
        &mut self,
        dit: &Dit,
        tape: &Tape,
        grad_output: &Matrix,
    ) -> Result<(), NetworkError> {
        let config = dit.config();
        let (heads, eps) = (config.num_heads, config.eps);
        let inverse = tape.positions.inverse();
        let width = config.d_model;

        let (_, scale) = final_modulation(dit, &tape.vector)?;
        let grad_normed = project_backward(&dit.final_out, &mut [], &tape.normed, grad_output);
        let grad_image = layer_norm_backward(&tape.image, eps, &scaled(&grad_normed, &scale));
        let mut grad = concatenate(&Matrix::new(tape.text_len, width), &grad_image);

        for (index, (block, step)) in dit.single.iter().zip(&tape.single).enumerate().rev() {
            let modulation = block.modulation.forward(&tape.vector)?;
            let [queries, keys, values, attended] = &step.attention;
            let activated = map(&step.pre, gelu);
            let grad_fused_out = project_backward(
                &block.fused_out,
                self.adapters_mut(&format!("single.{index}.fused_out")),
                &concatenate_columns(attended, &activated),
                &gated(&grad, &modulation[0][2]),
            );
            let grad_attended = columns(&grad_fused_out, 0..width);
            let grad_pre = derivative(
                &columns(&grad_fused_out, width..grad_fused_out.cols),
                &step.pre,
            );
            let (mut grad_queries, mut grad_keys, grad_values) =
                attention_backward(queries, keys, values, heads, &grad_attended);
            inverse.apply(&mut grad_queries, heads);
            inverse.apply(&mut grad_keys, heads);
            let grad_queries =
                head_norm_backward(&block.query_norm, &step.raw[0], heads, &grad_queries);
            let grad_keys = head_norm_backward(&block.key_norm, &step.raw[1], heads, &grad_keys);
            let grad_fused = concatenate_columns(
                &concatenate_columns(
                    &concatenate_columns(&grad_queries, &grad_keys),
                    &grad_values,
                ),
                &grad_pre,
            );
            let grad_normed = project_backward(
                &block.fused_in,
                self.adapters_mut(&format!("single.{index}.fused_in")),
                &step.normed,
                &grad_fused,
            );
            add(
                &mut grad,
                &layer_norm_backward(&step.input, eps, &scaled(&grad_normed, &modulation[0][1])),
            );
        }

        let (mut grad_text, mut grad_image) = split(&grad, tape.text_len);
        for (index, (block, step)) in dit.double.iter().zip(&tape.double).enumerate().rev() {
            let key = |stream: &str, name: &str| format!("double.{index}.{stream}.{name}");
            let image_mod = block.image.modulation.forward(&tape.vector)?;
            let text_mod = block.text.modulation.forward(&tape.vector)?;

            // The feed-forward half, back to the stream between the halves.
            let mut feed_back = |grad: &mut Matrix,
                                 mid: &Matrix,
                                 pre: &Matrix,
                                 name: &str,
                                 stream: &crate::mmdit::Stream,
                                 modulation: &[[Vec<f32>; 3]]| {
                let grad_activated = project_backward(
                    &stream.mlp_out,
                    self.adapters_mut(&key(name, "mlp_out")),
                    &map(pre, gelu),
                    &gated(grad, &modulation[1][2]),
                );
                let normed = modulate(&layer_norm(mid, eps), &modulation[1][0], &modulation[1][1]);
                let grad_normed = project_backward(
                    &stream.mlp_in,
                    self.adapters_mut(&key(name, "mlp_in")),
                    &normed,
                    &derivative(&grad_activated, pre),
                );
                add(
                    grad,
                    &layer_norm_backward(mid, eps, &scaled(&grad_normed, &modulation[1][1])),
                );
            };
            feed_back(
                &mut grad_image,
                &step.mid.0,
                &step.pre.0,
                "image",
                &block.image,
                &image_mod,
            );
            feed_back(
                &mut grad_text,
                &step.mid.1,
                &step.pre.1,
                "text",
                &block.text,
                &text_mod,
            );

            // The attention half, back to the block's input.
            let [queries, keys, values, attended] = &step.attention;
            let (text_attended, image_attended) = split(attended, tape.text_len);
            let grad_image_attended = project_backward(
                &block.image.projection,
                self.adapters_mut(&key("image", "projection")),
                &image_attended,
                &gated(&grad_image, &image_mod[0][2]),
            );
            let grad_text_attended = project_backward(
                &block.text.projection,
                self.adapters_mut(&key("text", "projection")),
                &text_attended,
                &gated(&grad_text, &text_mod[0][2]),
            );
            let (mut grad_queries, mut grad_keys, grad_values) = attention_backward(
                queries,
                keys,
                values,
                heads,
                &concatenate(&grad_text_attended, &grad_image_attended),
            );
            inverse.apply(&mut grad_queries, heads);
            inverse.apply(&mut grad_keys, heads);
            let (text_queries, image_queries) = split(&grad_queries, tape.text_len);
            let (text_keys, image_keys) = split(&grad_keys, tape.text_len);
            let (text_values, image_values) = split(&grad_values, tape.text_len);
            let [text_raw, image_raw, text_keys_raw, image_keys_raw] = &step.raw;

            let mut attention_back = |grad: &mut Matrix,
                                      input: &Matrix,
                                      normed: &Matrix,
                                      name: &str,
                                      stream: &crate::mmdit::Stream,
                                      modulation: &[[Vec<f32>; 3]],
                                      parts: [Matrix; 3]| {
                let grad_qkv =
                    concatenate_columns(&concatenate_columns(&parts[0], &parts[1]), &parts[2]);
                let grad_normed = project_backward(
                    &stream.qkv,
                    self.adapters_mut(&key(name, "qkv")),
                    normed,
                    &grad_qkv,
                );
                add(
                    grad,
                    &layer_norm_backward(input, eps, &scaled(&grad_normed, &modulation[0][1])),
                );
            };
            let image_parts = [
                head_norm_backward(&block.image.query_norm, image_raw, heads, &image_queries),
                head_norm_backward(&block.image.key_norm, image_keys_raw, heads, &image_keys),
                image_values,
            ];
            let text_parts = [
                head_norm_backward(&block.text.query_norm, text_raw, heads, &text_queries),
                head_norm_backward(&block.text.key_norm, text_keys_raw, heads, &text_keys),
                text_values,
            ];
            attention_back(
                &mut grad_image,
                &step.input.0,
                &step.normed.0,
                "image",
                &block.image,
                &image_mod,
                image_parts,
            );
            attention_back(
                &mut grad_text,
                &step.input.1,
                &step.normed.1,
                "text",
                &block.text,
                &text_mod,
                text_parts,
            );
        }
        Ok(())
    }

    fn adapters_mut(&mut self, key: &str) -> &mut [Adapter] {
        self.adapters
            .get_mut(key)
            .map_or(&mut [], Vec::as_mut_slice)
    }
}

/// What one forward pass keeps for the backward one.
pub struct Tape {
    vector: Vec<f32>,
    positions: Positions,
    text_len: usize,
    double: Vec<DoubleTape>,
    single: Vec<SingleTape>,
    /// The image half of the stream after the last block, and it normed and
    /// modulated for the output projection.
    image: Matrix,
    normed: Matrix,
}

/// A double block's activations; pairs are `(image, text)`.
struct DoubleTape {
    input: (Matrix, Matrix),
    normed: (Matrix, Matrix),
    /// Queries and keys before their norms: text then image queries, then
    /// text then image keys.
    raw: [Matrix; 4],
    /// Queries, keys and values as attention read them, and its output.
    attention: [Matrix; 4],
    mid: (Matrix, Matrix),
    pre: (Matrix, Matrix),
}

struct SingleTape {
    input: Matrix,
    normed: Matrix,
    raw: [Matrix; 2],
    attention: [Matrix; 4],
    pre: Matrix,
}

/// The base projection plus every adapter on it.
fn project(dense: &Dense, adapters: &[Adapter], input: &Matrix) -> Result<Matrix, NetworkError> {
    let mut output = dense.forward(input)?;
    for adapter in adapters {
        let hidden = bottleneck(&adapter.lora, input);
        let mut delta = Matrix::new(input.rows, adapter.rows.len());
        hidden.dot_rhs_transposed(&adapter.lora.up.value, &mut delta);
        for (row, delta) in output
            .data
            .chunks_mut(output.cols)
            .zip(delta.data.chunks(delta.cols))
        {
            for (slot, value) in row[adapter.rows.clone()].iter_mut().zip(delta) {
                *slot += value;
            }
        }
    }
    Ok(output)
}

/// `scale * input * down^T`.
fn bottleneck(lora: &Lora, input: &Matrix) -> Matrix {
    let mut hidden = Matrix::new(input.rows, lora.down.value.rows);
    input.dot_rhs_transposed(&lora.down.value, &mut hidden);
    hidden
        .data
        .iter_mut()
        .for_each(|value| *value *= lora.scale);
    hidden
}

/// `dL/d(input)` of [`project`], adding each adapter's gradients to it.
fn project_backward(
    dense: &Dense,
    adapters: &mut [Adapter],
    input: &Matrix,
    grad_output: &Matrix,
) -> Matrix {
    let mut grad_input = Matrix::new(input.rows, input.cols);
    grad_output.dot(&dense.weight_f32(), &mut grad_input);
    for adapter in adapters {
        let lora = &mut adapter.lora;
        let grad = columns(grad_output, adapter.rows.clone());
        grad.dot_self_transposed_accumulate(&bottleneck(lora, input), &mut lora.up.grad);
        let mut grad_hidden = Matrix::new(input.rows, lora.down.value.rows);
        grad.dot(&lora.up.value, &mut grad_hidden);
        grad_hidden
            .data
            .iter_mut()
            .for_each(|value| *value *= lora.scale);
        grad_hidden.dot_self_transposed_accumulate(input, &mut lora.down.grad);
        grad_hidden.dot_accumulate(&lora.down.value, &mut grad_input);
    }
    grad_input
}

/// `dL/dx` of the affine-free [`layer_norm`].
fn layer_norm_backward(input: &Matrix, eps: f32, grad: &Matrix) -> Matrix {
    let mut output = Matrix::new(input.rows, input.cols);
    let width = input.cols as f32;
    for ((row, grad), target) in input
        .data
        .chunks(input.cols)
        .zip(grad.data.chunks(input.cols))
        .zip(output.data.chunks_mut(input.cols))
    {
        let mean = row.iter().sum::<f32>() / width;
        let variance = row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / width;
        let inverse = (variance + eps).sqrt().recip();
        let grad_mean = grad.iter().sum::<f32>() / width;
        let projection = row
            .iter()
            .zip(grad)
            .map(|(value, grad)| (value - mean) * inverse * grad)
            .sum::<f32>()
            / width;
        for ((target, value), grad) in target.iter_mut().zip(row).zip(grad) {
            *target = inverse * (grad - grad_mean - (value - mean) * inverse * projection);
        }
    }
    output
}

/// `dL/dx` of [`HeadNorm::apply`], given the values it was applied to.
fn head_norm_backward(norm: &HeadNorm, input: &Matrix, heads: usize, grad: &Matrix) -> Matrix {
    let head_dim = norm.scale.len();
    let mut output = Matrix::new(input.rows, input.cols);
    for ((row, grad), target) in input
        .data
        .chunks(heads * head_dim)
        .zip(grad.data.chunks(heads * head_dim))
        .zip(output.data.chunks_mut(heads * head_dim))
    {
        for head in 0..heads {
            let span = head * head_dim..(head + 1) * head_dim;
            let (row, grad, target) = (&row[span.clone()], &grad[span.clone()], &mut target[span]);
            let mean_square = row.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
            let inverse = (mean_square + 1e-6).sqrt().recip();
            let dot = row
                .iter()
                .zip(grad)
                .zip(&norm.scale)
                .map(|((value, grad), scale)| value * grad * scale)
                .sum::<f32>();
            for (((target, value), grad), scale) in
                target.iter_mut().zip(row).zip(grad).zip(&norm.scale)
            {
                *target = inverse * grad * scale - value * inverse.powi(3) * dot / head_dim as f32;
            }
        }
    }
    output
}

/// `dL/dq`, `dL/dk` and `dL/dv` of non-causal [`attention`] with as many key
/// heads as query heads.
fn attention_backward(
    queries: &Matrix,
    keys: &Matrix,
    values: &Matrix,
    heads: usize,
    grad: &Matrix,
) -> (Matrix, Matrix, Matrix) {
    let tokens = queries.rows;
    let head_dim = queries.cols / heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut grads = [
        Matrix::new(tokens, queries.cols),
        Matrix::new(keys.rows, keys.cols),
        Matrix::new(values.rows, values.cols),
    ];
    let mut weights = vec![0.0; keys.rows];
    let mut grad_weights = vec![0.0; keys.rows];
    for head in 0..heads {
        let span = head * head_dim..(head + 1) * head_dim;
        for query in 0..tokens {
            let q = &queries.row(query)[span.clone()];
            let g = &grad.row(query)[span.clone()];
            for (key, weight) in weights.iter_mut().enumerate() {
                *weight = dot(q, &keys.row(key)[span.clone()]) * scale;
            }
            let largest = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            weights
                .iter_mut()
                .for_each(|weight| *weight = (*weight - largest).exp());
            let total: f32 = weights.iter().sum();
            weights.iter_mut().for_each(|weight| *weight /= total);
            for (key, grad_weight) in grad_weights.iter_mut().enumerate() {
                *grad_weight = dot(g, &values.row(key)[span.clone()]);
            }
            let expected: f32 = weights.iter().zip(&grad_weights).map(|(w, g)| w * g).sum();
            for key in 0..keys.rows {
                let grad_score = weights[key] * (grad_weights[key] - expected) * scale;
                for index in 0..head_dim {
                    let column = span.start + index;
                    grads[0].row_mut(query)[column] += grad_score * keys.row(key)[column];
                    grads[1].row_mut(key)[column] += grad_score * q[index];
                    grads[2].row_mut(key)[column] += weights[key] * g[index];
                }
            }
        }
    }
    let [grad_queries, grad_keys, grad_values] = grads;
    (grad_queries, grad_keys, grad_values)
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

/// The final layer's shift and scale.
fn final_modulation(dit: &Dit, vector: &[f32]) -> Result<(Vec<f32>, Vec<f32>), NetworkError> {
    let activated: Vec<f32> = vector
        .iter()
        .map(|value| crate::ffn::silu(*value))
        .collect();
    let mut shift = dit.final_modulation.apply(&activated)?;
    let scale = shift.split_off(shift.len() / 2);
    Ok((shift, scale))
}

fn thirds(qkv: &Matrix) -> (Matrix, Matrix, Matrix) {
    let width = qkv.cols / 3;
    (
        columns(qkv, 0..width),
        columns(qkv, width..2 * width),
        columns(qkv, 2 * width..3 * width),
    )
}

fn columns(matrix: &Matrix, range: Range<usize>) -> Matrix {
    let mut output = Matrix::new(matrix.rows, range.len());
    for row in 0..matrix.rows {
        output
            .row_mut(row)
            .copy_from_slice(&matrix.row(row)[range.clone()]);
    }
    output
}

fn normed(norm: &HeadNorm, values: &Matrix, heads: usize) -> Matrix {
    normed_by(norm, values, heads)
}

fn normed_by(norm: &HeadNorm, values: &Matrix, heads: usize) -> Matrix {
    let mut output = values.clone();
    norm.apply(&mut output, heads);
    output
}

fn map(matrix: &Matrix, function: fn(f32) -> f32) -> Matrix {
    Matrix::from_vec(
        matrix.rows,
        matrix.cols,
        matrix.data.iter().map(|value| function(*value)).collect(),
    )
}

/// `grad * gelu'(pre)`.
fn derivative(grad: &Matrix, pre: &Matrix) -> Matrix {
    let data = grad
        .data
        .iter()
        .zip(&pre.data)
        .map(|(grad, pre)| grad * gelu_derivative(*pre))
        .collect();
    Matrix::from_vec(grad.rows, grad.cols, data)
}

/// Every row times `gate`, column by column.
fn gated(grad: &Matrix, gate: &[f32]) -> Matrix {
    let mut output = grad.clone();
    output.data.chunks_mut(grad.cols).for_each(|row| {
        row.iter_mut()
            .zip(gate)
            .for_each(|(value, gate)| *value *= gate)
    });
    output
}

/// Every row times `1 + scale`, which is what [`modulate`] does to a gradient.
fn scaled(grad: &Matrix, scale: &[f32]) -> Matrix {
    let one_plus: Vec<f32> = scale.iter().map(|scale| 1.0 + scale).collect();
    gated(grad, &one_plus)
}

fn add(target: &mut Matrix, source: &Matrix) {
    target
        .data
        .iter_mut()
        .zip(&source.data)
        .for_each(|(target, source)| *target += source);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs `RB_DIT_LORA_REF` from `scripts/dit_lora_reference.py`.
    #[test]
    #[ignore]
    fn dit_lora_gradients_match_pytorch() {
        let dir = std::path::PathBuf::from(std::env::var("RB_DIT_LORA_REF").unwrap());
        let reference: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
                .unwrap();
        let floats = |value: &serde_json::Value| -> Vec<f32> {
            let values = value.as_array().unwrap().iter();
            values.map(|x| x.as_f64().unwrap() as f32).collect()
        };
        let number = |key: &str| reference[key].as_f64().unwrap();
        let config =
            crate::mmdit::DitConfig::from_file(dir.join("transformer/config.json")).unwrap();
        let dit = Dit::load(
            dir.join("transformer/diffusion_pytorch_model.safetensors"),
            "",
            config.clone(),
        )
        .unwrap();
        let mut lora = DitLora::read_peft(
            dir.join("adapters.safetensors"),
            &dit,
            number("scale") as f32,
        )
        .unwrap();
        let (height, width) = (number("height") as usize, number("width") as usize);
        let text = floats(&reference["text"]);
        let conditioning = Conditioning {
            text: Matrix::from_vec(text.len() / config.text_dim, config.text_dim, text),
            pooled: floats(&reference["pooled"]),
            height,
            width,
            guidance: 0.0,
        };
        let patches = Matrix::from_vec(
            height * width,
            config.in_channels,
            floats(&reference["patches"]),
        );

        let (output, tape) = lora
            .forward(&dit, &patches, number("timestep") as f32, &conditioning)
            .unwrap();
        let worst = |got: &[f32], want: &[f32]| {
            assert_eq!(got.len(), want.len());
            let largest = want.iter().fold(0.0f32, |max, value| max.max(value.abs()));
            let error = got
                .iter()
                .zip(want)
                .fold(0.0f32, |max, (got, want)| max.max((got - want).abs()));
            error / largest
        };
        let error = worst(&output.data, &floats(&reference["output"]));
        assert!(error < 1e-4, "the prediction is off by {error} relative");

        let grad = Matrix::from_vec(output.rows, output.cols, floats(&reference["weights"]));
        lora.backward(&dit, &tape, &grad).unwrap();
        let gradients = reference["gradients"].as_object().unwrap();
        let mut checked = 0;
        for (name, want) in gradients {
            let module = name.split(".lora_").next().unwrap();
            let (key, rows) = peft_target(module, config.d_model, config.mlp_dim()).unwrap();
            let adapter = lora.adapters[&key]
                .iter()
                .find(|adapter| adapter.rows == rows)
                .unwrap();
            let got = match name.contains(".lora_A.") {
                true => &adapter.lora.down.grad.data,
                false => &adapter.lora.up.grad.data,
            };
            let error = worst(got, &floats(want));
            assert!(error < 1e-4, "{name} is off by {error} relative");
            checked += 1;
        }
        assert_eq!(checked, 68);
    }
}
