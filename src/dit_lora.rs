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

/// The diffusers FLUX modules [`peft_target`] places, under a block's prefix.
const DOUBLE_MODULES: [&str; 12] = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff_context.net.0.proj",
    "ff_context.net.2",
];
const SINGLE_MODULES: [&str; 5] = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "proj_mlp",
    "proj_out",
];

/// Every diffusers module of the model's blocks.
fn peft_modules(dit: &Dit) -> Vec<String> {
    let config = dit.config();
    let double = (0..config.double_blocks).flat_map(|block| {
        DOUBLE_MODULES
            .iter()
            .map(move |name| format!("transformer_blocks.{block}.{name}"))
    });
    let single = (0..config.single_blocks).flat_map(|block| {
        SINGLE_MODULES
            .iter()
            .map(move |name| format!("single_transformer_blocks.{block}.{name}"))
    });
    double.chain(single).collect()
}

/// The projection an adapter key names, if the model has that block.
pub(crate) fn projection<'a>(dit: &'a Dit, key: &str) -> Option<&'a Dense> {
    let mut parts = key.split('.');
    let (kind, block) = (parts.next()?, parts.next()?.parse::<usize>().ok()?);
    if kind == "single" {
        let block = dit.single.get(block)?;
        return match parts.next()? {
            "fused_in" => Some(&block.fused_in),
            "fused_out" => Some(&block.fused_out),
            _ => None,
        };
    }
    let block = dit.double.get(block)?;
    let stream = match parts.next()? {
        "image" => &block.image,
        _ => &block.text,
    };
    match parts.next()? {
        "qkv" => Some(&stream.qkv),
        "projection" => Some(&stream.projection),
        "mlp_in" => Some(&stream.mlp_in),
        _ => Some(&stream.mlp_out),
    }
}

impl DitLora {
    /// Fresh adapters of `rank` on every module a diffusers FLUX LoRA covers,
    /// `up` at zero so the model starts unchanged.
    pub fn new(dit: &Dit, rank: usize, alpha: f32, rng: &mut rand::rngs::StdRng) -> Self {
        let config = dit.config();
        let mut result = Self::default();
        for module in peft_modules(dit) {
            let (key, rows) = peft_target(&module, config.d_model, config.mlp_dim())
                .expect("every listed module has a target");
            let inner = projection(dit, &key)
                .expect("the module's block exists")
                .in_dim();
            let lora = Lora::new(inner, rows.len(), rank, alpha, rng);
            result
                .adapters
                .entry(key)
                .or_default()
                .push(Adapter { lora, rows });
        }
        result
    }

    /// Writes the adapters as a diffusers FLUX PEFT file, with each adapter's
    /// scale folded into `lora_B`, so [`read_peft`](Self::read_peft) at scale
    /// 1 and diffusers' `load_lora_weights` (alpha defaults to the rank) give
    /// back the same model. Each adapter must cover one module's rows, with
    /// its factors on the host.
    pub fn write_peft(
        &self,
        path: impl AsRef<std::path::Path>,
        dit: &Dit,
    ) -> Result<(), NetworkError> {
        self.check(dit)?;
        let config = dit.config();
        let targets: Vec<_> = peft_modules(dit)
            .into_iter()
            .filter_map(|module| {
                let target = peft_target(&module, config.d_model, config.mlp_dim())?;
                Some((target, module))
            })
            .collect();
        let mut tensors = Vec::new();
        for (key, list) in &self.adapters {
            for adapter in list {
                let module = targets
                    .iter()
                    .find(|((k, rows), _)| k == key && *rows == adapter.rows)
                    .map(|(_, module)| module)
                    .ok_or_else(|| {
                        NetworkError::InvalidConfig(format!(
                            "rows {:?} of {key} are no one diffusers module",
                            adapter.rows
                        ))
                    })?;
                let (down, up) = (&adapter.lora.down.value, &adapter.lora.up.value);
                let scaled: Vec<f32> = up.data.iter().map(|v| v * adapter.lora.scale).collect();
                tensors.push((
                    format!("transformer.{module}.lora_A.weight"),
                    vec![down.rows, down.cols],
                    down.data.clone(),
                ));
                tensors.push((
                    format!("transformer.{module}.lora_B.weight"),
                    vec![up.rows, up.cols],
                    scaled,
                ));
            }
        }
        let views: Vec<_> = tensors
            .iter()
            .map(|(name, shape, data)| (name.as_str(), &shape[..], &data[..]))
            .collect();
        crate::safetensors::write(path, &views, &BTreeMap::new())
    }

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
        result.check(dit)?;
        Ok(result)
    }

    /// Checks that every adapter fits the projection its key names. The
    /// device path writes where an adapter's rows say, so it refuses one
    /// that does not.
    pub(crate) fn check(&self, dit: &Dit) -> Result<(), NetworkError> {
        for (key, list) in &self.adapters {
            let dense = projection(dit, key).ok_or_else(|| {
                NetworkError::InvalidConfig(format!("{key} names a block the model lacks"))
            })?;
            for adapter in list {
                let (down, up) = (&adapter.lora.down.value, &adapter.lora.up.value);
                if down.cols != dense.in_dim()
                    || adapter.rows.is_empty()
                    || adapter.rows.end > dense.out_dim()
                    || up.rows != adapter.rows.len()
                    || up.cols != down.rows
                {
                    return Err(NetworkError::InvalidConfig(format!(
                        "an adapter on {key} does not fit its {}x{} projection",
                        dense.out_dim(),
                        dense.in_dim()
                    )));
                }
            }
        }
        Ok(())
    }

    /// Puts every adapter on the device, for
    /// [`DeviceDit::forward_adapted`](crate::cuda_mmdit::DeviceDit::forward_adapted)
    /// and its backward pass.
    #[cfg(feature = "cuda")]
    pub fn move_to_cuda(
        &mut self,
        context: &std::sync::Arc<crate::gpu_transformer::GpuContext>,
    ) -> Result<(), NetworkError> {
        for adapter in self.adapters.values_mut().flatten() {
            adapter.lora.down.move_to_cuda(context)?;
            adapter.lora.up.move_to_cuda(context)?;
        }
        Ok(())
    }

    /// Brings every adapter back, values and gradients.
    #[cfg(feature = "cuda")]
    pub fn move_to_cpu(&mut self) -> Result<(), NetworkError> {
        for adapter in self.adapters.values_mut().flatten() {
            adapter.lora.down.move_to_cpu()?;
            adapter.lora.up.move_to_cpu()?;
        }
        Ok(())
    }

    /// Every adapter factor, for an optimizer.
    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        self.adapters
            .values_mut()
            .flatten()
            .flat_map(|adapter| adapter.lora.params_mut())
            .collect()
    }

    pub(crate) fn get(&self, key: &str) -> &[Adapter] {
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

    pub(crate) fn adapters_mut(&mut self, key: &str) -> &mut [Adapter] {
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
pub(crate) fn layer_norm_backward(input: &Matrix, eps: f32, grad: &Matrix) -> Matrix {
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
pub(crate) fn head_norm_backward(
    norm: &HeadNorm,
    input: &Matrix,
    heads: usize,
    grad: &Matrix,
) -> Matrix {
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
pub(crate) fn attention_backward(
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

pub(crate) fn columns(matrix: &Matrix, range: Range<usize>) -> Matrix {
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

    #[test]
    fn a_peft_file_must_fit_the_projections_it_names() {
        let config = crate::mmdit::tests::tiny();
        let dir = std::env::temp_dir();
        let path = dir.join("rusting_brain_dit_lora_model.safetensors");
        crate::mmdit::tests::checkpoint(&config, &path);
        let dit = Dit::load(&path, "", config.clone()).unwrap();
        std::fs::remove_file(&path).ok();
        let (d, rank) = (config.d_model, 2);
        let read = |module: &str, inner: usize| {
            let path = dir.join("rusting_brain_dit_lora_peft.safetensors");
            let (down, up) = (vec![0.1; rank * inner], vec![0.1; d * rank]);
            let names = [
                format!("transformer.{module}.lora_A.weight"),
                format!("transformer.{module}.lora_B.weight"),
            ];
            crate::safetensors::write(
                &path,
                &[
                    (&names[0], &[rank, inner][..], &down[..]),
                    (&names[1], &[d, rank][..], &up[..]),
                ],
                &Default::default(),
            )
            .unwrap();
            let lora = DitLora::read_peft(&path, &dit, 1.0);
            std::fs::remove_file(&path).ok();
            lora
        };
        let lora = read("transformer_blocks.1.attn.to_k", d).unwrap();
        assert_eq!(lora.adapters["double.1.image.qkv"][0].rows, d..2 * d);
        // The feed-forward's way out reads the wide hidden layer, not the stream.
        assert!(read("transformer_blocks.0.ff.net.2", d).is_err());
        assert!(read("transformer_blocks.0.ff.net.2", config.mlp_dim()).is_ok());
        assert!(read("transformer_blocks.2.attn.to_q", d).is_err());
        assert!(read("single_transformer_blocks.0.norm.linear", d).is_err());
    }

    #[test]
    fn written_adapters_read_back_to_the_same_prediction() {
        use rand::SeedableRng;
        let config = crate::mmdit::tests::tiny();
        let dir = std::env::temp_dir();
        let path = dir.join("rusting_brain_dit_lora_write_model.safetensors");
        crate::mmdit::tests::checkpoint(&config, &path);
        let dit = Dit::load(&path, "", config.clone()).unwrap();
        std::fs::remove_file(&path).ok();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut lora = DitLora::new(&dit, 2, 4.0, &mut rng);
        let modules = config.double_blocks * 12 + config.single_blocks * 5;
        assert_eq!(lora.adapters.values().map(Vec::len).sum::<usize>(), modules);
        for (index, adapter) in lora.adapters.values_mut().flatten().enumerate() {
            let up = &mut adapter.lora.up.value.data;
            up.iter_mut()
                .enumerate()
                .for_each(|(i, v)| *v = ((i + index) as f32 * 0.7).sin() * 0.05);
        }
        let path = dir.join("rusting_brain_dit_lora_written.safetensors");
        lora.write_peft(&path, &dit).unwrap();
        let read = DitLora::read_peft(&path, &dit, 1.0).unwrap();
        std::fs::remove_file(&path).ok();

        let (height, width) = (2, 3);
        let patches = Matrix::from_vec(
            height * width,
            config.in_channels,
            (0..height * width * config.in_channels)
                .map(|i| (i as f32 * 0.3).cos())
                .collect(),
        );
        let conditioning = crate::mmdit::tests::conditioning(&config, height, width);
        let plain = dit.forward_with(&patches, 0.5, &conditioning).unwrap();
        let (expected, _) = lora.forward(&dit, &patches, 0.5, &conditioning).unwrap();
        let (actual, _) = read.forward(&dit, &patches, 0.5, &conditioning).unwrap();
        let distance = |a: &Matrix, b: &Matrix| {
            let pairs = a.data.iter().zip(&b.data);
            pairs.map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        };
        assert!(
            distance(&expected, &plain) > 1e-3,
            "the adapters change nothing"
        );
        assert!(distance(&expected, &actual) < 1e-5);

        // An adapter over two modules' rows has no one name to be written under.
        lora.adapters.get_mut("double.0.image.qkv").unwrap()[0].rows = 0..2 * config.d_model;
        let up = &mut lora.adapters.get_mut("double.0.image.qkv").unwrap()[0]
            .lora
            .up;
        *up = Param::new(Matrix::new(2 * config.d_model, 2));
        assert!(lora.write_peft(&path, &dit).is_err());
    }

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

    /// Needs `RB_DIT_LORA_TRAINED` from `scripts/dit_lora_trained_check.py
    /// check`, run on what `train_dit_lora` wrote.
    #[test]
    #[ignore]
    fn a_trained_lora_loads_as_diffusers_loads_it() {
        let dir = std::path::PathBuf::from(std::env::var("RB_DIT_LORA_TRAINED").unwrap());
        let reference: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
                .unwrap();
        let floats = |value: &serde_json::Value| -> Vec<f32> {
            let values = value.as_array().unwrap().iter();
            values.map(|x| x.as_f64().unwrap() as f32).collect()
        };
        let number = |key: &str| reference[key].as_f64().unwrap();
        let model = std::path::PathBuf::from(reference["model"].as_str().unwrap());
        let file = std::path::PathBuf::from(reference["lora"].as_str().unwrap());
        let config =
            crate::mmdit::DitConfig::from_file(model.join("transformer/config.json")).unwrap();
        let dit = Dit::load(
            model.join("transformer/diffusion_pytorch_model.safetensors"),
            "",
            config.clone(),
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
        let timestep = number("timestep") as f32;
        let worst = |got: &[f32], want: &[f32]| {
            let largest = want.iter().fold(0.0f32, |max, value| max.max(value.abs()));
            let error = got
                .iter()
                .zip(want)
                .fold(0.0f32, |max, (got, want)| max.max((got - want).abs()));
            error / largest
        };
        let (plain, adapted) = (floats(&reference["plain"]), floats(&reference["adapted"]));
        // Training has to have moved the model, or matching says nothing.
        let moved = worst(&adapted, &plain);
        assert!(
            moved > 1e-3,
            "the LoRA moves diffusers' prediction by {moved}"
        );
        assert_eq!(number("adapters") as usize, peft_modules(&dit).len());

        let lora = DitLora::read_peft(&file, &dit, 1.0).unwrap();
        let (ours, _) = lora
            .forward(&dit, &patches, timestep, &conditioning)
            .unwrap();
        let error = worst(&ours.data, &adapted);
        assert!(
            error < 1e-4,
            "read_peft is off diffusers by {error}, moved {moved}"
        );

        let pipeline = crate::pipeline::ImagePipeline::load_with_loras(
            &model,
            crate::transformer::Precision::F32,
            &[(file, 1.0)],
        )
        .unwrap();
        let crate::pipeline::ImageDenoiser::Transformer(merged) = &pipeline.denoiser else {
            panic!("the model is a DiT");
        };
        let merged = merged
            .forward_with(&patches, timestep, &conditioning)
            .unwrap();
        let error = worst(&merged.data, &adapted);
        assert!(
            error < 1e-4,
            "load_with_loras is off diffusers by {error}, moved {moved}"
        );
        println!("moved {moved}, merged off diffusers by {error}");
    }
}
