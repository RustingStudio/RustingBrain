//! The LLaMA-family prompt encoder on the device, the twin of
//! [`TextEncoder::forward_layers`]. FLUX.2 reads its prompt through Qwen3-4B
//! this way.
//!
//! Like [`crate::cuda_t5`], it runs on the BF16 kernels the language model
//! trains with: the residual stream and every GEMM accumulator are FP32, and
//! only the weights and the operands a GEMM reads are BF16 (int8 where the
//! host holds them at one byte). Attention is the fused BF16 kernel, causal,
//! with the padding's keys cut off by its key count, which is the mask a
//! padded prompt is encoded with: every position, padding included, sees the
//! real tokens up to its own.
//!
//! The embedding table stays on the host. It is read by lookup, and for
//! Qwen3-4B it is 1.5 GB that no GEMM reads.

use crate::cuda_flash::compile_ptx;
use crate::cuda_training::{cfg, cuda_err};
use crate::gpu_model::{Act, FlashShape, Gpu};
use crate::gpu_transformer::GpuContext;
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::text_encoder::TextEncoder;
use cudarc::driver::{CudaFunction, CudaSlice, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::sync::{Arc, OnceLock};

// The per-head RMS norm on a query or key (skipped when `norm` is 0), then the
// rotary position in the half-split order Hugging Face checkpoints use:
// dimension `p` pairs with `p + head_dim / 2`. One thread per (row, head).
const KERNELS: &str = r#"
extern "C" __global__ void head_norm_rope_half(float*x,const float*scale,const float*cs,const float*sn,int rows,int heads,int head_dim,int stride,int off,int norm,float eps){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*heads)return;int r=i/heads,h=i-r*heads;
  float*v=x+(size_t)r*stride+off+h*head_dim;int half=head_dim/2;
  float inv=1.f;
  if(norm){float s=0.f;for(int d=0;d<head_dim;d++)s+=v[d]*v[d];inv=rsqrtf(s/head_dim+eps);}
  const float*c=cs+(size_t)r*half;const float*n=sn+(size_t)r*half;
  for(int p=0;p<half;p++){
    float a=v[p]*inv,b=v[p+half]*inv;
    if(norm){a*=scale[p];b*=scale[p+half];}
    v[p]=a*c[p]-b*n[p];v[p+half]=b*c[p]+a*n[p];
  }
}
"#;

fn ptx() -> Result<&'static Ptx, NetworkError> {
    static PTX: OnceLock<Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| compile_ptx(KERNELS).map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| {
            NetworkError::Cuda(format!("text encoder kernel compilation failed: {error}"))
        })
}

struct DeviceLayer {
    input_norm: CudaSlice<f32>,
    /// Query, key and value stacked, `[(heads + 2 * kv_heads) * head_dim, d_model]`.
    qkv: Act,
    /// The per-head norms, or ones where the model has none.
    query_norm: CudaSlice<f32>,
    key_norm: CudaSlice<f32>,
    output: Act,
    post_norm: CudaSlice<f32>,
    /// The gate half then the up half, `[2 * mlp_dim, d_model]`.
    gate_up: Act,
    down: Act,
}

/// A [`TextEncoder`]'s layers resident on the device.
pub struct DeviceTextEncoder {
    context: Arc<GpuContext>,
    head_norm_rope: CudaFunction,
    /// Whether the layers normalize each query and key head, as Qwen3's do.
    head_norms: bool,
    /// Whether activations go to the GEMMs in BF16. Int8 weights keep them
    /// in FP32: BF16 on top of Q8's rounding put FLUX.2-klein's embeddings
    /// at 0.0259 against the host's 0.025, FP32 brings them back to 0.0249.
    narrow: bool,
    layers: Vec<DeviceLayer>,
}

impl std::fmt::Debug for DeviceTextEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceTextEncoder").finish_non_exhaustive()
    }
}

/// Every weight's shape against the configuration the kernels are launched
/// with, so a malformed checkpoint is refused instead of read past on the device.
fn check_shapes(encoder: &TextEncoder) -> Result<(), NetworkError> {
    let config = encoder.config();
    let (d_model, mlp, head_dim) = (config.d_model, config.mlp_dim, config.head_dim);
    let (query_width, kv_width) = (config.num_heads * head_dim, config.kv_heads * head_dim);
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(NetworkError::InvalidConfig(format!(
            "rotary heads need an even width, not {head_dim}"
        )));
    }
    for (index, layer) in encoder.layers.iter().enumerate() {
        let dense = [
            ("q_proj", &layer.query, query_width, d_model),
            ("k_proj", &layer.key, kv_width, d_model),
            ("v_proj", &layer.value, kv_width, d_model),
            ("o_proj", &layer.output, d_model, query_width),
            ("gate_proj", &layer.gate, mlp, d_model),
            ("up_proj", &layer.up, mlp, d_model),
            ("down_proj", &layer.down, d_model, mlp),
        ];
        for (name, weight, out_dim, in_dim) in dense {
            if (weight.out_dim(), weight.in_dim()) != (out_dim, in_dim) {
                return Err(NetworkError::InvalidConfig(format!(
                    "layer {index} {name} is {}x{}, the configuration wants {out_dim}x{in_dim}",
                    weight.out_dim(),
                    weight.in_dim()
                )));
            }
        }
        let norms = [
            ("input_layernorm", Some(&layer.input_norm), d_model),
            ("post_attention_layernorm", Some(&layer.post_norm), d_model),
            ("q_norm", layer.query_norm.as_ref(), head_dim),
            ("k_norm", layer.key_norm.as_ref(), head_dim),
        ];
        for (name, weight, width) in norms {
            if let Some(weight) = weight.filter(|weight| weight.len() != width) {
                return Err(NetworkError::InvalidConfig(format!(
                    "layer {index} {name} has {} values, the configuration wants {width}",
                    weight.len()
                )));
            }
        }
    }
    Ok(())
}

impl DeviceTextEncoder {
    /// Uploads `encoder`'s layers as BF16, or as int8 where it holds them at
    /// one byte. The host encoder stays the source of the embeddings, so it is
    /// borrowed again by [`DeviceTextEncoder::forward_layers`].
    pub fn upload(context: &Arc<GpuContext>, encoder: &TextEncoder) -> Result<Self, NetworkError> {
        check_shapes(encoder)?;
        let config = encoder.config();
        if context.half_accumulate {
            return Err(NetworkError::InvalidConfig(
                "the text encoder needs a BF16 context, not an FP16 one".into(),
            ));
        }
        if context.flash.is_none() || !crate::cuda_flash::eligible(true, config.head_dim) {
            return Err(NetworkError::InvalidConfig(format!(
                "the device text encoder needs the fused attention kernel, which takes heads \
                 of 64 or 128 on Ampere or later; this one has {}",
                config.head_dim
            )));
        }
        if config.kv_heads == 0 || !config.num_heads.is_multiple_of(config.kv_heads) {
            return Err(NetworkError::InvalidConfig(format!(
                "{} query heads do not share {} key heads evenly",
                config.num_heads, config.kv_heads
            )));
        }
        let first = encoder.layers.first();
        let head_norms = first.is_some_and(|layer| layer.query_norm.is_some());
        // NOTE: Qwen2's attention biases and a model that normalizes only
        // some heads stay on the host; add a bias pass if one is ever served.
        if encoder.layers.iter().any(|layer| {
            [
                &layer.query,
                &layer.key,
                &layer.value,
                &layer.output,
                &layer.gate,
                &layer.up,
                &layer.down,
            ]
            .iter()
            .any(|dense| dense.bias.is_some())
                || layer.query_norm.is_some() != head_norms
                || layer.key_norm.is_some() != head_norms
        }) {
            return Err(NetworkError::InvalidConfig(
                "the device text encoder takes no biases and the same head norms in every layer"
                    .into(),
            ));
        }

        let gpu = Gpu { context };
        let module = context
            .stream
            .context()
            .load_module(ptx()?.clone())
            .map_err(cuda_err("text encoder module loading"))?;
        let head_norm_rope = module
            .load_function("head_norm_rope_half")
            .map_err(cuda_err("text encoder kernel lookup"))?;
        let pack = |parts: &[&crate::conv::Dense]| -> Result<Act, NetworkError> {
            let inner = parts[0].in_dim();
            // Int8 parts with a scale a row go up stacked as they are: no
            // widened copy on the host, and none on the device, where one
            // beside a denoiser is what a 12 GB card ran out of room for.
            let int8: Option<Vec<_>> = parts
                .iter()
                .map(|part| part.int8().filter(|&(.., group)| group == inner))
                .collect();
            if let Some(int8) = int8 {
                let q: Vec<i8> = int8.iter().flat_map(|&(q, ..)| q).copied().collect();
                let scales: Vec<f32> = int8.iter().flat_map(|&(_, s, _)| s).copied().collect();
                return gpu.int8(&q, &scales, q.len() / inner, inner, inner);
            }
            let mut values = Vec::new();
            for part in parts {
                values.extend_from_slice(&part.weight_f32().data);
            }
            let units = values.len() / inner;
            let wide = gpu.upload(&values)?;
            match parts.iter().any(|part| part.is_quantized()) {
                true => gpu.quantized(&wide.slice(..), units, inner),
                false => gpu.narrowed(&wide.slice(..), units * inner, true),
            }
        };
        let identity = vec![1.0; config.head_dim];
        let norm = |weight: &Option<Vec<f32>>| gpu.upload(weight.as_deref().unwrap_or(&identity));
        let layers = encoder
            .layers
            .iter()
            .map(|layer| {
                Ok(DeviceLayer {
                    input_norm: gpu.upload(&layer.input_norm)?,
                    qkv: pack(&[&layer.query, &layer.key, &layer.value])?,
                    query_norm: norm(&layer.query_norm)?,
                    key_norm: norm(&layer.key_norm)?,
                    output: pack(&[&layer.output])?,
                    post_norm: gpu.upload(&layer.post_norm)?,
                    gate_up: pack(&[&layer.gate, &layer.up])?,
                    down: pack(&[&layer.down])?,
                })
            })
            .collect::<Result<_, NetworkError>>()?;
        let narrow = !encoder
            .layers
            .iter()
            .any(|layer| layer.query.is_quantized());
        Ok(Self {
            context: context.clone(),
            head_norm_rope,
            head_norms,
            narrow,
            layers,
        })
    }

    /// The states after each of `layers` layers, side by side per token, of
    /// which the first `real` of `ids` are attended to. `encoder` must be the
    /// one this was uploaded from, and has checked `layers` already.
    pub fn forward_layers(
        &self,
        encoder: &TextEncoder,
        ids: &[u32],
        real: usize,
        layers: &[usize],
    ) -> Result<Matrix, NetworkError> {
        let gpu = Gpu {
            context: &self.context,
        };
        let flash = self.context.flash.as_ref().expect("checked at upload");
        let config = encoder.config();
        let (tokens, d_model, eps) = (ids.len(), config.d_model, config.eps);
        let (heads, kv_heads, head_dim) = (config.num_heads, config.kv_heads, config.head_dim);
        let (query_width, kv_width) = (heads * head_dim, kv_heads * head_dim);
        let qkv_width = query_width + 2 * kv_width;
        let width = config.mlp_dim;
        let deepest = layers.iter().copied().max().unwrap_or(0);
        let narrow = self.narrow;

        let mut hidden = gpu.upload(&encoder.embed(ids)?.data)?;
        let rotary = encoder.rotary(tokens, real);
        let (cos, sin) = (gpu.upload(&rotary.cos)?, gpu.upload(&rotary.sin)?);
        let shape = FlashShape {
            rows: tokens,
            seq_len: tokens,
            // The padding's keys are never read: they are the last ones.
            kv_len: real.clamp(1, tokens),
            sequences: 1,
            heads,
            group: heads / kv_heads,
            qkv_width,
            kv_width: qkv_width,
            query_width,
            key_base: query_width,
            value_base: query_width + kv_width,
            causal: config.causal,
        };
        let scale = (head_dim as f32).sqrt().recip();
        let mut output = Matrix::new(tokens, layers.len() * d_model);

        for (index, layer) in self.layers[..deepest].iter().enumerate() {
            let (normed, _, _) = gpu.rmsnorm(
                &hidden,
                &layer.input_norm.as_view(),
                tokens,
                d_model,
                eps,
                narrow,
                false,
            )?;
            let mut qkv = gpu.uninit(tokens * qkv_width)?;
            gpu.project(
                &layer.qkv,
                qkv_width,
                d_model,
                &normed.all(),
                narrow,
                &mut qkv.slice_mut(..),
                tokens,
                0.0,
            )?;
            for (norm, count, offset) in [
                (&layer.query_norm, heads, 0),
                (&layer.key_norm, kv_heads, query_width),
            ] {
                unsafe {
                    self.context
                        .stream
                        .launch_builder(&self.head_norm_rope)
                        .arg(&mut qkv)
                        .arg(norm)
                        .arg(&cos)
                        .arg(&sin)
                        .arg(&(tokens as i32))
                        .arg(&(count as i32))
                        .arg(&(head_dim as i32))
                        .arg(&(qkv_width as i32))
                        .arg(&(offset as i32))
                        .arg(&i32::from(self.head_norms))
                        .arg(&eps)
                        .launch(cfg(tokens * count))
                        .map_err(cuda_err("text encoder head norm kernel"))?;
                }
            }
            let qkv = gpu.narrowed(&qkv.slice(..), tokens * qkv_width, true)?;
            let mut merged = gpu.act(tokens * query_width, true)?;
            let mut log_sum_exp = gpu.uninit(heads * tokens)?;
            gpu.flash_attention(
                flash,
                &qkv,
                &qkv,
                &mut merged,
                &mut log_sum_exp,
                shape,
                scale,
            )?;
            gpu.project(
                &layer.output,
                d_model,
                query_width,
                &merged.all(),
                true,
                &mut hidden.slice_mut(..),
                tokens,
                1.0,
            )?;

            let (normed, _, _) = gpu.rmsnorm(
                &hidden,
                &layer.post_norm.as_view(),
                tokens,
                d_model,
                eps,
                narrow,
                false,
            )?;
            let mut gate_up = gpu.uninit(tokens * 2 * width)?;
            gpu.project(
                &layer.gate_up,
                2 * width,
                d_model,
                &normed.all(),
                narrow,
                &mut gate_up.slice_mut(..),
                tokens,
                0.0,
            )?;
            let wide = gpu.gated(&gate_up, tokens, width, narrow, false)?;
            gpu.project(
                &layer.down,
                d_model,
                width,
                &wide.all(),
                narrow,
                &mut hidden.slice_mut(..),
                tokens,
                1.0,
            )?;

            let done = index + 1;
            if layers.contains(&done) {
                let states = gpu.download(&hidden)?;
                for (slot, _) in layers
                    .iter()
                    .enumerate()
                    .filter(|(_, layer)| **layer == done)
                {
                    for token in 0..tokens {
                        output.row_mut(token)[slot * d_model..(slot + 1) * d_model]
                            .copy_from_slice(&states[token * d_model..(token + 1) * d_model]);
                    }
                }
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::check_shapes;
    use crate::cuda_training::cuda_doctor;
    use crate::gpu_transformer::GpuContext;
    use crate::network::NetworkError;
    use crate::text_encoder::tests::checkpoint;
    use crate::text_encoder::{TextEncoder, TextEncoderConfig};
    use crate::transformer::Precision;
    use std::sync::Arc;

    fn cuda_or_skip() -> Option<Arc<GpuContext>> {
        match cuda_doctor(0, 8192) {
            Ok(_) => Some(GpuContext::new(0).expect("the CUDA doctor passed")),
            Err(NetworkError::Cuda(message))
                if message.contains("NO_DEVICE") || message.contains("no CUDA-capable device") =>
            {
                crate::cuda_training::skip_without_cuda();
                None
            }
            Err(error) => panic!("CUDA is present but the CUDA doctor failed: {error}"),
        }
    }

    #[test]
    fn a_checkpoint_whose_shapes_disagree_with_its_configuration_is_refused() {
        let config = TextEncoderConfig {
            vocab_size: 16,
            d_model: 64,
            layers: 2,
            num_heads: 4,
            kv_heads: 2,
            head_dim: 64,
            mlp_dim: 96,
            rope_theta: 10_000.0,
            eps: 1e-6,
            causal: true,
        };
        let path = std::env::temp_dir().join(format!(
            "rusting_brain_cuda_text_shapes_{}.safetensors",
            std::process::id()
        ));
        checkpoint(&config, &path, true);
        let load =
            || TextEncoder::load_at(&path, "model.", config.clone(), Precision::F32).unwrap();
        assert!(check_shapes(&load()).is_ok());
        let breaks: [fn(&mut crate::text_encoder::Layer); 4] = [
            |layer| std::mem::swap(&mut layer.gate, &mut layer.down),
            |layer| std::mem::swap(&mut layer.query, &mut layer.key),
            |layer| layer.post_norm.truncate(63),
            |layer| layer.key_norm = Some(vec![1.0; 3]),
        ];
        let context = cuda_or_skip();
        for (case, broken) in breaks.into_iter().enumerate() {
            let mut encoder = load();
            broken(&mut encoder.layers[1]);
            assert!(
                matches!(check_shapes(&encoder), Err(NetworkError::InvalidConfig(_))),
                "case {case}"
            );
            if let Some(context) = &context {
                assert!(encoder.attach_device(context).is_err(), "case {case}");
                assert!(!encoder.on_device());
            }
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_device_text_encoder_matches_the_host_or_skips_without_device() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        // The fused attention kernel takes heads of 64 or 128, so this is
        // the smallest shape it runs: four query heads over two key heads.
        let config = TextEncoderConfig {
            vocab_size: 16,
            d_model: 64,
            layers: 3,
            num_heads: 4,
            kv_heads: 2,
            head_dim: 64,
            mlp_dim: 96,
            rope_theta: 10_000.0,
            eps: 1e-6,
            causal: true,
        };
        for head_norms in [true, false] {
            let path = std::env::temp_dir()
                .join(format!("rusting_brain_cuda_text_{head_norms}.safetensors"));
            checkpoint(&config, &path, head_norms);
            for precision in [Precision::F32, Precision::Q8] {
                let mut encoder =
                    TextEncoder::load_at(&path, "model.", config.clone(), precision).unwrap();
                // 70 tokens, 9 of them real: two query tiles, padding in both.
                let ids: Vec<u32> = (0..70).map(|index| (index * 7 % 16) as u32).collect();
                let expected = encoder.forward_layers(&ids, 9, &[1, 3]).unwrap();
                let plain = encoder.forward(&ids[..9]).unwrap();
                encoder.attach_device(&context).unwrap();
                assert!(encoder.on_device());
                let actual = encoder.forward_layers(&ids, 9, &[1, 3]).unwrap();
                let actual_plain = encoder.forward(&ids[..9]).unwrap();

                for (expected, actual) in [(&expected, &actual), (&plain, &actual_plain)] {
                    assert_eq!((actual.rows, actual.cols), (expected.rows, expected.cols));
                    let (gap, norm) = actual
                        .data
                        .iter()
                        .zip(&expected.data)
                        .fold((0.0f64, 0.0f64), |(g, n), (a, b)| {
                            (g + ((a - b) as f64).powi(2), n + (*b as f64).powi(2))
                        });
                    // BF16 operands over three layers: 7e-4 to 4.3e-3 measured. Reading
                    // the padding's keys puts it at 0.11.
                    let error = (gap / norm).sqrt();
                    assert!(error < 1e-2, "{precision:?}, norms {head_norms}: {error}");
                }
                encoder.detach_device();
                assert!(!encoder.on_device());
            }
            std::fs::remove_file(path).ok();
        }
    }
}
