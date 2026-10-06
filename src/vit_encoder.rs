//! The CLIP image tower, frozen, as a source of conditioning tokens.
//!
//! [`crate::clip`] is the text half of the same model and [`crate::vision`] is
//! a vision transformer trained from scratch. Neither is this one.
//! `crate::vision` uses RMSNorm, rotary positions and mean pooling, so it
//! cannot read a published CLIP checkpoint; this module reads one and runs it
//! forward, which is all an image-to-3D pipeline needs, because the tower is
//! frozen and never sees a gradient.
//!
//! The tower is: a patch embedding, a class token, a learned position table, a
//! layer norm, twelve of [`crate::clip`]'s layers with attention that is not
//! causal, and a final layer norm. [`VitEncoder::encode`] returns every token,
//! `[patches + 1, d_model]`, with no pooling and no projection head: a
//! cross-attention condition wants the whole grid, and the class token at row 0
//! is there for a caller that wants a single vector too.
//!
//! ponytail: the patch embedding is a convolution in the checkpoint, with the
//! kernel the same size as the stride. That makes it a linear map of the pixels
//! in each patch, so the `[d_model, channels, patch, patch]` weight is read as
//! `[d_model, channels * patch * patch]` and run through [`Dense`] instead of
//! [`Conv2d`](crate::conv::Conv2d). Same arithmetic, no convolution.
//!
//! Inference only. With the `cuda` feature, [`VitEncoder::attach_device`] moves
//! the tower onto a card and [`VitEncoder::encode_batch`] runs many images as
//! one stack there, which is what a corpus of a hundred thousand images needs.

use crate::clip::{Layer, Norm, dense, norm, run_layer};
use crate::conv::{Dense, FeatureMap};
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::safetensors::ShardedSafeTensors;
use crate::transformer::Precision;

/// The two things about an image tower that its weights do not state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VitEncoderConfig {
    /// Query heads. CLIP's image towers use 64-wide heads throughout, so this
    /// is `d_model / 64`: 12 for ViT-B, 16 for ViT-L.
    pub num_heads: usize,
    /// LayerNorm epsilon, `1e-5` in every published tower.
    pub eps: f32,
    /// `x * sigmoid(1.702x)` rather than the exact GELU. True for the OpenAI
    /// checkpoints and the ones converted from them.
    pub quick_gelu: bool,
}

impl Default for VitEncoderConfig {
    /// ViT-B/16.
    fn default() -> Self {
        Self {
            num_heads: 12,
            eps: 1e-5,
            quick_gelu: true,
        }
    }
}

/// A frozen CLIP image tower.
pub struct VitEncoder {
    config: VitEncoderConfig,
    /// `[d_model, channels * patch * patch]`, the convolution read flat.
    pub(crate) patch: Dense,
    class_token: Vec<f32>,
    /// `[patches + 1, d_model]`.
    positions: Matrix,
    pub(crate) pre_norm: Norm,
    pub(crate) layers: Vec<Layer>,
    pub(crate) post_norm: Norm,
    channels: usize,
    patch_size: usize,
    /// The same weights on a CUDA device, when one was attached.
    #[cfg(feature = "cuda")]
    device: Option<crate::cuda_image::DeviceVitEncoder>,
}

impl VitEncoder {
    /// Reads a tower out of a checkpoint.
    ///
    /// `prefix` is what comes before `vision_model` in the tensor names: empty
    /// for a standalone CLIP checkpoint, something like `image_encoder.` where
    /// the tower is one model among several. Everything but
    /// [`VitEncoderConfig`] is taken from the shapes, so a ViT-L checkpoint
    /// loads by changing `num_heads` and nothing else.
    pub fn load<P: AsRef<std::path::Path>>(
        path: P,
        prefix: &str,
        config: VitEncoderConfig,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        let mut file = ShardedSafeTensors::open(path)?;
        Self::read_at(&mut file, prefix, config, precision)
    }

    /// The same, from a checkpoint that is already open.
    pub fn read_at(
        file: &mut ShardedSafeTensors,
        prefix: &str,
        config: VitEncoderConfig,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        if config.num_heads == 0 {
            return Err(NetworkError::InvalidConfig(
                "an image tower needs at least one attention head".into(),
            ));
        }
        let base = format!("{prefix}vision_model");

        let (values, shape) = file.tensor(&format!("{base}.embeddings.patch_embedding.weight"))?;
        let [d_model, channels, patch_rows, patch_cols] = shape[..] else {
            return Err(NetworkError::InvalidConfig(format!(
                "the patch embedding is {shape:?}, which is not a convolution weight"
            )));
        };
        if patch_rows != patch_cols {
            return Err(NetworkError::InvalidConfig(format!(
                "the patch embedding reads a {patch_rows}x{patch_cols} patch, which is not square"
            )));
        }
        // No bias: the published towers train this convolution without one.
        let patch = Dense::new(
            Matrix::from_vec(d_model, channels * patch_rows * patch_cols, values),
            None,
        )?;

        let class_token = file
            .tensor(&format!("{base}.embeddings.class_embedding"))?
            .0;
        if class_token.len() != d_model {
            return Err(NetworkError::InvalidTarget {
                expected: d_model,
                actual: class_token.len(),
            });
        }
        let positions = file.matrix(&format!("{base}.embeddings.position_embedding.weight"))?;
        if positions.cols != d_model {
            return Err(NetworkError::InvalidConfig(format!(
                "the position table is {} wide and the patch embedding produces {d_model}",
                positions.cols
            )));
        }
        let patches = positions.rows - 1;
        // A square grid of patches plus the class token is the only layout the
        // flattening in `encode` knows how to undo.
        if patches.isqrt().pow(2) != patches || patches == 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "the position table holds {} rows, which is not a square grid plus a class token",
                positions.rows
            )));
        }

        // Transformers shipped this one misspelled and kept the name for
        // compatibility, so both spellings are in the wild.
        let pre_norm = norm(file, &format!("{base}.pre_layrnorm"))
            .or_else(|_| norm(file, &format!("{base}.pre_layernorm")))?;

        let mut layers = Vec::new();
        while let Ok(layer) = read_layer(file, &base, layers.len(), precision) {
            layers.push(layer);
        }
        if layers.is_empty() {
            return Err(NetworkError::InvalidConfig(format!(
                "{base}.encoder.layers.0 is missing, so this is not an image tower"
            )));
        }
        if d_model % config.num_heads != 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "{} heads do not divide a width of {d_model}",
                config.num_heads
            )));
        }

        Ok(Self {
            post_norm: norm(file, &format!("{base}.post_layernorm"))?,
            config,
            patch,
            class_token,
            positions,
            pre_norm,
            layers,
            channels,
            patch_size: patch_rows,
            #[cfg(feature = "cuda")]
            device: None,
        })
    }

    /// The configuration this tower was built for.
    pub fn config(&self) -> &VitEncoderConfig {
        &self.config
    }

    /// Uploads the tower to a CUDA device, and [`VitEncoder::encode`] and
    /// [`VitEncoder::encode_batch`] run there from now on.
    ///
    /// The device computes in FP16, so its tokens differ slightly from the
    /// host's. A failed upload leaves the tower on the CPU and returns the
    /// error; a failure later, during a pass, is returned too and never falls
    /// back to the CPU.
    #[cfg(feature = "cuda")]
    pub fn attach_device(
        &mut self,
        gpu: std::sync::Arc<crate::cuda_image::ImageGpu>,
    ) -> Result<(), NetworkError> {
        self.device = Some(crate::cuda_image::DeviceVitEncoder::upload(&gpu, self)?);
        Ok(())
    }

    /// Side of the square image the tower expects, in pixels.
    pub fn image_size(&self) -> usize {
        self.patch_size * (self.positions.rows - 1).isqrt()
    }

    /// Colour channels the tower expects, 3 in every published checkpoint.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Width of one token.
    pub fn d_model(&self) -> usize {
        self.positions.cols
    }

    /// Tokens [`VitEncoder::encode`] returns: one per patch, plus the class
    /// token.
    pub fn tokens(&self) -> usize {
        self.positions.rows
    }

    /// Runs the tower and returns every token, `[tokens, d_model]`.
    ///
    /// Row 0 is the class token and the rest are the patches in row-major
    /// order. The final layer norm is applied to all of them, not to the class
    /// token alone: a condition that mixes normalized and unnormalized rows
    /// would make the cross-attention learn the difference for no reason.
    ///
    /// The image is expected already resized to [`VitEncoder::image_size`] and
    /// already normalized the way the checkpoint was trained, which for CLIP is
    /// per-channel `(pixel - mean) / std` over `[0, 1]` values; see
    /// [`CLIP_MEAN`] and [`CLIP_STD`].
    pub fn encode(&self, image: &FeatureMap) -> Result<Matrix, NetworkError> {
        #[cfg(feature = "cuda")]
        if self.device.is_some() {
            return Ok(self.encode_batch(std::slice::from_ref(image))?.remove(0));
        }

        let flat = self.flatten(image)?;
        let projected = self.patch.forward(&flat)?;
        let mut hidden = Matrix::new(self.positions.rows, self.d_model());
        hidden.row_mut(0).copy_from_slice(&self.class_token);
        for index in 0..projected.rows {
            hidden
                .row_mut(index + 1)
                .copy_from_slice(projected.row(index));
        }
        for index in 0..hidden.rows {
            for (value, learned) in hidden
                .row_mut(index)
                .iter_mut()
                .zip(self.positions.row(index))
            {
                *value += learned;
            }
        }

        let mut hidden = self.pre_norm.forward(&hidden, self.config.eps);
        for layer in &self.layers {
            run_layer(
                layer,
                &mut hidden,
                self.config.num_heads,
                self.config.eps,
                self.config.quick_gelu,
                false,
            )?;
        }
        Ok(self.post_norm.forward(&hidden, self.config.eps))
    }

    /// [`VitEncoder::encode`] over several images, one token matrix each.
    ///
    /// On the host this is a loop. On a device the images run as one stack,
    /// `[images * tokens, d_model]`, so every projection is one GEMM over all
    /// of them, and only attention is split per image. The whole slice is one
    /// stack, so the caller picks the batch size by how many images it passes.
    pub fn encode_batch(&self, images: &[FeatureMap]) -> Result<Vec<Matrix>, NetworkError> {
        #[cfg(feature = "cuda")]
        if let Some(device) = self.device.as_ref().filter(|_| !images.is_empty()) {
            let tokens = self.tokens();
            let width = self.d_model();
            let mut patches = Matrix::new(images.len() * tokens, self.patch.in_dim());
            let mut embedding = Matrix::new(images.len() * tokens, width);
            for (index, image) in images.iter().enumerate() {
                let flat = self.flatten(image)?;
                let first = index * tokens;
                // Row `first` stays zero, so the class token gets no projection.
                let start = (first + 1) * flat.cols;
                patches.data[start..start + flat.data.len()].copy_from_slice(&flat.data);
                let rows = first * width..(first + tokens) * width;
                embedding.data[rows].copy_from_slice(&self.positions.data);
                for (value, class) in embedding.row_mut(first).iter_mut().zip(&self.class_token) {
                    *value += class;
                }
            }
            let stacked = device.forward(&patches, &embedding, tokens)?;
            return Ok(stacked
                .data
                .chunks_exact(tokens * width)
                .map(|image| Matrix::from_vec(tokens, width, image.to_vec()))
                .collect());
        }
        images.iter().map(|image| self.encode(image)).collect()
    }

    /// The image cut into patches, `[patches, channels * patch * patch]`, one
    /// row per patch in row-major order.
    fn flatten(&self, image: &FeatureMap) -> Result<Matrix, NetworkError> {
        let size = self.image_size();
        if image.channels != self.channels || image.height != size || image.width != size {
            return Err(NetworkError::InvalidConfig(format!(
                "the tower reads a {}x{size}x{size} image and this one is {}x{}x{}",
                self.channels, image.channels, image.height, image.width
            )));
        }

        let patch = self.patch_size;
        let grid = size / patch;
        let mut flat = Matrix::new(grid * grid, self.patch.in_dim());
        for row in 0..grid {
            for column in 0..grid {
                let target = flat.row_mut(row * grid + column);
                for channel in 0..self.channels {
                    for y in 0..patch {
                        let source = (channel * size + row * patch + y) * size + column * patch;
                        let start = (channel * patch + y) * patch;
                        target[start..start + patch]
                            .copy_from_slice(&image.data[source..source + patch]);
                    }
                }
            }
        }
        Ok(flat)
    }
}

/// Per-channel mean CLIP's images were normalized with, over `[0, 1]` pixels.
pub const CLIP_MEAN: [f32; 3] = [0.481_454_82, 0.457_827_5, 0.408_210_74];
/// Per-channel standard deviation to match [`CLIP_MEAN`].
pub const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Composites an RGBA image onto one flat colour and normalizes it for the
/// tower.
///
/// Renders of a 3D asset come out with an alpha channel, and the tower has
/// three input channels. Compositing is the cheap answer; training a fourth
/// weight column would mean unfreezing the tower.
///
/// `pixels` is `[r, g, b, a]` per pixel in row-major order with values in
/// `[0, 1]`, which is what an image decoder gives after dividing by 255.
pub fn composite_rgba(
    pixels: &[f32],
    size: usize,
    background: [f32; 3],
) -> Result<FeatureMap, NetworkError> {
    if pixels.len() != size * size * 4 {
        return Err(NetworkError::InvalidTarget {
            expected: size * size * 4,
            actual: pixels.len(),
        });
    }
    let mut map = FeatureMap::new(3, size, size);
    for (index, pixel) in pixels.chunks_exact(4).enumerate() {
        let alpha = pixel[3];
        for channel in 0..3 {
            let composited = pixel[channel] * alpha + background[channel] * (1.0 - alpha);
            map.data[channel * size * size + index] =
                (composited - CLIP_MEAN[channel]) / CLIP_STD[channel];
        }
    }
    Ok(map)
}

/// Decodes, resizes and normalizes one image the way a CLIP tower was trained.
///
/// `size` is [`VitEncoder::image_size`] and `background` is the grey level a
/// transparent pixel is composited onto, which for a render on white is 1.0.
///
/// This is the one place the `image` crate is used for reading, so a caller
/// that already has pixels builds a [`FeatureMap`] itself or goes through
/// [`composite_rgba`].
#[cfg(feature = "images")]
pub fn read_image<P: AsRef<std::path::Path>>(
    path: P,
    size: usize,
    background: f32,
) -> Result<FeatureMap, NetworkError> {
    let path = path.as_ref();
    let decoded = image::open(path)
        .map_err(|error| NetworkError::InvalidDataset(format!("{}: {error}", path.display())))?;
    let side = size as u32;
    let decoded = decoded.resize_exact(side, side, image::imageops::FilterType::CatmullRom);

    // A render of a 3D asset is usually cut out, and the tower has three input
    // channels, so the alpha is composited rather than fed.
    if decoded.color().has_alpha() {
        let rgba: Vec<f32> = decoded
            .to_rgba8()
            .iter()
            .map(|&value| f32::from(value) / 255.0)
            .collect();
        return composite_rgba(&rgba, size, [background; 3]);
    }

    let mut map = FeatureMap::new(3, size, size);
    for (index, pixel) in decoded.to_rgb8().pixels().enumerate() {
        for channel in 0..3 {
            map.data[channel * size * size + index] =
                (f32::from(pixel[channel]) / 255.0 - CLIP_MEAN[channel]) / CLIP_STD[channel];
        }
    }
    Ok(map)
}

fn read_layer(
    file: &mut ShardedSafeTensors,
    base: &str,
    index: usize,
    precision: Precision,
) -> Result<Layer, NetworkError> {
    let layer = format!("{base}.encoder.layers.{index}");
    Ok(Layer {
        attention_norm: norm(file, &format!("{layer}.layer_norm1"))?,
        query: dense(file, &format!("{layer}.self_attn.q_proj"), precision)?,
        key: dense(file, &format!("{layer}.self_attn.k_proj"), precision)?,
        value: dense(file, &format!("{layer}.self_attn.v_proj"), precision)?,
        output: dense(file, &format!("{layer}.self_attn.out_proj"), precision)?,
        mlp_norm: norm(file, &format!("{layer}.layer_norm2"))?,
        mlp_in: dense(file, &format!("{layer}.mlp.fc1"), precision)?,
        mlp_out: dense(file, &format!("{layer}.mlp.fc2"), precision)?,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const D_MODEL: usize = 8;
    const PATCH: usize = 2;
    const GRID: usize = 3;
    const LAYERS: usize = 2;

    /// A tower whose every residual branch is zeroed, so the layers are the
    /// identity and the output is the embedding through the two norms. That
    /// gives the forward pass a reference to be checked against rather than
    /// only a shape.
    pub(crate) fn checkpoint(path: &std::path::Path, identity_layers: bool) {
        let mut tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
        let mut add = |name: String, shape: Vec<usize>, zero: bool| {
            let count = shape.iter().product::<usize>();
            let values = (0..count)
                .map(|index| match zero {
                    true => 0.0,
                    false => ((index % 11) as f32 - 5.0) * 0.05,
                })
                .collect();
            tensors.insert(name, (shape, values));
        };

        add(
            "vision_model.embeddings.patch_embedding.weight".into(),
            vec![D_MODEL, 3, PATCH, PATCH],
            false,
        );
        add(
            "vision_model.embeddings.class_embedding".into(),
            vec![D_MODEL],
            false,
        );
        add(
            "vision_model.embeddings.position_embedding.weight".into(),
            vec![GRID * GRID + 1, D_MODEL],
            false,
        );
        for name in ["pre_layrnorm", "post_layernorm"] {
            add(format!("vision_model.{name}.weight"), vec![D_MODEL], false);
            add(format!("vision_model.{name}.bias"), vec![D_MODEL], false);
        }
        for index in 0..LAYERS {
            let base = format!("vision_model.encoder.layers.{index}");
            for norm in ["layer_norm1", "layer_norm2"] {
                add(format!("{base}.{norm}.weight"), vec![D_MODEL], false);
                add(format!("{base}.{norm}.bias"), vec![D_MODEL], false);
            }
            for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                let zero = identity_layers && name == "out_proj";
                add(
                    format!("{base}.self_attn.{name}.weight"),
                    vec![D_MODEL, D_MODEL],
                    zero,
                );
                add(format!("{base}.self_attn.{name}.bias"), vec![D_MODEL], zero);
            }
            add(
                format!("{base}.mlp.fc1.weight"),
                vec![D_MODEL * 2, D_MODEL],
                false,
            );
            add(format!("{base}.mlp.fc1.bias"), vec![D_MODEL * 2], false);
            add(
                format!("{base}.mlp.fc2.weight"),
                vec![D_MODEL, D_MODEL * 2],
                identity_layers,
            );
            add(
                format!("{base}.mlp.fc2.bias"),
                vec![D_MODEL],
                identity_layers,
            );
        }
        crate::safetensors::write_checkpoint(path, &tensors);
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "rusting-brain-vit-{name}-{}.safetensors",
            std::process::id()
        ))
    }

    pub(crate) fn tiny() -> VitEncoderConfig {
        VitEncoderConfig {
            num_heads: 2,
            ..VitEncoderConfig::default()
        }
    }

    pub(crate) fn ramp(size: usize) -> FeatureMap {
        FeatureMap::from_vec(
            3,
            size,
            size,
            (0..3 * size * size)
                .map(|index| ((index % 7) as f32 - 3.0) * 0.1)
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn the_shape_is_read_out_of_the_checkpoint() {
        let path = scratch("shape");
        checkpoint(&path, false);
        let tower = VitEncoder::load(&path, "", tiny(), Precision::F32).unwrap();
        assert_eq!(tower.d_model(), D_MODEL);
        assert_eq!(tower.channels(), 3);
        assert_eq!(tower.image_size(), GRID * PATCH);
        assert_eq!(tower.tokens(), GRID * GRID + 1);

        let tokens = tower.encode(&ramp(tower.image_size())).unwrap();
        assert_eq!((tokens.rows, tokens.cols), (GRID * GRID + 1, D_MODEL));
        assert!(tokens.data.iter().all(|value| value.is_finite()));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn zeroed_residual_branches_leave_the_embedding_alone() {
        let path = scratch("identity");
        checkpoint(&path, true);
        let tower = VitEncoder::load(&path, "", tiny(), Precision::F32).unwrap();
        let size = tower.image_size();
        let image = ramp(size);
        let tokens = tower.encode(&image).unwrap();

        // The same embedding the forward pass builds, by hand: the class token
        // and the patch projections, each plus its learned position.
        let mut expected = Matrix::new(tower.tokens(), D_MODEL);
        expected.row_mut(0).copy_from_slice(&tower.class_token);
        for row in 0..GRID {
            for column in 0..GRID {
                let mut patch = Vec::new();
                for channel in 0..3 {
                    for y in 0..PATCH {
                        for x in 0..PATCH {
                            let index =
                                (channel * size + row * PATCH + y) * size + column * PATCH + x;
                            patch.push(image.data[index]);
                        }
                    }
                }
                let target = expected.row_mut(row * GRID + column + 1);
                for (unit, weight) in target.iter_mut().enumerate() {
                    *weight = tower
                        .patch
                        .weight
                        .row(unit)
                        .iter()
                        .zip(&patch)
                        .map(|(a, b)| a * b)
                        .sum();
                }
            }
        }
        for index in 0..expected.rows {
            for (value, learned) in expected
                .row_mut(index)
                .iter_mut()
                .zip(tower.positions.row(index))
            {
                *value += learned;
            }
        }
        let expected = tower.pre_norm.forward(&expected, tiny().eps);
        let expected = tower.post_norm.forward(&expected, tiny().eps);

        for (got, want) in tokens.data.iter().zip(&expected.data) {
            assert!((got - want).abs() < 1e-5, "{got} against {want}");
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn an_image_of_the_wrong_size_is_refused() {
        let path = scratch("wrong-size");
        checkpoint(&path, false);
        let tower = VitEncoder::load(&path, "", tiny(), Precision::F32).unwrap();
        assert!(tower.encode(&ramp(tower.image_size() + PATCH)).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn heads_that_do_not_divide_the_width_are_refused() {
        let path = scratch("bad-heads");
        checkpoint(&path, false);
        let config = VitEncoderConfig {
            num_heads: 3,
            ..tiny()
        };
        assert!(VitEncoder::load(&path, "", config, Precision::F32).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn alpha_chooses_between_the_pixel_and_the_background() {
        let opaque = composite_rgba(&[0.5, 0.5, 0.5, 1.0], 1, [0.0, 0.0, 0.0]).unwrap();
        let clear = composite_rgba(&[0.5, 0.5, 0.5, 0.0], 1, [1.0, 1.0, 1.0]).unwrap();
        for channel in 0..3 {
            let expected = (0.5 - CLIP_MEAN[channel]) / CLIP_STD[channel];
            assert!((opaque.data[channel] - expected).abs() < 1e-6);
            let expected = (1.0 - CLIP_MEAN[channel]) / CLIP_STD[channel];
            assert!((clear.data[channel] - expected).abs() < 1e-6);
        }
        assert!(composite_rgba(&[0.0; 3], 1, [0.0; 3]).is_err());
    }
}
