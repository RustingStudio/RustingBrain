//! Vision transformer: an image as a sequence of patches.
//!
//! The expensive parts are already here. A patch embedding is a linear
//! projection of `channels * patch * patch` pixels, the blocks are the same
//! [`TransformerBlock`] a language model stacks, and the only thing a vision
//! model needs that a decoder does not is bidirectional attention, which
//! [`MultiHeadAttention::set_causal`](crate::MultiHeadAttention::set_causal) gives it: a patch in the top-left corner
//! has to be able to read one in the bottom-right.
//!
//! Positions come from the same rotary embedding the language models use,
//! applied over the patches in row-major order, rather than from a learned or
//! sinusoidal table.
//!
//! NOTE: rotary positions over a flattened grid know that two patches are
//! `n` patches apart in reading order, not that they are neighbours one row
//! down. A learned `[num_patches, d_model]` table added after the patch
//! projection is the usual answer and is a `Param` plus an add in the forward
//! pass; add it if accuracy on a real dataset asks for it.
//!
//! The pooled representation is the mean over patches rather than a class
//! token, which saves a parameter and a special-cased row and costs nothing
//! measurable.

use crate::batch::Layout;
use crate::causal_lm_loss::cross_entropy_rows;
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::norm::RmsNorm;
use crate::optimizers::Optimizer;
use crate::param::{Linear, Param};
use crate::rope::Rope;
use crate::transformer_block::{FeedForward, TransformerBlock, TransformerBlockCache};
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Shape of a vision transformer.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct VitConfig {
    /// Side of the square image, in pixels.
    pub image_size: usize,
    /// Side of a square patch. Must divide `image_size`.
    pub patch_size: usize,
    /// 1 for grayscale, 3 for RGB, matching what the dataset holds.
    pub channels: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub d_ff: usize,
    pub num_classes: usize,
    pub rmsnorm_eps: f32,
    pub seed: u64,
}

impl Default for VitConfig {
    fn default() -> Self {
        Self {
            image_size: 32,
            patch_size: 4,
            channels: 3,
            d_model: 192,
            n_layers: 6,
            num_heads: 3,
            num_kv_heads: 3,
            head_dim: 64,
            d_ff: 768,
            num_classes: 10,
            rmsnorm_eps: 1e-5,
            seed: 42,
        }
    }
}

impl VitConfig {
    /// Pixels per patch, which is what the patch projection reads.
    pub fn patch_dim(&self) -> usize {
        self.channels * self.patch_size * self.patch_size
    }

    /// Patches per image, which is the sequence length.
    pub fn num_patches(&self) -> usize {
        let side = self.image_size / self.patch_size;
        side * side
    }

    /// Values per image, which is the width of a dataset row.
    pub fn image_len(&self) -> usize {
        self.channels * self.image_size * self.image_size
    }

    fn check(&self) -> Result<(), NetworkError> {
        if self.patch_size == 0 || !self.image_size.is_multiple_of(self.patch_size) {
            return Err(NetworkError::InvalidConfig(format!(
                "a {}-pixel image does not divide into {}-pixel patches",
                self.image_size, self.patch_size
            )));
        }
        if self.channels == 0 || self.num_classes < 2 {
            return Err(NetworkError::InvalidConfig(
                "a classifier needs at least one channel and two classes".into(),
            ));
        }
        Ok(())
    }
}

/// What the backward pass needs from the forward pass.
#[derive(Clone, Debug)]
pub struct VitCache {
    patches: Matrix,
    blocks: Vec<TransformerBlockCache>,
    final_input: Matrix,
    pooled: Matrix,
    /// Set when the blocks ran on a device, in which case `blocks` is empty.
    #[cfg(feature = "cuda")]
    device: Option<std::sync::Arc<crate::gpu_model::GpuStack>>,
}

impl VitCache {
    /// Sum of every MoE layer's auxiliary losses. Zero here, since the vision
    /// blocks are dense, and kept so the caller reports the same total a
    /// language model does.
    pub fn auxiliary_loss(&self) -> f32 {
        self.blocks.iter().map(|block| block.auxiliary_loss()).sum()
    }
}

/// An image classifier built from the transformer stack.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisionTransformer {
    pub config: VitConfig,
    /// `patch_dim -> d_model`.
    pub patch: Linear,
    pub blocks: Vec<TransformerBlock>,
    pub norm: RmsNorm,
    /// `d_model -> num_classes`, over the mean of the patches.
    pub head: Linear,
    pub optimizer: Optimizer,
    #[serde(default)]
    optimizer_step: usize,
    /// Set by [`VisionTransformer::to_cuda`]. Never serialized.
    #[cfg(feature = "cuda")]
    #[serde(skip)]
    device: Option<std::sync::Arc<crate::gpu_transformer::GpuContext>>,
}

impl VisionTransformer {
    /// Builds the model `config` describes, with bidirectional attention in
    /// every block.
    ///
    /// ```
    /// # use rusting_brain::{VisionTransformer, VitConfig};
    /// let model = VisionTransformer::new(VitConfig {
    ///     image_size: 8,
    ///     patch_size: 4,
    ///     channels: 1,
    ///     d_model: 16,
    ///     n_layers: 2,
    ///     num_heads: 2,
    ///     num_kv_heads: 2,
    ///     head_dim: 8,
    ///     d_ff: 32,
    ///     num_classes: 3,
    ///     ..VitConfig::default()
    /// })?;
    /// assert_eq!(model.config.num_patches(), 4);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(config: VitConfig) -> Result<Self, NetworkError> {
        config.check()?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let rope = Rope::new(config.head_dim, config.num_patches(), 10_000.0)?;

        let mut blocks = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            let mut block = TransformerBlock::new(
                config.d_model,
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                rope.clone(),
                FeedForward::swiglu(config.d_model, config.d_ff, &mut rng),
                config.rmsnorm_eps,
                &mut rng,
            )?;
            block.attention.set_causal(false);
            blocks.push(block);
        }

        Ok(Self {
            patch: Linear::new(config.patch_dim(), config.d_model, &mut rng),
            blocks,
            norm: RmsNorm::new(config.d_model, config.rmsnorm_eps),
            head: Linear::new(config.d_model, config.num_classes, &mut rng),
            optimizer: Optimizer::adam(3e-4),
            optimizer_step: 0,
            #[cfg(feature = "cuda")]
            device: None,
            config,
        })
    }

    /// Replaces the optimizer, which defaults to Adam at `3e-4`.
    pub fn with_optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Cuts each image into patches: `[batch, image_len]` in, `[batch *
    /// num_patches, patch_dim]` out.
    ///
    /// The dataset stores an image plane by plane, each plane row-major, which
    /// is what [`Dataset::from_image_folder`](crate::Dataset::from_image_folder)
    /// writes, so a patch is `channels` blocks of `patch_size` short runs.
    pub fn patchify(&self, images: &Matrix) -> Result<Matrix, NetworkError> {
        let config = &self.config;
        if images.cols != config.image_len() {
            return Err(NetworkError::InvalidInput {
                expected: config.image_len(),
                actual: images.cols,
            });
        }

        let side = config.image_size / config.patch_size;
        let (patch, size) = (config.patch_size, config.image_size);
        let mut out = Matrix::new(images.rows * config.num_patches(), config.patch_dim());

        out.data
            .par_chunks_mut(config.num_patches() * config.patch_dim())
            .enumerate()
            .for_each(|(image, rows)| {
                let pixels = images.row(image);
                for index in 0..config.num_patches() {
                    let (top, left) = (index / side * patch, index % side * patch);
                    let row = &mut rows[index * config.patch_dim()..][..config.patch_dim()];
                    for channel in 0..config.channels {
                        let plane = channel * size * size;
                        for line in 0..patch {
                            let source = plane + (top + line) * size + left;
                            let target = (channel * patch + line) * patch;
                            row[target..target + patch]
                                .copy_from_slice(&pixels[source..source + patch]);
                        }
                    }
                }
            });

        Ok(out)
    }

    /// Logits for a batch of images, `[batch, num_classes]`, with the cache the
    /// backward pass needs.
    pub fn forward_train(&self, images: &Matrix) -> Result<(Matrix, VitCache), NetworkError> {
        let patches = self.patchify(images)?;
        let mut hidden = self.patch.forward(&patches);

        #[cfg(feature = "cuda")]
        if let Some(context) = &self.device {
            let (hidden, stack) = crate::gpu_model::stack_forward(
                &self.blocks,
                context,
                &hidden,
                self.config.num_patches(),
            )?;
            let (logits, mut cache) = self.head_forward(patches, Vec::new(), hidden);
            cache.device = Some(std::sync::Arc::new(stack));
            return Ok((logits, cache));
        }

        let layout = Layout {
            seq_len: Some(self.config.num_patches()),
            valid: None,
        };
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (output, cache) = block.forward_train(&hidden, layout)?;
            hidden = output;
            blocks.push(cache);
        }

        Ok(self.head_forward(patches, blocks, hidden))
    }

    /// The norm, the pooling and the head over the last block's output.
    fn head_forward(
        &self,
        patches: Matrix,
        blocks: Vec<TransformerBlockCache>,
        hidden: Matrix,
    ) -> (Matrix, VitCache) {
        let normed = self.norm.forward(&hidden);
        let pooled = self.mean_pool(&normed);
        let logits = self.head.forward(&pooled);
        (
            logits,
            VitCache {
                patches,
                blocks,
                final_input: hidden,
                pooled,
                #[cfg(feature = "cuda")]
                device: None,
            },
        )
    }

    /// Logits alone, for inference.
    pub fn forward(&self, images: &Matrix) -> Result<Matrix, NetworkError> {
        Ok(self.forward_train(images)?.0)
    }

    /// The predicted class of each image.
    pub fn predict(&self, images: &Matrix) -> Result<Vec<usize>, NetworkError> {
        let logits = self.forward(images)?;
        Ok((0..logits.rows)
            .map(|row| {
                logits
                    .row(row)
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(class, _)| class)
                    .unwrap_or(0)
            })
            .collect())
    }

    /// Mean over the patches of each image: `[rows, d_model]` in, `[batch,
    /// d_model]` out.
    fn mean_pool(&self, hidden: &Matrix) -> Matrix {
        let patches = self.config.num_patches();
        let scale = 1.0 / patches as f32;
        let mut pooled = Matrix::new(hidden.rows / patches, hidden.cols);
        for row in 0..hidden.rows {
            let target = pooled.row_mut(row / patches);
            for (sum, value) in target.iter_mut().zip(hidden.row(row)) {
                *sum += value * scale;
            }
        }
        pooled
    }

    /// Cross-entropy of the logits against `labels`, one class index per image.
    pub fn loss(
        &self,
        logits: &Matrix,
        labels: &[u32],
    ) -> Result<crate::CausalLmLoss, NetworkError> {
        if labels.len() != logits.rows {
            return Err(NetworkError::InvalidTarget {
                expected: logits.rows,
                actual: labels.len(),
            });
        }
        cross_entropy_rows(logits, logits.rows, |row| Some(labels[row]))
    }

    /// Accumulates gradients from `grad_logits`, which is what
    /// [`VisionTransformer::loss`] returns.
    pub fn backward(&mut self, cache: &VitCache, grad_logits: &Matrix) -> Result<(), NetworkError> {
        let grad_pooled = self.head.backward(&cache.pooled, grad_logits);

        // The mean sends the same gradient to every patch of its image.
        let patches = self.config.num_patches();
        let scale = 1.0 / patches as f32;
        let mut grad_normed = Matrix::new(cache.final_input.rows, cache.final_input.cols);
        for row in 0..grad_normed.rows {
            let source = grad_pooled.row(row / patches);
            for (target, value) in grad_normed.row_mut(row).iter_mut().zip(source) {
                *target = value * scale;
            }
        }

        let mut grad_hidden = self.norm.backward(&cache.final_input, &grad_normed);
        #[cfg(feature = "cuda")]
        if let Some(stack) = &cache.device {
            let context = self.device.clone().ok_or_else(|| {
                NetworkError::Cuda(
                    "the cache is from a device pass but the model is on the host".into(),
                )
            })?;
            grad_hidden =
                crate::gpu_model::stack_backward(&mut self.blocks, &context, stack, &grad_hidden)?;
            self.patch.backward(&cache.patches, &grad_hidden);
            return Ok(());
        }
        for (block, block_cache) in self.blocks.iter_mut().zip(&cache.blocks).rev() {
            grad_hidden = block.backward(block_cache, &grad_hidden)?;
        }
        // The gradient with respect to the pixels is not wanted: nothing
        // upstream of the patch projection has parameters.
        self.patch.backward(&cache.patches, &grad_hidden);
        Ok(())
    }

    /// Forward, loss, backward and one optimizer update. Returns the loss.
    pub fn train_step(&mut self, images: &Matrix, labels: &[u32]) -> Result<f32, NetworkError> {
        let (logits, cache) = self.forward_train(images)?;
        let loss = self.loss(&logits, labels)?;

        self.zero_grad();
        self.backward(&cache, &loss.grad_logits)?;
        self.step();

        Ok(loss.loss)
    }

    /// The loss over a batch, without gradients and without a step.
    pub fn evaluate(&self, images: &Matrix, labels: &[u32]) -> Result<f32, NetworkError> {
        let logits = self.forward(images)?;
        Ok(self.loss(&logits, labels)?.loss)
    }

    /// The fraction of `images` whose predicted class is the labelled one.
    pub fn accuracy(&self, images: &Matrix, labels: &[u32]) -> Result<f32, NetworkError> {
        let predictions = self.predict(images)?;
        if labels.len() != predictions.len() {
            return Err(NetworkError::InvalidTarget {
                expected: predictions.len(),
                actual: labels.len(),
            });
        }
        let correct = predictions
            .iter()
            .zip(labels)
            .filter(|(predicted, label)| **predicted as u32 == **label)
            .count();
        Ok(correct as f32 / predictions.len() as f32)
    }

    /// Trains for `epochs` passes over `dataset`, shuffled each time, calling
    /// `on_epoch` with the epoch index and its mean loss and stopping when it
    /// returns `false`.
    ///
    /// The dataset is what the image loaders produce: a flattened image per
    /// row, and a one-hot target whose hot column is the class. Both
    /// [`Dataset::from_idx`](crate::Dataset::from_idx) and
    /// `Dataset::from_image_folder` (`--features images`) write that shape, and
    /// the config has to agree with it — `image_size`, `channels` and
    /// `num_classes` are not read off the data.
    ///
    /// ```no_run
    /// # use rusting_brain::{Dataset, VisionTransformer, VitConfig};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut data = Dataset::from_idx("train-images-idx3-ubyte", "train-labels-idx1-ubyte")?;
    /// data.one_hot_targets(10)?;
    /// let mut model = VisionTransformer::new(VitConfig {
    ///     image_size: 28,
    ///     patch_size: 7,
    ///     channels: 1,
    ///     num_classes: 10,
    ///     ..VitConfig::default()
    /// })?;
    /// model.fit(&mut data, 32, 10, |epoch, loss| {
    ///     println!("epoch {epoch}: {loss:.4}");
    ///     true
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn fit(
        &mut self,
        dataset: &mut crate::dataset::Dataset,
        batch_size: usize,
        epochs: usize,
        mut on_epoch: impl FnMut(usize, f32) -> bool,
    ) -> Result<f32, NetworkError> {
        if batch_size == 0 {
            return Err(NetworkError::InvalidConfig(
                "a batch of no images trains nothing".into(),
            ));
        }
        if dataset.is_empty() {
            return Err(NetworkError::EmptyDataset);
        }

        let mut mean = 0.0;
        for epoch in 0..epochs {
            dataset.shuffle(None);
            let mut total = 0.0;
            let mut steps = 0;
            for batch in dataset.batches(batch_size) {
                let (images, labels) = Self::pack(batch.inputs, batch.targets)?;
                total += self.train_step(&images, &labels)?;
                steps += 1;
            }
            mean = total / steps.max(1) as f32;
            if !on_epoch(epoch, mean) {
                break;
            }
        }
        Ok(mean)
    }

    /// Packs dataset rows into the image matrix and class indices a step takes.
    ///
    /// The target is one-hot, as every dataset loader here writes it, so the
    /// class is the column holding the largest value.
    fn pack(inputs: &[Vec<f32>], targets: &[Vec<f32>]) -> Result<(Matrix, Vec<u32>), NetworkError> {
        let cols = inputs.first().map(Vec::len).unwrap_or(0);
        if inputs.iter().any(|row| row.len() != cols) {
            return Err(NetworkError::InvalidDataset(
                "the images in a batch are not all the same size".into(),
            ));
        }
        let images = Matrix::from_vec(inputs.len(), cols, inputs.concat());
        let labels = targets
            .iter()
            .map(|target| {
                target
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(class, _)| class as u32)
                    .ok_or_else(|| {
                        NetworkError::InvalidDataset("an image has no class at all".into())
                    })
            })
            .collect::<Result<Vec<u32>, NetworkError>>()?;
        Ok((images, labels))
    }

    pub fn zero_grad(&mut self) {
        crate::optimizers::zero_grad(&mut self.params_mut());
    }

    pub fn step(&mut self) {
        self.optimizer_step += 1;
        let step = self.optimizer_step;
        let optimizer = self.optimizer.clone();
        crate::optimizers::apply_step(&mut self.params_mut(), &optimizer, step, 1.0);
    }

    /// Moves the blocks' projections onto CUDA device `device`, where
    /// [`forward_train`](Self::forward_train) and [`backward`](Self::backward)
    /// then run the block stack. The patch projection, the norms, the pooling
    /// and the head stay on the host: they are one small matmul each, and the
    /// stack is the rest of the step.
    ///
    /// `mixed_precision` rounds the matmul operands to BF16, which is also what
    /// lets a 64-wide head take the fused attention kernel. Call
    /// [`sync_from_device`](Self::sync_from_device) before saving.
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&mut self, device: usize, mixed_precision: bool) -> Result<(), NetworkError> {
        let context = crate::gpu_transformer::GpuContext::with_precision(device, mixed_precision)?;
        for block in &mut self.blocks {
            for linear in block.linears_mut() {
                for param in linear.params_mut() {
                    param.move_to_cuda(&context)?;
                }
            }
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
    #[cfg(feature = "cuda")]
    pub fn sync_from_device(&mut self) -> Result<(), NetworkError> {
        for param in self.params_mut() {
            param.sync_from_device()?;
        }
        match &self.device {
            Some(context) => context.check(),
            None => Ok(()),
        }
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = vec![&mut self.patch.weight];
        for block in &mut self.blocks {
            params.extend(block.params_mut());
        }
        params.extend(self.norm.params_mut());
        params.push(&mut self.head.weight);
        params
    }

    pub fn num_parameters(&self) -> usize {
        self.patch.weight.len()
            + self
                .blocks
                .iter()
                .map(|block| block.num_parameters())
                .sum::<usize>()
            + self.norm.weight.len()
            + self.head.weight.len()
    }

    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), NetworkError> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        serde_json::to_writer(&mut writer, self)?;
        std::io::Write::flush(&mut writer)?;
        Ok(())
    }

    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, NetworkError> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> VitConfig {
        VitConfig {
            image_size: 8,
            patch_size: 4,
            channels: 1,
            d_model: 16,
            n_layers: 2,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            d_ff: 32,
            num_classes: 3,
            rmsnorm_eps: 1e-5,
            seed: 3,
        }
    }

    #[test]
    fn patches_read_the_square_they_cover() {
        let model = VisionTransformer::new(tiny()).unwrap();
        // One 8x8 grayscale image numbered 0..64 in reading order.
        let image = Matrix::from_vec(1, 64, (0..64).map(|value| value as f32).collect());
        let patches = model.patchify(&image).unwrap();

        assert_eq!(patches.rows, 4);
        assert_eq!(patches.cols, 16);
        // Top-left patch: the first four values of the first four image rows.
        assert_eq!(&patches.row(0)[..4], &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(&patches.row(0)[4..8], &[8.0, 9.0, 10.0, 11.0]);
        // Bottom-right patch starts at row 4, column 4.
        assert_eq!(&patches.row(3)[..4], &[36.0, 37.0, 38.0, 39.0]);
    }

    #[test]
    fn colour_planes_land_side_by_side_in_a_patch() {
        let config = VitConfig {
            channels: 3,
            ..tiny()
        };
        let model = VisionTransformer::new(config).unwrap();
        // Plane `c` is filled with `c`, so a patch is 16 zeros, 16 ones, 16 twos.
        let image = Matrix::from_vec(
            1,
            3 * 64,
            (0..3 * 64).map(|index| (index / 64) as f32).collect(),
        );
        let patches = model.patchify(&image).unwrap();

        assert_eq!(patches.cols, 48);
        assert!(patches.row(0)[..16].iter().all(|&value| value == 0.0));
        assert!(patches.row(0)[16..32].iter().all(|&value| value == 1.0));
        assert!(patches.row(0)[32..].iter().all(|&value| value == 2.0));
    }

    #[test]
    fn a_wrong_sized_image_is_refused() {
        let model = VisionTransformer::new(tiny()).unwrap();
        assert!(model.patchify(&Matrix::new(1, 63)).is_err());
        assert!(
            VisionTransformer::new(VitConfig {
                patch_size: 3,
                ..tiny()
            })
            .is_err()
        );
    }

    #[test]
    fn every_patch_reads_every_other_one() {
        let model = VisionTransformer::new(tiny()).unwrap();
        assert!(
            model
                .blocks
                .iter()
                .all(|block| !block.attention.is_causal())
        );

        // Changing the last patch changes the logits, which under a causal
        // mask and mean pooling it could only do for itself.
        let mut image = Matrix::from_vec(1, 64, (0..64).map(|value| value as f32 / 64.0).collect());
        let before = model.forward(&image).unwrap();
        for index in 0..64 {
            if index / 8 >= 4 && index % 8 >= 4 {
                image.data[index] = 1.0;
            }
        }
        let after = model.forward(&image).unwrap();
        assert!(
            before
                .data
                .iter()
                .zip(&after.data)
                .any(|(before, after)| (before - after).abs() > 1e-6)
        );
    }

    fn images(count: usize) -> (Matrix, Vec<u32>) {
        // Class 0 is a bright left half, class 1 a bright right half, class 2
        // a bright top half: separable, and only from the whole image.
        let mut data = vec![0.0; count * 64];
        let labels: Vec<u32> = (0..count).map(|index| index as u32 % 3).collect();
        for (image, &label) in labels.iter().enumerate() {
            let pixels = &mut data[image * 64..(image + 1) * 64];
            for row in 0..8 {
                for col in 0..8 {
                    let bright = match label {
                        0 => col < 4,
                        1 => col >= 4,
                        _ => row < 4,
                    };
                    pixels[row * 8 + col] = if bright { 1.0 } else { 0.0 };
                }
            }
        }
        (Matrix::from_vec(count, 64, data), labels)
    }

    #[test]
    fn training_separates_three_shapes() {
        let (images, labels) = images(9);
        let mut model = VisionTransformer::new(tiny())
            .unwrap()
            .with_optimizer(Optimizer::adam(3e-3));

        let first = model.train_step(&images, &labels).unwrap();
        for _ in 0..60 {
            model.train_step(&images, &labels).unwrap();
        }

        let last = model.evaluate(&images, &labels).unwrap();
        assert!(last < first * 0.5, "{first} -> {last}");
        assert_eq!(model.accuracy(&images, &labels).unwrap(), 1.0);
    }

    #[test]
    fn the_patch_gradient_matches_finite_differences() {
        let (images, labels) = images(2);
        let mut model = VisionTransformer::new(tiny()).unwrap();

        let (logits, cache) = model.forward_train(&images).unwrap();
        let loss = model.loss(&logits, &labels).unwrap();
        model.zero_grad();
        model.backward(&cache, &loss.grad_logits).unwrap();
        let analytic = model.patch.weight.grad.data.clone();

        let epsilon = 1e-3;
        for (index, &expected) in analytic.iter().enumerate() {
            let mut probe = model.clone();
            probe.patch.weight.value.data[index] += epsilon;
            let high = probe.evaluate(&images, &labels).unwrap();
            probe.patch.weight.value.data[index] -= 2.0 * epsilon;
            let low = probe.evaluate(&images, &labels).unwrap();
            let numeric = (high - low) / (2.0 * epsilon);
            assert!((expected - numeric).abs() < 1e-2, "index {index}");
        }
    }

    #[test]
    fn fit_runs_over_a_dataset_and_stops_when_the_callback_says_so() {
        let (images, labels) = images(6);
        let mut dataset = crate::dataset::Dataset::new(
            (0..6).map(|row| images.row(row).to_vec()).collect(),
            labels
                .iter()
                .map(|&label| {
                    let mut target = vec![0.0; 3];
                    target[label as usize] = 1.0;
                    target
                })
                .collect(),
        );
        let mut model = VisionTransformer::new(tiny())
            .unwrap()
            .with_optimizer(Optimizer::adam(3e-3));

        let mut epochs = 0;
        model
            .fit(&mut dataset, 3, 10, |_, _| {
                epochs += 1;
                epochs < 4
            })
            .unwrap();

        assert_eq!(epochs, 4);
        assert!(model.fit(&mut dataset, 0, 1, |_, _| true).is_err());
    }

    #[test]
    fn a_checkpoint_round_trips() {
        let (images, labels) = images(3);
        let mut model = VisionTransformer::new(tiny()).unwrap();
        model.train_step(&images, &labels).unwrap();

        let path = std::env::temp_dir().join(format!("rb_vit_{}.json", std::process::id()));
        model.save_json(&path).unwrap();
        let loaded = VisionTransformer::load_json(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.config, model.config);
        assert!(
            loaded
                .blocks
                .iter()
                .all(|block| !block.attention.is_causal())
        );
        assert_eq!(
            loaded.forward(&images).unwrap().data,
            model.forward(&images).unwrap().data
        );
    }

    /// One SGD step on the device against the same step on the host.
    ///
    /// 100 patches is more than one 64-wide attention tile and not a multiple
    /// of it, so the fused kernel's bidirectional loop and its ragged tail are
    /// both exercised. FP32 takes the three-kernel attention and measured
    /// 1.4e-6 from the host on logits up to 2.2; BF16 with a 64-wide head
    /// takes the fused kernels and measured 1.4e-2. A causal mask where the
    /// model has none puts the FP32 run 1.8 away.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_device_step_matches_the_host_or_skips_without_device() {
        if crate::gpu_transformer::GpuContext::new(0).is_err() {
            crate::cuda_training::skip_without_cuda();
            return;
        }
        let config = VitConfig {
            image_size: 40,
            patch_size: 4,
            channels: 1,
            d_model: 128,
            n_layers: 2,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 64,
            d_ff: 128,
            num_classes: 3,
            ..VitConfig::default()
        };
        let count = 3;
        let pixels: Vec<f32> = (0..count * config.image_len())
            .map(|index| ((index * 37 % 101) as f32 / 101.0) - 0.5)
            .collect();
        let images = Matrix::from_vec(count, config.image_len(), pixels);
        let labels = [0u32, 1, 2];
        let model = || {
            VisionTransformer::new(config)
                .unwrap()
                .with_optimizer(Optimizer::sgd(0.5))
        };

        let mut host = model();
        host.train_step(&images, &labels).unwrap();
        let expected = host.forward(&images).unwrap();

        for (mixed_precision, band) in [(false, 1e-4), (true, 3e-2)] {
            let mut device = model();
            device.to_cuda(0, mixed_precision).unwrap();
            device.train_step(&images, &labels).unwrap();
            let logits = device.forward(&images).unwrap();
            let error = logits
                .data
                .iter()
                .zip(&expected.data)
                .map(|(device, host)| (device - host).abs())
                .fold(0.0f32, f32::max);
            assert!(
                error < band,
                "mixed precision {mixed_precision}: the device is {error} from the host"
            );

            // The trained weights come back with the model.
            device.to_cpu().unwrap();
            let back = device.forward(&images).unwrap();
            let moved = back
                .data
                .iter()
                .zip(&expected.data)
                .map(|(device, host)| (device - host).abs())
                .fold(0.0f32, f32::max);
            assert!(
                moved < band,
                "after to_cpu the model is {moved} from the host"
            );
        }
    }
}
