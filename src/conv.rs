//! Convolutions and the layers that surround them in an image decoder.
//!
//! A diffusion model's latent becomes pixels through a stack that is almost
//! entirely convolutional: `Conv2d`, [`GroupNorm`], SiLU, nearest-neighbour
//! upsampling and, in the FLUX.2 line, a pixel shuffle. None of that existed
//! here, because a language model needs none of it.
//!
//! The convolution is lowered to the matrix multiply the crate already has.
//! `im2col` copies each patch the kernel will see into a row, so one
//! `[pixels, in_channels * kh * kw]` by `[out_channels, in_channels * kh * kw]`
//! multiply produces the whole output. That costs `kh * kw` times the memory of
//! the input for the copy, and buys the tuned `matrixmultiply` kernel and the
//! rayon threading that comes with it — a much better trade than a hand-written
//! seven-deep loop nest.
//!
//! These layers are inference only: they hold plain buffers rather than
//! [`crate::param::Param`]s and have no backward pass. Hosting a published
//! decoder is a forward pass; training one is a different project.
//!
//! NOTE: [`Conv2d`] has no backward pass and no CUDA. Both are additive —
//! `Param` and a device buffer can replace the plain `Matrix` without changing
//! the shapes — and neither is needed to run a checkpoint someone else trained.
//!
//! # Training a convolution
//!
//! [`TrainableConv2d`] is the other half: a batched convolution whose weight
//! and bias are [`Param`]s, with a backward pass and a
//! device path. It is a separate type rather than a mode of [`Conv2d`] because
//! the two want different layouts. A published checkpoint stores
//! `[out, in, kh, kw]`, which flattens to a column ordered channel-slowest, and
//! that is what [`Conv2d`] reads. A trained weight is only ever read by the
//! code that wrote it, so [`TrainableConv2d`] orders its columns
//! channel-fastest over an [`ImageBatch`] of `[batch * height * width,
//! channels]` rows, which is the layout the rest of an image training stack
//! already holds its activations in and which makes im2col one contiguous copy
//! per pixel.
//!
//! The decision behind the device path, since the alternative was to publish
//! `im2col`/`col2im` for an outside crate to drive: a convolution's column
//! matrix is `kernel * kernel` times its input — 452 MiB for one 3x3 layer over
//! 96 channels of 256x256 at batch 2 — and a caller that drove the pair itself
//! would still have to hand the columns to the GEMM across the bus. Owning both
//! sides here means the columns are built on the device and consumed there, and
//! only the activation and its gradient cross, which is the nine-fold
//! difference the layer's cost is made of.

use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::param::{Linear, Param};
#[cfg(feature = "cuda")]
use cudarc::driver::PushKernelArg;
use rand::rngs::StdRng;
use rayon::prelude::*;

/// An image-shaped buffer: channels, then rows, then columns, which is the
/// layout every published checkpoint stores and every convolution reads.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureMap {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

impl FeatureMap {
    /// A zeroed map of the given shape.
    pub fn new(channels: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            height,
            width,
            data: vec![0.0; channels * height * width],
        }
    }

    /// Wraps data that is already in channel-height-width order.
    pub fn from_vec(
        channels: usize,
        height: usize,
        width: usize,
        data: Vec<f32>,
    ) -> Result<Self, NetworkError> {
        if data.len() != channels * height * width {
            return Err(NetworkError::InvalidTarget {
                expected: channels * height * width,
                actual: data.len(),
            });
        }
        Ok(Self {
            channels,
            height,
            width,
            data,
        })
    }

    /// Pixels per channel.
    pub fn pixels(&self) -> usize {
        self.height * self.width
    }

    /// One channel's plane.
    pub fn plane(&self, channel: usize) -> &[f32] {
        let pixels = self.pixels();
        &self.data[channel * pixels..(channel + 1) * pixels]
    }

    /// The map as a `[pixels, channels]` matrix, which is the shape attention
    /// and a linear layer want.
    pub fn to_tokens(&self) -> Matrix {
        let pixels = self.pixels();
        let mut tokens = Matrix::new(pixels, self.channels);
        for channel in 0..self.channels {
            let plane = self.plane(channel);
            for (pixel, value) in plane.iter().enumerate() {
                tokens.data[pixel * self.channels + channel] = *value;
            }
        }
        tokens
    }

    /// The reverse of [`FeatureMap::to_tokens`].
    pub fn from_tokens(tokens: &Matrix, height: usize, width: usize) -> Result<Self, NetworkError> {
        if tokens.rows != height * width {
            return Err(NetworkError::InvalidTarget {
                expected: height * width,
                actual: tokens.rows,
            });
        }
        let mut map = Self::new(tokens.cols, height, width);
        for channel in 0..tokens.cols {
            for pixel in 0..tokens.rows {
                map.data[channel * tokens.rows + pixel] =
                    tokens.data[pixel * tokens.cols + channel];
            }
        }
        Ok(map)
    }
}

/// A linear map applied to every row of a `[tokens, in]` matrix, with the bias
/// a checkpoint's layer usually carries.
///
/// [`crate::param::Linear`] is the trainable, bias-free version. This one holds
/// plain buffers, because a layer read out of someone else's checkpoint is only
/// ever run forward.
#[derive(Clone, Debug)]
pub struct Dense {
    pub weight: Matrix,
    pub bias: Option<Vec<f32>>,
    /// The same weight at one byte per value, once [`Dense::quantize`] has
    /// been called. The `f32` copy is dropped then, which is the point.
    quantized: Option<crate::quantized::Quantized>,
}

impl Dense {
    /// Wraps an `[out, in]` weight and its bias.
    pub fn new(weight: Matrix, bias: Option<Vec<f32>>) -> Result<Self, NetworkError> {
        if bias.as_ref().is_some_and(|bias| bias.len() != weight.rows) {
            return Err(NetworkError::InvalidTarget {
                expected: weight.rows,
                actual: bias.as_ref().map_or(0, |bias| bias.len()),
            });
        }
        Ok(Self {
            weight,
            bias,
            quantized: None,
        })
    }

    /// Output width.
    pub fn out_dim(&self) -> usize {
        self.weight.rows
    }

    /// Input width.
    pub fn in_dim(&self) -> usize {
        self.weight.cols
    }

    /// Stores the weight as one byte per value and drops the `f32` copy.
    ///
    /// A quarter of the memory, for about 0.4% relative error per weight. The
    /// shape is kept, so [`Dense::in_dim`] and [`Dense::out_dim`] still answer.
    /// There is no way back: the `f32` values are gone.
    pub fn quantize(&mut self) {
        if self.quantized.is_none() {
            self.quantized = Some(crate::quantized::Quantized::from_matrix(&self.weight));
            self.weight.data = Vec::new();
        }
    }

    /// The weight as `f32`, widening it first if it is stored quantized.
    ///
    /// Borrowed in the common case, so a float layer costs nothing to ask.
    pub(crate) fn weight_f32(&self) -> std::borrow::Cow<'_, Matrix> {
        match &self.quantized {
            Some(quantized) => std::borrow::Cow::Owned(quantized.dequantize()),
            None => std::borrow::Cow::Borrowed(&self.weight),
        }
    }

    /// Whether the weight is stored quantized.
    pub fn is_quantized(&self) -> bool {
        self.quantized.is_some()
    }

    /// Applies the map to every row.
    pub fn forward(&self, tokens: &Matrix) -> Result<Matrix, NetworkError> {
        if tokens.cols != self.weight.cols {
            return Err(NetworkError::InvalidTarget {
                expected: self.weight.cols,
                actual: tokens.cols,
            });
        }
        let mut output = match &self.quantized {
            Some(weight) => weight.matmul_rhs_transposed(tokens),
            None => {
                let mut output = Matrix::new(tokens.rows, self.weight.rows);
                tokens.dot_rhs_transposed(&self.weight, &mut output);
                output
            }
        };
        if let Some(bias) = &self.bias {
            output
                .data
                .par_chunks_mut(self.weight.rows)
                .for_each(|row| {
                    for (value, bias) in row.iter_mut().zip(bias) {
                        *value += bias;
                    }
                });
        }
        Ok(output)
    }

    /// Applies the map to a single vector.
    pub fn apply(&self, input: &[f32]) -> Result<Vec<f32>, NetworkError> {
        let row = Matrix::from_vec(1, input.len(), input.to_vec());
        Ok(self.forward(&row)?.data)
    }
}

/// A two-dimensional convolution with zero padding.
#[derive(Clone, Debug)]
pub struct Conv2d {
    /// `[out_channels, in_channels * kernel * kernel]`, the layout a
    /// checkpoint's `[out, in, kh, kw]` tensor already has once flattened.
    pub weight: Matrix,
    pub bias: Option<Vec<f32>>,
    pub in_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
}

impl Conv2d {
    /// Wraps weights read from a checkpoint.
    ///
    /// `weight` is the `[out_channels, in_channels * kernel * kernel]` matrix a
    /// `[out, in, kh, kw]` tensor flattens to, which is what
    /// [`crate::safetensors::SafeTensors::tensor`] hands back.
    pub fn new(
        weight: Matrix,
        bias: Option<Vec<f32>>,
        in_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self, NetworkError> {
        if kernel == 0 || stride == 0 {
            return Err(NetworkError::InvalidConfig(
                "a convolution needs a kernel and a stride of at least one".into(),
            ));
        }
        if weight.cols != in_channels * kernel * kernel {
            return Err(NetworkError::InvalidConfig(format!(
                "a {kernel}x{kernel} convolution over {in_channels} channels needs {} weights per \
                 output channel, and this one has {}",
                in_channels * kernel * kernel,
                weight.cols
            )));
        }
        if bias.as_ref().is_some_and(|bias| bias.len() != weight.rows) {
            return Err(NetworkError::InvalidConfig(
                "a convolution's bias has one value per output channel".into(),
            ));
        }
        Ok(Self {
            weight,
            bias,
            in_channels,
            kernel,
            stride,
            padding,
        })
    }

    /// Output channels.
    pub fn out_channels(&self) -> usize {
        self.weight.rows
    }

    /// The output shape for an input of the given size.
    pub fn output_size(&self, height: usize, width: usize) -> (usize, usize) {
        // A kernel wider than the padded input fits nowhere, which is zero
        // outputs rather than one built out of padding.
        let size = |length: usize| match (length + 2 * self.padding).checked_sub(self.kernel) {
            Some(span) => span / self.stride + 1,
            None => 0,
        };
        (size(height), size(width))
    }

    /// Convolves `input`.
    pub fn forward(&self, input: &FeatureMap) -> Result<FeatureMap, NetworkError> {
        if input.channels != self.in_channels {
            return Err(NetworkError::InvalidConfig(format!(
                "this convolution reads {} channels and was handed {}",
                self.in_channels, input.channels
            )));
        }
        let (out_height, out_width) = self.output_size(input.height, input.width);
        if out_height == 0 || out_width == 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "a {}x{} kernel over a {}x{} input leaves nothing",
                self.kernel, self.kernel, input.height, input.width
            )));
        }

        let columns = self.im2col(input, out_height, out_width);
        let mut product = Matrix::new(out_height * out_width, self.out_channels());
        columns.dot_rhs_transposed(&self.weight, &mut product);

        // Back to channel-major, adding the bias on the way.
        let mut output = FeatureMap::new(self.out_channels(), out_height, out_width);
        let pixels = out_height * out_width;
        output
            .data
            .par_chunks_mut(pixels)
            .enumerate()
            .for_each(|(channel, plane)| {
                let bias = self.bias.as_ref().map_or(0.0, |bias| bias[channel]);
                for (pixel, value) in plane.iter_mut().enumerate() {
                    *value = product.data[pixel * self.out_channels() + channel] + bias;
                }
            });
        Ok(output)
    }

    /// One row per output pixel, holding every input value that pixel's kernel
    /// reads. Padding shows up as the zeros the row was created with.
    fn im2col(&self, input: &FeatureMap, out_height: usize, out_width: usize) -> Matrix {
        let patch = self.in_channels * self.kernel * self.kernel;
        let mut columns = Matrix::new(out_height * out_width, patch);
        columns
            .data
            .par_chunks_mut(patch)
            .enumerate()
            .for_each(|(pixel, row)| {
                let (out_y, out_x) = (pixel / out_width, pixel % out_width);
                let top = (out_y * self.stride) as isize - self.padding as isize;
                let left = (out_x * self.stride) as isize - self.padding as isize;
                for channel in 0..self.in_channels {
                    let plane = channel * input.height * input.width;
                    for row_offset in 0..self.kernel {
                        let y = top + row_offset as isize;
                        if y < 0 || y >= input.height as isize {
                            continue;
                        }
                        let source = plane + y as usize * input.width;
                        let target = (channel * self.kernel + row_offset) * self.kernel;
                        for column_offset in 0..self.kernel {
                            let x = left + column_offset as isize;
                            if x >= 0 && x < input.width as isize {
                                row[target + column_offset] = input.data[source + x as usize];
                            }
                        }
                    }
                }
            });
        columns
    }
}

/// A batch of image-shaped activations, one row per pixel.
///
/// `[batch * height * width, channels]`, which is what makes a convolution a
/// matrix multiply and an activation function a pass over a slice. [`Conv2d`]'s
/// [`FeatureMap`] is the other arrangement, channel-major and unbatched,
/// because that is the one a published checkpoint's decoder is written against.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageBatch {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
    pub tokens: Matrix,
}

impl ImageBatch {
    pub fn new(batch: usize, height: usize, width: usize, channels: usize) -> Self {
        Self {
            batch,
            height,
            width,
            tokens: Matrix::new(batch * height * width, channels),
        }
    }

    /// Wraps rows that are already one pixel each.
    pub fn from_tokens(batch: usize, height: usize, width: usize, tokens: Matrix) -> Self {
        debug_assert_eq!(tokens.rows, batch * height * width);
        Self {
            batch,
            height,
            width,
            tokens,
        }
    }

    pub fn channels(&self) -> usize {
        self.tokens.cols
    }

    /// Pixels in one sample.
    pub fn pixels(&self) -> usize {
        self.height * self.width
    }

    /// A zeroed map of the same size with a different channel count.
    pub fn like(&self, channels: usize) -> Self {
        Self::new(self.batch, self.height, self.width, channels)
    }

    /// One pixel's channels.
    pub fn pixel(&self, sample: usize, y: usize, x: usize) -> &[f32] {
        self.tokens.row((sample * self.height + y) * self.width + x)
    }

    pub fn pixel_mut(&mut self, sample: usize, y: usize, x: usize) -> &mut [f32] {
        let row = (sample * self.height + y) * self.width + x;
        self.tokens.row_mut(row)
    }

    /// Adds another map of the same shape in place.
    pub fn add(&mut self, other: &ImageBatch) {
        debug_assert_eq!(self.tokens.data.len(), other.tokens.data.len());
        for (slot, value) in self.tokens.data.iter_mut().zip(&other.tokens.data) {
            *slot += value;
        }
    }
}

/// An [`ImageBatch`] held on a device, in the same layout, so a chain of
/// device layers passes activations without crossing the bus. Made by
/// [`crate::param::CudaDevice::upload`].
#[cfg(feature = "cuda")]
pub struct DeviceImageBatch {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
    channels: usize,
    pub(crate) data: cudarc::driver::CudaSlice<f32>,
    context: std::sync::Arc<crate::gpu_transformer::GpuContext>,
}

#[cfg(feature = "cuda")]
impl DeviceImageBatch {
    pub(crate) fn upload(
        context: &std::sync::Arc<crate::gpu_transformer::GpuContext>,
        batch: &ImageBatch,
    ) -> Result<Self, NetworkError> {
        Ok(Self {
            batch: batch.batch,
            height: batch.height,
            width: batch.width,
            channels: batch.channels(),
            data: context.upload(&batch.tokens)?,
            context: context.clone(),
        })
    }

    /// Copies the batch back to the host.
    pub fn download(&self) -> Result<ImageBatch, NetworkError> {
        let mut tokens = Matrix::new(self.batch * self.pixels(), self.channels);
        self.context.download(&self.data, &mut tokens)?;
        Ok(ImageBatch::from_tokens(
            self.batch,
            self.height,
            self.width,
            tokens,
        ))
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn pixels(&self) -> usize {
        self.height * self.width
    }

    /// Zeros in this batch's shape with `channels` channels.
    pub fn like(&self, channels: usize) -> Result<Self, NetworkError> {
        let data = self.context.zeros(self.batch * self.pixels() * channels)?;
        Ok(Self {
            channels,
            ..self.with_data(data)
        })
    }

    /// Another buffer of this shape.
    fn with_data(&self, data: cudarc::driver::CudaSlice<f32>) -> Self {
        Self {
            data,
            context: self.context.clone(),
            ..*self
        }
    }
}

#[cfg(feature = "cuda")]
impl Clone for DeviceImageBatch {
    fn clone(&self) -> Self {
        self.with_data(self.data.clone())
    }
}

#[cfg(feature = "cuda")]
impl crate::param::CudaDevice {
    /// Puts a batch on this device for [`TrainableConv2d::forward_device`].
    pub fn upload(&self, batch: &ImageBatch) -> Result<DeviceImageBatch, NetworkError> {
        DeviceImageBatch::upload(self.context(), batch)
    }

    /// [`DeviceImageBatch::download`].
    pub fn download(&self, batch: &DeviceImageBatch) -> Result<ImageBatch, NetworkError> {
        batch.download()
    }
}

/// Everything `im2col` and `col2im` need to know about one convolution, which
/// is what the device kernels take as their arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvGeometry {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub out_height: usize,
    pub out_width: usize,
}

impl ConvGeometry {
    /// A 1x1 kernel at stride 1 without padding: each column row is the
    /// input pixel itself.
    pub fn is_pointwise(&self) -> bool {
        self.kernel == 1 && self.stride == 1 && self.padding == 0
    }

    /// Values in the input map.
    pub fn input_elements(&self) -> usize {
        self.batch * self.height * self.width * self.channels
    }

    /// Values in the column matrix: `kernel * kernel` times the output's
    /// pixels times the input's channels.
    pub fn column_elements(&self) -> usize {
        self.batch * self.out_height * self.out_width * self.patch()
    }

    /// Values in one column row.
    pub fn patch(&self) -> usize {
        self.channels * self.kernel * self.kernel
    }

    /// Output pixels across the whole batch, which is the column matrix's row
    /// count and the GEMM's.
    pub fn rows(&self) -> usize {
        self.batch * self.out_height * self.out_width
    }
}

/// A two-dimensional convolution that trains.
///
/// The weight is a [`Linear`] over `im2col` columns, so the matrix multiply,
/// the weight gradient and the input gradient are the ones the crate already
/// has, on the host and on a device alike. See the module documentation for why
/// this is not a mode of [`Conv2d`].
#[derive(Clone, Debug)]
pub struct TrainableConv2d {
    /// `[out_channels, in_channels * kernel * kernel]`.
    pub weight: Linear,
    /// `[1, out_channels]`. Zero and unused after
    /// [`TrainableConv2d::without_bias`].
    pub bias: Param,
    has_bias: bool,
    in_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    /// Whether the backward pass rebuilds the columns instead of the forward
    /// pass keeping them. See [`TrainableConv2d::set_recompute_columns`].
    recompute_columns: bool,
}

/// What [`TrainableConv2d::backward`] needs from the forward pass.
///
/// The columns are the largest thing a convolution holds, so the cache keeps
/// them wherever the forward pass built them: on the host, or on the device,
/// where they never have to cross the bus at all. Under
/// [`TrainableConv2d::set_recompute_columns`] it keeps the input instead and
/// the backward pass builds the columns again.
pub struct ConvCache {
    columns: Columns,
    shape: ConvGeometry,
}

impl ConvCache {
    /// Whether this cache holds the columns themselves rather than the input
    /// they were built from, which is the difference between `kernel * kernel`
    /// values per input value and one.
    pub fn holds_columns(&self) -> bool {
        match &self.columns {
            Columns::Input(_) => false,
            #[cfg(feature = "cuda")]
            Columns::DeviceInput(_) => false,
            _ => true,
        }
    }
}

enum Columns {
    Host(Matrix),
    #[cfg(feature = "cuda")]
    Device(cudarc::driver::CudaSlice<f32>),
    /// The forward input, kept in place of the columns it built.
    Input(ImageBatch),
    /// The forward input, left on the device where it was already uploaded, so
    /// rebuilding the columns costs a kernel and not a second crossing.
    #[cfg(feature = "cuda")]
    DeviceInput(cudarc::driver::CudaSlice<f32>),
    /// Device columns as bf16, from a mixed-precision context.
    #[cfg(feature = "cuda")]
    DeviceNarrow(cudarc::driver::CudaSlice<u8>),
}

impl std::fmt::Debug for ConvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvCache")
            .field("shape", &self.shape)
            .finish()
    }
}

impl TrainableConv2d {
    /// A `kernel x kernel` convolution, padded to keep the resolution when
    /// `stride` is one.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        rng: &mut StdRng,
    ) -> Result<Self, NetworkError> {
        if kernel == 0 || stride == 0 || in_channels == 0 || out_channels == 0 {
            return Err(NetworkError::InvalidConfig(
                "a convolution needs a non-zero kernel, stride and channel count".into(),
            ));
        }
        let fan_in = in_channels * kernel * kernel;
        Ok(Self {
            weight: Linear {
                weight: Param::he_uniform(out_channels, fan_in, fan_in, rng),
                lora: None,
            },
            bias: Param::zeros(1, out_channels),
            has_bias: true,
            in_channels,
            kernel,
            stride,
            padding: kernel / 2,
            recompute_columns: false,
        })
    }

    /// [`TrainableConv2d::new`] with the weight set to zero, which is how a
    /// residual branch's last convolution starts so the block begins as the
    /// identity.
    pub fn zeroed(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        rng: &mut StdRng,
    ) -> Result<Self, NetworkError> {
        let mut conv = Self::new(in_channels, out_channels, kernel, 1, rng)?;
        conv.weight.weight = Param::zeros(out_channels, in_channels * kernel * kernel);
        Ok(conv)
    }

    /// Drops the bias: the forward pass adds none, the backward pass gathers
    /// none, and [`TrainableConv2d::params_mut`] leaves it out. For a
    /// convolution before a norm, whose mean subtraction would cancel it.
    pub fn without_bias(mut self) -> Self {
        self.has_bias = false;
        self.bias = Param::zeros(1, self.bias.value.cols);
        self
    }

    pub fn has_bias(&self) -> bool {
        self.has_bias
    }

    pub fn out_channels(&self) -> usize {
        self.bias.value.cols
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn output_size(&self, height: usize, width: usize) -> (usize, usize) {
        let size = |length: usize| {
            (length + 2 * self.padding).saturating_sub(self.kernel) / self.stride + 1
        };
        (size(height), size(width))
    }

    /// The geometry of this convolution over an input of the given size.
    pub fn geometry(&self, batch: usize, height: usize, width: usize) -> ConvGeometry {
        let (out_height, out_width) = self.output_size(height, width);
        ConvGeometry {
            batch,
            channels: self.in_channels,
            height,
            width,
            kernel: self.kernel,
            stride: self.stride,
            padding: self.padding,
            out_height,
            out_width,
        }
    }

    /// Trades one `im2col` pass per backward for the memory the columns take.
    ///
    /// One convolution's columns are
    ///
    /// ```text
    /// batch * out_height * out_width * in_channels * kernel * kernel * 4 bytes
    /// ```
    ///
    /// which is `kernel * kernel` times its input, and they stay live from the
    /// forward pass until the backward one. A single convolution should keep
    /// them: the memory is paid once and the work is saved. A network should
    /// weigh it layer by layer, because every layer's columns are alive at
    /// once, and the total is what bounds the batch. Put the batch, the
    /// resolution and the channel count of the widest few layers through the
    /// formula above before deciding; in a deep net the sum is usually larger
    /// than the weights and the optimizer moments together.
    ///
    /// What it costs is one extra `im2col` per backward pass. On a device the
    /// input stays on the card, so nothing crosses the bus twice. Worked
    /// example, on an RTX 3060: a 3x3 convolution from 192 channels to 96 over
    /// twelve 64x64 images measures 33.5 ms per forward and backward with the
    /// columns kept and 41.0 ms with them rebuilt, and the formula gives the
    /// 324 MiB it stops holding against the 36 MiB of input it holds instead.
    ///
    /// The columns are only needed for the weight gradient at all; the input
    /// gradient needs the weight and the upstream gradient alone.
    pub fn set_recompute_columns(&mut self, recompute: bool) {
        self.recompute_columns = recompute;
    }

    /// Whether the backward pass rebuilds the columns.
    pub fn recomputes_columns(&self) -> bool {
        self.recompute_columns
    }

    /// Moves the weight and the bias onto a device a caller already opened.
    /// Every later [`TrainableConv2d::forward`] and
    /// [`TrainableConv2d::backward`] then runs entirely there, and only the
    /// activation and its gradient cross the bus; `forward_device` and
    /// `backward_device` skip even that.
    pub fn to_cuda_on(&mut self, device: &crate::param::CudaDevice) -> Result<(), NetworkError> {
        self.weight.to_cuda_on(device)?;
        if self.has_bias {
            self.bias.to_cuda_on(device)?;
        }
        Ok(())
    }

    /// Brings the weight and the bias back to the host.
    pub fn to_cpu(&mut self) -> Result<(), NetworkError> {
        self.weight.to_cpu()?;
        self.bias.to_cpu()
    }

    pub fn forward(&self, input: &ImageBatch) -> Result<(ImageBatch, ConvCache), NetworkError> {
        if input.channels() != self.in_channels {
            return Err(NetworkError::InvalidConfig(format!(
                "this convolution reads {} channels and was given {}",
                self.in_channels,
                input.channels()
            )));
        }
        let shape = self.geometry(input.batch, input.height, input.width);
        #[cfg(feature = "cuda")]
        if let Some(device) = &self.weight.weight.device {
            return self.forward_on_device(input, device);
        }
        let columns = self.im2col(input, &shape);
        let mut tokens = self.weight.forward(&columns);
        self.add_bias(&mut tokens);
        Ok((
            ImageBatch::from_tokens(input.batch, shape.out_height, shape.out_width, tokens),
            ConvCache {
                columns: match self.recompute_columns {
                    true => Columns::Input(input.clone()),
                    false => Columns::Host(columns),
                },
                shape,
            },
        ))
    }

    /// Accumulates the weight and bias gradients and returns `dL/dinput`.
    pub fn backward(
        &mut self,
        cache: &ConvCache,
        grad_output: &ImageBatch,
    ) -> Result<ImageBatch, NetworkError> {
        match &cache.columns {
            Columns::Host(columns) => {
                self.accumulate_bias_grad(&grad_output.tokens);
                let grad_columns = self.weight.backward(columns, &grad_output.tokens);
                Ok(self.col2im(&grad_columns, &cache.shape))
            }
            #[cfg(feature = "cuda")]
            Columns::Device(_) | Columns::DeviceInput(_) | Columns::DeviceNarrow(_) => {
                self.backward_on_device(cache, grad_output)
            }
            Columns::Input(input) => {
                self.accumulate_bias_grad(&grad_output.tokens);
                let columns = self.im2col(input, &cache.shape);
                let grad_columns = self.weight.backward(&columns, &grad_output.tokens);
                Ok(self.col2im(&grad_columns, &cache.shape))
            }
        }
    }

    /// [`TrainableConv2d::forward_device`] for inference: no cache, SiLU under
    /// `silu` fused into the bias, and a 1x1 stride-1 kernel reads the input
    /// as its columns instead of copying it into them.
    #[cfg(feature = "cuda")]
    pub fn infer_device(
        &self,
        input: &DeviceImageBatch,
        silu: bool,
    ) -> Result<DeviceImageBatch, NetworkError> {
        self.infer_device_with(input, silu, None)
    }

    /// [`TrainableConv2d::infer_device`] with the weight also given as bf16
    /// bytes, from [`TrainableConv2d::bf16_weight`]: a convolution that needs
    /// `im2col` then builds its columns as bf16 too, half the bytes, and runs
    /// a bf16 GEMM with an FP32 accumulator.
    #[cfg(feature = "cuda")]
    pub(crate) fn infer_device_with(
        &self,
        input: &DeviceImageBatch,
        silu: bool,
        narrow_weight: Option<&cudarc::driver::CudaSlice<u8>>,
    ) -> Result<DeviceImageBatch, NetworkError> {
        if input.channels != self.in_channels {
            return Err(NetworkError::InvalidConfig(format!(
                "this convolution reads {} channels and was given {}",
                self.in_channels, input.channels
            )));
        }
        let device = self.weight.weight.device.as_ref().ok_or_else(|| {
            NetworkError::InvalidConfig("infer_device needs the weight on a device".into())
        })?;
        let shape = self.geometry(input.batch, input.height, input.width);
        let context = device.context();
        let mut product = match ((self.kernel, self.stride, self.padding), narrow_weight) {
            ((1, 1, 0), _) => device.matmul_rhs_transposed_device(&input.data, shape.rows())?,
            (_, Some(weight)) => device.matmul_rhs_transposed_bf16(
                &context.im2col_bf16(&input.data, &shape)?,
                weight,
                shape.rows(),
            )?,
            _ => device.matmul_rhs_transposed_device(
                &context.im2col(&input.data, &shape)?,
                shape.rows(),
            )?,
        };
        let bias = self
            .bias
            .device
            .as_ref()
            .filter(|_| self.has_bias)
            .ok_or_else(|| {
                NetworkError::InvalidConfig("infer_device needs a bias on the device".into())
            })?;
        context.add_bias_rows_then(&mut product, bias.value(), self.out_channels(), silu)?;
        Ok(DeviceImageBatch {
            batch: input.batch,
            height: shape.out_height,
            width: shape.out_width,
            channels: self.out_channels(),
            data: product,
            context: context.clone(),
        })
    }

    /// The device weight as bf16 bytes for [`TrainableConv2d::infer_device_with`],
    /// when its context runs mixed precision and this convolution needs
    /// `im2col`; `None` otherwise.
    #[cfg(feature = "cuda")]
    pub(crate) fn bf16_weight(
        &self,
    ) -> Result<Option<cudarc::driver::CudaSlice<u8>>, NetworkError> {
        let Some(device) = self.weight.weight.device.as_ref() else {
            return Ok(None);
        };
        let context = device.context();
        match context.mixed_precision && (self.kernel, self.stride, self.padding) != (1, 1, 0) {
            true => context.bf16_bytes(device.value()).map(Some),
            false => Ok(None),
        }
    }

    /// The forward pass for an input that is already on the device, leaving
    /// the output there too. Needs [`TrainableConv2d::to_cuda_on`] first.
    #[cfg(feature = "cuda")]
    pub fn forward_device(
        &self,
        input: &DeviceImageBatch,
    ) -> Result<(DeviceImageBatch, ConvCache), NetworkError> {
        if input.channels != self.in_channels {
            return Err(NetworkError::InvalidConfig(format!(
                "this convolution reads {} channels and was given {}",
                self.in_channels, input.channels
            )));
        }
        let device = self.weight.weight.device.as_ref().ok_or_else(|| {
            NetworkError::InvalidConfig("forward_device needs the weight on a device".into())
        })?;
        let shape = self.geometry(input.batch, input.height, input.width);
        let context = device.context();
        // A 1x1 stride-1 convolution's columns are its input; a mixed-precision
        // context builds the others as bf16 for a bf16 GEMM.
        let (mut product, columns) = match (shape.is_pointwise(), context.mixed_precision) {
            (true, _) => (
                device.matmul_rhs_transposed_device(&input.data, shape.rows())?,
                None,
            ),
            (false, true) => {
                let columns = context.im2col_bf16(&input.data, &shape)?;
                let weight = context.bf16_bytes(device.value())?;
                (
                    device.matmul_rhs_transposed_bf16(&columns, &weight, shape.rows())?,
                    Some(Columns::DeviceNarrow(columns)),
                )
            }
            (false, false) => {
                let columns = context.im2col(&input.data, &shape)?;
                (
                    device.matmul_rhs_transposed_device(&columns, shape.rows())?,
                    Some(Columns::Device(columns)),
                )
            }
        };
        if let Some(bias) = self.bias.device.as_ref().filter(|_| self.has_bias) {
            context.add_bias_rows(&mut product, bias.value(), self.out_channels())?;
        }
        Ok((
            DeviceImageBatch {
                batch: input.batch,
                height: shape.out_height,
                width: shape.out_width,
                channels: self.out_channels(),
                data: product,
                context: context.clone(),
            },
            ConvCache {
                // Dropping the device columns here is the whole point of the
                // option: they are freed before the next layer allocates its
                // own, so a deep network holds one layer's worth rather than
                // every layer's.
                columns: match columns.filter(|_| !self.recompute_columns) {
                    Some(columns) => columns,
                    None => Columns::DeviceInput(input.data.clone()),
                },
                shape,
            },
        ))
    }

    /// The backward pass for a gradient that is already on the device, for a
    /// cache [`TrainableConv2d::forward_device`] (or a device-resident
    /// [`TrainableConv2d::forward`]) built. Accumulates the weight and bias
    /// gradients on the device and returns `dL/dinput` there.
    #[cfg(feature = "cuda")]
    pub fn backward_device(
        &mut self,
        cache: &ConvCache,
        grad_output: &DeviceImageBatch,
    ) -> Result<DeviceImageBatch, NetworkError> {
        let shape = &cache.shape;
        let device = self.weight.weight.device.as_mut().ok_or_else(|| {
            NetworkError::InvalidConfig("backward_device needs the weight on a device".into())
        })?;
        let context = device.context().clone();
        if context.mixed_precision && !shape.is_pointwise() {
            let rebuilt;
            let columns = match &cache.columns {
                Columns::DeviceNarrow(columns) => columns,
                Columns::DeviceInput(input) => {
                    rebuilt = context.im2col_bf16(input, shape)?;
                    &rebuilt
                }
                _ => {
                    return Err(NetworkError::InvalidConfig(
                        "backward_device needs a cache from a mixed-precision forward pass".into(),
                    ));
                }
            };
            let grad = context.bf16_bytes(&grad_output.data)?;
            device.accumulate_grad_bf16(&grad, columns, shape.rows())?;
            let weight = context.bf16_bytes(device.value())?;
            let grad_columns = device.matmul_bf16(&grad, &weight, shape.rows())?;
            self.accumulate_bias_device(&context, grad_output, shape.rows())?;
            return Ok(DeviceImageBatch {
                batch: shape.batch,
                height: shape.height,
                width: shape.width,
                channels: shape.channels,
                data: context.col2im_bf16(&grad_columns, shape)?,
                context,
            });
        }
        let rebuilt;
        let columns = match &cache.columns {
            Columns::Device(columns) => columns,
            Columns::DeviceInput(input) if shape.is_pointwise() => input,
            Columns::DeviceInput(input) => {
                rebuilt = context.im2col(input, shape)?;
                &rebuilt
            }
            _ => {
                return Err(NetworkError::InvalidConfig(
                    "backward_device needs a cache from a device forward pass".into(),
                ));
            }
        };
        device.accumulate_grad_device(&grad_output.data, columns, shape.rows())?;
        let grad_columns = device.matmul_device(&grad_output.data, shape.rows())?;
        self.accumulate_bias_device(&context, grad_output, shape.rows())?;
        Ok(DeviceImageBatch {
            batch: shape.batch,
            height: shape.height,
            width: shape.width,
            channels: shape.channels,
            data: match shape.is_pointwise() {
                true => grad_columns,
                false => context.col2im(&grad_columns, shape)?,
            },
            context,
        })
    }

    #[cfg(feature = "cuda")]
    fn accumulate_bias_device(
        &mut self,
        context: &std::sync::Arc<crate::gpu_transformer::GpuContext>,
        grad_output: &DeviceImageBatch,
        rows: usize,
    ) -> Result<(), NetworkError> {
        if let Some(bias) = self.bias.device.as_mut().filter(|_| self.has_bias) {
            // The column sum as a GEMM against ones, which keeps the negated
            // accumulate convention in one place.
            let ones = context.filled(rows, 1.0)?;
            bias.accumulate_grad_device(&ones, &grad_output.data, rows)?;
        }
        Ok(())
    }

    /// [`TrainableConv2d::forward_device`] between an upload and a download.
    #[cfg(feature = "cuda")]
    fn forward_on_device(
        &self,
        input: &ImageBatch,
        device: &crate::gpu_transformer::DeviceParam,
    ) -> Result<(ImageBatch, ConvCache), NetworkError> {
        let resident = DeviceImageBatch::upload(device.context(), input)?;
        let (output, cache) = self.forward_device(&resident)?;
        Ok((output.download()?, cache))
    }

    #[cfg(feature = "cuda")]
    fn backward_on_device(
        &mut self,
        cache: &ConvCache,
        grad_output: &ImageBatch,
    ) -> Result<ImageBatch, NetworkError> {
        let context = self
            .weight
            .weight
            .device
            .as_ref()
            .expect("a device-resident cache means a device-resident weight")
            .context()
            .clone();
        let resident = DeviceImageBatch::upload(&context, grad_output)?;
        self.backward_device(cache, &resident)?.download()
    }

    fn add_bias(&self, tokens: &mut Matrix) {
        if !self.has_bias {
            return;
        }
        let bias = &self.bias.value.data;
        tokens.data.par_chunks_mut(tokens.cols).for_each(|row| {
            for (value, bias) in row.iter_mut().zip(bias) {
                *value += bias;
            }
        });
    }

    /// The bias gradient is the upstream gradient summed over every pixel.
    ///
    /// Always on the host: it is one pass over a buffer that is already here,
    /// and the bias is one value per output channel rather than one per weight.
    fn accumulate_bias_grad(&mut self, grad_output: &Matrix) {
        if self.bias.is_frozen() || !self.has_bias {
            return;
        }
        for row in 0..grad_output.rows {
            for (slot, value) in self.bias.grad.data.iter_mut().zip(grad_output.row(row)) {
                *slot += value;
            }
        }
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = self.weight.params_mut();
        if self.has_bias {
            params.push(&mut self.bias);
        }
        params
    }

    /// `[batch * out_height * out_width, in_channels * kernel * kernel]`, one
    /// row per output pixel.
    fn im2col(&self, input: &ImageBatch, shape: &ConvGeometry) -> Matrix {
        let patch = shape.patch();
        let mut columns = Matrix::new(shape.rows(), patch);
        let (kernel, stride, padding) = (self.kernel, self.stride, self.padding);
        let channels = self.in_channels;
        columns
            .data
            .par_chunks_mut(patch)
            .enumerate()
            .for_each(|(index, row)| {
                let plane = shape.out_height * shape.out_width;
                let (sample, rest) = (index / plane, index % plane);
                let (out_y, out_x) = (rest / shape.out_width, rest % shape.out_width);
                for ky in 0..kernel {
                    let y = (out_y * stride + ky) as isize - padding as isize;
                    if y < 0 || y as usize >= input.height {
                        continue;
                    }
                    for kx in 0..kernel {
                        let x = (out_x * stride + kx) as isize - padding as isize;
                        if x < 0 || x as usize >= input.width {
                            continue;
                        }
                        let source = input.pixel(sample, y as usize, x as usize);
                        let offset = (ky * kernel + kx) * channels;
                        row[offset..offset + channels].copy_from_slice(source);
                    }
                }
            });
        columns
    }

    /// The transpose of [`TrainableConv2d::im2col`]: gradients scattered back
    /// and summed where the patches overlapped.
    fn col2im(&self, columns: &Matrix, shape: &ConvGeometry) -> ImageBatch {
        let mut grad = ImageBatch::new(shape.batch, shape.height, shape.width, self.in_channels);
        for sample in 0..shape.batch {
            for out_y in 0..shape.out_height {
                for out_x in 0..shape.out_width {
                    let row =
                        columns.row((sample * shape.out_height + out_y) * shape.out_width + out_x);
                    for ky in 0..self.kernel {
                        let y = (out_y * self.stride + ky) as isize - self.padding as isize;
                        if y < 0 || y as usize >= shape.height {
                            continue;
                        }
                        for kx in 0..self.kernel {
                            let x = (out_x * self.stride + kx) as isize - self.padding as isize;
                            if x < 0 || x as usize >= shape.width {
                                continue;
                            }
                            let offset = (ky * self.kernel + kx) * self.in_channels;
                            let target = grad.pixel_mut(sample, y as usize, x as usize);
                            for channel in 0..self.in_channels {
                                target[channel] += row[offset + channel];
                            }
                        }
                    }
                }
            }
        }
        grad
    }
}

/// Normalization over groups of channels, which is what an image model uses
/// where a language model uses [`crate::norm::RmsNorm`].
///
/// A batch norm would need statistics collected at training time and a layer
/// norm would mix unrelated channels; a group norm normalizes each group of
/// channels over its own pixels, which is stable at a batch size of one.
#[derive(Clone, Debug)]
pub struct GroupNorm {
    pub groups: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub eps: f32,
}

impl GroupNorm {
    /// Wraps the per-channel scale and shift a checkpoint stores.
    pub fn new(
        groups: usize,
        weight: Vec<f32>,
        bias: Vec<f32>,
        eps: f32,
    ) -> Result<Self, NetworkError> {
        if groups == 0 || weight.is_empty() || weight.len() % groups != 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "{} channels do not divide into {groups} groups",
                weight.len()
            )));
        }
        if bias.len() != weight.len() {
            return Err(NetworkError::InvalidTarget {
                expected: weight.len(),
                actual: bias.len(),
            });
        }
        Ok(Self {
            groups,
            weight,
            bias,
            eps,
        })
    }

    /// Normalizes in place.
    pub fn forward(&self, map: &mut FeatureMap) -> Result<(), NetworkError> {
        if map.channels != self.weight.len() {
            return Err(NetworkError::InvalidTarget {
                expected: self.weight.len(),
                actual: map.channels,
            });
        }
        let per_group = map.channels / self.groups;
        let pixels = map.pixels();
        let span = per_group * pixels;

        map.data
            .par_chunks_mut(span)
            .enumerate()
            .for_each(|(group, values)| {
                let mean = values.iter().sum::<f32>() / span as f32;
                let variance = values
                    .iter()
                    .map(|value| (value - mean).powi(2))
                    .sum::<f32>()
                    / span as f32;
                let scale = (variance + self.eps).sqrt().recip();
                for (index, value) in values.iter_mut().enumerate() {
                    let channel = group * per_group + index / pixels;
                    *value = (*value - mean) * scale * self.weight[channel] + self.bias[channel];
                }
            });
        Ok(())
    }
}

/// [`GroupNorm`] with a backward pass, over an [`ImageBatch`].
///
/// Statistics are summed channel by channel in the same order as
/// [`GroupNorm::forward`], so the two agree to the bit.
#[derive(Clone, Debug)]
pub struct TrainableGroupNorm {
    pub groups: usize,
    pub eps: f32,
    pub gamma: Param,
    pub beta: Param,
}

/// What [`TrainableGroupNorm::backward`] needs from the forward pass.
#[derive(Clone, Debug)]
pub struct GroupNormCache {
    normalized: Matrix,
    /// `1 / sqrt(variance + eps)` per sample and group.
    inv_std: Vec<f32>,
}

impl TrainableGroupNorm {
    /// Scale ones, shift zeros.
    pub fn new(groups: usize, channels: usize, eps: f32) -> Result<Self, NetworkError> {
        if groups == 0 || channels == 0 || channels % groups != 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "{channels} channels do not divide into {groups} groups"
            )));
        }
        Ok(Self {
            groups,
            eps,
            gamma: Param::filled(1, channels, 1.0),
            beta: Param::zeros(1, channels),
        })
    }

    pub fn forward(
        &self,
        input: &ImageBatch,
    ) -> Result<(ImageBatch, GroupNormCache), NetworkError> {
        let channels = self.gamma.value.cols;
        if input.channels() != channels {
            return Err(NetworkError::InvalidTarget {
                expected: channels,
                actual: input.channels(),
            });
        }
        let (per_group, pixels) = (channels / self.groups, input.pixels());
        let span = (per_group * pixels) as f32;
        let mut normalized = Matrix::new(input.tokens.rows, channels);
        let mut inv_std = vec![0f32; input.batch * self.groups];
        normalized
            .data
            .par_chunks_mut(pixels * channels)
            .zip(input.tokens.data.par_chunks(pixels * channels))
            .zip(inv_std.par_chunks_mut(self.groups))
            .for_each(|((out, sample), inv_std)| {
                for (group, inv_std) in inv_std.iter_mut().enumerate() {
                    let range = group * per_group..(group + 1) * per_group;
                    let values = || {
                        range.clone().flat_map(|channel| {
                            (0..pixels).map(move |p| sample[p * channels + channel])
                        })
                    };
                    let mean = values().sum::<f32>() / span;
                    let variance = values().map(|value| (value - mean).powi(2)).sum::<f32>() / span;
                    *inv_std = (variance + self.eps).sqrt().recip();
                    for p in 0..pixels {
                        for channel in range.clone() {
                            let at = p * channels + channel;
                            out[at] = (sample[at] - mean) * *inv_std;
                        }
                    }
                }
            });
        let (gamma, beta) = (&self.gamma.value.data, &self.beta.value.data);
        let mut output = input.like(channels);
        for (out, row) in output
            .tokens
            .data
            .chunks_exact_mut(channels)
            .zip(normalized.data.chunks_exact(channels))
        {
            for channel in 0..channels {
                out[channel] = row[channel] * gamma[channel] + beta[channel];
            }
        }
        Ok((
            output,
            GroupNormCache {
                normalized,
                inv_std,
            },
        ))
    }

    /// Adds to `gamma` and `beta`'s gradients and returns the input's.
    pub fn backward(&mut self, cache: &GroupNormCache, grad: &ImageBatch) -> ImageBatch {
        let channels = grad.channels();
        let (per_group, pixels) = (channels / self.groups, grad.pixels());
        let span = (per_group * pixels) as f32;
        for (g, row) in grad
            .tokens
            .data
            .chunks_exact(channels)
            .zip(cache.normalized.data.chunks_exact(channels))
        {
            for channel in 0..channels {
                self.gamma.grad.data[channel] += g[channel] * row[channel];
                self.beta.grad.data[channel] += g[channel];
            }
        }
        let gamma = &self.gamma.value.data;
        let mut input_grad = grad.like(channels);
        input_grad
            .tokens
            .data
            .par_chunks_mut(pixels * channels)
            .zip(grad.tokens.data.par_chunks(pixels * channels))
            .zip(cache.normalized.data.par_chunks(pixels * channels))
            .zip(cache.inv_std.par_chunks(self.groups))
            .for_each(|(((out, g), normalized), inv_std)| {
                for (group, &inv_std) in inv_std.iter().enumerate() {
                    let range = group * per_group..(group + 1) * per_group;
                    let (mut mean, mut along) = (0f32, 0f32);
                    for p in 0..pixels {
                        for channel in range.clone() {
                            let at = p * channels + channel;
                            let d = g[at] * gamma[channel];
                            mean += d;
                            along += d * normalized[at];
                        }
                    }
                    let (mean, along) = (mean / span, along / span);
                    for p in 0..pixels {
                        for channel in range.clone() {
                            let at = p * channels + channel;
                            out[at] =
                                inv_std * (g[at] * gamma[channel] - mean - normalized[at] * along);
                        }
                    }
                }
            });
        input_grad
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        vec![&mut self.gamma, &mut self.beta]
    }

    /// Moves `gamma` and `beta` onto a device for
    /// [`TrainableGroupNorm::forward_device`].
    pub fn to_cuda_on(&mut self, device: &crate::param::CudaDevice) -> Result<(), NetworkError> {
        self.gamma.to_cuda_on(device)?;
        self.beta.to_cuda_on(device)
    }

    pub fn to_cpu(&mut self) -> Result<(), NetworkError> {
        self.gamma.to_cpu()?;
        self.beta.to_cpu()
    }

    /// [`TrainableGroupNorm::forward`] for a batch already on the device.
    #[cfg(feature = "cuda")]
    pub fn forward_device(
        &self,
        input: &DeviceImageBatch,
    ) -> Result<(DeviceImageBatch, DeviceGroupNormCache), NetworkError> {
        let channels = self.gamma.value.cols;
        if input.channels != channels {
            return Err(NetworkError::InvalidTarget {
                expected: channels,
                actual: input.channels,
            });
        }
        let (gamma, beta) = match (&self.gamma.device, &self.beta.device) {
            (Some(gamma), Some(beta)) => (gamma.value(), beta.value()),
            _ => {
                return Err(NetworkError::InvalidConfig(
                    "forward_device needs gamma and beta on a device".into(),
                ));
            }
        };
        let context = &input.context;
        let err = crate::cuda_training::cuda_err;
        let total = input.data.len();
        let blocks = input.batch * self.groups;
        let (mut mean, mut inv_std) = (context.zeros(blocks)?, context.zeros(blocks)?);
        let (mut output, mut normalized) = (context.zeros(total)?, context.zeros(total)?);
        let [n, pixels, channels, per_group] =
            [total, input.pixels(), channels, channels / self.groups].map(|v| v as i32);
        unsafe {
            context
                .stream
                .launch_builder(&context.image.gn_stats)
                .arg(&mut mean)
                .arg(&mut inv_std)
                .arg(&input.data)
                .arg(&pixels)
                .arg(&channels)
                .arg(&per_group)
                .arg(&self.eps)
                .launch(group_blocks(blocks))
                .map_err(err("group norm statistics kernel"))?;
            context
                .stream
                .launch_builder(&context.image.gn_apply)
                .arg(&mut output)
                .arg(&mut normalized)
                .arg(&input.data)
                .arg(&mean)
                .arg(&inv_std)
                .arg(gamma)
                .arg(beta)
                .arg(&n)
                .arg(&pixels)
                .arg(&channels)
                .arg(&per_group)
                .launch(crate::cuda_training::cfg(total))
                .map_err(err("group norm kernel"))?;
        }
        Ok((
            input.with_data(output),
            DeviceGroupNormCache {
                normalized,
                inv_std,
            },
        ))
    }

    /// [`TrainableGroupNorm::backward`] on the device: adds to `gamma` and
    /// `beta`'s device gradients and returns the input's.
    #[cfg(feature = "cuda")]
    pub fn backward_device(
        &mut self,
        cache: &DeviceGroupNormCache,
        grad: &DeviceImageBatch,
    ) -> Result<DeviceImageBatch, NetworkError> {
        let context = grad.context.clone();
        let err = crate::cuda_training::cuda_err;
        let (total, rows) = (grad.data.len(), grad.batch * grad.pixels());
        let blocks = grad.batch * self.groups;
        let channels = self.gamma.value.cols;
        let (Some(gamma), Some(beta)) = (self.gamma.device.as_mut(), self.beta.device.as_mut())
        else {
            return Err(NetworkError::InvalidConfig(
                "backward_device needs gamma and beta on a device".into(),
            ));
        };
        let ones = context.filled(rows, 1.0)?;
        let mut product = context.zeros(total)?;
        let (mut first, mut second) = (context.zeros(blocks)?, context.zeros(blocks)?);
        let mut input_grad = context.zeros(total)?;
        let [n, pixels, channels, per_group] =
            [total, grad.pixels(), channels, channels / self.groups].map(|v| v as i32);
        unsafe {
            context
                .stream
                .launch_builder(&context.image.mul)
                .arg(&mut product)
                .arg(&grad.data)
                .arg(&cache.normalized)
                .arg(&n)
                .launch(crate::cuda_training::cfg(total))
                .map_err(err("multiply kernel"))?;
            context
                .stream
                .launch_builder(&context.image.gn_bwd_stats)
                .arg(&mut first)
                .arg(&mut second)
                .arg(&grad.data)
                .arg(&cache.normalized)
                .arg(gamma.value())
                .arg(&pixels)
                .arg(&channels)
                .arg(&per_group)
                .launch(group_blocks(blocks))
                .map_err(err("group norm backward statistics kernel"))?;
            context
                .stream
                .launch_builder(&context.image.gn_bwd_apply)
                .arg(&mut input_grad)
                .arg(&grad.data)
                .arg(&cache.normalized)
                .arg(&cache.inv_std)
                .arg(&first)
                .arg(&second)
                .arg(gamma.value())
                .arg(&n)
                .arg(&pixels)
                .arg(&channels)
                .arg(&per_group)
                .launch(crate::cuda_training::cfg(total))
                .map_err(err("group norm backward kernel"))?;
        }
        gamma.accumulate_grad_device(&ones, &product, rows)?;
        beta.accumulate_grad_device(&ones, &grad.data, rows)?;
        Ok(grad.with_data(input_grad))
    }
}

/// What [`TrainableGroupNorm::backward_device`] needs from the forward pass.
#[cfg(feature = "cuda")]
pub struct DeviceGroupNormCache {
    normalized: cudarc::driver::CudaSlice<f32>,
    inv_std: cudarc::driver::CudaSlice<f32>,
}

/// One block per sample and group, for the group norm reductions.
#[cfg(feature = "cuda")]
fn group_blocks(blocks: usize) -> cudarc::driver::LaunchConfig {
    cudarc::driver::LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// SiLU over a device-resident batch. Keep the input: it is what
/// [`silu_backward_device`] differentiates at.
#[cfg(feature = "cuda")]
pub fn silu_device(input: &DeviceImageBatch) -> Result<DeviceImageBatch, NetworkError> {
    silu_launch(input, None)
}

/// `grad * silu'(input)`, on the device.
#[cfg(feature = "cuda")]
pub fn silu_backward_device(
    input: &DeviceImageBatch,
    grad: &DeviceImageBatch,
) -> Result<DeviceImageBatch, NetworkError> {
    silu_launch(input, Some(grad))
}

/// What [`spatial_softmax_backward_device`] needs from the forward pass.
#[cfg(feature = "cuda")]
pub struct DeviceSpatialSoftmaxCache {
    probabilities: DeviceImageBatch,
    expected: cudarc::driver::CudaSlice<f32>,
}

/// [`spatial_softmax`] on the device. The output is a `[batch, 1, 1,
/// 2 * channels]` batch, laid out like the host `Matrix`.
#[cfg(feature = "cuda")]
pub fn spatial_softmax_device(
    input: &DeviceImageBatch,
) -> Result<(DeviceImageBatch, DeviceSpatialSoftmaxCache), NetworkError> {
    let context = &input.context;
    let channels = input.channels;
    let mut probabilities = context.zeros(input.data.len())?;
    let mut expected = context.zeros(input.batch * 2 * channels)?;
    let [height, width, channels_arg] = [input.height, input.width, channels].map(|v| v as i32);
    unsafe {
        context
            .stream
            .launch_builder(&context.image.ssm_fwd)
            .arg(&mut probabilities)
            .arg(&mut expected)
            .arg(&input.data)
            .arg(&height)
            .arg(&width)
            .arg(&channels_arg)
            .launch(group_blocks(input.batch * channels))
            .map_err(crate::cuda_training::cuda_err("spatial softmax kernel"))?;
    }
    let output = DeviceImageBatch {
        batch: input.batch,
        height: 1,
        width: 1,
        channels: 2 * channels,
        data: expected.clone(),
        context: context.clone(),
    };
    let probabilities = input.with_data(probabilities);
    Ok((
        output,
        DeviceSpatialSoftmaxCache {
            probabilities,
            expected,
        },
    ))
}

/// [`spatial_softmax_backward`] on the device.
#[cfg(feature = "cuda")]
pub fn spatial_softmax_backward_device(
    cache: &DeviceSpatialSoftmaxCache,
    grad: &DeviceImageBatch,
) -> Result<DeviceImageBatch, NetworkError> {
    let probabilities = &cache.probabilities;
    let context = &probabilities.context;
    let total = probabilities.data.len();
    let mut out = context.zeros(total)?;
    let [n, height, width, channels] = [
        total,
        probabilities.height,
        probabilities.width,
        probabilities.channels,
    ]
    .map(|v| v as i32);
    unsafe {
        context
            .stream
            .launch_builder(&context.image.ssm_bwd)
            .arg(&mut out)
            .arg(&probabilities.data)
            .arg(&grad.data)
            .arg(&cache.expected)
            .arg(&n)
            .arg(&height)
            .arg(&width)
            .arg(&channels)
            .launch(crate::cuda_training::cfg(total))
            .map_err(crate::cuda_training::cuda_err(
                "spatial softmax backward kernel",
            ))?;
    }
    Ok(probabilities.with_data(out))
}

/// [`global_avg_pool`] on the device: a `[batch, 1, 1, channels]` batch.
#[cfg(feature = "cuda")]
pub fn global_avg_pool_device(input: &DeviceImageBatch) -> Result<DeviceImageBatch, NetworkError> {
    let context = &input.context;
    let mut out = context.zeros(input.batch * input.channels)?;
    let [pixels, channels] = [input.pixels(), input.channels].map(|v| v as i32);
    unsafe {
        context
            .stream
            .launch_builder(&context.image.gap_fwd)
            .arg(&mut out)
            .arg(&input.data)
            .arg(&pixels)
            .arg(&channels)
            .launch(group_blocks(input.batch * input.channels))
            .map_err(crate::cuda_training::cuda_err("average pool kernel"))?;
    }
    Ok(DeviceImageBatch {
        height: 1,
        width: 1,
        ..input.with_data(out)
    })
}

/// [`global_avg_pool_backward`] on the device, back to a `height` by `width`
/// batch.
#[cfg(feature = "cuda")]
pub fn global_avg_pool_backward_device(
    grad: &DeviceImageBatch,
    height: usize,
    width: usize,
) -> Result<DeviceImageBatch, NetworkError> {
    let context = &grad.context;
    let total = grad.batch * height * width * grad.channels;
    let mut out = context.zeros(total)?;
    let [n, pixels, channels] = [total, height * width, grad.channels].map(|v| v as i32);
    unsafe {
        context
            .stream
            .launch_builder(&context.image.gap_bwd)
            .arg(&mut out)
            .arg(&grad.data)
            .arg(&n)
            .arg(&pixels)
            .arg(&channels)
            .launch(crate::cuda_training::cfg(total))
            .map_err(crate::cuda_training::cuda_err(
                "average pool backward kernel",
            ))?;
    }
    Ok(DeviceImageBatch {
        height,
        width,
        ..grad.with_data(out)
    })
}

#[cfg(feature = "cuda")]
fn silu_launch(
    input: &DeviceImageBatch,
    grad: Option<&DeviceImageBatch>,
) -> Result<DeviceImageBatch, NetworkError> {
    let context = &input.context;
    let total = input.data.len();
    let mut output = context.uninit(total)?;
    let n = total as i32;
    let mut launch = match grad {
        None => context.stream.launch_builder(&context.image.silu),
        Some(_) => context.stream.launch_builder(&context.image.silu_bwd),
    };
    launch.arg(&mut output);
    if let Some(grad) = grad {
        launch.arg(&grad.data);
    }
    launch.arg(&input.data).arg(&n);
    unsafe { launch.launch(crate::cuda_training::cfg(total)) }
        .map_err(crate::cuda_training::cuda_err("SiLU kernel"))?;
    Ok(input.with_data(output))
}

/// SiLU over a whole map, the activation every one of these decoders uses.
pub fn silu(map: &mut FeatureMap) {
    map.data
        .par_iter_mut()
        .for_each(|value| *value = crate::ffn::silu(*value));
}

/// Doubles each side by repeating pixels, which is how a decoder climbs from a
/// latent's resolution to an image's.
pub fn upsample_nearest(map: &FeatureMap, factor: usize) -> FeatureMap {
    let (height, width) = (map.height * factor, map.width * factor);
    let mut output = FeatureMap::new(map.channels, height, width);
    let pixels = height * width;
    output
        .data
        .par_chunks_mut(pixels)
        .enumerate()
        .for_each(|(channel, plane)| {
            let source = map.plane(channel);
            for (pixel, value) in plane.iter_mut().enumerate() {
                let (y, x) = (pixel / width / factor, pixel % width / factor);
                *value = source[y * map.width + x];
            }
        });
    output
}

/// Trades channels for resolution: `[c * factor^2, h, w]` becomes
/// `[c, h * factor, w * factor]`.
///
/// FLUX.2's decoder ends in one of these, and the ordering here is the one
/// PyTorch's `pixel_shuffle` uses, so a checkpoint's weights line up.
pub fn pixel_shuffle(map: &FeatureMap, factor: usize) -> Result<FeatureMap, NetworkError> {
    let square = factor * factor;
    if factor == 0 || map.channels % square != 0 {
        return Err(NetworkError::InvalidConfig(format!(
            "{} channels do not shuffle by {factor}",
            map.channels
        )));
    }
    let (channels, height, width) = (
        map.channels / square,
        map.height * factor,
        map.width * factor,
    );
    let mut output = FeatureMap::new(channels, height, width);
    let pixels = height * width;
    output
        .data
        .par_chunks_mut(pixels)
        .enumerate()
        .for_each(|(channel, plane)| {
            for (pixel, value) in plane.iter_mut().enumerate() {
                let (y, x) = (pixel / width, pixel % width);
                let source = (channel * square + (y % factor) * factor + x % factor)
                    * map.height
                    * map.width
                    + (y / factor) * map.width
                    + x / factor;
                *value = map.data[source];
            }
        });
    Ok(output)
}

/// The inverse of [`pixel_shuffle`]: resolution back into channels, which is
/// how FLUX.2 packs a latent before the transformer sees it.
pub fn pixel_unshuffle(map: &FeatureMap, factor: usize) -> Result<FeatureMap, NetworkError> {
    if factor == 0 || map.height % factor != 0 || map.width % factor != 0 {
        return Err(NetworkError::InvalidConfig(format!(
            "a {}x{} map does not unshuffle by {factor}",
            map.height, map.width
        )));
    }
    let square = factor * factor;
    let (channels, height, width) = (
        map.channels * square,
        map.height / factor,
        map.width / factor,
    );
    let mut output = FeatureMap::new(channels, height, width);
    let pixels = height * width;
    output
        .data
        .par_chunks_mut(pixels)
        .enumerate()
        .for_each(|(channel, plane)| {
            let (source_channel, offset) = (channel / square, channel % square);
            let (row_offset, column_offset) = (offset / factor, offset % factor);
            let source = map.plane(source_channel);
            for (pixel, value) in plane.iter_mut().enumerate() {
                let (y, x) = (pixel / width, pixel % width);
                *value = source[(y * factor + row_offset) * map.width + x * factor + column_offset];
            }
        });
    Ok(output)
}

/// `n` evenly spaced points from -1 to 1; the middle for one.
fn unit_coordinates(n: usize) -> Vec<f32> {
    match n {
        1 => vec![0.0],
        _ => (0..n)
            .map(|i| -1.0 + 2.0 * i as f32 / (n - 1) as f32)
            .collect(),
    }
}

/// What [`spatial_softmax_backward`] needs from the forward pass.
pub struct SpatialSoftmaxCache {
    /// The softmax over pixels of each sample's channels, laid out like the
    /// input.
    probabilities: ImageBatch,
    /// The forward output.
    expected: Matrix,
}

/// Each channel's softmax over its pixels, reduced to the expected `x` and
/// `y` with both axes running from -1 to 1: `[batch, 2 * channels]`, every
/// channel's `x` first, then every channel's `y`. A keypoint layer with no
/// parameters.
pub fn spatial_softmax(input: &ImageBatch) -> (Matrix, SpatialSoftmaxCache) {
    let (channels, pixels, width) = (input.channels(), input.pixels(), input.width);
    let (xs, ys) = (
        unit_coordinates(input.width),
        unit_coordinates(input.height),
    );
    let mut probabilities = input.like(channels);
    let mut expected = Matrix::new(input.batch, 2 * channels);
    probabilities
        .tokens
        .data
        .par_chunks_mut(pixels * channels)
        .zip(input.tokens.data.par_chunks(pixels * channels))
        .zip(expected.data.par_chunks_mut(2 * channels))
        .for_each(|((probability, logits), expected)| {
            for channel in 0..channels {
                let logit = |pixel: usize| logits[pixel * channels + channel];
                let max = (0..pixels).map(logit).fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for pixel in 0..pixels {
                    let value = (logit(pixel) - max).exp();
                    probability[pixel * channels + channel] = value;
                    sum += value;
                }
                let (mut x, mut y) = (0.0, 0.0);
                for pixel in 0..pixels {
                    let slot = &mut probability[pixel * channels + channel];
                    *slot /= sum;
                    x += *slot * xs[pixel % width];
                    y += *slot * ys[pixel / width];
                }
                (expected[channel], expected[channels + channel]) = (x, y);
            }
        });
    let cache = SpatialSoftmaxCache {
        probabilities,
        expected: expected.clone(),
    };
    (expected, cache)
}

/// `dL/d(input)` of [`spatial_softmax`], given `dL/d(output)`.
pub fn spatial_softmax_backward(cache: &SpatialSoftmaxCache, grad: &Matrix) -> ImageBatch {
    let probabilities = &cache.probabilities;
    let (channels, pixels, width) = (
        probabilities.channels(),
        probabilities.pixels(),
        probabilities.width,
    );
    let (xs, ys) = (
        unit_coordinates(probabilities.width),
        unit_coordinates(probabilities.height),
    );
    let mut out = probabilities.like(channels);
    out.tokens
        .data
        .par_chunks_mut(pixels * channels)
        .zip(probabilities.tokens.data.par_chunks(pixels * channels))
        .zip(
            grad.data
                .par_chunks(2 * channels)
                .zip(cache.expected.data.par_chunks(2 * channels)),
        )
        .for_each(|((out, probability), (grad, expected))| {
            for pixel in 0..pixels {
                for channel in 0..channels {
                    // The softmax Jacobian: `q_i (dL/dq_i - sum_j q_j dL/dq_j)`,
                    // where `dL/dq_i` is the coordinate times the upstream.
                    let (gx, gy) = (grad[channel], grad[channels + channel]);
                    let along = gx * (xs[pixel % width] - expected[channel])
                        + gy * (ys[pixel / width] - expected[channels + channel]);
                    let slot = pixel * channels + channel;
                    out[slot] = probability[slot] * along;
                }
            }
        });
    out
}

/// The mean of each channel over its pixels: `[batch, channels]`.
pub fn global_avg_pool(input: &ImageBatch) -> Matrix {
    let (channels, pixels) = (input.channels(), input.pixels());
    let mut out = Matrix::new(input.batch, channels);
    out.data
        .par_chunks_mut(channels)
        .zip(input.tokens.data.par_chunks(pixels * channels))
        .for_each(|(out, sample)| {
            for pixel in sample.chunks_exact(channels) {
                out.iter_mut()
                    .zip(pixel)
                    .for_each(|(sum, value)| *sum += value);
            }
            out.iter_mut().for_each(|sum| *sum /= pixels as f32);
        });
    out
}

/// `dL/d(input)` of [`global_avg_pool`] over a `height` by `width` input.
pub fn global_avg_pool_backward(grad: &Matrix, height: usize, width: usize) -> ImageBatch {
    let (channels, pixels) = (grad.cols, height * width);
    let mut out = ImageBatch::new(grad.rows, height, width, channels);
    out.tokens
        .data
        .par_chunks_mut(pixels * channels)
        .zip(grad.data.par_chunks(channels))
        .for_each(|(out, grad)| {
            for pixel in out.chunks_exact_mut(channels) {
                pixel
                    .iter_mut()
                    .zip(grad)
                    .for_each(|(slot, value)| *slot = value / pixels as f32);
            }
        });
    out
}

/// PyTorch's `BatchNorm2d` over an image batch's channels, training on batch
/// statistics and evaluating on running ones. YOLO's convolutions use it with
/// [`TrainableBatchNorm2d::yolo`].
#[derive(Clone, Debug)]
pub struct TrainableBatchNorm2d {
    pub eps: f32,
    /// Weight of the newest batch in the running statistics.
    pub momentum: f32,
    pub gamma: Param,
    pub beta: Param,
    /// `[2, channels]`: the running mean, then the running unbiased variance.
    /// A `Param` so it moves to and from a device with the rest; the
    /// optimizer never sees it.
    pub running: Param,
}

/// What [`TrainableBatchNorm2d::backward`] needs from the forward pass.
#[derive(Clone, Debug)]
pub struct BatchNormCache {
    normalized: Matrix,
    inv_std: Vec<f32>,
    training: bool,
}

impl TrainableBatchNorm2d {
    /// Scale ones, shift zeros, running mean zeros and variance ones.
    pub fn new(channels: usize, eps: f32, momentum: f32) -> Self {
        let mut running = Matrix::new(2, channels);
        running.data[channels..].fill(1.0);
        Self {
            eps,
            momentum,
            gamma: Param::filled(1, channels, 1.0),
            beta: Param::zeros(1, channels),
            running: Param::new(running),
        }
    }

    /// ultralytics' settings: eps 1e-3, momentum 0.03.
    pub fn yolo(channels: usize) -> Self {
        Self::new(channels, 1e-3, 0.03)
    }

    pub fn channels(&self) -> usize {
        self.gamma.value.cols
    }

    /// Normalizes with this batch's statistics and folds them into the
    /// running ones when `training`, else with the running ones.
    pub fn forward(
        &mut self,
        input: &ImageBatch,
        training: bool,
    ) -> Result<(ImageBatch, BatchNormCache), NetworkError> {
        let channels = self.channels();
        if input.channels() != channels {
            return Err(NetworkError::InvalidTarget {
                expected: channels,
                actual: input.channels(),
            });
        }
        let rows = input.tokens.rows;
        let (mut mean, mut variance) = (vec![0f32; channels], vec![0f32; channels]);
        if training {
            for row in input.tokens.data.chunks_exact(channels) {
                mean.iter_mut().zip(row).for_each(|(m, v)| *m += v);
            }
            mean.iter_mut().for_each(|m| *m /= rows as f32);
            for row in input.tokens.data.chunks_exact(channels) {
                for c in 0..channels {
                    variance[c] += (row[c] - mean[c]).powi(2);
                }
            }
            variance.iter_mut().for_each(|v| *v /= rows as f32);
            let unbias = rows as f32 / (rows as f32 - 1.0).max(1.0);
            let m = self.momentum;
            let running = &mut self.running.value.data;
            for c in 0..channels {
                running[c] = (1.0 - m) * running[c] + m * mean[c];
                running[channels + c] =
                    (1.0 - m) * running[channels + c] + m * variance[c] * unbias;
            }
        } else {
            mean.copy_from_slice(&self.running.value.data[..channels]);
            variance.copy_from_slice(&self.running.value.data[channels..]);
        }
        let inv_std: Vec<f32> = variance
            .iter()
            .map(|v| 1.0 / (v + self.eps).sqrt())
            .collect();
        let mut normalized = Matrix::new(rows, channels);
        let mut output = input.like(channels);
        let (gamma, beta) = (&self.gamma.value.data, &self.beta.value.data);
        for ((out, h), row) in output
            .tokens
            .data
            .chunks_exact_mut(channels)
            .zip(normalized.data.chunks_exact_mut(channels))
            .zip(input.tokens.data.chunks_exact(channels))
        {
            for c in 0..channels {
                h[c] = (row[c] - mean[c]) * inv_std[c];
                out[c] = h[c] * gamma[c] + beta[c];
            }
        }
        Ok((
            output,
            BatchNormCache {
                normalized,
                inv_std,
                training,
            },
        ))
    }

    /// Adds to `gamma` and `beta`'s gradients and returns the input's.
    pub fn backward(&mut self, cache: &BatchNormCache, grad: &ImageBatch) -> ImageBatch {
        let channels = self.channels();
        let rows = grad.tokens.rows as f32;
        let (mut mean, mut along) = (vec![0f32; channels], vec![0f32; channels]);
        for (g, h) in grad
            .tokens
            .data
            .chunks_exact(channels)
            .zip(cache.normalized.data.chunks_exact(channels))
        {
            for c in 0..channels {
                mean[c] += g[c];
                along[c] += g[c] * h[c];
            }
        }
        for c in 0..channels {
            self.gamma.grad.data[c] += along[c];
            self.beta.grad.data[c] += mean[c];
        }
        let gamma = &self.gamma.value.data;
        let mut input_grad = grad.like(channels);
        for ((out, g), h) in input_grad
            .tokens
            .data
            .chunks_exact_mut(channels)
            .zip(grad.tokens.data.chunks_exact(channels))
            .zip(cache.normalized.data.chunks_exact(channels))
        {
            for c in 0..channels {
                let centred = match cache.training {
                    true => g[c] - (mean[c] + h[c] * along[c]) / rows,
                    false => g[c],
                };
                out[c] = cache.inv_std[c] * gamma[c] * centred;
            }
        }
        input_grad
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        vec![&mut self.gamma, &mut self.beta]
    }

    /// Moves `gamma`, `beta` and the running statistics onto a device.
    pub fn to_cuda_on(&mut self, device: &crate::param::CudaDevice) -> Result<(), NetworkError> {
        self.gamma.to_cuda_on(device)?;
        self.beta.to_cuda_on(device)?;
        self.running.to_cuda_on(device)
    }

    pub fn to_cpu(&mut self) -> Result<(), NetworkError> {
        self.gamma.to_cpu()?;
        self.beta.to_cpu()?;
        self.running.to_cpu()
    }

    /// The convolution that computes this norm, evaluating, after `conv`:
    /// each output channel's weights scaled by `gamma / sqrt(var + eps)` and
    /// the shift folded into the bias. Both must be on the host.
    pub fn fuse_into(&self, conv: &TrainableConv2d) -> TrainableConv2d {
        let channels = self.channels();
        let mut fused = conv.clone();
        let patch = fused.weight.weight.value.cols;
        let running = &self.running.value.data;
        for c in 0..channels {
            let scale = self.gamma.value.data[c] / (running[channels + c] + self.eps).sqrt();
            fused.weight.weight.value.data[c * patch..(c + 1) * patch]
                .iter_mut()
                .for_each(|w| *w *= scale);
            let bias = if conv.has_bias {
                conv.bias.value.data[c]
            } else {
                0.0
            };
            fused.bias.value.data[c] = (bias - running[c]) * scale + self.beta.value.data[c];
        }
        fused.has_bias = true;
        fused
    }

    /// [`TrainableBatchNorm2d::forward`] for a batch already on the device.
    #[cfg(feature = "cuda")]
    pub fn forward_device(
        &mut self,
        input: &DeviceImageBatch,
        training: bool,
    ) -> Result<(DeviceImageBatch, DeviceBatchNormCache), NetworkError> {
        self.forward_device_with(input, training, false)
    }

    /// [`TrainableBatchNorm2d::forward_device`] with SiLU over the output
    /// under `silu`, in the same kernel; the backward pass differentiates it too.
    #[cfg(feature = "cuda")]
    pub(crate) fn forward_device_with(
        &mut self,
        input: &DeviceImageBatch,
        training: bool,
        silu: bool,
    ) -> Result<(DeviceImageBatch, DeviceBatchNormCache), NetworkError> {
        let channels = self.channels();
        if input.channels != channels {
            return Err(NetworkError::InvalidTarget {
                expected: channels,
                actual: input.channels,
            });
        }
        let (Some(gamma), Some(beta), Some(running)) = (
            self.gamma.device.as_ref(),
            self.beta.device.as_ref(),
            self.running.device.as_mut(),
        ) else {
            return Err(NetworkError::InvalidConfig(
                "forward_device needs the batch norm on a device".into(),
            ));
        };
        let context = &input.context;
        let err = crate::cuda_training::cuda_err;
        let (total, rows) = (input.data.len(), input.batch * input.pixels());
        let (mut mean, mut variance) = (context.zeros(channels)?, context.zeros(channels)?);
        let mut inv_std = context.zeros(channels)?;
        if training {
            let ones = context.filled(rows, 1.0)?;
            let scale = 1.0 / rows as f32;
            let sum = |x: &cudarc::driver::CudaSlice<f32>,
                       out: &mut cudarc::driver::CudaSlice<f32>| {
                crate::gpu_transformer::gemm_lhs_transposed(
                    context, &ones, 1, x, channels, out, channels, rows, 1, channels, scale, 0.0,
                )
            };
            sum(&input.data, &mut mean)?;
            let mut squares = context.uninit(total)?;
            let [n, c] = [total, channels].map(|v| v as i32);
            unsafe {
                context
                    .stream
                    .launch_builder(&context.image.bn_center_sq)
                    .arg(&mut squares)
                    .arg(&input.data)
                    .arg(&mean)
                    .arg(&n)
                    .arg(&c)
                    .launch(crate::cuda_training::cfg(total))
                    .map_err(err("batch norm variance kernel"))?;
            }
            sum(&squares, &mut variance)?;
        }
        let unbias = rows as f32 / (rows as f32 - 1.0).max(1.0);
        let [c, train] = [channels as i32, training as i32];
        unsafe {
            context
                .stream
                .launch_builder(&context.image.bn_finish)
                .arg(&mut mean)
                .arg(&mut variance)
                .arg(&mut inv_std)
                .arg(running.value_mut())
                .arg(&c)
                .arg(&self.eps)
                .arg(&self.momentum)
                .arg(&unbias)
                .arg(&train)
                .launch(crate::cuda_training::cfg(channels))
                .map_err(err("batch norm statistics kernel"))?;
        }
        let (mut output, mut normalized) = (context.uninit(total)?, context.uninit(total)?);
        let n = total as i32;
        unsafe {
            context
                .stream
                .launch_builder(&context.image.bn_apply)
                .arg(&mut output)
                .arg(&mut normalized)
                .arg(&input.data)
                .arg(&mean)
                .arg(&inv_std)
                .arg(gamma.value())
                .arg(beta.value())
                .arg(&n)
                .arg(&c)
                .arg(&(silu as i32))
                .launch(crate::cuda_training::cfg(total))
                .map_err(err("batch norm kernel"))?;
        }
        Ok((
            input.with_data(output),
            DeviceBatchNormCache {
                normalized,
                inv_std,
                training,
                silu,
            },
        ))
    }

    /// [`TrainableBatchNorm2d::backward`] on the device.
    #[cfg(feature = "cuda")]
    pub fn backward_device(
        &mut self,
        cache: &DeviceBatchNormCache,
        grad: &DeviceImageBatch,
    ) -> Result<DeviceImageBatch, NetworkError> {
        let context = grad.context.clone();
        let err = crate::cuda_training::cuda_err;
        let (total, rows) = (grad.data.len(), grad.batch * grad.pixels());
        let channels = self.channels();
        let (Some(gamma), Some(beta)) = (self.gamma.device.as_mut(), self.beta.device.as_mut())
        else {
            return Err(NetworkError::InvalidConfig(
                "backward_device needs gamma and beta on a device".into(),
            ));
        };
        let ones = context.filled(rows, 1.0)?;
        let (mut product, mut grad_y) = (context.uninit(total)?, context.uninit(total)?);
        let [n, c] = [total, channels].map(|v| v as i32);
        unsafe {
            context
                .stream
                .launch_builder(&context.image.bn_bwd_prep)
                .arg(&mut grad_y)
                .arg(&mut product)
                .arg(&grad.data)
                .arg(&cache.normalized)
                .arg(gamma.value())
                .arg(beta.value())
                .arg(&n)
                .arg(&c)
                .arg(&(cache.silu as i32))
                .launch(crate::cuda_training::cfg(total))
                .map_err(err("batch norm backward kernel"))?;
        }
        let (mut mean, mut along) = (context.zeros(channels)?, context.zeros(channels)?);
        if cache.training {
            let scale = 1.0 / rows as f32;
            for (x, out) in [(&grad_y, &mut mean), (&product, &mut along)] {
                crate::gpu_transformer::gemm_lhs_transposed(
                    &context, &ones, 1, x, channels, out, channels, rows, 1, channels, scale, 0.0,
                )?;
            }
        }
        let mut input_grad = context.uninit(total)?;
        unsafe {
            context
                .stream
                .launch_builder(&context.image.bn_bwd_apply)
                .arg(&mut input_grad)
                .arg(&grad_y)
                .arg(&cache.normalized)
                .arg(&cache.inv_std)
                .arg(&mean)
                .arg(&along)
                .arg(gamma.value())
                .arg(&n)
                .arg(&c)
                .launch(crate::cuda_training::cfg(total))
                .map_err(err("batch norm backward kernel"))?;
        }
        gamma.accumulate_grad_device(&ones, &product, rows)?;
        beta.accumulate_grad_device(&ones, &grad_y, rows)?;
        Ok(grad.with_data(input_grad))
    }
}

/// What [`TrainableBatchNorm2d::backward_device`] needs from the forward pass.
#[cfg(feature = "cuda")]
pub struct DeviceBatchNormCache {
    normalized: cudarc::driver::CudaSlice<f32>,
    inv_std: cudarc::driver::CudaSlice<f32>,
    training: bool,
    silu: bool,
}

/// A depthwise convolution without bias, as ultralytics' `DWConv` and C2PSA's
/// positional encoding use before a batch norm: each channel convolved with
/// its own `kernel x kernel` filter, padded by `kernel / 2`.
#[derive(Clone, Debug)]
pub struct DepthwiseConv2d {
    /// `[1, channels * kernel * kernel]`, PyTorch's `[C, 1, k, k]` flattened.
    pub weight: Param,
    kernel: usize,
    stride: usize,
}

/// What [`DepthwiseConv2d::backward`] needs: the input.
#[derive(Clone, Debug)]
pub struct DepthwiseCache {
    input: ImageBatch,
}

impl DepthwiseConv2d {
    pub fn new(channels: usize, kernel: usize, stride: usize, rng: &mut StdRng) -> Self {
        let fan_in = kernel * kernel;
        Self {
            weight: Param::he_uniform(1, channels * fan_in, fan_in, rng),
            kernel,
            stride,
        }
    }

    pub fn channels(&self) -> usize {
        self.weight.value.cols / (self.kernel * self.kernel)
    }

    pub fn output_size(&self, height: usize, width: usize) -> (usize, usize) {
        let size = |n: usize| (n + 2 * (self.kernel / 2) - self.kernel) / self.stride + 1;
        (size(height), size(width))
    }

    /// Every `(output index, input index, weight index)` the convolution
    /// multiplies, in a fixed order.
    fn taps(&self, input: &ImageBatch) -> impl Iterator<Item = (usize, usize, usize)> + use<> {
        let (h, w, c) = (input.height, input.width, input.channels());
        let (oh, ow) = self.output_size(h, w);
        let (k, s, pad) = (self.kernel, self.stride, self.kernel / 2);
        let batch = input.batch;
        (0..batch * oh * ow).flat_map(move |row| {
            let (ox, oy, b) = (row % ow, row / ow % oh, row / ow / oh);
            (0..k * k).flat_map(move |t| {
                let (sy, sx) = (
                    (oy * s + t / k).wrapping_sub(pad),
                    (ox * s + t % k).wrapping_sub(pad),
                );
                let inside = sy < h && sx < w;
                (0..c).filter(move |_| inside).map(move |ch| {
                    (
                        row * c + ch,
                        ((b * h + sy) * w + sx) * c + ch,
                        ch * k * k + t,
                    )
                })
            })
        })
    }

    pub fn forward(
        &self,
        input: &ImageBatch,
    ) -> Result<(ImageBatch, DepthwiseCache), NetworkError> {
        if input.channels() != self.channels() {
            return Err(NetworkError::InvalidTarget {
                expected: self.channels(),
                actual: input.channels(),
            });
        }
        let (oh, ow) = self.output_size(input.height, input.width);
        let mut output = ImageBatch::new(input.batch, oh, ow, input.channels());
        let weight = &self.weight.value.data;
        for (o, i, t) in self.taps(input) {
            output.tokens.data[o] += input.tokens.data[i] * weight[t];
        }
        Ok((
            output,
            DepthwiseCache {
                input: input.clone(),
            },
        ))
    }

    /// Adds to the weight's gradient and returns the input's.
    pub fn backward(&mut self, cache: &DepthwiseCache, grad: &ImageBatch) -> ImageBatch {
        let mut input_grad = cache.input.like(cache.input.channels());
        let weight = self.weight.value.data.clone();
        for (o, i, t) in self.taps(&cache.input) {
            input_grad.tokens.data[i] += grad.tokens.data[o] * weight[t];
            self.weight.grad.data[t] += grad.tokens.data[o] * cache.input.tokens.data[i];
        }
        input_grad
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        vec![&mut self.weight]
    }

    pub fn to_cuda_on(&mut self, device: &crate::param::CudaDevice) -> Result<(), NetworkError> {
        self.weight.to_cuda_on(device)
    }

    pub fn to_cpu(&mut self) -> Result<(), NetworkError> {
        self.weight.to_cpu()
    }

    /// [`DepthwiseConv2d::forward`] on the device. The cache is the input,
    /// which the caller passes back to [`DepthwiseConv2d::backward_device`].
    #[cfg(feature = "cuda")]
    pub fn forward_device(
        &self,
        input: &DeviceImageBatch,
    ) -> Result<DeviceImageBatch, NetworkError> {
        let Some(weight) = self.weight.device.as_ref() else {
            return Err(NetworkError::InvalidConfig(
                "forward_device needs the weight on a device".into(),
            ));
        };
        if input.channels != self.channels() {
            return Err(NetworkError::InvalidTarget {
                expected: self.channels(),
                actual: input.channels,
            });
        }
        let (oh, ow) = self.output_size(input.height, input.width);
        let context = &input.context;
        let total = input.batch * oh * ow * input.channels;
        let mut out = context.uninit(total)?;
        let dims = self.dims(input);
        unsafe {
            let mut launch = context.stream.launch_builder(&context.image.dw_fwd);
            launch.arg(&mut out).arg(&input.data).arg(weight.value());
            dims.iter().for_each(|d| {
                launch.arg(d);
            });
            launch
                .launch(crate::cuda_training::cfg(total))
                .map_err(crate::cuda_training::cuda_err("depthwise kernel"))?;
        }
        Ok(DeviceImageBatch {
            height: oh,
            width: ow,
            ..input.with_data(out)
        })
    }

    /// [`DepthwiseConv2d::backward`] on the device, given the forward input.
    #[cfg(feature = "cuda")]
    pub fn backward_device(
        &mut self,
        input: &DeviceImageBatch,
        grad: &DeviceImageBatch,
    ) -> Result<DeviceImageBatch, NetworkError> {
        let dims = self.dims(input);
        let taps = self.weight.value.cols;
        let Some(weight) = self.weight.device.as_mut() else {
            return Err(NetworkError::InvalidConfig(
                "backward_device needs the weight on a device".into(),
            ));
        };
        let context = input.context.clone();
        let err = crate::cuda_training::cuda_err;
        let total = input.data.len();
        let mut input_grad = context.uninit(total)?;
        let mut weight_grad = context.zeros(taps)?;
        unsafe {
            let mut launch = context.stream.launch_builder(&context.image.dw_bwd_input);
            launch
                .arg(&mut input_grad)
                .arg(&grad.data)
                .arg(weight.value());
            dims.iter().for_each(|d| {
                launch.arg(d);
            });
            launch
                .launch(crate::cuda_training::cfg(total))
                .map_err(err("depthwise input gradient kernel"))?;
            let mut launch = context.stream.launch_builder(&context.image.dw_bwd_weight);
            launch
                .arg(&mut weight_grad)
                .arg(&grad.data)
                .arg(&input.data);
            dims.iter().for_each(|d| {
                launch.arg(d);
            });
            // DW_ROWS output pixels a thread, as the kernel reads them.
            let pixels = input.batch * dims[6] as usize * dims[7] as usize;
            launch
                .launch(crate::cuda_training::cfg(pixels.div_ceil(64) * taps))
                .map_err(err("depthwise weight gradient kernel"))?;
        }
        let one = context.filled(1, 1.0)?;
        weight.accumulate_grad_device(&one, &weight_grad, 1)?;
        Ok(input.with_data(input_grad))
    }

    /// `[batch, height, width, channels, kernel, stride, out_height, out_width]`.
    #[cfg(feature = "cuda")]
    fn dims(&self, input: &DeviceImageBatch) -> [i32; 8] {
        let (oh, ow) = self.output_size(input.height, input.width);
        [
            input.batch,
            input.height,
            input.width,
            input.channels,
            self.kernel,
            self.stride,
            oh,
            ow,
        ]
        .map(|v| v as i32)
    }
}

/// The head layout of a `qkv` map for [`pixel_attention`]: each head's
/// channels are its `key_dim` queries, `key_dim` keys and `head_dim` values,
/// in that order, as ultralytics' `Attention` views them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelAttentionShape {
    pub heads: usize,
    pub key_dim: usize,
    pub head_dim: usize,
}

impl PixelAttentionShape {
    /// The shape of ultralytics' `Attention(dim, num_heads, attn_ratio=0.5)`.
    pub fn yolo(dim: usize, heads: usize) -> Self {
        let head_dim = dim / heads;
        Self {
            heads,
            key_dim: head_dim / 2,
            head_dim,
        }
    }

    fn per_head(&self) -> usize {
        2 * self.key_dim + self.head_dim
    }

    /// Channels of the attended output and of the values, `heads * head_dim`.
    pub fn dim(&self) -> usize {
        self.heads * self.head_dim
    }

    fn scale(&self) -> f32 {
        (self.key_dim as f32).powf(-0.5)
    }

    fn check(&self, channels: usize) -> Result<(), NetworkError> {
        match channels == self.heads * self.per_head() {
            true => Ok(()),
            false => Err(NetworkError::InvalidTarget {
                expected: self.heads * self.per_head(),
                actual: channels,
            }),
        }
    }
}

/// What [`pixel_attention_backward`] needs from the forward pass.
#[derive(Clone, Debug)]
pub struct PixelAttentionCache {
    qkv: ImageBatch,
    shape: PixelAttentionShape,
    /// Per sample and head, `[pixels, pixels]`.
    probabilities: Vec<Matrix>,
}

/// One head's `width` channels from `offset` within it, `[pixels, width]`.
fn head_part(map: &ImageBatch, sample: usize, start: usize, width: usize) -> Matrix {
    let (pixels, channels) = (map.pixels(), map.channels());
    let mut part = Matrix::new(pixels, width);
    for p in 0..pixels {
        let row = &map.tokens.data[(sample * pixels + p) * channels..][start..start + width];
        part.data[p * width..(p + 1) * width].copy_from_slice(row);
    }
    part
}

fn add_head_part(map: &mut ImageBatch, sample: usize, start: usize, part: &Matrix) {
    let (pixels, channels, width) = (map.pixels(), map.channels(), part.cols);
    for p in 0..pixels {
        map.tokens.data[(sample * pixels + p) * channels..][start..start + width]
            .iter_mut()
            .zip(part.row(p))
            .for_each(|(slot, v)| *slot += v);
    }
}

/// ultralytics' `Attention` between its `qkv` and `pe`/`proj` convolutions:
/// every pixel attends over all pixels of its sample, per head. Returns the
/// attended map and the values (the input to the positional encoding), both
/// `shape.dim()` channels with head `h` at `h * head_dim`.
pub fn pixel_attention(
    qkv: &ImageBatch,
    shape: PixelAttentionShape,
) -> Result<(ImageBatch, ImageBatch, PixelAttentionCache), NetworkError> {
    shape.check(qkv.channels())?;
    let (pixels, kd, hd) = (qkv.pixels(), shape.key_dim, shape.head_dim);
    let mut attended = qkv.like(shape.dim());
    let mut values = qkv.like(shape.dim());
    let mut probabilities = Vec::new();
    for b in 0..qkv.batch {
        for h in 0..shape.heads {
            let start = h * shape.per_head();
            let q = head_part(qkv, b, start, kd);
            let k = head_part(qkv, b, start + kd, kd);
            let v = head_part(qkv, b, start + 2 * kd, hd);
            let mut p = Matrix::new(pixels, pixels);
            q.dot_rhs_transposed(&k, &mut p);
            for row in p.data.chunks_exact_mut(pixels) {
                let max = row.iter().fold(f32::MIN, |m, &s| m.max(s * shape.scale()));
                let mut sum = 0.0;
                row.iter_mut().for_each(|s| {
                    *s = (*s * shape.scale() - max).exp();
                    sum += *s;
                });
                row.iter_mut().for_each(|s| *s /= sum);
            }
            let mut out = Matrix::new(pixels, hd);
            p.dot(&v, &mut out);
            add_head_part(&mut attended, b, h * hd, &out);
            add_head_part(&mut values, b, h * hd, &v);
            probabilities.push(p);
        }
    }
    Ok((
        attended,
        values,
        PixelAttentionCache {
            qkv: qkv.clone(),
            shape,
            probabilities,
        },
    ))
}

/// The `qkv` gradient from the attended map's and the values' gradients.
pub fn pixel_attention_backward(
    cache: &PixelAttentionCache,
    grad_attended: &ImageBatch,
    grad_values: &ImageBatch,
) -> ImageBatch {
    let (qkv, shape) = (&cache.qkv, cache.shape);
    let (pixels, kd, hd) = (qkv.pixels(), shape.key_dim, shape.head_dim);
    let mut grad = qkv.like(qkv.channels());
    for b in 0..qkv.batch {
        for h in 0..shape.heads {
            let start = h * shape.per_head();
            let p = &cache.probabilities[b * shape.heads + h];
            let q = head_part(qkv, b, start, kd);
            let k = head_part(qkv, b, start + kd, kd);
            let v = head_part(qkv, b, start + 2 * kd, hd);
            let d_out = head_part(grad_attended, b, h * hd, hd);
            let mut d_v = head_part(grad_values, b, h * hd, hd);
            let mut through = Matrix::new(pixels, hd);
            p.dot_self_transposed(&d_out, &mut through);
            d_v.data
                .iter_mut()
                .zip(&through.data)
                .for_each(|(a, b)| *a += b);
            let mut d_s = Matrix::new(pixels, pixels);
            d_out.dot_rhs_transposed(&v, &mut d_s);
            for (ds, pr) in d_s
                .data
                .chunks_exact_mut(pixels)
                .zip(p.data.chunks_exact(pixels))
            {
                let along: f32 = ds.iter().zip(pr).map(|(d, p)| d * p).sum();
                ds.iter_mut()
                    .zip(pr)
                    .for_each(|(d, p)| *d = p * (*d - along) * shape.scale());
            }
            let (mut d_q, mut d_k) = (Matrix::new(pixels, kd), Matrix::new(pixels, kd));
            d_s.dot(&k, &mut d_q);
            d_s.dot_self_transposed(&q, &mut d_k);
            add_head_part(&mut grad, b, start, &d_q);
            add_head_part(&mut grad, b, start + kd, &d_k);
            add_head_part(&mut grad, b, start + 2 * kd, &d_v);
        }
    }
    grad
}

/// What [`pixel_attention_backward_device`] needs from the forward pass.
#[cfg(feature = "cuda")]
pub struct DevicePixelAttentionCache {
    qkv: DeviceImageBatch,
    shape: PixelAttentionShape,
    /// `[batch, heads, pixels, pixels]`.
    probabilities: cudarc::driver::CudaSlice<f32>,
}

/// [`pixel_attention`] on the device, through cuBLAS batched over the heads.
// NOTE: the cache holds every head's `pixels x pixels` probabilities,
// 4 x 400 x 400 floats a sample at the P5 map of a 1280 input (2.6 MB);
// a flash kernel replaces it if larger maps ever attend.
#[cfg(feature = "cuda")]
pub fn pixel_attention_device(
    qkv: &DeviceImageBatch,
    shape: PixelAttentionShape,
) -> Result<
    (
        DeviceImageBatch,
        DeviceImageBatch,
        DevicePixelAttentionCache,
    ),
    NetworkError,
> {
    use crate::gpu_transformer::{gemm_plain_batched, gemm_rhs_transposed_batched};
    shape.check(qkv.channels)?;
    let context = &qkv.context;
    let (n, kd, hd, s, cq, dim) = (
        qkv.pixels(),
        shape.key_dim,
        shape.head_dim,
        shape.per_head(),
        qkv.channels,
        shape.dim(),
    );
    let (rows, span) = (qkv.batch * n, shape.heads * n * n);
    let mut probabilities = context.zeros(qkv.batch * span)?;
    let mut lse = context.zeros(qkv.batch * shape.heads * n)?;
    let mut attended = context.zeros(rows * dim)?;
    let mut values = context.zeros(rows * dim)?;
    for b in 0..qkv.batch {
        let base = b * n * cq;
        gemm_rhs_transposed_batched(
            context,
            &qkv.data.slice(base..),
            cq,
            &qkv.data.slice(base + kd..),
            cq,
            &mut probabilities.slice_mut(b * span..),
            n,
            n,
            n,
            kd,
            shape.scale(),
            0.0,
            shape.heads,
            s,
            s,
            n * n,
        )?;
    }
    let gpu = crate::gpu_model::Gpu { context };
    gpu.attention_softmax(
        &mut probabilities,
        &mut lse,
        qkv.batch * shape.heads * n,
        n,
        false,
        n,
        0,
    )?;
    for b in 0..qkv.batch {
        gemm_plain_batched(
            context,
            &probabilities.slice(b * span..),
            n,
            &qkv.data.slice(b * n * cq + 2 * kd..),
            cq,
            &mut attended.slice_mut(b * n * dim..),
            dim,
            n,
            hd,
            n,
            1.0,
            0.0,
            shape.heads,
            n * n,
            s,
            hd,
        )?;
    }
    for h in 0..shape.heads {
        copy_channels(
            context,
            &mut values,
            dim,
            h * hd,
            &qkv.data,
            cq,
            h * s + 2 * kd,
            hd,
            rows,
        )?;
    }
    Ok((
        DeviceImageBatch {
            channels: dim,
            ..qkv.with_data(attended)
        },
        DeviceImageBatch {
            channels: dim,
            ..qkv.with_data(values)
        },
        DevicePixelAttentionCache {
            qkv: qkv.clone(),
            shape,
            probabilities,
        },
    ))
}

/// [`pixel_attention_backward`] on the device.
#[cfg(feature = "cuda")]
pub fn pixel_attention_backward_device(
    cache: &DevicePixelAttentionCache,
    grad_attended: &DeviceImageBatch,
    grad_values: &DeviceImageBatch,
) -> Result<DeviceImageBatch, NetworkError> {
    use crate::gpu_transformer::{
        gemm_lhs_transposed_batched, gemm_plain_batched, gemm_rhs_transposed_batched,
    };
    let (qkv, shape) = (&cache.qkv, cache.shape);
    let context = &qkv.context;
    let (n, kd, hd, s, cq, dim) = (
        qkv.pixels(),
        shape.key_dim,
        shape.head_dim,
        shape.per_head(),
        qkv.channels,
        shape.dim(),
    );
    let (rows, span) = (qkv.batch * n, shape.heads * n * n);
    let mut grad = context.zeros(rows * cq)?;
    for h in 0..shape.heads {
        copy_channels(
            context,
            &mut grad,
            cq,
            h * s + 2 * kd,
            &grad_values.data,
            dim,
            h * hd,
            hd,
            rows,
        )?;
    }
    let mut d_s = context.zeros(qkv.batch * span)?;
    for b in 0..qkv.batch {
        let base = b * n * cq;
        let d_out = grad_attended.data.slice(b * n * dim..);
        // dV += P^T dO
        gemm_lhs_transposed_batched(
            context,
            &cache.probabilities.slice(b * span..),
            n,
            &d_out,
            dim,
            &mut grad.slice_mut(base + 2 * kd..),
            cq,
            n,
            n,
            hd,
            1.0,
            1.0,
            shape.heads,
            n * n,
            hd,
            s,
        )?;
        // dP = dO V^T
        gemm_rhs_transposed_batched(
            context,
            &d_out,
            dim,
            &qkv.data.slice(base + 2 * kd..),
            cq,
            &mut d_s.slice_mut(b * span..),
            n,
            n,
            n,
            hd,
            1.0,
            0.0,
            shape.heads,
            hd,
            s,
            n * n,
        )?;
    }
    let gpu = crate::gpu_model::Gpu { context };
    gpu.softmax_backward(
        &mut d_s,
        &cache.probabilities,
        qkv.batch * shape.heads * n,
        n,
        false,
    )?;
    for b in 0..qkv.batch {
        let base = b * n * cq;
        // dQ = scale dS K, dK = scale dS^T Q
        gemm_plain_batched(
            context,
            &d_s.slice(b * span..),
            n,
            &qkv.data.slice(base + kd..),
            cq,
            &mut grad.slice_mut(base..),
            cq,
            n,
            kd,
            n,
            shape.scale(),
            0.0,
            shape.heads,
            n * n,
            s,
            s,
        )?;
        gemm_lhs_transposed_batched(
            context,
            &d_s.slice(b * span..),
            n,
            &qkv.data.slice(base..),
            cq,
            &mut grad.slice_mut(base + kd..),
            cq,
            n,
            n,
            kd,
            shape.scale(),
            0.0,
            shape.heads,
            n * n,
            s,
            s,
        )?;
    }
    Ok(qkv.with_data(grad))
}

/// Nearest-neighbour upsampling by `factor` along both axes.
pub fn upsample_nearest_batch(input: &ImageBatch, factor: usize) -> ImageBatch {
    let mut output = ImageBatch::new(
        input.batch,
        input.height * factor,
        input.width * factor,
        input.channels(),
    );
    for b in 0..input.batch {
        for y in 0..output.height {
            for x in 0..output.width {
                let source = input.pixel(b, y / factor, x / factor).to_vec();
                output.pixel_mut(b, y, x).copy_from_slice(&source);
            }
        }
    }
    output
}

/// The gradient of [`upsample_nearest_batch`]: each input pixel sums the
/// `factor x factor` block it was copied to.
pub fn upsample_nearest_backward(grad: &ImageBatch, factor: usize) -> ImageBatch {
    let mut input_grad = ImageBatch::new(
        grad.batch,
        grad.height / factor,
        grad.width / factor,
        grad.channels(),
    );
    for b in 0..grad.batch {
        for y in 0..grad.height {
            for x in 0..grad.width {
                let g = grad.pixel(b, y, x).to_vec();
                input_grad
                    .pixel_mut(b, y / factor, x / factor)
                    .iter_mut()
                    .zip(g)
                    .for_each(|(s, g)| *s += g);
            }
        }
    }
    input_grad
}

/// The maps side by side along the channels, which must share their size.
/// Its gradient is [`split_channels`].
pub fn concat_channels(parts: &[&ImageBatch]) -> ImageBatch {
    let channels = parts.iter().map(|p| p.channels()).sum();
    let mut output = parts[0].like(channels);
    let mut offset = 0;
    for part in parts {
        let width = part.channels();
        for (out, row) in output
            .tokens
            .data
            .chunks_exact_mut(channels)
            .zip(part.tokens.data.chunks_exact(width))
        {
            out[offset..offset + width].copy_from_slice(row);
        }
        offset += width;
    }
    output
}

/// The map cut along the channels into parts of the given widths, which must
/// add up to its channels. Its gradient is [`concat_channels`].
pub fn split_channels(input: &ImageBatch, widths: &[usize]) -> Vec<ImageBatch> {
    let mut offset = 0;
    widths
        .iter()
        .map(|&width| {
            let mut part = input.like(width);
            for (out, row) in part
                .tokens
                .data
                .chunks_exact_mut(width)
                .zip(input.tokens.data.chunks_exact(input.channels()))
            {
                out.copy_from_slice(&row[offset..offset + width]);
            }
            offset += width;
            part
        })
        .collect()
}

/// Which window position won each output of [`max_pool_same`].
#[derive(Clone, Debug)]
pub struct MaxPoolCache {
    kernel: usize,
    argmax: Vec<u32>,
}

/// A `kernel x kernel` max pool with stride one, padded so the size stays,
/// as SPPF's `nn.MaxPool2d(5, 1, 2)`. The first maximum in the window wins a
/// tie.
pub fn max_pool_same(input: &ImageBatch, kernel: usize) -> (ImageBatch, MaxPoolCache) {
    let (height, width, channels, pad) = (input.height, input.width, input.channels(), kernel / 2);
    let mut output = input.like(channels);
    let mut argmax = vec![0u32; input.tokens.data.len()];
    for b in 0..input.batch {
        for y in 0..height {
            for x in 0..width {
                let at = ((b * height + y) * width + x) * channels;
                for c in 0..channels {
                    let (mut best, mut arg) = (f32::MIN, 0);
                    for ky in 0..kernel {
                        for kx in 0..kernel {
                            let (sy, sx) = ((y + ky).wrapping_sub(pad), (x + kx).wrapping_sub(pad));
                            if sy < height && sx < width {
                                let v = input.pixel(b, sy, sx)[c];
                                if v > best {
                                    (best, arg) = (v, ky * kernel + kx);
                                }
                            }
                        }
                    }
                    output.tokens.data[at + c] = best;
                    argmax[at + c] = arg as u32;
                }
            }
        }
    }
    (output, MaxPoolCache { kernel, argmax })
}

/// Routes each output's gradient to the input that won its window.
pub fn max_pool_same_backward(cache: &MaxPoolCache, grad: &ImageBatch) -> ImageBatch {
    let (height, width, channels, kernel) =
        (grad.height, grad.width, grad.channels(), cache.kernel);
    let pad = kernel / 2;
    let mut input_grad = grad.like(channels);
    for (index, (&g, &arg)) in grad.tokens.data.iter().zip(&cache.argmax).enumerate() {
        let (c, pixel) = (index % channels, index / channels);
        let (x, y, b) = (
            pixel % width,
            pixel / width % height,
            pixel / width / height,
        );
        let (ky, kx) = (arg as usize / kernel, arg as usize % kernel);
        let (sy, sx) = (y + ky - pad, x + kx - pad);
        input_grad.tokens.data[((b * height + sy) * width + sx) * channels + c] += g;
    }
    input_grad
}

/// [`upsample_nearest_batch`] on the device.
#[cfg(feature = "cuda")]
pub fn upsample_nearest_device(
    input: &DeviceImageBatch,
    factor: usize,
) -> Result<DeviceImageBatch, NetworkError> {
    let total = input.data.len() * factor * factor;
    let out = image_launch(
        input,
        total,
        "upsample kernel",
        |context| &context.image.upsample,
        &[input.height, input.width, input.channels, factor],
    )?;
    Ok(DeviceImageBatch {
        height: input.height * factor,
        width: input.width * factor,
        ..input.with_data(out)
    })
}

/// [`upsample_nearest_backward`] on the device.
#[cfg(feature = "cuda")]
pub fn upsample_nearest_backward_device(
    grad: &DeviceImageBatch,
    factor: usize,
) -> Result<DeviceImageBatch, NetworkError> {
    let (height, width) = (grad.height / factor, grad.width / factor);
    let total = grad.data.len() / (factor * factor);
    let out = image_launch(
        grad,
        total,
        "upsample backward kernel",
        |context| &context.image.upsample_bwd,
        &[height, width, grad.channels, factor],
    )?;
    Ok(DeviceImageBatch {
        height,
        width,
        ..grad.with_data(out)
    })
}

/// `a + b` on the device, for residual connections.
#[cfg(feature = "cuda")]
pub fn add_device(
    a: &DeviceImageBatch,
    b: &DeviceImageBatch,
) -> Result<DeviceImageBatch, NetworkError> {
    let context = &a.context;
    let total = a.data.len();
    if b.data.len() != total {
        return Err(NetworkError::InvalidTarget {
            expected: total,
            actual: b.data.len(),
        });
    }
    let mut out = context.uninit(total)?;
    let n = total as i32;
    unsafe {
        context
            .stream
            .launch_builder(&context.image.add)
            .arg(&mut out)
            .arg(&a.data)
            .arg(&b.data)
            .arg(&n)
            .launch(crate::cuda_training::cfg(total))
            .map_err(crate::cuda_training::cuda_err("add kernel"))?;
    }
    Ok(a.with_data(out))
}

/// [`concat_channels`] on the device.
#[cfg(feature = "cuda")]
pub fn concat_channels_device(
    parts: &[&DeviceImageBatch],
) -> Result<DeviceImageBatch, NetworkError> {
    let first = parts[0];
    let channels: usize = parts.iter().map(|p| p.channels).sum();
    let rows = first.batch * first.pixels();
    let mut out = first.context.uninit(rows * channels)?;
    let mut offset = 0;
    for part in parts {
        copy_channels(
            &first.context,
            &mut out,
            channels,
            offset,
            &part.data,
            part.channels,
            0,
            part.channels,
            rows,
        )?;
        offset += part.channels;
    }
    Ok(DeviceImageBatch {
        channels,
        ..first.with_data(out)
    })
}

/// [`split_channels`] on the device.
#[cfg(feature = "cuda")]
pub fn split_channels_device(
    input: &DeviceImageBatch,
    widths: &[usize],
) -> Result<Vec<DeviceImageBatch>, NetworkError> {
    let mut offset = 0;
    widths
        .iter()
        .map(|&width| {
            offset += width;
            slice_channels_device(input, offset - width, width)
        })
        .collect()
}

/// `width` channels from `offset`, copied out on the device.
#[cfg(feature = "cuda")]
pub fn slice_channels_device(
    input: &DeviceImageBatch,
    offset: usize,
    width: usize,
) -> Result<DeviceImageBatch, NetworkError> {
    let rows = input.batch * input.pixels();
    let mut out = input.context.uninit(rows * width)?;
    copy_channels(
        &input.context,
        &mut out,
        width,
        0,
        &input.data,
        input.channels,
        offset,
        width,
        rows,
    )?;
    Ok(DeviceImageBatch {
        channels: width,
        ..input.with_data(out)
    })
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn copy_channels(
    context: &crate::gpu_transformer::GpuContext,
    dst: &mut cudarc::driver::CudaSlice<f32>,
    dst_channels: usize,
    dst_offset: usize,
    src: &cudarc::driver::CudaSlice<f32>,
    src_channels: usize,
    src_offset: usize,
    width: usize,
    rows: usize,
) -> Result<(), NetworkError> {
    let [rows_arg, width_arg, dc, doff, sc, soff] = [
        rows,
        width,
        dst_channels,
        dst_offset,
        src_channels,
        src_offset,
    ]
    .map(|v| v as i32);
    unsafe {
        context
            .stream
            .launch_builder(&context.image.copy_channels)
            .arg(dst)
            .arg(src)
            .arg(&rows_arg)
            .arg(&width_arg)
            .arg(&dc)
            .arg(&doff)
            .arg(&sc)
            .arg(&soff)
            .launch(crate::cuda_training::cfg(rows * width))
            .map_err(crate::cuda_training::cuda_err("channel copy kernel"))?;
    }
    Ok(())
}

/// What [`max_pool_same_backward_device`] needs from the forward pass.
#[cfg(feature = "cuda")]
pub struct DeviceMaxPoolCache {
    kernel: usize,
    argmax: cudarc::driver::CudaSlice<i32>,
}

/// [`max_pool_same`] on the device.
#[cfg(feature = "cuda")]
pub fn max_pool_same_device(
    input: &DeviceImageBatch,
    kernel: usize,
) -> Result<(DeviceImageBatch, DeviceMaxPoolCache), NetworkError> {
    let context = &input.context;
    let total = input.data.len();
    let mut out = context.uninit(total)?;
    let mut argmax = context
        .stream
        .alloc_zeros::<i32>(total)
        .map_err(crate::cuda_training::cuda_err("max pool allocation"))?;
    let [n, h, w, c, k] =
        [total, input.height, input.width, input.channels, kernel].map(|v| v as i32);
    unsafe {
        context
            .stream
            .launch_builder(&context.image.maxpool)
            .arg(&mut out)
            .arg(&mut argmax)
            .arg(&input.data)
            .arg(&n)
            .arg(&h)
            .arg(&w)
            .arg(&c)
            .arg(&k)
            .launch(crate::cuda_training::cfg(total))
            .map_err(crate::cuda_training::cuda_err("max pool kernel"))?;
    }
    Ok((input.with_data(out), DeviceMaxPoolCache { kernel, argmax }))
}

/// [`max_pool_same_backward`] on the device, gathering rather than
/// scattering so it is deterministic.
#[cfg(feature = "cuda")]
pub fn max_pool_same_backward_device(
    cache: &DeviceMaxPoolCache,
    grad: &DeviceImageBatch,
) -> Result<DeviceImageBatch, NetworkError> {
    let context = &grad.context;
    let total = grad.data.len();
    let mut out = context.zeros(total)?;
    let [n, h, w, c, k] =
        [total, grad.height, grad.width, grad.channels, cache.kernel].map(|v| v as i32);
    unsafe {
        context
            .stream
            .launch_builder(&context.image.maxpool_bwd)
            .arg(&mut out)
            .arg(&grad.data)
            .arg(&cache.argmax)
            .arg(&n)
            .arg(&h)
            .arg(&w)
            .arg(&c)
            .arg(&k)
            .launch(crate::cuda_training::cfg(total))
            .map_err(crate::cuda_training::cuda_err("max pool backward kernel"))?;
    }
    Ok(grad.with_data(out))
}

/// One thread per output element of a kernel taking `(out, in, total, dims...)`.
#[cfg(feature = "cuda")]
fn image_launch(
    input: &DeviceImageBatch,
    total: usize,
    what: &'static str,
    kernel: impl Fn(&crate::gpu_transformer::GpuContext) -> &cudarc::driver::CudaFunction,
    dims: &[usize],
) -> Result<cudarc::driver::CudaSlice<f32>, NetworkError> {
    let context = &input.context;
    let mut out = context.uninit(total)?;
    let n = total as i32;
    let dims: Vec<i32> = dims.iter().map(|&d| d as i32).collect();
    let mut launch = context.stream.launch_builder(kernel(context));
    launch.arg(&mut out).arg(&input.data).arg(&n);
    for d in &dims {
        launch.arg(d);
    }
    unsafe { launch.launch(crate::cuda_training::cfg(total)) }
        .map_err(crate::cuda_training::cuda_err(what))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::Optimizer;
    use rand::SeedableRng;

    /// A deterministic batch of activations, spread either side of zero.
    fn noise(
        batch: usize,
        height: usize,
        width: usize,
        channels: usize,
        seed: usize,
    ) -> ImageBatch {
        let data = (0..batch * height * width * channels)
            .map(|index| (((index * 37 + seed * 11) % 197) as f32 / 98.0) - 1.0)
            .collect();
        ImageBatch::from_tokens(
            batch,
            height,
            width,
            Matrix::from_vec(batch * height * width, channels, data),
        )
    }

    /// Mean squared error against a fixed target, and its gradient.
    fn mse(prediction: &ImageBatch, target: &ImageBatch) -> (f32, ImageBatch) {
        let count = prediction.tokens.data.len() as f32;
        let mut grad = prediction.like(prediction.channels());
        let mut loss = 0.0;
        for (index, (value, wanted)) in prediction
            .tokens
            .data
            .iter()
            .zip(&target.tokens.data)
            .enumerate()
        {
            let difference = value - wanted;
            loss += difference * difference / count;
            grad.tokens.data[index] = 2.0 * difference / count;
        }
        (loss, grad)
    }

    /// One hundred Adam steps, returning the loss before each one.
    fn loss_curve(
        conv: &mut TrainableConv2d,
        input: &ImageBatch,
        target: &ImageBatch,
        steps: usize,
    ) -> Result<Vec<f32>, NetworkError> {
        let optimizer = Optimizer::adam(1e-2);
        let mut curve = Vec::with_capacity(steps);
        for step in 1..=steps {
            let (prediction, cache) = conv.forward(input)?;
            let (loss, grad_output) = mse(&prediction, target);
            curve.push(loss);
            for param in conv.params_mut() {
                param.zero_grad();
            }
            conv.backward(&cache, &grad_output)?;
            for param in conv.params_mut() {
                param.step(&optimizer, step, 1.0);
            }
        }
        Ok(curve)
    }

    /// `sum(f(input) * weights)`'s gradient by central differences, against
    /// `analytic`, relative to the largest analytic value.
    fn pooled_gap(
        f: impl Fn(&ImageBatch) -> Matrix,
        input: &ImageBatch,
        weights: &[f32],
        analytic: &[f32],
    ) -> f32 {
        let loss = |input: &ImageBatch| -> f32 {
            f(input).data.iter().zip(weights).map(|(a, b)| a * b).sum()
        };
        let scale = analytic.iter().fold(0f32, |m, g| m.max(g.abs()));
        let mut moved = input.clone();
        let mut worst = 0f32;
        for (index, analytic) in analytic.iter().enumerate() {
            let h = 1e-2;
            moved.tokens.data[index] += h;
            let up = loss(&moved);
            moved.tokens.data[index] -= 2.0 * h;
            let down = loss(&moved);
            moved.tokens.data[index] += h;
            worst = worst.max(((up - down) / (2.0 * h) - analytic).abs() / scale);
        }
        worst
    }

    #[test]
    fn spatial_softmax_matches_finite_differences_and_finds_a_peak() {
        let mut input = noise(2, 3, 4, 5, 3);
        input.tokens.data.iter_mut().for_each(|v| *v *= 3.0);
        let weights = noise(1, 1, 20, 1, 4).tokens.data;
        let (_, cache) = spatial_softmax(&input);
        let grad = spatial_softmax_backward(&cache, &Matrix::from_vec(2, 10, weights.clone()));
        let gap = pooled_gap(
            |x| spatial_softmax(x).0,
            &input,
            &weights,
            &grad.tokens.data,
        );
        assert!(gap < 1e-2, "{gap}");

        // A spike at sample 0, row 0, column 3, channel 2 is the top right.
        let mut spike = ImageBatch::new(2, 3, 4, 5);
        spike.tokens.data[3 * 5 + 2] = 50.0;
        let (expected, _) = spatial_softmax(&spike);
        assert!((expected.data[2] - 1.0).abs() < 1e-4);
        assert!((expected.data[5 + 2] + 1.0).abs() < 1e-4);
    }

    #[test]
    fn trainable_group_norm_matches_the_inference_one_and_finite_differences() {
        let input = noise(2, 3, 4, 8, 3);
        let mut norm = TrainableGroupNorm::new(4, 8, 1e-5).unwrap();
        norm.gamma.value.data = noise(1, 1, 8, 1, 4)
            .tokens
            .data
            .iter()
            .map(|v| v + 1.5)
            .collect();
        norm.beta.value.data = noise(1, 1, 8, 1, 5).tokens.data;
        let fixed = GroupNorm::new(
            4,
            norm.gamma.value.data.clone(),
            norm.beta.value.data.clone(),
            1e-5,
        )
        .unwrap();
        let (output, cache) = norm.forward(&input).unwrap();
        for sample in 0..2 {
            let rows = 12 * sample..12 * (sample + 1);
            let tokens = Matrix::from_vec(
                12,
                8,
                input.tokens.data[rows.start * 8..rows.end * 8].to_vec(),
            );
            let mut map = FeatureMap::from_tokens(&tokens, 3, 4).unwrap();
            fixed.forward(&mut map).unwrap();
            assert_eq!(
                map.to_tokens().data,
                output.tokens.data[rows.start * 8..rows.end * 8]
            );
        }

        let weights = noise(2, 3, 4, 8, 6);
        let grad = norm.backward(&cache, &weights);
        let through = |input: &ImageBatch| norm.forward(input).unwrap().0.tokens;
        let gap = pooled_gap(through, &input, &weights.tokens.data, &grad.tokens.data);
        assert!(gap < 1e-3, "input {gap}");
        for (which, analytic) in [(0, &norm.gamma.grad.data), (1, &norm.beta.grad.data)] {
            let params = ImageBatch::from_tokens(
                1,
                1,
                8,
                Matrix::from_vec(
                    8,
                    1,
                    if which == 0 {
                        norm.gamma.value.data.clone()
                    } else {
                        norm.beta.value.data.clone()
                    },
                ),
            );
            let through = |params: &ImageBatch| {
                let mut moved = norm.clone();
                let target = if which == 0 {
                    &mut moved.gamma
                } else {
                    &mut moved.beta
                };
                target.value.data.copy_from_slice(&params.tokens.data);
                moved.forward(&input).unwrap().0.tokens
            };
            let gap = pooled_gap(through, &params, &weights.tokens.data, analytic);
            assert!(gap < 1e-3, "param {which}: {gap}");
        }
        assert!(TrainableGroupNorm::new(3, 8, 1e-5).is_err());
    }

    #[test]
    fn global_avg_pool_matches_finite_differences() {
        let input = noise(2, 3, 4, 5, 5);
        let weights = noise(1, 1, 10, 1, 6).tokens.data;
        let grad = global_avg_pool_backward(&Matrix::from_vec(2, 5, weights.clone()), 3, 4);
        let gap = pooled_gap(global_avg_pool, &input, &weights, &grad.tokens.data);
        assert!(gap < 1e-2, "{gap}");
        assert!((global_avg_pool(&ImageBatch::new(1, 2, 2, 3)).data[0]).abs() < 1e-9);
    }

    #[test]
    fn a_convolution_without_bias_adds_and_trains_none() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(12);
        let mut biased = TrainableConv2d::new(3, 4, 3, 2, &mut rng).unwrap();
        biased.bias.value.data.fill(1.0);
        let mut plain = biased.clone().without_bias();
        assert!(!plain.has_bias());
        assert_eq!(plain.params_mut().len() + 1, biased.params_mut().len());

        let (input, target) = (noise(2, 5, 6, 3, 1), noise(2, 3, 3, 4, 2));
        let (with, with_cache) = biased.forward(&input).unwrap();
        let (without, cache) = plain.forward(&input).unwrap();
        for (with, without) in with.tokens.data.iter().zip(&without.tokens.data) {
            assert!((with - without - 1.0).abs() < 1e-6);
        }
        let (_, grad) = mse(&without, &target);
        biased.backward(&with_cache, &grad).unwrap();
        plain.backward(&cache, &grad).unwrap();
        assert_eq!(
            plain.weight.weight.grad.data,
            biased.weight.weight.grad.data
        );
        assert!(plain.bias.grad.data.iter().all(|g| *g == 0.0));
        assert!(plain.bias.value.data.iter().all(|b| *b == 0.0));
    }

    #[test]
    fn a_trainable_convolution_matches_finite_differences() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let mut conv = TrainableConv2d::new(3, 4, 3, 1, &mut rng).unwrap();
        let input = noise(2, 5, 6, 3, 1);
        let target = noise(2, 5, 6, 4, 2);

        let (prediction, cache) = conv.forward(&input).unwrap();
        let (_, grad_output) = mse(&prediction, &target);
        let grad_input = conv.backward(&cache, &grad_output).unwrap();

        let epsilon = 1e-3;
        for index in [0usize, 17, 43, 88] {
            let mut moved = input.clone();
            moved.tokens.data[index] += epsilon;
            let (up, _) = conv.forward(&moved).unwrap();
            moved.tokens.data[index] -= 2.0 * epsilon;
            let (down, _) = conv.forward(&moved).unwrap();
            let numerical = (mse(&up, &target).0 - mse(&down, &target).0) / (2.0 * epsilon);
            let analytic = grad_input.tokens.data[index];
            assert!(
                (numerical - analytic).abs() < 1e-3,
                "dL/dinput[{index}]: {analytic} against {numerical} by finite difference"
            );
        }

        for index in [0usize, 7, 26] {
            let mut moved = conv.clone();
            moved.weight.weight.value.data[index] += epsilon;
            let up = mse(&moved.forward(&input).unwrap().0, &target).0;
            moved.weight.weight.value.data[index] -= 2.0 * epsilon;
            let down = mse(&moved.forward(&input).unwrap().0, &target).0;
            let numerical = (up - down) / (2.0 * epsilon);
            let analytic = conv.weight.weight.grad.data[index];
            assert!(
                (numerical - analytic).abs() < 1e-3,
                "dL/dweight[{index}]: {analytic} against {numerical} by finite difference"
            );
        }
    }

    #[test]
    fn training_a_convolution_drives_its_loss_down() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(17);
        let mut conv = TrainableConv2d::new(8, 8, 3, 1, &mut rng).unwrap();
        let input = noise(1, 8, 8, 8, 9);
        let target = noise(1, 8, 8, 8, 10);
        let curve = loss_curve(&mut conv, &input, &target, 20).unwrap();
        assert!(
            curve[19] < curve[0] / 4.0,
            "twenty Adam steps went from {} to {}",
            curve[0],
            curve[19]
        );
    }

    #[test]
    fn a_strided_convolution_shrinks_the_map_and_still_reaches_every_pixel() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let mut conv = TrainableConv2d::new(2, 3, 3, 2, &mut rng).unwrap();
        let input = noise(1, 8, 8, 2, 3);
        let (output, cache) = conv.forward(&input).unwrap();
        assert_eq!((output.height, output.width), (4, 4));

        let grad_output = noise(1, 4, 4, 3, 4);
        let grad_input = conv.backward(&cache, &grad_output).unwrap();
        assert_eq!(grad_input.tokens.rows, input.tokens.rows);
        assert!(
            grad_input.tokens.data.iter().any(|value| *value != 0.0),
            "a stride of two still reads most of the input"
        );
    }

    #[test]
    fn rebuilding_the_columns_reaches_the_same_gradients_as_keeping_them() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(31);
        let mut keeping = TrainableConv2d::new(6, 8, 3, 2, &mut rng).unwrap();
        let mut rebuilding = keeping.clone();
        rebuilding.set_recompute_columns(true);
        let input = noise(2, 11, 9, 6, 3);

        let (kept, kept_cache) = keeping.forward(&input).unwrap();
        let (rebuilt, rebuilt_cache) = rebuilding.forward(&input).unwrap();
        assert_eq!(
            kept.tokens.data, rebuilt.tokens.data,
            "the forward pass is the same either way"
        );
        assert!(kept_cache.holds_columns());
        assert!(
            !rebuilt_cache.holds_columns(),
            "the whole point is that the columns are not held"
        );

        let grad_output = noise(2, kept.height, kept.width, 8, 4);
        let kept_input = keeping.backward(&kept_cache, &grad_output).unwrap();
        let rebuilt_input = rebuilding.backward(&rebuilt_cache, &grad_output).unwrap();

        assert_eq!(kept_input.tokens.data, rebuilt_input.tokens.data);
        assert_eq!(
            keeping.weight.weight.grad.data,
            rebuilding.weight.weight.grad.data
        );
        assert_eq!(keeping.bias.grad.data, rebuilding.bias.grad.data);
    }

    /// The option has to survive a whole run, not one step: the cache it builds
    /// is read by a backward pass whose weight has already moved twice.
    #[test]
    fn a_convolution_that_rebuilds_its_columns_trains_to_the_same_place() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(37);
        let mut keeping = TrainableConv2d::new(8, 8, 3, 1, &mut rng).unwrap();
        let mut rebuilding = keeping.clone();
        rebuilding.set_recompute_columns(true);
        let input = noise(1, 12, 12, 8, 5);
        let target = noise(1, 12, 12, 8, 6);

        let kept = loss_curve(&mut keeping, &input, &target, 40).unwrap();
        let rebuilt = loss_curve(&mut rebuilding, &input, &target, 40).unwrap();
        assert!(
            kept[39] < kept[0] / 2.0,
            "the reference run has to be learning"
        );
        assert_eq!(kept, rebuilt);
    }

    #[cfg(feature = "cuda")]
    fn cuda_or_skip() -> Option<crate::param::CudaDevice> {
        match crate::cuda_training::cuda_doctor(0, 4096) {
            Ok(_) => Some(crate::param::CudaDevice::new(0, 4096).expect("the CUDA doctor passed")),
            Err(NetworkError::Cuda(message))
                if message.contains("NO_DEVICE") || message.contains("no CUDA-capable device") =>
            {
                crate::cuda_training::skip_without_cuda();
                None
            }
            Err(error) => panic!("CUDA is present but the CUDA doctor failed: {error}"),
        }
    }

    /// The acceptance test for the device path: a 96-channel 3x3 convolution
    /// over a 64x64 map trains on the device and its loss curve follows the
    /// host's for a hundred steps.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_convolution_trained_on_the_device_follows_the_hosts_loss_curve() {
        let Some(device) = cuda_or_skip() else {
            return;
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(29);
        let mut host = TrainableConv2d::new(96, 96, 3, 1, &mut rng).unwrap();
        let mut resident = host.clone();
        let input = noise(1, 64, 64, 96, 7);
        let target = noise(1, 64, 64, 96, 8);

        let on_host = loss_curve(&mut host, &input, &target, 100).unwrap();
        resident.to_cuda_on(&device).unwrap();
        let on_device = loss_curve(&mut resident, &input, &target, 100).unwrap();

        assert!(
            on_host[99] < on_host[0] / 2.0,
            "the host path has to be learning"
        );
        for (step, (a, b)) in on_host.iter().zip(&on_device).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 + 1e-3 * a.abs(),
                "step {step}: {a} on the host against {b} on the device"
            );
        }

        // And the weights themselves agree, not only the losses they produced.
        resident.to_cpu().unwrap();
        for (index, (a, b)) in host
            .weight
            .weight
            .value
            .data
            .iter()
            .zip(&resident.weight.weight.value.data)
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-3,
                "weight[{index}]: {a} on the host against {b} on the device"
            );
        }
    }

    /// A 3x3 convolution on a mixed-precision device, where `im2col` builds
    /// bf16 columns and all three GEMMs run in bf16, stays within bf16
    /// rounding of the host.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_mixed_precision_convolution_tracks_the_host() {
        let Some(_) = cuda_or_skip() else {
            return;
        };
        let device = crate::param::CudaDevice::with_mixed_precision(0, 1024).unwrap();
        // Columns kept and rebuilt, with channels in fours and not.
        for (channels, recompute) in [(8, false), (6, true)] {
            let mut rng = rand::rngs::StdRng::seed_from_u64(32);
            let mut conv = TrainableConv2d::new(channels, 16, 3, 1, &mut rng).unwrap();
            conv.bias.value.data = noise(1, 1, 16, 1, 3).tokens.data;
            conv.set_recompute_columns(recompute);
            let mut host = conv.clone();
            conv.to_cuda_on(&device).unwrap();
            let input = noise(2, 10, 9, channels, 1);
            let grad = noise(2, 10, 9, 16, 2);
            let (expected, cache) = host.forward(&input).unwrap();
            let expected_grad = host.backward(&cache, &grad).unwrap();
            let (output, cache) = conv.forward_device(&device.upload(&input).unwrap()).unwrap();
            let input_grad = conv
                .backward_device(&cache, &device.upload(&grad).unwrap())
                .unwrap();
            // bf16 keeps 8 bits of mantissa: about 0.4% per operand.
            let near = |a: &[f32], b: &[f32], what: &str| {
                let gap: f32 = a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum();
                let scale: f32 = b.iter().map(|b| b * b).sum();
                assert!(gap.sqrt() < 1e-2 * scale.sqrt(), "{what}: {gap} against {scale}");
            };
            near(
                &output.download().unwrap().tokens.data,
                &expected.tokens.data,
                "output",
            );
            near(
                &input_grad.download().unwrap().tokens.data,
                &expected_grad.tokens.data,
                "input grad",
            );
            conv.to_cpu().unwrap();
            near(
                &conv.weight.weight.grad.data,
                &host.weight.weight.grad.data,
                "weight grad",
            );
            near(&conv.bias.grad.data, &host.bias.grad.data, "bias grad");
        }
    }

    /// Two biased convolutions chained on the device: activations and
    /// gradients cross the bus only at the two ends, and every output and
    /// gradient matches the host chain.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_device_chain_stays_on_the_card_and_matches_the_host() {
        let Some(device) = cuda_or_skip() else {
            return;
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(31);
        let mut first = TrainableConv2d::new(6, 8, 3, 2, &mut rng).unwrap();
        // A 1x1 convolution, whose columns are its input.
        let mut second = TrainableConv2d::new(8, 5, 1, 1, &mut rng).unwrap();
        first.bias.value.data = noise(1, 1, 8, 1, 3).tokens.data;
        second.bias.value.data = noise(1, 1, 5, 1, 4).tokens.data;
        second.set_recompute_columns(true);
        let (mut host_first, mut host_second) = (first.clone(), second.clone());
        first.to_cuda_on(&device).unwrap();
        second.to_cuda_on(&device).unwrap();

        let input = noise(3, 11, 9, 6, 1);
        let (middle, first_cache) = host_first.forward(&input).unwrap();
        let (expected, second_cache) = host_second.forward(&middle).unwrap();
        let grad = noise(3, expected.height, expected.width, 5, 2);
        let expected_grad = host_first
            .backward(
                &first_cache,
                &host_second.backward(&second_cache, &grad).unwrap(),
            )
            .unwrap();

        let context = device.context();
        let resident = device.upload(&input).unwrap();
        let resident_grad = device.upload(&grad).unwrap();
        let before = context.transfers();
        let (middle, first_cache) = first.forward_device(&resident).unwrap();
        let (output, second_cache) = second.forward_device(&middle).unwrap();
        let back = second
            .backward_device(&second_cache, &resident_grad)
            .unwrap();
        let input_grad = first.backward_device(&first_cache, &back).unwrap();
        assert_eq!(
            context.transfers(),
            before,
            "nothing crosses inside the chain"
        );

        let close = |a: &[f32], b: &[f32], what: &str| {
            assert_eq!(a.len(), b.len(), "{what}");
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4 + 1e-4 * b.abs(),
                    "{what}[{index}]: {a} against {b}"
                );
            }
        };
        close(
            &device.download(&output).unwrap().tokens.data,
            &expected.tokens.data,
            "output",
        );
        close(
            &input_grad.download().unwrap().tokens.data,
            &expected_grad.tokens.data,
            "input grad",
        );
        first.to_cpu().unwrap();
        second.to_cpu().unwrap();
        for (device_side, host_side, what) in [
            (&first, &host_first, "first"),
            (&second, &host_second, "second"),
        ] {
            close(
                &device_side.weight.weight.grad.data,
                &host_side.weight.weight.grad.data,
                what,
            );
            close(&device_side.bias.grad.data, &host_side.bias.grad.data, what);
        }
    }

    /// Group norm then SiLU on the device, forward and back, against the host
    /// layer and the host SiLU, with nothing crossing the bus in between.
    #[cfg(feature = "cuda")]
    #[test]
    fn device_group_norm_and_silu_match_the_host() {
        let Some(device) = cuda_or_skip() else {
            return;
        };
        let input = noise(2, 16, 17, 12, 1);
        let grad = noise(2, 16, 17, 12, 2);
        let mut host = TrainableGroupNorm::new(4, 12, 1e-5).unwrap();
        host.gamma.value.data = noise(1, 1, 12, 1, 3)
            .tokens
            .data
            .iter()
            .map(|v| v + 1.2)
            .collect();
        host.beta.value.data = noise(1, 1, 12, 1, 4).tokens.data;
        let mut resident = host.clone();
        resident.to_cuda_on(&device).unwrap();

        let (normed, cache) = host.forward(&input).unwrap();
        let mut activated = normed.clone();
        let mut back = grad.clone();
        for ((value, grad), pre) in activated
            .tokens
            .data
            .iter_mut()
            .zip(&mut back.tokens.data)
            .zip(&normed.tokens.data)
        {
            *value = crate::ffn::silu(*pre);
            *grad *= crate::ffn::silu_derivative(*pre);
        }
        let expected_grad = host.backward(&cache, &back);

        let context = device.context();
        let (on_card, on_card_grad) = (
            device.upload(&input).unwrap(),
            device.upload(&grad).unwrap(),
        );
        let before = context.transfers();
        let (normed, cache) = resident.forward_device(&on_card).unwrap();
        let output = silu_device(&normed).unwrap();
        let back = silu_backward_device(&normed, &on_card_grad).unwrap();
        let input_grad = resident.backward_device(&cache, &back).unwrap();
        assert_eq!(
            context.transfers(),
            before,
            "nothing crosses inside the chain"
        );

        let close = |a: &[f32], b: &[f32], what: &str| {
            assert_eq!(a.len(), b.len(), "{what}");
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4 + 1e-4 * b.abs(),
                    "{what}[{index}]: {a} against {b}"
                );
            }
        };
        close(
            &output.download().unwrap().tokens.data,
            &activated.tokens.data,
            "output",
        );
        close(
            &input_grad.download().unwrap().tokens.data,
            &expected_grad.tokens.data,
            "input grad",
        );
        resident.to_cpu().unwrap();
        close(
            &resident.gamma.grad.data,
            &host.gamma.grad.data,
            "gamma grad",
        );
        close(&resident.beta.grad.data, &host.beta.grad.data, "beta grad");
    }

    /// Spatial softmax and average pooling on the device against the host.
    #[cfg(feature = "cuda")]
    #[test]
    fn device_spatial_softmax_and_pooling_match_the_host() {
        let Some(device) = cuda_or_skip() else {
            return;
        };
        let input = noise(3, 17, 19, 5, 1);
        let (expected, cache) = spatial_softmax(&input);
        let grad = noise(3, 1, 1, 10, 2).tokens;
        let expected_grad = spatial_softmax_backward(&cache, &grad);
        let pooled = global_avg_pool(&input);
        let pool_grad = noise(3, 1, 1, 5, 3).tokens;
        let expected_pool_grad = global_avg_pool_backward(&pool_grad, 17, 19);

        let context = device.context();
        let on_card = device.upload(&input).unwrap();
        let on_card_grad = device
            .upload(&ImageBatch::from_tokens(3, 1, 1, grad))
            .unwrap();
        let on_card_pool_grad = device
            .upload(&ImageBatch::from_tokens(3, 1, 1, pool_grad))
            .unwrap();
        let before = context.transfers();
        let (output, cache) = spatial_softmax_device(&on_card).unwrap();
        let back = spatial_softmax_backward_device(&cache, &on_card_grad).unwrap();
        let mean = global_avg_pool_device(&on_card).unwrap();
        let spread = global_avg_pool_backward_device(&on_card_pool_grad, 17, 19).unwrap();
        assert_eq!(
            context.transfers(),
            before,
            "nothing crosses inside the chain"
        );

        let close = |a: &[f32], b: &[f32], what: &str| {
            assert_eq!(a.len(), b.len(), "{what}");
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5 + 1e-4 * b.abs(),
                    "{what}[{index}]: {a} against {b}"
                );
            }
        };
        close(
            &output.download().unwrap().tokens.data,
            &expected.data,
            "keypoints",
        );
        close(
            &back.download().unwrap().tokens.data,
            &expected_grad.tokens.data,
            "keypoint grad",
        );
        close(&mean.download().unwrap().tokens.data, &pooled.data, "pool");
        close(
            &spread.download().unwrap().tokens.data,
            &expected_pool_grad.tokens.data,
            "pool grad",
        );
    }

    /// The host path is unchanged by a device being present, which is what
    /// makes the parity test above meaningful.
    #[cfg(feature = "cuda")]
    #[test]
    fn im2col_on_the_device_builds_the_same_columns_as_the_host() {
        let Some(device) = cuda_or_skip() else {
            return;
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(13);
        let conv = TrainableConv2d::new(5, 7, 3, 2, &mut rng).unwrap();
        let input = noise(2, 9, 11, 5, 6);
        let shape = conv.geometry(input.batch, input.height, input.width);
        let expected = conv.im2col(&input, &shape);

        let context = device.context();
        let resident = context.upload(&input.tokens).unwrap();
        let columns = context.im2col(&resident, &shape).unwrap();
        let mut actual = Matrix::new(expected.rows, expected.cols);
        context.download(&columns, &mut actual).unwrap();
        assert_eq!(actual.data, expected.data);

        // col2im is its transpose, so scattering the columns back has to match
        // the host's scatter exactly too.
        let scattered = context.col2im(&columns, &shape).unwrap();
        let mut actual = Matrix::new(input.tokens.rows, input.tokens.cols);
        context.download(&scattered, &mut actual).unwrap();
        let expected = conv.col2im(&expected, &shape);
        for (index, (a, b)) in actual.data.iter().zip(&expected.tokens.data).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "col2im[{index}]: {a} on the device against {b} on the host"
            );
        }
    }

    /// The device path has its own cache, so it needs its own parity check:
    /// rebuilding the columns from an uploaded input has to reach the same
    /// gradients as reading the ones the forward pass left on the card.
    #[cfg(feature = "cuda")]
    #[test]
    fn rebuilding_the_columns_on_the_device_reaches_the_same_gradients() {
        let Some(device) = cuda_or_skip() else {
            return;
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(41);
        let mut keeping = TrainableConv2d::new(16, 24, 3, 1, &mut rng).unwrap();
        let mut rebuilding = keeping.clone();
        rebuilding.set_recompute_columns(true);
        keeping.to_cuda_on(&device).unwrap();
        rebuilding.to_cuda_on(&device).unwrap();
        let input = noise(2, 16, 16, 16, 9);

        let (kept, kept_cache) = keeping.forward(&input).unwrap();
        let (rebuilt, rebuilt_cache) = rebuilding.forward(&input).unwrap();
        assert_eq!(kept.tokens.data, rebuilt.tokens.data);
        assert!(!rebuilt_cache.holds_columns());

        let grad_output = noise(2, kept.height, kept.width, 24, 10);
        let kept_input = keeping.backward(&kept_cache, &grad_output).unwrap();
        let rebuilt_input = rebuilding.backward(&rebuilt_cache, &grad_output).unwrap();
        assert_eq!(kept_input.tokens.data, rebuilt_input.tokens.data);

        keeping.to_cpu().unwrap();
        rebuilding.to_cpu().unwrap();
        assert_eq!(
            keeping.weight.weight.grad.data,
            rebuilding.weight.weight.grad.data
        );
        assert_eq!(keeping.bias.grad.data, rebuilding.bias.grad.data);
    }

    fn ramp(channels: usize, height: usize, width: usize) -> FeatureMap {
        let data = (0..channels * height * width)
            .map(|value| value as f32)
            .collect();
        FeatureMap::from_vec(channels, height, width, data).unwrap()
    }

    #[test]
    fn quantizing_a_dense_keeps_its_answer_and_drops_its_floats() {
        let weight = Matrix::from_vec(
            3,
            4,
            (0..12).map(|index| (index as f32 * 0.7).sin()).collect(),
        );
        let tokens = Matrix::from_vec(2, 4, (0..8).map(|index| index as f32 * 0.25).collect());
        let mut layer = Dense::new(weight, Some(vec![0.5, -0.5, 0.25])).unwrap();
        let expected = layer.forward(&tokens).unwrap();

        layer.quantize();
        assert!(layer.is_quantized());
        assert!(layer.weight.data.is_empty());
        assert_eq!((layer.out_dim(), layer.in_dim()), (3, 4));

        let actual = layer.forward(&tokens).unwrap();
        for (actual, expected) in actual.data.iter().zip(&expected.data) {
            assert!(
                (actual - expected).abs() < 0.02,
                "{actual} is not {expected}"
            );
        }
    }

    #[test]
    fn a_one_by_one_convolution_is_a_per_pixel_linear_layer() {
        // Two output channels: the first sums the inputs, the second negates
        // the second input.
        let weight = Matrix::from_vec(2, 2, vec![1.0, 1.0, 0.0, -1.0]);
        let conv = Conv2d::new(weight, Some(vec![0.5, 0.0]), 2, 1, 1, 0).unwrap();
        let input = ramp(2, 2, 2);

        let output = conv.forward(&input).unwrap();

        assert_eq!(output.channels, 2);
        assert_eq!((output.height, output.width), (2, 2));
        // Channel zero: input[0] + input[1] + 0.5, over pixels 0..4 where the
        // second plane starts at 4.
        assert_eq!(output.plane(0), &[4.5, 6.5, 8.5, 10.5]);
        assert_eq!(output.plane(1), &[-4.0, -5.0, -6.0, -7.0]);
    }

    #[test]
    fn a_padded_three_by_three_convolution_keeps_the_size_and_reads_zeros_outside() {
        // A kernel that picks the pixel above, so the top row reads padding.
        let mut weight = vec![0.0; 9];
        weight[1] = 1.0;
        let conv = Conv2d::new(Matrix::from_vec(1, 9, weight), None, 1, 3, 1, 1).unwrap();
        let input = ramp(1, 3, 3);

        let output = conv.forward(&input).unwrap();

        assert_eq!((output.height, output.width), (3, 3));
        assert_eq!(
            output.data,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn a_strided_convolution_halves_the_resolution() {
        let conv = Conv2d::new(Matrix::from_vec(1, 4, vec![0.25; 4]), None, 1, 2, 2, 0).unwrap();
        let output = conv.forward(&ramp(1, 4, 4)).unwrap();

        assert_eq!((output.height, output.width), (2, 2));
        // Each output is the mean of its 2x2 block.
        assert_eq!(output.data, vec![2.5, 4.5, 10.5, 12.5]);
    }

    #[test]
    fn a_convolution_refuses_shapes_it_cannot_read() {
        let conv = Conv2d::new(Matrix::from_vec(1, 4, vec![1.0; 4]), None, 1, 2, 1, 0).unwrap();
        assert!(conv.forward(&ramp(2, 4, 4)).is_err());
        assert!(conv.forward(&ramp(1, 1, 1)).is_err());
        assert!(Conv2d::new(Matrix::from_vec(1, 3, vec![1.0; 3]), None, 1, 2, 1, 0).is_err());
        assert!(
            Conv2d::new(
                Matrix::from_vec(1, 4, vec![1.0; 4]),
                Some(vec![0.0; 2]),
                1,
                2,
                1,
                0
            )
            .is_err()
        );
    }

    #[test]
    fn group_norm_standardizes_each_group_over_its_own_pixels() {
        let mut map = ramp(4, 2, 2);
        let norm = GroupNorm::new(2, vec![1.0; 4], vec![0.0; 4], 1e-5).unwrap();
        norm.forward(&mut map).unwrap();

        for group in 0..2 {
            let values = &map.data[group * 8..(group + 1) * 8];
            let mean = values.iter().sum::<f32>() / 8.0;
            let variance = values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f32>()
                / 8.0;
            assert!(mean.abs() < 1e-5, "{mean}");
            assert!((variance - 1.0).abs() < 1e-3, "{variance}");
        }

        // The scale and shift are per channel, not per group.
        let mut map = ramp(2, 1, 2);
        let norm = GroupNorm::new(1, vec![2.0, 0.5], vec![1.0, -1.0], 1e-5).unwrap();
        norm.forward(&mut map).unwrap();
        assert!(
            (map.data[0] - (-1.341_640_8 * 2.0 + 1.0)).abs() < 1e-4,
            "{:?}",
            map.data
        );

        assert!(GroupNorm::new(3, vec![1.0; 4], vec![0.0; 4], 1e-5).is_err());
    }

    #[test]
    fn upsampling_repeats_pixels_and_shuffling_round_trips() {
        let map = ramp(1, 2, 2);
        let large = upsample_nearest(&map, 2);
        assert_eq!((large.height, large.width), (4, 4));
        assert_eq!(&large.data[..4], &[0.0, 0.0, 1.0, 1.0]);
        assert_eq!(&large.data[4..8], &[0.0, 0.0, 1.0, 1.0]);

        let packed = ramp(8, 2, 3);
        let shuffled = pixel_shuffle(&packed, 2).unwrap();
        assert_eq!(
            (shuffled.channels, shuffled.height, shuffled.width),
            (2, 4, 6)
        );
        assert_eq!(pixel_unshuffle(&shuffled, 2).unwrap(), packed);

        assert!(pixel_shuffle(&packed, 3).is_err());
        assert!(pixel_unshuffle(&ramp(1, 3, 3), 2).is_err());
    }

    #[test]
    fn a_map_survives_the_trip_through_token_shape() {
        let map = ramp(3, 2, 4);
        let tokens = map.to_tokens();
        assert_eq!((tokens.rows, tokens.cols), (8, 3));
        assert_eq!(tokens.row(1), &[1.0, 9.0, 17.0]);
        assert_eq!(FeatureMap::from_tokens(&tokens, 2, 4).unwrap(), map);
        assert!(FeatureMap::from_tokens(&tokens, 3, 4).is_err());
    }
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(a, b)| a * b).sum()
    }

    fn batch_norm_for_tests() -> TrainableBatchNorm2d {
        let mut norm = TrainableBatchNorm2d::yolo(6);
        norm.gamma.value.data = noise(1, 1, 6, 1, 4)
            .tokens
            .data
            .iter()
            .map(|v| v + 1.5)
            .collect();
        norm.beta.value.data = noise(1, 1, 6, 1, 5).tokens.data;
        norm.running.value.data = noise(1, 2, 6, 1, 9)
            .tokens
            .data
            .iter()
            .enumerate()
            .map(|(i, v)| if i < 6 { *v } else { v.abs() + 0.5 })
            .collect();
        norm
    }

    #[test]
    fn batch_norm_matches_finite_differences_in_both_modes_and_tracks_running_statistics() {
        let input = noise(2, 3, 5, 6, 3);
        let weights = noise(2, 3, 5, 6, 6);
        for training in [true, false] {
            let mut norm = batch_norm_for_tests();
            let before = norm.running.value.data.clone();
            let (output, cache) = norm.forward(&input, training).unwrap();
            let grad = norm.backward(&cache, &weights);
            let through = |input: &ImageBatch| {
                batch_norm_for_tests()
                    .forward(input, training)
                    .unwrap()
                    .0
                    .tokens
            };
            let gap = pooled_gap(through, &input, &weights.tokens.data, &grad.tokens.data);
            assert!(gap < 1e-3, "training {training}: input {gap}");
            // gamma and beta enter linearly, so their gradients are exact sums.
            let normalized: Vec<f32> = output
                .tokens
                .data
                .iter()
                .enumerate()
                .map(|(i, y)| (y - norm.beta.value.data[i % 6]) / norm.gamma.value.data[i % 6])
                .collect();
            for c in 0..6 {
                let (mut dg, mut db) = (0.0, 0.0);
                for r in 0..30 {
                    dg += weights.tokens.data[r * 6 + c] * normalized[r * 6 + c];
                    db += weights.tokens.data[r * 6 + c];
                }
                assert!((norm.gamma.grad.data[c] - dg).abs() < 1e-4);
                assert!((norm.beta.grad.data[c] - db).abs() < 1e-4);
            }
            if training {
                let c = 2;
                let column: Vec<f32> = (0..30).map(|r| input.tokens.data[r * 6 + c]).collect();
                let mean = column.iter().sum::<f32>() / 30.0;
                let unbiased = column.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 29.0;
                let running = &norm.running.value.data;
                assert!((running[c] - (0.97 * before[c] + 0.03 * mean)).abs() < 1e-6);
                assert!((running[6 + c] - (0.97 * before[6 + c] + 0.03 * unbiased)).abs() < 1e-6);
            } else {
                assert_eq!(norm.running.value.data, before);
            }
        }
    }

    #[test]
    fn a_fused_convolution_equals_the_convolution_then_the_evaluating_norm() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let conv = TrainableConv2d::new(4, 6, 3, 2, &mut rng)
            .unwrap()
            .without_bias();
        let mut norm = batch_norm_for_tests();
        let input = noise(2, 7, 9, 4, 1);
        let (expected, _) = norm
            .forward(&conv.forward(&input).unwrap().0, false)
            .unwrap();
        let fused = norm.fuse_into(&conv).forward(&input).unwrap().0;
        for (a, b) in fused.tokens.data.iter().zip(&expected.tokens.data) {
            assert!((a - b).abs() < 1e-5, "{a} against {b}");
        }
    }

    #[test]
    fn upsample_concat_split_and_max_pool_backwards_are_their_adjoints() {
        let input = noise(2, 3, 4, 5, 1);
        let up = upsample_nearest_batch(&input, 2);
        assert_eq!((up.height, up.width), (6, 8));
        assert_eq!(up.pixel(1, 5, 7), input.pixel(1, 2, 3));
        let grad = noise(2, 6, 8, 5, 2);
        let back = upsample_nearest_backward(&grad, 2);
        let gap =
            dot(&up.tokens.data, &grad.tokens.data) - dot(&input.tokens.data, &back.tokens.data);
        assert!(gap.abs() < 1e-3, "upsample {gap}");

        let other = noise(2, 3, 4, 2, 7);
        let joined = concat_channels(&[&input, &other]);
        assert_eq!(joined.pixel(1, 2, 3)[5..], *other.pixel(1, 2, 3));
        let parts = split_channels(&joined, &[5, 2]);
        assert_eq!(parts[0].tokens.data, input.tokens.data);
        assert_eq!(parts[1].tokens.data, other.tokens.data);

        // Distinct values keep every window's winner clear of a tie.
        let mut distinct = noise(2, 6, 7, 3, 4);
        distinct
            .tokens
            .data
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v += i as f32 * 1e-3);
        let (pooled, cache) = max_pool_same(&distinct, 5);
        assert_eq!(
            pooled.pixel(0, 0, 0)[0],
            (0..3)
                .flat_map(|y| (0..3).map(move |x| (y, x)))
                .map(|(y, x)| distinct.pixel(0, y, x)[0])
                .fold(f32::MIN, f32::max)
        );
        let weights = noise(2, 6, 7, 3, 8);
        let back = max_pool_same_backward(&cache, &weights);
        let through = |input: &ImageBatch| max_pool_same(input, 5).0.tokens;
        let gap = pooled_gap(through, &distinct, &weights.tokens.data, &back.tokens.data);
        assert!(gap < 1e-3, "max pool {gap}");
    }

    /// The detection layers on the device against the host.
    #[cfg(feature = "cuda")]
    #[test]
    fn device_detection_layers_match_the_host() {
        let Some(device) = cuda_or_skip() else {
            return;
        };
        let close = |a: &[f32], b: &[f32], what: &str| {
            assert_eq!(a.len(), b.len(), "{what}");
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4 + 1e-4 * b.abs(),
                    "{what}[{index}]: {a} against {b}"
                );
            }
        };
        let input = noise(3, 9, 11, 6, 1);
        let weights = noise(3, 9, 11, 6, 2);
        let on_card = device.upload(&input).unwrap();
        let on_card_weights = device.upload(&weights).unwrap();
        for training in [true, false] {
            let mut host = batch_norm_for_tests();
            let mut resident = batch_norm_for_tests();
            resident.to_cuda_on(&device).unwrap();
            let (expected, cache) = host.forward(&input, training).unwrap();
            let expected_grad = host.backward(&cache, &weights);
            let (output, cache) = resident.forward_device(&on_card, training).unwrap();
            let grad = resident.backward_device(&cache, &on_card_weights).unwrap();
            close(
                &output.download().unwrap().tokens.data,
                &expected.tokens.data,
                "norm",
            );
            close(
                &grad.download().unwrap().tokens.data,
                &expected_grad.tokens.data,
                "norm grad",
            );
            resident.to_cpu().unwrap();
            close(
                &resident.gamma.grad.data,
                &host.gamma.grad.data,
                "gamma grad",
            );
            close(&resident.beta.grad.data, &host.beta.grad.data, "beta grad");
            close(
                &resident.running.value.data,
                &host.running.value.data,
                "running",
            );
        }

        let up = upsample_nearest_device(&on_card, 2).unwrap();
        close(
            &up.download().unwrap().tokens.data,
            &upsample_nearest_batch(&input, 2).tokens.data,
            "upsample",
        );
        let back = upsample_nearest_backward_device(&up, 2).unwrap();
        close(
            &back.download().unwrap().tokens.data,
            &upsample_nearest_backward(&upsample_nearest_batch(&input, 2), 2)
                .tokens
                .data,
            "upsample grad",
        );

        let joined = concat_channels_device(&[&on_card, &on_card_weights]).unwrap();
        close(
            &joined.download().unwrap().tokens.data,
            &concat_channels(&[&input, &weights]).tokens.data,
            "concat",
        );
        let parts = split_channels_device(&joined, &[4, 8]).unwrap();
        for (part, expected) in parts.iter().zip(split_channels(
            &concat_channels(&[&input, &weights]),
            &[4, 8],
        )) {
            close(
                &part.download().unwrap().tokens.data,
                &expected.tokens.data,
                "split",
            );
        }
        let sum = add_device(&on_card, &on_card_weights).unwrap();
        let mut expected = input.clone();
        expected.add(&weights);
        close(
            &sum.download().unwrap().tokens.data,
            &expected.tokens.data,
            "add",
        );

        let (pooled, cache) = max_pool_same(&input, 5);
        let (on_card_pooled, device_cache) = max_pool_same_device(&on_card, 5).unwrap();
        close(
            &on_card_pooled.download().unwrap().tokens.data,
            &pooled.tokens.data,
            "max pool",
        );
        let back = max_pool_same_backward_device(&device_cache, &on_card_weights).unwrap();
        close(
            &back.download().unwrap().tokens.data,
            &max_pool_same_backward(&cache, &weights).tokens.data,
            "max pool grad",
        );
    }
    #[test]
    fn depthwise_convolution_matches_finite_differences() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        for stride in [1, 2] {
            let mut conv = DepthwiseConv2d::new(4, 3, stride, &mut rng);
            let input = noise(2, 7, 6, 4, 1);
            let (output, cache) = conv.forward(&input).unwrap();
            assert_eq!((output.height, output.width), conv.output_size(7, 6));
            // Channel 1's centre pixel against a direct sum.
            let w = &conv.weight.value.data[9..18];
            let (y, x) = (1, 2 / stride);
            let direct: f32 = (0..9)
                .map(|t| input.pixel(1, y * stride + t / 3 - 1, x * stride + t % 3 - 1)[1] * w[t])
                .sum();
            assert!((output.pixel(1, y, x)[1] - direct).abs() < 1e-5);
            let weights = noise(2, output.height, output.width, 4, 6);
            let grad = conv.backward(&cache, &weights);
            let through = |input: &ImageBatch| conv.forward(input).unwrap().0.tokens;
            let gap = pooled_gap(through, &input, &weights.tokens.data, &grad.tokens.data);
            assert!(gap < 1e-3, "stride {stride}: input {gap}");
            let as_batch =
                |data: Vec<f32>| ImageBatch::from_tokens(1, 1, 36, Matrix::from_vec(36, 1, data));
            let through = |params: &ImageBatch| {
                let mut moved = conv.clone();
                moved.weight.value.data.copy_from_slice(&params.tokens.data);
                moved.forward(&input).unwrap().0.tokens
            };
            let gap = pooled_gap(
                through,
                &as_batch(conv.weight.value.data.clone()),
                &weights.tokens.data,
                &conv.weight.grad.data,
            );
            assert!(gap < 1e-3, "stride {stride}: weight {gap}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn device_depthwise_convolution_matches_the_host() {
        let Some(device) = cuda_or_skip() else {
            return;
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        for stride in [1, 2] {
            let mut host = DepthwiseConv2d::new(6, 3, stride, &mut rng);
            let mut resident = host.clone();
            resident.to_cuda_on(&device).unwrap();
            let input = noise(3, 13, 10, 6, 1);
            let (expected, cache) = host.forward(&input).unwrap();
            let weights = noise(3, expected.height, expected.width, 6, 2);
            let expected_grad = host.backward(&cache, &weights);
            let on_card = device.upload(&input).unwrap();
            let output = resident.forward_device(&on_card).unwrap();
            let grad = resident
                .backward_device(&on_card, &device.upload(&weights).unwrap())
                .unwrap();
            let close = |a: &[f32], b: &[f32], what: &str| {
                assert_eq!(a.len(), b.len(), "{what}");
                for (index, (a, b)) in a.iter().zip(b).enumerate() {
                    assert!(
                        (a - b).abs() < 1e-4 + 1e-4 * b.abs(),
                        "{what}[{index}]: {a} against {b}"
                    );
                }
            };
            close(
                &output.download().unwrap().tokens.data,
                &expected.tokens.data,
                "output",
            );
            close(
                &grad.download().unwrap().tokens.data,
                &expected_grad.tokens.data,
                "input grad",
            );
            resident.to_cpu().unwrap();
            close(
                &resident.weight.grad.data,
                &host.weight.grad.data,
                "weight grad",
            );
        }
    }

    #[test]
    fn pixel_attention_matches_ultralytics() {
        let fixture = std::path::Path::new("/home/vasylt/cs2/fixtures/k13a_attention.json");
        let Ok(text) = std::fs::read_to_string(fixture) else {
            eprintln!("skipped: no {}", fixture.display());
            return;
        };
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        let size = |key: &str| json[key].as_u64().unwrap() as usize;
        let map = |key: &str| {
            let data: Vec<f32> = serde_json::from_value(json[key].clone()).unwrap();
            let rows = size("batch") * size("height") * size("width");
            let tokens = Matrix::from_vec(rows, data.len() / rows, data);
            ImageBatch::from_tokens(size("batch"), size("height"), size("width"), tokens)
        };
        let shape = PixelAttentionShape::yolo(size("dim"), size("heads"));
        let (attended, values, cache) = pixel_attention(&map("qkv"), shape).unwrap();
        let grad = pixel_attention_backward(&cache, &map("grad_attended"), &map("grad_values"));
        for (got, key) in [
            (attended, "attended"),
            (values, "values"),
            (grad, "grad_qkv"),
        ] {
            let gap = got
                .tokens
                .data
                .iter()
                .zip(&map(key).tokens.data)
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(gap < 1e-5, "{key}: {gap}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn device_pixel_attention_matches_the_host() {
        let Some(device) = cuda_or_skip() else {
            return;
        };
        let shape = PixelAttentionShape::yolo(16, 2);
        let qkv = noise(3, 5, 4, 32, 1);
        let (attended, values, cache) = pixel_attention(&qkv, shape).unwrap();
        let (grad_attended, grad_values) = (noise(3, 5, 4, 16, 2), noise(3, 5, 4, 16, 3));
        let grad = pixel_attention_backward(&cache, &grad_attended, &grad_values);
        let (on_attended, on_values, on_cache) =
            pixel_attention_device(&device.upload(&qkv).unwrap(), shape).unwrap();
        let on_grad = pixel_attention_backward_device(
            &on_cache,
            &device.upload(&grad_attended).unwrap(),
            &device.upload(&grad_values).unwrap(),
        )
        .unwrap();
        for (got, expected, what) in [
            (on_attended, attended, "attended"),
            (on_values, values, "values"),
            (on_grad, grad, "qkv grad"),
        ] {
            let got = got.download().unwrap();
            for (index, (a, b)) in got
                .tokens
                .data
                .iter()
                .zip(&expected.tokens.data)
                .enumerate()
            {
                assert!(
                    (a - b).abs() < 1e-4 + 1e-4 * b.abs(),
                    "{what}[{index}]: {a} against {b}"
                );
            }
        }
    }
}
