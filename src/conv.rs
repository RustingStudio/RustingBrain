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
//! ponytail: [`Conv2d`] has no backward pass and no CUDA. Both are additive —
//! `Param` and a device buffer can replace the plain `Matrix` without changing
//! the shapes — and neither is needed to run a checkpoint someone else trained.
//!
//! # Training a convolution
//!
//! [`TrainableConv2d`] is the other half: a batched convolution whose weight
//! and bias are [`Param`](crate::param::Param)s, with a backward pass and a
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
    #[cfg(feature = "cuda")]
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
    /// `[1, out_channels]`.
    pub bias: Param,
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

    /// Moves the weight onto a device a caller already opened. Every later
    /// [`TrainableConv2d::forward`] and [`TrainableConv2d::backward`] then
    /// builds its columns and runs its GEMMs there, and only the activation
    /// and its gradient cross the bus.
    ///
    /// The bias stays on the host: both its use and its gradient are host
    /// passes over buffers already there. Moving it too left the device copy
    /// owning the value while the gradient landed in the host copy, so the
    /// optimizer stepped it with zero and it never trained.
    pub fn to_cuda_on(&mut self, device: &crate::param::CudaDevice) -> Result<(), NetworkError> {
        self.weight.to_cuda_on(device)
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
        if self.weight.weight.device.is_some() {
            return self.forward_on_device(input, shape);
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
            Columns::Device(columns) => self.backward_on_device(columns, &cache.shape, grad_output),
            #[cfg(feature = "cuda")]
            Columns::DeviceInput(input) => {
                let device = self
                    .weight
                    .weight
                    .device
                    .as_ref()
                    .expect("a device-resident cache means a device-resident weight");
                let context = device.context().clone();
                let columns = context.im2col(input, &cache.shape)?;
                self.backward_on_device(&columns, &cache.shape, grad_output)
            }
            Columns::Input(input) => {
                self.accumulate_bias_grad(&grad_output.tokens);
                let columns = self.im2col(input, &cache.shape);
                let grad_columns = self.weight.backward(&columns, &grad_output.tokens);
                Ok(self.col2im(&grad_columns, &cache.shape))
            }
        }
    }

    /// The forward pass with the columns built and consumed on the device.
    #[cfg(feature = "cuda")]
    fn forward_on_device(
        &self,
        input: &ImageBatch,
        shape: ConvGeometry,
    ) -> Result<(ImageBatch, ConvCache), NetworkError> {
        let device = self
            .weight
            .weight
            .device
            .as_ref()
            .expect("the caller checked residency");
        let context = device.context();
        let resident_input = context.upload(&input.tokens)?;
        let columns = context.im2col(&resident_input, &shape)?;
        let product = device.matmul_rhs_transposed_device(&columns, shape.rows())?;
        let mut tokens = Matrix::new(shape.rows(), self.out_channels());
        context.download(&product, &mut tokens)?;
        self.add_bias(&mut tokens);
        Ok((
            ImageBatch::from_tokens(input.batch, shape.out_height, shape.out_width, tokens),
            ConvCache {
                // Dropping the device columns here is the whole point of the
                // option: they are freed before the next layer allocates its
                // own, so a deep network holds one layer's worth rather than
                // every layer's.
                columns: match self.recompute_columns {
                    true => Columns::DeviceInput(resident_input),
                    false => Columns::Device(columns),
                },
                shape,
            },
        ))
    }

    #[cfg(feature = "cuda")]
    fn backward_on_device(
        &mut self,
        columns: &cudarc::driver::CudaSlice<f32>,
        shape: &ConvGeometry,
        grad_output: &ImageBatch,
    ) -> Result<ImageBatch, NetworkError> {
        self.accumulate_bias_grad(&grad_output.tokens);
        let device = self
            .weight
            .weight
            .device
            .as_mut()
            .expect("a device-resident cache means a device-resident weight");
        let context = device.context().clone();
        let resident_grad = context.upload(&grad_output.tokens)?;
        device.accumulate_grad_device(&resident_grad, columns, shape.rows())?;
        let grad_columns = device.matmul_device(&resident_grad, shape.rows())?;
        let grad_input = context.col2im(&grad_columns, shape)?;
        let mut tokens = Matrix::new(shape.batch * shape.height * shape.width, shape.channels);
        context.download(&grad_input, &mut tokens)?;
        Ok(ImageBatch::from_tokens(
            shape.batch,
            shape.height,
            shape.width,
            tokens,
        ))
    }

    fn add_bias(&self, tokens: &mut Matrix) {
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
        if self.bias.is_frozen() {
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
        params.push(&mut self.bias);
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
}
