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
//! NOTE: no backward pass and no CUDA. Both are additive — `Param` and a
//! device buffer can replace the plain `Matrix` without changing the shapes —
//! and neither is needed to run a checkpoint someone else trained.

use crate::matrix::Matrix;
use crate::network::NetworkError;
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
