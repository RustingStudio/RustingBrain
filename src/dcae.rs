//! The DC-AE decoder (`AutoencoderDC`, `dc-ae-f32c32-sana-1.1`): 32 latent
//! channels at 1/32 scale back to RGB.
//!
//! Activations are token-major, `[pixels, channels]`, so every 1x1
//! convolution is a [`Dense`] and a 3x3 one is a banded im2col over whole
//! neighbour rows.

use crate::conv::{Dense, FeatureMap};
use crate::ffn::silu;
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::safetensors::ShardedSafeTensors;
use crate::sana::{linear, linear_attention, transpose};
use rayon::prelude::*;
use std::path::Path;

/// `decoder_block_out_channels`, shallowest first.
pub(crate) const CHANNELS: [usize; 6] = [128, 256, 512, 512, 1024, 1024];
pub(crate) const HEAD_DIM: usize = 32;
const EPS: f32 = 1e-5;

/// A 3x3 convolution, padding 1, with its weight reordered to
/// `[out, (ky, kx, in)]` so a patch row is nine contiguous channel rows.
pub(crate) struct Conv3 {
    pub(crate) weight: Matrix,
    pub(crate) bias: Option<Vec<f32>>,
    pub(crate) inputs: usize,
}

impl Conv3 {
    pub(crate) fn read(
        file: &mut ShardedSafeTensors,
        name: &str,
        bias: bool,
    ) -> Result<Self, NetworkError> {
        let (values, shape) = file.tensor(&format!("{name}.weight"))?;
        let (outputs, inputs) = (shape[0], shape[1]);
        let mut weight = Matrix::new(outputs, 9 * inputs);
        for o in 0..outputs {
            for i in 0..inputs {
                for k in 0..9 {
                    weight.data[o * 9 * inputs + k * inputs + i] = values[(o * inputs + i) * 9 + k];
                }
            }
        }
        let bias = match bias {
            true => Some(file.tensor(&format!("{name}.bias"))?.0),
            false => None,
        };
        Ok(Self {
            weight,
            bias,
            inputs,
        })
    }

    /// Convolves a `height x width` map. With `upsample`, the input is half
    /// that size and read through a 2x nearest upsampling.
    pub(crate) fn forward(
        &self,
        input: &Matrix,
        height: usize,
        width: usize,
        upsample: bool,
    ) -> Matrix {
        let (c, outputs) = (self.inputs, self.weight.rows);
        let source_width = if upsample { width / 2 } else { width };
        let source = |y: usize, x: usize| match upsample {
            true => (y / 2) * source_width + x / 2,
            false => y * source_width + x,
        };
        let mut out = Matrix::new(height * width, outputs);
        // NOTE: bands of ~256 MB of im2col; a direct conv kernel if the
        // host decoder ever needs to be fast.
        let band = ((64 << 20) / (width * 9 * c)).max(1);
        for top in (0..height).step_by(band) {
            let rows = band.min(height - top);
            let mut columns = Matrix::new(rows * width, 9 * c);
            columns
                .data
                .par_chunks_mut(9 * c)
                .enumerate()
                .for_each(|(pixel, patch)| {
                    let (y, x) = (top + pixel / width, pixel % width);
                    for k in 0..9 {
                        let (sy, sx) = (
                            y as isize + k as isize / 3 - 1,
                            x as isize + k as isize % 3 - 1,
                        );
                        if sy < 0 || sx < 0 || sy >= height as isize || sx >= width as isize {
                            continue;
                        }
                        let row = source(sy as usize, sx as usize);
                        patch[k * c..(k + 1) * c]
                            .copy_from_slice(&input.data[row * c..(row + 1) * c]);
                    }
                });
            let mut product = Matrix::new(rows * width, outputs);
            columns.dot_rhs_transposed(&self.weight, &mut product);
            if let Some(bias) = &self.bias {
                product.data.par_chunks_mut(outputs).for_each(|row| {
                    row.iter_mut().zip(bias).for_each(|(v, b)| *v += b);
                });
            }
            out.data[top * width * outputs..][..rows * width * outputs]
                .copy_from_slice(&product.data);
        }
        out
    }
}

/// A `k x k` depthwise convolution, zero padded, over token-major input;
/// `weight` is `[channels, k, k]`.
pub(crate) fn depthwise(
    x: &Matrix,
    height: usize,
    width: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    k: usize,
) -> Matrix {
    let c = x.cols;
    let half = (k / 2) as isize;
    let mut out = Matrix::new(x.rows, c);
    out.data
        .par_chunks_mut(width * c)
        .enumerate()
        .for_each(|(y, rows)| {
            for xx in 0..width {
                let target = &mut rows[xx * c..(xx + 1) * c];
                if let Some(bias) = bias {
                    target.copy_from_slice(bias);
                }
                for dy in 0..k {
                    let sy = y as isize + dy as isize - half;
                    if sy < 0 || sy >= height as isize {
                        continue;
                    }
                    for dx in 0..k {
                        let sx = xx as isize + dx as isize - half;
                        if sx < 0 || sx >= width as isize {
                            continue;
                        }
                        let source = &x.data[(sy as usize * width + sx as usize) * c..][..c];
                        let tap = dy * k + dx;
                        for ch in 0..c {
                            target[ch] += weight[ch * k * k + tap] * source[ch];
                        }
                    }
                }
            }
        });
    out
}

/// RMS norm over channels with a gain and a bias, in place.
pub(crate) fn rms_affine(x: &mut Matrix, weight: &[f32], bias: &[f32]) {
    x.data.par_chunks_mut(weight.len()).for_each(|row| {
        let mean = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let inverse = (mean + EPS).sqrt().recip();
        for ((v, w), b) in row.iter_mut().zip(weight).zip(bias) {
            *v = *v * inverse * w + b;
        }
    });
}

fn add(target: &mut Matrix, other: &Matrix) {
    target
        .data
        .par_iter_mut()
        .zip(&other.data)
        .for_each(|(a, b)| *a += b);
}

pub(crate) struct Norm {
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}

impl Norm {
    fn read(file: &mut ShardedSafeTensors, name: &str) -> Result<Self, NetworkError> {
        Ok(Self {
            weight: file.tensor(&format!("{name}.weight"))?.0,
            bias: file.tensor(&format!("{name}.bias"))?.0,
        })
    }
}

pub(crate) enum Block {
    /// `DCUpBlock2d`, interpolate mode, with its channel-repeat shortcut.
    Up {
        conv: Conv3,
        repeats: usize,
    },
    Res {
        conv1: Conv3,
        conv2: Conv3,
        norm: Norm,
    },
    Vit(Box<Vit>),
}

/// `EfficientViTBlock`: multiscale linear attention, then GLUMBConv.
pub(crate) struct Vit {
    pub(crate) query: Dense,
    pub(crate) key: Dense,
    pub(crate) value: Dense,
    /// Depthwise 5x5 over the `3C` qkv channels, `[3C, 25]`.
    pub(crate) multiscale_in: Vec<f32>,
    /// Grouped 1x1, 32 channels per group, `[3C, 32]`.
    pub(crate) multiscale_out: Vec<f32>,
    pub(crate) out: Dense,
    pub(crate) attention_norm: Norm,
    pub(crate) inverted: Dense,
    pub(crate) depth_weight: Vec<f32>,
    pub(crate) depth_bias: Vec<f32>,
    pub(crate) point: Dense,
    pub(crate) norm: Norm,
}

pub struct DcaeDecoder {
    pub(crate) conv_in: Conv3,
    /// `up_blocks.5` first, as the decoder runs them.
    pub(crate) stages: Vec<Vec<Block>>,
    pub(crate) norm_out: Norm,
    pub(crate) conv_out: Conv3,
}

impl DcaeDecoder {
    /// Reads the decoder half of `vae/diffusion_pytorch_model.safetensors`.
    pub fn load(model: impl AsRef<Path>) -> Result<Self, NetworkError> {
        let mut file = ShardedSafeTensors::open(
            model
                .as_ref()
                .join("vae/diffusion_pytorch_model.safetensors"),
        )?;
        let f = &mut file;
        let mut stages = Vec::new();
        for stage in (0..6).rev() {
            let channels = CHANNELS[stage];
            let mut blocks = Vec::new();
            let mut at = 0;
            if stage < 5 {
                blocks.push(Block::Up {
                    conv: Conv3::read(f, &format!("decoder.up_blocks.{stage}.0.conv"), true)?,
                    repeats: channels * 4 / CHANNELS[stage + 1],
                });
                at = 1;
            }
            for index in at..at + 3 {
                let name = |part: &str| format!("decoder.up_blocks.{stage}.{index}.{part}");
                blocks.push(match stage >= 3 {
                    false => Block::Res {
                        conv1: Conv3::read(f, &name("conv1"), true)?,
                        conv2: Conv3::read(f, &name("conv2"), false)?,
                        norm: Norm::read(f, &name("norm"))?,
                    },
                    true => Block::Vit(Box::new(Vit {
                        query: linear(f, &name("attn.to_q"), false)?,
                        key: linear(f, &name("attn.to_k"), false)?,
                        value: linear(f, &name("attn.to_v"), false)?,
                        multiscale_in: f
                            .tensor(&name("attn.to_qkv_multiscale.0.proj_in.weight"))?
                            .0,
                        multiscale_out: f
                            .tensor(&name("attn.to_qkv_multiscale.0.proj_out.weight"))?
                            .0,
                        out: linear(f, &name("attn.to_out"), false)?,
                        attention_norm: Norm::read(f, &name("attn.norm_out"))?,
                        inverted: linear(f, &name("conv_out.conv_inverted"), true)?,
                        depth_weight: f.tensor(&name("conv_out.conv_depth.weight"))?.0,
                        depth_bias: f.tensor(&name("conv_out.conv_depth.bias"))?.0,
                        point: linear(f, &name("conv_out.conv_point"), false)?,
                        norm: Norm::read(f, &name("conv_out.norm"))?,
                    })),
                });
            }
            stages.push(blocks);
        }
        Ok(Self {
            conv_in: Conv3::read(f, "decoder.conv_in", true)?,
            stages,
            norm_out: Norm::read(f, "decoder.norm_out")?,
            conv_out: Conv3::read(f, "decoder.conv_out", true)?,
        })
    }

    /// Decodes a latent already divided by the scaling factor to an image in
    /// `[-1, 1]`.
    pub fn decode(&self, latent: &FeatureMap) -> Result<FeatureMap, NetworkError> {
        self.decode_probed(latent, |_, _| {})
    }

    /// [`DcaeDecoder::decode`], handing each stage's output (`up_blocks.5`
    /// first) to `probe` as a channel-major map.
    pub(crate) fn decode_probed(
        &self,
        latent: &FeatureMap,
        mut probe: impl FnMut(usize, FeatureMap),
    ) -> Result<FeatureMap, NetworkError> {
        if latent.channels != crate::sana::LATENT_CHANNELS {
            return Err(NetworkError::InvalidConfig(format!(
                "the DC-AE decodes 32 latent channels, not {}",
                latent.channels
            )));
        }
        let (mut height, mut width) = (latent.height, latent.width);
        let tokens = transpose(&Matrix::from_vec(
            latent.channels,
            height * width,
            latent.data.clone(),
        ));
        let mut hidden = self.conv_in.forward(&tokens, height, width, false);
        let repeats = hidden.cols / tokens.cols;
        hidden
            .data
            .par_chunks_mut(CHANNELS[5])
            .zip(tokens.data.par_chunks(tokens.cols))
            .for_each(|(row, source)| {
                for (k, v) in row.iter_mut().enumerate() {
                    *v += source[k / repeats];
                }
            });
        for (index, stage) in self.stages.iter().enumerate() {
            for block in stage {
                match block {
                    Block::Up { conv, repeats } => {
                        let source_width = width;
                        height *= 2;
                        width *= 2;
                        let mut up = conv.forward(&hidden, height, width, true);
                        let source_channels = hidden.cols;
                        let channels = up.cols;
                        up.data
                            .par_chunks_mut(channels)
                            .enumerate()
                            .for_each(|(pixel, row)| {
                                let (y, x) = (pixel / width, pixel % width);
                                let sub = (y % 2) * 2 + x % 2;
                                let source = &hidden.data
                                    [((y / 2) * source_width + x / 2) * source_channels..]
                                    [..source_channels];
                                for (c, v) in row.iter_mut().enumerate() {
                                    *v += source[(c * 4 + sub) / repeats];
                                }
                            });
                        hidden = up;
                    }
                    Block::Res { conv1, conv2, norm } => {
                        let mut x = conv1.forward(&hidden, height, width, false);
                        x.data.par_iter_mut().for_each(|v| *v = silu(*v));
                        let mut x = conv2.forward(&x, height, width, false);
                        rms_affine(&mut x, &norm.weight, &norm.bias);
                        add(&mut hidden, &x);
                    }
                    Block::Vit(vit) => vit.forward(&mut hidden, height, width)?,
                }
            }
            probe(index, channel_major(&hidden, height, width));
        }
        rms_affine(&mut hidden, &self.norm_out.weight, &self.norm_out.bias);
        hidden.data.par_iter_mut().for_each(|v| *v = v.max(0.0));
        let rgb = self.conv_out.forward(&hidden, height, width, false);
        Ok(channel_major(&rgb, height, width))
    }
}

pub(crate) fn channel_major(tokens: &Matrix, height: usize, width: usize) -> FeatureMap {
    FeatureMap {
        channels: tokens.cols,
        height,
        width,
        data: transpose(tokens).data,
    }
}

impl Vit {
    fn forward(
        &self,
        hidden: &mut Matrix,
        height: usize,
        width: usize,
    ) -> Result<(), NetworkError> {
        let (n, c) = (hidden.rows, hidden.cols);
        // q | k | v, then their 5x5 + grouped 1x1 aggregate: 6C channels.
        let mut qkv = Matrix::new(n, 3 * c);
        for (part, layer) in [&self.query, &self.key, &self.value]
            .into_iter()
            .enumerate()
        {
            let projected = layer.forward(hidden)?;
            qkv.data
                .par_chunks_mut(3 * c)
                .zip(projected.data.par_chunks(c))
                .for_each(|(row, source)| row[part * c..(part + 1) * c].copy_from_slice(source));
        }
        let spread = depthwise(&qkv, height, width, &self.multiscale_in, None, 5);
        let mut multiscale = Matrix::new(n, 3 * c);
        multiscale
            .data
            .par_chunks_mut(3 * c)
            .zip(spread.data.par_chunks(3 * c))
            .for_each(|(row, source)| {
                for (o, v) in row.iter_mut().enumerate() {
                    let group = &source[(o / HEAD_DIM) * HEAD_DIM..][..HEAD_DIM];
                    let weight = &self.multiscale_out[o * HEAD_DIM..(o + 1) * HEAD_DIM];
                    *v = group.iter().zip(weight).map(|(a, b)| a * b).sum();
                }
            });
        // Groups of 96 consecutive channels across [qkv, multiscale]: the
        // first 32 are that group's queries, then keys, then values.
        let groups = 6 * c / (3 * HEAD_DIM);
        let width_out = groups * HEAD_DIM;
        let mut split = [
            Matrix::new(n, width_out),
            Matrix::new(n, width_out),
            Matrix::new(n, width_out),
        ];
        for (part, target) in split.iter_mut().enumerate() {
            target
                .data
                .par_chunks_mut(width_out)
                .enumerate()
                .for_each(|(token, row)| {
                    for group in 0..groups {
                        for i in 0..HEAD_DIM {
                            let channel = group * 3 * HEAD_DIM + part * HEAD_DIM + i;
                            let value = match channel < 3 * c {
                                true => qkv.data[token * 3 * c + channel],
                                false => multiscale.data[token * 3 * c + channel - 3 * c],
                            };
                            row[group * HEAD_DIM + i] = match part < 2 {
                                true => value.max(0.0),
                                false => value,
                            };
                        }
                    }
                });
        }
        let attended = linear_attention(&split[0], &split[1], &split[2], groups, HEAD_DIM);
        let mut out = self.out.forward(&attended)?;
        rms_affine(
            &mut out,
            &self.attention_norm.weight,
            &self.attention_norm.bias,
        );
        add(hidden, &out);

        // GLUMBConv with a norm and a residual.
        let mut x = self.inverted.forward(hidden)?;
        x.data.par_iter_mut().for_each(|v| *v = silu(*v));
        let x = depthwise(
            &x,
            height,
            width,
            &self.depth_weight,
            Some(&self.depth_bias),
            3,
        );
        let half = x.cols / 2;
        let mut glu = Matrix::new(n, half);
        glu.data
            .par_chunks_mut(half)
            .zip(x.data.par_chunks(2 * half))
            .for_each(|(out, row)| {
                for i in 0..half {
                    out[i] = row[i] * silu(row[half + i]);
                }
            });
        let mut out = self.point.forward(&glu)?;
        rms_affine(&mut out, &self.norm.weight, &self.norm.bias);
        add(hidden, &out);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sana::tests::{gate, reference, tensor};

    #[test]
    fn a_3x3_convolution_matches_the_direct_sum() {
        // One input channel, one output, the centre tap and its right
        // neighbour set: out(y, x) = in(y, x) + 2 in(y, x + 1).
        let mut weight = Matrix::new(1, 9);
        weight.data[4] = 1.0;
        weight.data[5] = 2.0;
        let conv = Conv3 {
            weight,
            bias: Some(vec![0.5]),
            inputs: 1,
        };
        let input = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let out = conv.forward(&input, 2, 2, false);
        assert_eq!(out.data, vec![5.5, 2.5, 11.5, 4.5]);
        // Through a 2x nearest upsampling of a single pixel, every output
        // reads the same value: right neighbours exist only in column 0.
        let out = conv.forward(&Matrix::from_vec(1, 1, vec![1.0]), 2, 2, true);
        assert_eq!(out.data, vec![3.5, 1.5, 3.5, 1.5]);
    }

    #[test]
    #[ignore = "needs RB_SANA_REF and the SANA-Sprint weights"]
    fn sana_dcae_matches_diffusers() {
        let Some((mut reference, model)) = reference() else {
            panic!("set RB_SANA_REF to the sana_refs.py output folder");
        };
        let decoder = DcaeDecoder::load(&model).unwrap();
        let latent = FeatureMap {
            channels: 32,
            height: 32,
            width: 32,
            data: tensor(&mut reference, "vae_input"),
        };
        let mut stages = Vec::new();
        let start = std::time::Instant::now();
        let image = decoder
            .decode_probed(&latent, |index, map| {
                if index < 3 {
                    stages.push(map)
                }
            })
            .unwrap();
        println!("host dc-ae: {:.1} s", start.elapsed().as_secs_f32());
        for (index, map) in stages.iter().enumerate() {
            let name = format!("vae_up{}", 5 - index);
            gate(&name, &map.data, &tensor(&mut reference, &name), 1e-2);
        }
        gate("image", &image.data, &tensor(&mut reference, "image"), 3e-2);
    }
}
