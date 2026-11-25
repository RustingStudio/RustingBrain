//! Qwen3-VL's vision tower: the eyes of Qwen-Image 2.1's prompt encoder.
//!
//! To edit a picture, Qwen-Image 2.1 hands its language model the picture
//! along with the instruction, and conditions on what the model makes of the
//! two together. This is the part that turns the picture into rows the
//! language model reads: a ViT over 16-pixel patches whose every four
//! neighbouring patches merge into one token of the language model's width.
//! Three earlier depths of the ViT are merged the same way and added to the
//! language model's first layers ("deepstack"); see
//! [`crate::text_encoder::Visual`].
//!
//! Host only, forward only.

use crate::clip::{Norm, dense, norm};
use crate::conv::{Dense, FeatureMap};
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::safetensors::ShardedSafeTensors;
use crate::text_encoder::Rotary;
use crate::transformer::Precision;
use rayon::prelude::*;

// ponytail: the shape of the one tower this has been pointed at, Qwen3-VL
// 8B's inside Qwen-Image 2.1. Read `vision_config` when a second one shows up.
const PATCH: usize = 16;
const HEADS: usize = 16;
const DEPTH: usize = 27;
/// The blocks whose output the language model reads as well as the last.
const DEEPSTACK: [usize; 3] = [8, 16, 24];
const EPS: f32 = 1e-6;

struct Block {
    norm1: Norm,
    qkv: Dense,
    proj: Dense,
    norm2: Norm,
    fc1: Dense,
    fc2: Dense,
}

/// Four neighbouring patches into one token of the language model's width.
struct Merger {
    norm: Norm,
    fc1: Dense,
    fc2: Dense,
}

impl Merger {
    fn read(
        file: &mut ShardedSafeTensors,
        name: &str,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        Ok(Self {
            norm: norm(file, &format!("{name}.norm"))?,
            fc1: dense(file, &format!("{name}.linear_fc1"), precision)?,
            fc2: dense(file, &format!("{name}.linear_fc2"), precision)?,
        })
    }

    /// `tokens` hold each merge block's four patches in a row, so merging is
    /// a reshape. The last merger norms each patch and the deepstack ones norm
    /// the four together; the width of the norm says which this is.
    fn forward(&self, tokens: &Matrix) -> Result<Matrix, NetworkError> {
        let wide = |tokens: Matrix| Matrix::from_vec(tokens.rows / 4, tokens.cols * 4, tokens.data);
        let normed = match self.norm.weight.len() == tokens.cols {
            true => wide(self.norm.forward(tokens, EPS)),
            false => self.norm.forward(&wide(tokens.clone()), EPS),
        };
        let mut hidden = self.fc1.forward(&normed)?;
        hidden
            .data
            .par_iter_mut()
            .for_each(|value| *value = crate::ffn::gelu_erf(*value));
        self.fc2.forward(&hidden)
    }
}

/// The vision tower of a Qwen3-VL checkpoint.
pub struct QwenVision {
    patch: Dense,
    /// Learned positions on a square grid, stretched to the picture's.
    positions: Matrix,
    blocks: Vec<Block>,
    merger: Merger,
    deepstack: Vec<Merger>,
}

impl QwenVision {
    /// Reads the tower whose keys start with `prefix`: `"model.visual."` in
    /// Qwen-Image 2.1's text encoder.
    pub fn read(
        file: &mut ShardedSafeTensors,
        prefix: &str,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        // The patch embedding is a 3-D convolution over two frames, and a
        // still picture is its one frame twice: the two slices of the kernel
        // add up to a linear layer over a patch.
        let (kernel, shape) = file.tensor(&format!("{prefix}patch_embed.proj.weight"))?;
        let [width, channels, frames, PATCH, PATCH] = shape[..] else {
            return Err(NetworkError::InvalidConfig(format!(
                "a {shape:?} patch embedding is not Qwen3-VL's"
            )));
        };
        let area = PATCH * PATCH;
        let mut weight = Matrix::new(width, channels * area);
        for (row, kernel) in kernel.chunks(channels * frames * area).enumerate() {
            for channel in 0..channels {
                for frame in 0..frames {
                    let slice = &kernel[(channel * frames + frame) * area..][..area];
                    for (sum, value) in weight.row_mut(row)[channel * area..][..area]
                        .iter_mut()
                        .zip(slice)
                    {
                        *sum += value;
                    }
                }
            }
        }
        let bias = file.tensor(&format!("{prefix}patch_embed.proj.bias"))?.0;

        let blocks = (0..DEPTH)
            .map(|index| {
                let base = format!("{prefix}blocks.{index}");
                Ok(Block {
                    norm1: norm(file, &format!("{base}.norm1"))?,
                    qkv: dense(file, &format!("{base}.attn.qkv"), precision)?,
                    proj: dense(file, &format!("{base}.attn.proj"), precision)?,
                    norm2: norm(file, &format!("{base}.norm2"))?,
                    fc1: dense(file, &format!("{base}.mlp.linear_fc1"), precision)?,
                    fc2: dense(file, &format!("{base}.mlp.linear_fc2"), precision)?,
                })
            })
            .collect::<Result<Vec<_>, NetworkError>>()?;
        let deepstack = (0..DEEPSTACK.len())
            .map(|index| {
                Merger::read(
                    file,
                    &format!("{prefix}deepstack_merger_list.{index}"),
                    precision,
                )
            })
            .collect::<Result<Vec<_>, NetworkError>>()?;

        Ok(Self {
            patch: Dense::new(weight, Some(bias))?,
            positions: file.matrix(&format!("{prefix}pos_embed.weight"))?,
            blocks,
            merger: Merger::read(file, &format!("{prefix}merger"), precision)?,
            deepstack,
        })
    }

    /// What the language model reads of `image`: one row per 32-pixel square,
    /// in reading order, and the same rows from each deepstack depth.
    ///
    /// `image` is RGB in `[-1, 1]` with sides that are multiples of 32.
    pub fn forward(&self, image: &FeatureMap) -> Result<(Matrix, Vec<Matrix>), NetworkError> {
        let (rows, columns) = (image.height / PATCH, image.width / PATCH);
        if image.channels != 3
            || rows == 0
            || columns == 0
            || !image.height.is_multiple_of(2 * PATCH)
            || !image.width.is_multiple_of(2 * PATCH)
        {
            return Err(NetworkError::InvalidConfig(format!(
                "the vision tower reads RGB with sides that are multiples of {}, not {} \
                 channels at {}x{}",
                2 * PATCH,
                image.channels,
                image.width,
                image.height
            )));
        }
        // The tower's own order: the four patches of a merge block together,
        // the blocks in reading order.
        let order: Vec<(usize, usize)> = (0..rows / 2)
            .flat_map(|row| (0..columns / 2).map(move |column| (row, column)))
            .flat_map(|(row, column)| {
                [(0, 0), (0, 1), (1, 0), (1, 1)].map(|(down, right)| (2 * row + down, 2 * column + right))
            })
            .collect();

        let mut patches = Matrix::new(order.len(), 3 * PATCH * PATCH);
        for (token, &(row, column)) in order.iter().enumerate() {
            let target = patches.row_mut(token);
            for channel in 0..3 {
                let plane = image.plane(channel);
                for y in 0..PATCH {
                    let start = (row * PATCH + y) * image.width + column * PATCH;
                    target[(channel * PATCH + y) * PATCH..][..PATCH]
                        .copy_from_slice(&plane[start..start + PATCH]);
                }
            }
        }
        let mut hidden = self.patch.forward(&patches)?;

        // The learned positions, bilinear with the corners aligned.
        let side = self.positions.rows.isqrt();
        let taps = |index: usize, size: usize| {
            let source = index as f32 * (side - 1) as f32 / (size.max(2) - 1) as f32;
            let low = (source as usize).min(side - 1);
            let part = source - low as f32;
            [(low, 1.0 - part), ((low + 1).min(side - 1), part)]
        };
        for (token, &(row, column)) in order.iter().enumerate() {
            let target = hidden.row_mut(token);
            for (grid_row, down) in taps(row, rows) {
                for (grid_column, right) in taps(column, columns) {
                    let position = self.positions.row(grid_row * side + grid_column);
                    for (value, position) in target.iter_mut().zip(position) {
                        *value += down * right * position;
                    }
                }
            }
        }

        // Rotary positions in two axes: the first half of a head's pairs
        // turn with the patch's row and the second with its column.
        let half = hidden.cols / HEADS / 2;
        let axis = half / 2;
        let (mut cos, mut sin) = (vec![0.0; order.len() * half], vec![0.0; order.len() * half]);
        for (token, &(row, column)) in order.iter().enumerate() {
            for index in 0..half {
                let frequency = 10_000f32.powf(-((index % axis * 2) as f32) / half as f32);
                let position = if index < axis { row } else { column };
                let angle = position as f32 * frequency;
                cos[token * half + index] = angle.cos();
                sin[token * half + index] = angle.sin();
            }
        }
        let rotary = Rotary { cos, sin, half };

        let add = |hidden: &mut Matrix, delta: &Matrix| {
            for (value, delta) in hidden.data.iter_mut().zip(&delta.data) {
                *value += delta;
            }
        };
        let mut deep = Vec::new();
        for (index, block) in self.blocks.iter().enumerate() {
            let qkv = block.qkv.forward(&block.norm1.forward(&hidden, EPS))?;
            add(&mut hidden, &block.proj.forward(&attention(&qkv, &rotary))?);
            let mut inner = block.fc1.forward(&block.norm2.forward(&hidden, EPS))?;
            inner
                .data
                .par_iter_mut()
                .for_each(|value| *value = crate::ffn::gelu(*value));
            add(&mut hidden, &block.fc2.forward(&inner)?);
            if let Some(slot) = DEEPSTACK.iter().position(|at| *at == index) {
                deep.push(self.deepstack[slot].forward(&hidden)?);
            }
        }
        Ok((self.merger.forward(&hidden)?, deep))
    }
}

/// Self-attention of every patch over every patch, from a block's joined
/// query, key and value projection.
///
/// ponytail: a `[tokens, tokens]` score matrix per head, 67 MB at the 4096
/// patches of a megapixel times however many heads run at once. Walk the
/// queries in slabs if a much larger picture is ever read.
fn attention(qkv: &Matrix, rotary: &Rotary) -> Matrix {
    let (tokens, dim) = (qkv.rows, qkv.cols / 3);
    let head_dim = dim / HEADS;
    let columns = |source: &Matrix, start: usize, width: usize| {
        let data = source
            .data
            .chunks(source.cols)
            .flat_map(|row| row[start..start + width].iter().copied())
            .collect();
        Matrix::from_vec(tokens, width, data)
    };
    let (mut queries, mut keys) = (columns(qkv, 0, dim), columns(qkv, dim, dim));
    let values = columns(qkv, 2 * dim, dim);
    rotary.apply(&mut queries, HEADS);
    rotary.apply(&mut keys, HEADS);

    let scale = (head_dim as f32).sqrt().recip();
    let heads: Vec<Matrix> = (0..HEADS)
        .into_par_iter()
        .map(|head| {
            let of = |source: &Matrix| columns(source, head * head_dim, head_dim);
            let mut scores = Matrix::new(tokens, tokens);
            of(&queries).dot_rhs_transposed(&of(&keys), &mut scores);
            for row in scores.data.chunks_mut(tokens) {
                let largest = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut total = 0.0;
                for score in row.iter_mut() {
                    *score = ((*score - largest) * scale).exp();
                    total += *score;
                }
                row.iter_mut().for_each(|score| *score /= total);
            }
            let mut attended = Matrix::new(tokens, head_dim);
            scores.dot(&of(&values), &mut attended);
            attended
        })
        .collect();

    let mut output = Matrix::new(tokens, dim);
    for (head, attended) in heads.iter().enumerate() {
        for token in 0..tokens {
            output.row_mut(token)[head * head_dim..][..head_dim].copy_from_slice(attended.row(token));
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_encoder::{TextEncoder, TextEncoderConfig, Visual};

    /// The tower and the language model's first three layers against
    /// transformers' `Qwen3VLModel`, on the activations
    /// `RB_QWEN_VL_REFERENCE` holds (`image`, `ids`, `vision`, `deepstack.N`
    /// and `hidden`, the states going into the final norm of a model cut to
    /// three layers). `RB_QWEN_IMAGE_MODEL` is the Qwen-Image 2.1 directory.
    #[test]
    #[ignore = "needs Qwen-Image 2.1 and reference activations"]
    fn qwen_vision_matches_transformers() {
        let model = std::path::PathBuf::from(std::env::var("RB_QWEN_IMAGE_MODEL").unwrap());
        let mut reference =
            ShardedSafeTensors::open(std::env::var("RB_QWEN_VL_REFERENCE").unwrap()).unwrap();
        let (data, shape) = reference.tensor("image").unwrap();
        let image = FeatureMap {
            channels: shape[0],
            height: shape[1],
            width: shape[2],
            data,
        };
        let ids: Vec<u32> = reference.tensor("ids").unwrap().0.iter().map(|id| *id as u32).collect();
        let close = |name: &str, ours: &Matrix, reference: &mut ShardedSafeTensors| {
            let theirs = reference.matrix(name).unwrap();
            assert_eq!((ours.rows, ours.cols), (theirs.rows, theirs.cols), "{name}");
            let largest = theirs.data.iter().fold(0f32, |a, b| a.max(b.abs()));
            let apart = ours
                .data
                .iter()
                .zip(&theirs.data)
                .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
            println!("{name}: off by {apart} of {largest}");
            assert!(apart < 2e-3 * largest, "{name}");
        };

        let index = std::fs::read_dir(model.join("text_encoder"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| path.to_string_lossy().ends_with(".safetensors.index.json"))
            .unwrap();
        let mut file = ShardedSafeTensors::open(index).unwrap();
        let tower = QwenVision::read(&mut file, "model.visual.", Precision::F32).unwrap();
        let seen = tower.forward(&image).unwrap();
        close("vision", &seen.0, &mut reference);
        for (index, deep) in seen.1.iter().enumerate() {
            close(&format!("deepstack.{index}"), deep, &mut reference);
        }

        let mut config =
            TextEncoderConfig::from_file(model.join("text_encoder/config.json")).unwrap();
        config.layers = 3;
        let encoder =
            TextEncoder::read_at(&mut file, "model.language_model.", config, Precision::F32)
                .unwrap();
        // One picture, its image tokens in one run.
        let pad = ids.iter().filter(|id| **id == 151_655).count();
        let first = ids.iter().position(|id| *id == 151_655).unwrap();
        let mut visual = Visual::new(encoder.config().d_model);
        visual.text(first);
        visual.picture(image.height / 32, image.width / 32, seen);
        assert_eq!(visual.rows.len(), pad);
        let hidden = encoder.forward_visual(&ids, &visual).unwrap();
        close("hidden", &hidden, &mut reference);
    }
}
