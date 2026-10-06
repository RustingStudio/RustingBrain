//! Rotary position embeddings.
//!
//! RoPE has no learned parameters: it rotates each pair of channels in a head
//! by an angle proportional to the token's absolute position, so a dot product
//! between a query and a key depends only on their *relative* distance.

use crate::matrix::Matrix;
use crate::network::NetworkError;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Precomputed `cos`/`sin` tables, laid out `[position, head_dim / 2]`.
///
/// Channel `j` is paired with channel `j + head_dim / 2` (the "rotate half"
/// convention used by the Llama and Qwen reference implementations), not with
/// its immediate neighbour. The two conventions are a permutation apart and
/// both are self-consistent, but weights converted from those models assume
/// this one.
#[derive(Clone, Debug, PartialEq)]
pub struct Rope {
    head_dim: usize,
    max_seq_len: usize,
    base: f32,
    scaling: Option<RopeScaling>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

/// How a model trained at one context length stretches its rotary angles to
/// run at a longer one. Field names follow the `rope_scaling` entry of a
/// Hugging Face `config.json`, so the value reads and writes there directly.
///
/// ponytail: no YaRN or dynamic NTK yet. YaRN also rescales attention logits,
/// and dynamic NTK changes the tables with the sequence length; add them when
/// a checkpoint that needs one is in reach.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "rope_type", rename_all = "lowercase")]
pub enum RopeScaling {
    /// Every position divided by `factor`.
    Linear { factor: f32 },
    /// Llama 3.1's scheme: channels that turn many times within the original
    /// context keep their frequency, channels that turn less than once are
    /// slowed by `factor`, and the band between blends the two.
    Llama3 {
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_max_position_embeddings: usize,
    },
}

impl RopeScaling {
    fn apply(self, frequency: f32) -> f32 {
        match self {
            Self::Linear { factor } => frequency / factor,
            Self::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            } => {
                let context = original_max_position_embeddings as f32;
                let wavelength = 2.0 * std::f32::consts::PI / frequency;
                if wavelength < context / high_freq_factor {
                    frequency
                } else if wavelength > context / low_freq_factor {
                    frequency / factor
                } else {
                    let smooth = (context / wavelength - low_freq_factor)
                        / (high_freq_factor - low_freq_factor);
                    (1.0 - smooth) * frequency / factor + smooth * frequency
                }
            }
        }
    }
}

/// What a snapshot stores. The tables are a pure function of these three
/// numbers and run to hundreds of kilobytes per layer, so they are rebuilt on
/// load instead of written out.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename = "Rope")]
pub struct RopeSpec {
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub base: f32,
    /// Absent in snapshots written before scaling existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scaling: Option<RopeScaling>,
}

impl From<Rope> for RopeSpec {
    fn from(rope: Rope) -> Self {
        Self {
            head_dim: rope.head_dim,
            max_seq_len: rope.max_seq_len,
            base: rope.base,
            scaling: rope.scaling,
        }
    }
}

impl TryFrom<RopeSpec> for Rope {
    type Error = NetworkError;

    fn try_from(spec: RopeSpec) -> Result<Self, Self::Error> {
        Rope::scaled(spec.head_dim, spec.max_seq_len, spec.base, spec.scaling)
    }
}

impl Serialize for Rope {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        RopeSpec::from(self.clone()).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Rope {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let spec = RopeSpec::deserialize(deserializer)?;
        Rope::try_from(spec).map_err(serde::de::Error::custom)
    }
}

impl Rope {
    pub fn new(head_dim: usize, max_seq_len: usize, base: f32) -> Result<Self, NetworkError> {
        Self::scaled(head_dim, max_seq_len, base, None)
    }

    /// [`Rope::new`] with the angles stretched by `scaling`.
    pub fn scaled(
        head_dim: usize,
        max_seq_len: usize,
        base: f32,
        scaling: Option<RopeScaling>,
    ) -> Result<Self, NetworkError> {
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "rope head_dim must be even and non-zero, got {head_dim}"
            )));
        }

        let half = head_dim / 2;
        let frequencies: Vec<f32> = (0..half)
            .map(|channel| {
                let frequency = base.powf(-2.0 * channel as f32 / head_dim as f32);
                scaling.map_or(frequency, |scaling| scaling.apply(frequency))
            })
            .collect();
        let mut cos = Vec::with_capacity(max_seq_len * half);
        let mut sin = Vec::with_capacity(max_seq_len * half);

        for position in 0..max_seq_len {
            for &frequency in &frequencies {
                let angle = position as f32 * frequency;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }

        Ok(Self {
            head_dim,
            max_seq_len,
            base,
            scaling,
            cos,
            sin,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn base(&self) -> f32 {
        self.base
    }

    /// Rotates `tensor` in place. Rows are tokens, each row holding `heads`
    /// heads of `head_dim` channels laid out back to back.
    ///
    /// `position_offset` is the absolute position of the first row, which is
    /// non-zero whenever a cached decode step feeds a single new token.
    pub fn apply(
        &self,
        tensor: &mut Matrix,
        heads: usize,
        position_offset: usize,
    ) -> Result<(), NetworkError> {
        let rows = tensor.rows;
        self.rotate(tensor, heads, position_offset, rows, 1.0)
    }

    /// Rotates a packed batch, where every `seq_len` rows start a new sequence
    /// and therefore restart at position zero.
    pub fn apply_batched(
        &self,
        tensor: &mut Matrix,
        heads: usize,
        seq_len: usize,
    ) -> Result<(), NetworkError> {
        self.rotate(tensor, heads, 0, seq_len, 1.0)
    }

    /// Backward pass of [`Rope::apply_batched`].
    pub fn apply_inverse_batched(
        &self,
        tensor: &mut Matrix,
        heads: usize,
        seq_len: usize,
    ) -> Result<(), NetworkError> {
        self.rotate(tensor, heads, 0, seq_len, -1.0)
    }

    /// The `cos` table, `[max_seq_len, head_dim / 2]`.
    pub fn cos(&self) -> &[f32] {
        &self.cos
    }

    /// The `sin` table, laid out like [`Rope::cos`].
    pub fn sin(&self) -> &[f32] {
        &self.sin
    }

    /// Rotates by the negated angle, which undoes [`Rope::apply`] and is also
    /// its backward pass: a rotation is orthogonal, so transposing it is the
    /// same as reversing it.
    pub fn apply_inverse(
        &self,
        tensor: &mut Matrix,
        heads: usize,
        position_offset: usize,
    ) -> Result<(), NetworkError> {
        let rows = tensor.rows;
        self.rotate(tensor, heads, position_offset, rows, -1.0)
    }

    fn rotate(
        &self,
        tensor: &mut Matrix,
        heads: usize,
        position_offset: usize,
        seq_len: usize,
        direction: f32,
    ) -> Result<(), NetworkError> {
        if tensor.cols != heads * self.head_dim {
            return Err(NetworkError::InvalidConfig(format!(
                "rope expected {} columns for {heads} heads, got {}",
                heads * self.head_dim,
                tensor.cols
            )));
        }

        if seq_len == 0 || tensor.rows % seq_len != 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "rope got {} rows, which do not divide into sequences of {seq_len}",
                tensor.rows
            )));
        }

        let end = position_offset + seq_len;
        if end > self.max_seq_len {
            return Err(NetworkError::SequenceTooLong {
                length: end,
                max_seq_len: self.max_seq_len,
            });
        }

        let half = self.head_dim / 2;
        // Each row rotates on its own, and this runs four times per block per
        // step over the query and key projections.
        tensor
            .data
            .par_chunks_mut(tensor.cols)
            .enumerate()
            .for_each(|(row, values)| {
                let table = (position_offset + row % seq_len) * half;

                for head in 0..heads {
                    let base = head * self.head_dim;
                    for channel in 0..half {
                        let cos = self.cos[table + channel];
                        let sin = direction * self.sin[table + channel];
                        let low = values[base + channel];
                        let high = values[base + half + channel];
                        values[base + channel] = low * cos - high * sin;
                        values[base + half + channel] = high * cos + low * sin;
                    }
                }
            });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_zero_is_the_identity() {
        let rope = Rope::new(4, 8, 10000.0).unwrap();
        let mut tensor = Matrix::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]);

        rope.apply(&mut tensor, 1, 0).unwrap();

        assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rotation_matches_a_hand_computed_example() {
        // head_dim 2, base 1.0 => a single channel pair at angle = position.
        let rope = Rope::new(2, 4, 1.0).unwrap();
        let mut tensor = Matrix::from_vec(1, 2, vec![1.0, 0.0]);

        rope.apply(&mut tensor, 1, 1).unwrap();

        // (1, 0) rotated by one radian is (cos 1, sin 1).
        assert!((tensor.data[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((tensor.data[1] - 1.0f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn inverse_undoes_the_rotation() {
        let rope = Rope::new(8, 16, 10000.0).unwrap();
        let original = Matrix::from_vec(3, 16, (0..48).map(|v| v as f32 * 0.1).collect());
        let mut tensor = original.clone();

        rope.apply(&mut tensor, 2, 5).unwrap();
        rope.apply_inverse(&mut tensor, 2, 5).unwrap();

        for (restored, expected) in tensor.data.iter().zip(&original.data) {
            assert!((restored - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn rotation_preserves_norm() {
        let rope = Rope::new(6, 32, 10000.0).unwrap();
        let mut tensor = Matrix::from_vec(1, 6, vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5]);
        let before: f32 = tensor.data.iter().map(|v| v * v).sum();

        rope.apply(&mut tensor, 1, 17).unwrap();

        let after: f32 = tensor.data.iter().map(|v| v * v).sum();
        assert!((before - after).abs() < 1e-4);
    }

    #[test]
    fn dot_product_depends_only_on_relative_distance() {
        let rope = Rope::new(4, 64, 10000.0).unwrap();
        let query = vec![0.3, -1.1, 0.7, 2.0];
        let key = vec![1.3, 0.2, -0.9, 0.4];

        let score_at = |query_position: usize, key_position: usize| {
            let mut q = Matrix::from_vec(1, 4, query.clone());
            let mut k = Matrix::from_vec(1, 4, key.clone());
            rope.apply(&mut q, 1, query_position).unwrap();
            rope.apply(&mut k, 1, key_position).unwrap();
            q.data.iter().zip(&k.data).map(|(a, b)| a * b).sum::<f32>()
        };

        assert!((score_at(3, 1) - score_at(20, 18)).abs() < 1e-4);
    }

    #[test]
    fn applying_past_the_table_is_an_error() {
        let rope = Rope::new(2, 4, 10000.0).unwrap();
        let mut tensor = Matrix::new(3, 2);

        assert!(matches!(
            rope.apply(&mut tensor, 1, 2),
            Err(NetworkError::SequenceTooLong { length: 5, .. })
        ));
    }

    #[test]
    fn odd_head_dim_is_rejected() {
        assert!(Rope::new(3, 8, 10000.0).is_err());
    }

    #[test]
    fn snapshot_rebuilds_the_tables() {
        let rope = Rope::new(8, 128, 10000.0).unwrap();
        let json = serde_json::to_string(&rope).unwrap();

        // Only the three defining numbers are written out.
        assert!(!json.contains("cos"));
        assert_eq!(serde_json::from_str::<Rope>(&json).unwrap(), rope);
    }

    /// The angle one position turns channel pair `channel` by.
    fn frequency(rope: &Rope, channel: usize) -> f32 {
        let at = rope.head_dim() / 2 + channel;
        rope.sin()[at].atan2(rope.cos()[at])
    }

    #[test]
    fn linear_scaling_stretches_positions() {
        let plain = Rope::new(8, 16, 10000.0).unwrap();
        let scaled =
            Rope::scaled(8, 16, 10000.0, Some(RopeScaling::Linear { factor: 4.0 })).unwrap();

        // Position 8 under a factor of 4 is position 2 unscaled.
        for channel in 0..4 {
            assert!((scaled.cos()[8 * 4 + channel] - plain.cos()[2 * 4 + channel]).abs() < 1e-6);
            assert!((scaled.sin()[8 * 4 + channel] - plain.sin()[2 * 4 + channel]).abs() < 1e-6);
        }
    }

    #[test]
    fn llama3_scaling_keeps_fast_channels_and_slows_the_long_ones() {
        // Llama 3.1's published settings.
        let scaling = RopeScaling::Llama3 {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        };
        let plain = Rope::new(128, 2, 500_000.0).unwrap();
        let scaled = Rope::scaled(128, 2, 500_000.0, Some(scaling)).unwrap();

        assert_eq!(frequency(&scaled, 0), frequency(&plain, 0));
        let last = 63;
        assert!((frequency(&scaled, last) - frequency(&plain, last) / 8.0).abs() < 1e-12);
        // In between the factor blends smoothly: never outside [f / 8, f] and
        // still falling with the channel index, so no band jumps past another.
        let mut previous = f32::INFINITY;
        for channel in 0..64 {
            let (f, original) = (frequency(&scaled, channel), frequency(&plain, channel));
            assert!(
                f <= original * 1.0001 && f >= original / 8.0 * 0.9999,
                "channel {channel}"
            );
            assert!(f < previous, "channel {channel}");
            previous = f;
        }
    }

    #[test]
    fn a_snapshot_keeps_its_scaling_and_an_old_one_has_none() {
        let rope = Rope::scaled(8, 32, 10000.0, Some(RopeScaling::Linear { factor: 2.0 })).unwrap();
        let json = serde_json::to_string(&rope).unwrap();
        assert_eq!(serde_json::from_str::<Rope>(&json).unwrap(), rope);

        let old = r#"{"head_dim":8,"max_seq_len":32,"base":10000.0}"#;
        assert_eq!(
            serde_json::from_str::<Rope>(old).unwrap(),
            Rope::new(8, 32, 10000.0).unwrap()
        );
    }
}
