//! Rotary position embeddings.
//!
//! RoPE has no learned parameters: it rotates each pair of channels in a head
//! by an angle proportional to the token's absolute position, so a dot product
//! between a query and a key depends only on their *relative* distance.

use crate::matrix::Matrix;
use crate::network::NetworkError;
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
    cos: Vec<f32>,
    sin: Vec<f32>,
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
}

impl From<Rope> for RopeSpec {
    fn from(rope: Rope) -> Self {
        Self {
            head_dim: rope.head_dim,
            max_seq_len: rope.max_seq_len,
            base: rope.base,
        }
    }
}

impl TryFrom<RopeSpec> for Rope {
    type Error = NetworkError;

    fn try_from(spec: RopeSpec) -> Result<Self, Self::Error> {
        Rope::new(spec.head_dim, spec.max_seq_len, spec.base)
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
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "rope head_dim must be even and non-zero, got {head_dim}"
            )));
        }

        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(max_seq_len * half);
        let mut sin = Vec::with_capacity(max_seq_len * half);

        for position in 0..max_seq_len {
            for channel in 0..half {
                let frequency = base.powf(-2.0 * channel as f32 / head_dim as f32);
                let angle = position as f32 * frequency;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }

        Ok(Self {
            head_dim,
            max_seq_len,
            base,
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
        for row in 0..tensor.rows {
            let table = (position_offset + row % seq_len) * half;
            let values = tensor.row_mut(row);

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
        }

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
}
