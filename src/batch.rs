//! Packing several sequences into one activation matrix.
//!
//! Every module in the transformer stack works on a `[rows, width]` matrix
//! whose rows are tokens. One sequence per call makes every matmul as tall as
//! the sequence, which is far too small to keep a GPU busy and far too small to
//! amortize a kernel launch. A batch is therefore packed into a single matrix
//! of `batch * seq_len` rows, and the two places where rows are *not*
//! independent - the causal mask and the next-token shift in the loss - use the
//! [`Layout`] to find the sequence a row belongs to.
//!
//! Sequences are right-padded to the longest one in the batch. Padding is free
//! for attention (a pad row sits after every real token of its sequence, and
//! the causal mask already stops a real token from seeing it) but not for the
//! router or the loss, which is what [`Layout::valid`] is for. Equal-length
//! sequences - the usual case, since language-model corpora are chunked to a
//! fixed context - carry no padding at all.

use crate::network::NetworkError;

/// A right-padded batch of token id sequences.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenBatch {
    ids: Vec<u32>,
    lengths: Vec<usize>,
    seq_len: usize,
    valid: Vec<bool>,
    supervised: Option<Vec<bool>>,
}

impl TokenBatch {
    /// Packs `sequences` into `batch * seq_len` rows, padding short ones.
    ///
    /// Each sequence needs at least two tokens, because a next-token loss over
    /// one token predicts nothing.
    pub fn new<S: AsRef<[u32]>>(sequences: &[S]) -> Result<Self, NetworkError> {
        if sequences.is_empty() {
            return Err(NetworkError::EmptyDataset);
        }
        let lengths: Vec<usize> = sequences.iter().map(|s| s.as_ref().len()).collect();
        if let Some(short) = lengths.iter().find(|&&length| length < 2) {
            return Err(NetworkError::InvalidConfig(format!(
                "a causal language-modelling loss needs at least two tokens per sequence, got {short}"
            )));
        }

        let seq_len = lengths.iter().copied().max().unwrap_or(0);
        let mut ids = vec![0u32; sequences.len() * seq_len];
        let mut valid = vec![false; sequences.len() * seq_len];
        for (index, sequence) in sequences.iter().enumerate() {
            let sequence = sequence.as_ref();
            let base = index * seq_len;
            ids[base..base + sequence.len()].copy_from_slice(sequence);
            valid[base..base + sequence.len()].fill(true);
        }

        Ok(Self {
            ids,
            lengths,
            seq_len,
            valid,
            supervised: None,
        })
    }

    /// Restricts the loss to the token positions flagged `true`.
    ///
    /// `mask` holds one flag per row of [`TokenBatch::ids`] - `batch *
    /// seq_len` flags, padding included - and a flag marks its token as a
    /// *target*: when `mask[i]` is true the position that predicts token `i`
    /// contributes to the loss and to the gradient, and when it is false that
    /// position contributes exactly zero to both. This is the same convention
    /// as a Hugging Face `labels` tensor, where masked entries are `-100`.
    ///
    /// The first token of a sequence is never a target - nothing precedes it -
    /// so its flag is ignored, and padding stays excluded whatever the mask
    /// says.
    ///
    /// Masking changes the loss alone. Masked rows still run through the
    /// forward pass and still serve as context for the rows that do count,
    /// which is what makes this usable for supervised fine-tuning: flag the
    /// response span and the instruction span in front of it conditions the
    /// model without being learned.
    pub fn with_loss_mask(mut self, mask: &[bool]) -> Result<Self, NetworkError> {
        if mask.len() != self.ids.len() {
            return Err(NetworkError::InvalidConfig(format!(
                "a loss mask covers {} token positions, not {}",
                mask.len(),
                self.ids.len()
            )));
        }
        self.supervised = Some(mask.to_vec());
        if self.predicted() == 0 {
            return Err(NetworkError::InvalidConfig(
                "a loss mask left no position to predict".into(),
            ));
        }
        Ok(self)
    }

    /// Whether row `row` predicts a token that counts toward the loss.
    ///
    /// False for padding, for the last position of a sequence, and for a
    /// position whose target a loss mask excluded.
    pub fn predicts(&self, row: usize) -> bool {
        if row % self.seq_len + 1 >= self.lengths[row / self.seq_len] {
            return false;
        }
        self.supervised
            .as_ref()
            .is_none_or(|supervised| supervised[row + 1])
    }

    /// Number of sequences.
    pub fn batch(&self) -> usize {
        self.lengths.len()
    }

    /// Rows per sequence, which is the longest sequence in the batch.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Total rows, padding included.
    pub fn rows(&self) -> usize {
        self.ids.len()
    }

    /// Padded ids, `[batch * seq_len]`.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn lengths(&self) -> &[usize] {
        &self.lengths
    }

    /// Whether any row is padding. Equal-length batches are the fast case.
    pub fn is_padded(&self) -> bool {
        self.lengths.iter().any(|&length| length != self.seq_len)
    }

    /// How many positions a next-token loss covers.
    pub fn predicted(&self) -> usize {
        match self.supervised {
            None => self.lengths.iter().map(|length| length - 1).sum(),
            Some(_) => (0..self.rows()).filter(|&row| self.predicts(row)).count(),
        }
    }

    /// The row-to-sequence map every batched module needs.
    pub fn layout(&self) -> Layout<'_> {
        Layout {
            seq_len: Some(self.seq_len),
            valid: self.is_padded().then_some(&self.valid),
        }
    }
}

/// How the rows of a packed matrix map onto sequences.
///
/// The default - no `seq_len`, no mask - is a single unpadded sequence, which
/// is what every module did before batching existed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Layout<'a> {
    /// Rows per sequence. `None` means the whole matrix is one sequence.
    pub seq_len: Option<usize>,
    /// Per-row padding mask. `None` means every row is a real token.
    pub valid: Option<&'a [bool]>,
}

impl<'a> Layout<'a> {
    /// One sequence of `rows` tokens, no padding.
    pub fn single(rows: usize) -> Self {
        Self {
            seq_len: Some(rows),
            valid: None,
        }
    }

    /// Rows per sequence, resolving the "one sequence" default against the
    /// matrix actually being processed.
    pub fn seq_len(&self, rows: usize) -> usize {
        self.seq_len.unwrap_or(rows).max(1)
    }

    /// Whether row `row` holds a real token.
    pub fn is_valid(&self, row: usize) -> bool {
        self.valid.is_none_or(|mask| mask[row])
    }

    /// Errors unless `rows` splits evenly into sequences and any mask covers
    /// every row.
    pub fn check(&self, rows: usize) -> Result<(), NetworkError> {
        let seq_len = self.seq_len(rows);
        if rows % seq_len != 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "{rows} rows do not divide into sequences of {seq_len}"
            )));
        }
        match self.valid {
            Some(mask) if mask.len() != rows => Err(NetworkError::InvalidConfig(format!(
                "padding mask covers {} rows, not {rows}",
                mask.len()
            ))),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_length_sequences_need_no_padding() {
        let batch = TokenBatch::new(&[[1u32, 2, 3], [4, 5, 6]]).unwrap();

        assert_eq!(batch.rows(), 6);
        assert_eq!(batch.seq_len(), 3);
        assert!(!batch.is_padded());
        assert_eq!(batch.layout().valid, None);
        assert_eq!(batch.predicted(), 4);
    }

    #[test]
    fn a_short_sequence_is_padded_on_the_right_and_masked() {
        let batch = TokenBatch::new(&[vec![1u32, 2, 3, 4], vec![5, 6]]).unwrap();

        assert_eq!(batch.ids(), &[1, 2, 3, 4, 5, 6, 0, 0]);
        assert!(batch.is_padded());
        let layout = batch.layout();
        assert!(layout.is_valid(5));
        assert!(!layout.is_valid(6));
        assert_eq!(batch.predicted(), 3 + 1);
    }

    #[test]
    fn a_loss_mask_counts_only_the_flagged_targets() {
        let batch = TokenBatch::new(&[[1u32, 2, 3, 4]])
            .unwrap()
            // Flag the last two tokens: rows 1 and 2 predict them.
            .with_loss_mask(&[false, false, true, true])
            .unwrap();

        assert_eq!(batch.predicted(), 2);
        assert!(!batch.predicts(0));
        assert!(batch.predicts(1));
        assert!(batch.predicts(2));
        // Nothing follows the last token, whatever the mask says.
        assert!(!batch.predicts(3));
    }

    #[test]
    fn a_loss_mask_never_resurrects_padding() {
        let batch = TokenBatch::new(&[vec![1u32, 2, 3, 4], vec![5, 6]])
            .unwrap()
            .with_loss_mask(&[true; 8])
            .unwrap();

        // Same count as the unmasked batch: rows 5, 6 and 7 of the short
        // sequence are padding or its final token.
        assert_eq!(batch.predicted(), 4);
        assert!(batch.predicts(4));
        assert!(!batch.predicts(5));
        assert!(!batch.predicts(6));
    }

    #[test]
    fn a_loss_mask_of_the_wrong_length_or_all_false_is_rejected() {
        let batch = || TokenBatch::new(&[[1u32, 2, 3, 4]]).unwrap();

        assert!(batch().with_loss_mask(&[true; 3]).is_err());
        assert!(batch().with_loss_mask(&[false; 4]).is_err());
        // Flagging only the first token leaves nothing to predict: nothing
        // precedes it.
        assert!(
            batch()
                .with_loss_mask(&[true, false, false, false])
                .is_err()
        );
    }

    #[test]
    fn a_one_token_sequence_is_rejected() {
        assert!(TokenBatch::new(&[vec![1u32]]).is_err());
        assert!(TokenBatch::new::<Vec<u32>>(&[]).is_err());
    }

    #[test]
    fn the_default_layout_is_one_sequence() {
        let layout = Layout::default();

        assert_eq!(layout.seq_len(7), 7);
        assert!(layout.is_valid(6));
        assert!(layout.check(7).is_ok());
    }

    #[test]
    fn a_layout_that_does_not_divide_the_rows_is_rejected() {
        assert!(Layout::single(3).check(7).is_err());
        let mask = [true, true];
        let layout = Layout {
            seq_len: Some(2),
            valid: Some(&mask),
        };
        assert!(layout.check(4).is_err());
    }
}
