//! Training on a corpus that does not fit in memory.
//!
//! A pre-tokenized corpus is a flat array of ids, and the only thing a language
//! model asks of it is "give me `batch` windows of `seq_len` tokens". That does
//! not need the whole array resident: it needs `batch` reads of `seq_len * 4`
//! bytes. [`TokenFile`] is that, over a file of little-endian `u32` ids.
//!
//! Windows start at uniformly random offsets rather than walking the file in
//! order, which is how a corpus too large to shuffle gets shuffled. The offsets
//! are drawn from the step number alone, so a run that dies at step 40,000 and
//! resumes there sees the same data it would have seen — there is no cursor to
//! checkpoint, and none to get out of step with the weights.
//!
//! ```no_run
//! # use rusting_brain::TokenFile;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! TokenFile::write("corpus.bin", &[1, 2, 3, 4])?;          // once, offline
//!
//! let mut corpus = TokenFile::open("corpus.bin", 42)?;
//! for step in 0..1_000u64 {
//!     let batch = corpus.batch(step, 8, 512)?;
//!     // model.train_step_batch(&batch)?;
//! }
//! # Ok(())
//! # }
//! ```

use crate::batch::TokenBatch;
use crate::network::NetworkError;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

fn check_batch(sequences: usize, seq_len: usize) -> Result<(), NetworkError> {
    if seq_len < 2 || sequences == 0 {
        return Err(NetworkError::InvalidConfig(format!(
            "a batch of {sequences} sequences of {seq_len} tokens has nothing to predict"
        )));
    }
    Ok(())
}

/// The generator for one step's draws.
///
/// Mixing the step through a multiplier rather than seeding with it directly:
/// consecutive seeds give `StdRng` streams that are unrelated in principle but
/// neighbouring in practice, and this is cheaper than proving that does not
/// matter.
fn step_rng(seed: u64, step: u64) -> StdRng {
    StdRng::seed_from_u64(seed ^ step.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// Several corpora drawn from by weight, the way a pretraining mix of web
/// text, code and books is: each sequence of a batch picks its source, then
/// a window inside it.
///
/// Like [`TokenFile::batch`], everything is drawn from the step number and
/// the mix's own seed, so a resumed run sees the same batches. The seeds the
/// files were opened with are not used.
///
/// ```no_run
/// # use rusting_brain::{TokenFile, TokenMix};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut mix = TokenMix::new(
///     vec![
///         (TokenFile::open("web.bin", 0)?, 0.7),
///         (TokenFile::open("code.bin", 0)?, 0.3),
///     ],
///     42,
/// )?;
/// let batch = mix.batch(0, 8, 512)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct TokenMix {
    files: Vec<TokenFile>,
    weights: WeightedIndex<f64>,
    seed: u64,
}

impl TokenMix {
    /// Weights are relative and need not sum to one. A zero weight leaves a
    /// source out; a negative one, or none above zero, is an error.
    pub fn new(sources: Vec<(TokenFile, f64)>, seed: u64) -> Result<Self, NetworkError> {
        let (files, weights): (Vec<_>, Vec<_>) = sources.into_iter().unzip();
        let weights = WeightedIndex::new(&weights).map_err(|error| {
            NetworkError::InvalidConfig(format!("token mix weights {weights:?}: {error}"))
        })?;
        Ok(Self {
            files,
            weights,
            seed,
        })
    }

    /// `sequences` windows of `seq_len` tokens, drawn from `step`. See
    /// [`TokenFile::batch`].
    pub fn batch(
        &mut self,
        step: u64,
        sequences: usize,
        seq_len: usize,
    ) -> Result<TokenBatch, NetworkError> {
        check_batch(sequences, seq_len)?;
        let mut rng = step_rng(self.seed, step);
        let windows = (0..sequences)
            .map(|_| {
                let source = self.weights.sample(&mut rng);
                self.files[source].window(&mut rng, seq_len)
            })
            .collect::<Result<Vec<_>, _>>()?;
        TokenBatch::new(&windows)
    }
}

/// A memory-resident-free view of a pre-tokenized corpus on disk.
#[derive(Debug)]
pub struct TokenFile {
    file: File,
    tokens: u64,
    seed: u64,
}

impl TokenFile {
    /// Writes `ids` as little-endian `u32`, the format [`TokenFile::open`]
    /// expects.
    ///
    /// Little-endian regardless of the host, so a corpus tokenized on one
    /// machine trains on another.
    pub fn write(path: impl AsRef<Path>, ids: &[u32]) -> Result<(), NetworkError> {
        let mut file = std::io::BufWriter::new(File::create(path)?);
        for &id in ids {
            file.write_all(&id.to_le_bytes())?;
        }
        file.flush()?;
        Ok(())
    }

    /// Tokenizes a JSONL corpus into the same format, one document per line.
    ///
    /// JSONL is what public text datasets ship in, and the step between
    /// downloading one and training on it is otherwise a script the caller
    /// writes. `field` names the string member to read — `"text"` for most
    /// corpora — and `separator` is appended after every document, which is
    /// where an end-of-text id goes so the model does not learn to run one
    /// document into the next.
    ///
    /// The file is read a line at a time and ids are written as they are
    /// produced, so the corpus never has to fit in memory. Returns how many
    /// tokens were written.
    ///
    /// Tokenization is the caller's: the crate ships no tokenizer, and
    /// `tokenize` is where a `tokenizers` BPE vocabulary or a byte-level
    /// mapping goes.
    ///
    /// ```no_run
    /// # use rusting_brain::TokenFile;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tokens = TokenFile::write_jsonl("corpus.bin", "corpus.jsonl", "text", Some(0), |text| {
    ///     text.bytes().map(u32::from).collect()
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn write_jsonl(
        path: impl AsRef<Path>,
        jsonl: impl AsRef<Path>,
        field: &str,
        separator: Option<u32>,
        mut tokenize: impl FnMut(&str) -> Vec<u32>,
    ) -> Result<u64, NetworkError> {
        let source = std::io::BufReader::new(File::open(jsonl)?);
        let mut out = std::io::BufWriter::new(File::create(path)?);
        let mut written = 0u64;

        for (index, line) in std::io::BufRead::lines(source).enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let document: serde_json::Value = serde_json::from_str(&line).map_err(|error| {
                NetworkError::InvalidDataset(format!("line {}: {error}", index + 1))
            })?;
            let text = document
                .get(field)
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    NetworkError::InvalidDataset(format!(
                        "line {} has no string field {field:?}",
                        index + 1
                    ))
                })?;

            for id in tokenize(text).into_iter().chain(separator) {
                out.write_all(&id.to_le_bytes())?;
                written += 1;
            }
        }

        out.flush()?;
        Ok(written)
    }

    /// Opens a corpus. `seed` picks the window offsets; two runs with the same
    /// seed see the same data in the same order.
    pub fn open(path: impl AsRef<Path>, seed: u64) -> Result<Self, NetworkError> {
        let file = File::open(path)?;
        let bytes = file.metadata()?.len();
        if bytes == 0 || bytes % 4 != 0 {
            return Err(NetworkError::InvalidDataset(format!(
                "a token file holds 4-byte ids, so its length cannot be {bytes} bytes"
            )));
        }
        Ok(Self {
            file,
            tokens: bytes / 4,
            seed,
        })
    }

    /// Number of tokens in the corpus.
    pub fn tokens(&self) -> u64 {
        self.tokens
    }

    /// `sequences` windows of `seq_len` tokens, drawn from `step`.
    ///
    /// `step` is any counter the caller keeps — a training step, or a step
    /// times the accumulation count plus the micro-batch index. The same
    /// counter always returns the same batch, which is what makes a resumed run
    /// line up with the one it replaced.
    pub fn batch(
        &mut self,
        step: u64,
        sequences: usize,
        seq_len: usize,
    ) -> Result<TokenBatch, NetworkError> {
        check_batch(sequences, seq_len)?;
        let mut rng = step_rng(self.seed, step);
        let windows = (0..sequences)
            .map(|_| self.window(&mut rng, seq_len))
            .collect::<Result<Vec<_>, _>>()?;
        TokenBatch::new(&windows)
    }

    /// One window of `seq_len` tokens at an offset drawn from `rng`.
    fn window(&mut self, rng: &mut StdRng, seq_len: usize) -> Result<Vec<u32>, NetworkError> {
        if (seq_len as u64) > self.tokens {
            return Err(NetworkError::InvalidDataset(format!(
                "a window of {seq_len} tokens does not fit in a corpus of {}",
                self.tokens
            )));
        }
        let offset = rng.gen_range(0..=self.tokens - seq_len as u64);
        let mut window = vec![0u8; seq_len * 4];
        self.file.seek(SeekFrom::Start(offset * 4))?;
        self.file.read_exact(&mut window)?;
        Ok(window
            .chunks_exact(4)
            .map(|id| u32::from_le_bytes([id[0], id[1], id[2], id[3]]))
            .collect())
    }

    /// Consecutive, non-overlapping windows: batch `index` of a walk through
    /// the whole corpus, or `None` once it has been walked.
    ///
    /// [`batch`](TokenFile::batch) draws at random, which is what training
    /// wants and what makes it useless for a held-out loss — the number would
    /// move between evaluations of an unchanged model, and some tokens would be
    /// scored twice and others not at all. This covers every token once, in
    /// order.
    ///
    /// ```no_run
    /// # use rusting_brain::TokenFile;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut held_out = TokenFile::open("validation.bin", 0)?;
    /// let (mut total, mut batches) = (0.0, 0);
    /// while let Some(batch) = held_out.chunk(batches, 8, 512)? {
    ///     // total += model.evaluate(&batch)?.lm_loss;
    ///     batches += 1;
    /// }
    /// let validation_loss = total / batches as f32;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// The tokens left over at the end, fewer than one window, are dropped
    /// rather than padded: a padded window's loss is not comparable with a full
    /// one's.
    pub fn chunk(
        &mut self,
        index: u64,
        sequences: usize,
        seq_len: usize,
    ) -> Result<Option<TokenBatch>, NetworkError> {
        if seq_len < 2 || sequences == 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "a batch of {sequences} sequences of {seq_len} tokens has nothing to predict"
            )));
        }

        let start = index * (sequences * seq_len) as u64;
        let remaining = self.tokens.saturating_sub(start);
        let available = (remaining / seq_len as u64).min(sequences as u64) as usize;
        if available == 0 {
            return Ok(None);
        }

        self.file.seek(SeekFrom::Start(start * 4))?;
        let mut window = vec![0u8; seq_len * 4];
        let mut windows = Vec::with_capacity(available);
        for _ in 0..available {
            self.file.read_exact(&mut window)?;
            windows.push(
                window
                    .chunks_exact(4)
                    .map(|id| u32::from_le_bytes([id[0], id[1], id[2], id[3]]))
                    .collect::<Vec<u32>>(),
            );
        }

        Ok(Some(TokenBatch::new(&windows)?))
    }
}

/// A text corpus tokenized as it is read, for training without writing a
/// [`TokenFile`] first.
///
/// [`TokenFile`] wants a flat array of ids on disk, which means a tokenizing
/// pass over the corpus and a second copy of it. This reads the text itself,
/// tokenizes a document at a time and cuts the ids into windows, so the corpus
/// on disk is the only copy and training can start immediately.
///
/// Tokenization is the caller's, as it is for
/// [`write_jsonl`](TokenFile::write_jsonl): the crate ships no tokenizer, and
/// `tokenize` is where a `tokenizers` BPE vocabulary or a byte-level mapping
/// goes.
///
/// ```no_run
/// # use rusting_brain::TokenStream;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut corpus = TokenStream::jsonl("corpus.jsonl", "text", Some(0), |text| {
///     text.bytes().map(u32::from).collect()
/// })?;
///
/// while let Some(batch) = corpus.batch(8, 512)? {
///     // model.train_step_batch(&batch)?;
/// }
/// corpus.restart()?;                       // the next epoch
/// # Ok(())
/// # }
/// ```
///
/// ponytail: windows come out in file order, where [`TokenFile::batch`] draws
/// them from random offsets. A corpus whose documents are already unordered
/// does not care; one that is sorted by source or by date does, and wants
/// either a shuffle buffer here or the pre-tokenized path. Read the corpus
/// through [`TokenFile::write_jsonl`] to get the random-offset behaviour.
pub struct TokenStream<T: FnMut(&str) -> Vec<u32>> {
    path: std::path::PathBuf,
    lines: std::io::Lines<std::io::BufReader<File>>,
    /// The JSON member holding the text, or `None` when every line is itself a
    /// document.
    field: Option<String>,
    separator: Option<u32>,
    tokenize: T,
    /// Ids read but not yet handed out. It holds at most one batch plus the
    /// tail of the document that filled it.
    pending: Vec<u32>,
    exhausted: bool,
    line_number: usize,
}

impl<T: FnMut(&str) -> Vec<u32>> TokenStream<T> {
    /// Reads a plain text file, one document per line.
    pub fn text(
        path: impl AsRef<Path>,
        separator: Option<u32>,
        tokenize: T,
    ) -> Result<Self, NetworkError> {
        Self::open(path.as_ref(), None, separator, tokenize)
    }

    /// Reads a JSONL file, taking `field` of every line as the document.
    ///
    /// `separator` is appended after every document, which is where an
    /// end-of-text id goes so the model does not learn to run one document into
    /// the next.
    pub fn jsonl(
        path: impl AsRef<Path>,
        field: &str,
        separator: Option<u32>,
        tokenize: T,
    ) -> Result<Self, NetworkError> {
        Self::open(path.as_ref(), Some(field.to_string()), separator, tokenize)
    }

    fn open(
        path: &Path,
        field: Option<String>,
        separator: Option<u32>,
        tokenize: T,
    ) -> Result<Self, NetworkError> {
        Ok(Self {
            lines: read_lines(path)?,
            path: path.to_path_buf(),
            field,
            separator,
            tokenize,
            pending: Vec::new(),
            exhausted: false,
            line_number: 0,
        })
    }

    /// The next `sequences` windows of `seq_len` tokens, or `None` at the end
    /// of the corpus.
    ///
    /// The last batch of a pass is short when the corpus does not divide
    /// evenly, and its last window is shorter than `seq_len` when the tail is:
    /// [`TokenBatch`] pads and masks it. A tail of one token is dropped, a
    /// next-token loss over a single token having nothing to predict.
    pub fn batch(
        &mut self,
        sequences: usize,
        seq_len: usize,
    ) -> Result<Option<TokenBatch>, NetworkError> {
        if seq_len < 2 || sequences == 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "a batch of {sequences} sequences of {seq_len} tokens has nothing to predict"
            )));
        }

        self.fill(sequences * seq_len)?;

        let mut windows = Vec::with_capacity(sequences);
        while windows.len() < sequences && self.pending.len() >= seq_len {
            windows.push(self.pending.drain(..seq_len).collect::<Vec<u32>>());
        }
        // A tail too short for a window is still worth a step, but only once
        // the file behind it is spent.
        if windows.len() < sequences && self.exhausted && self.pending.len() >= 2 {
            windows.push(std::mem::take(&mut self.pending));
        }

        if windows.is_empty() {
            return Ok(None);
        }
        Ok(Some(TokenBatch::new(&windows)?))
    }

    /// Returns to the start of the corpus, which is what the next epoch wants.
    pub fn restart(&mut self) -> Result<(), NetworkError> {
        self.lines = read_lines(&self.path)?;
        self.pending.clear();
        self.exhausted = false;
        self.line_number = 0;
        Ok(())
    }

    /// Tokenizes documents until `wanted` ids are pending or the file runs out.
    fn fill(&mut self, wanted: usize) -> Result<(), NetworkError> {
        while self.pending.len() < wanted {
            let Some(line) = self.lines.next() else {
                self.exhausted = true;
                break;
            };
            let line = line?;
            self.line_number += 1;
            if line.trim().is_empty() {
                continue;
            }

            let document = match &self.field {
                None => line,
                Some(field) => {
                    let parsed: serde_json::Value =
                        serde_json::from_str(&line).map_err(|error| {
                            NetworkError::InvalidDataset(format!(
                                "{}:{}: {error}",
                                self.path.display(),
                                self.line_number
                            ))
                        })?;
                    parsed
                        .get(field)
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| {
                            NetworkError::InvalidDataset(format!(
                                "{}:{}: no string field `{field}`",
                                self.path.display(),
                                self.line_number
                            ))
                        })?
                        .to_string()
                }
            };

            let ids = (self.tokenize)(&document);
            self.pending.extend(ids);
            if let Some(separator) = self.separator {
                self.pending.push(separator);
            }
        }
        Ok(())
    }
}

fn read_lines(path: &Path) -> Result<std::io::Lines<std::io::BufReader<File>>, NetworkError> {
    use std::io::BufRead;
    Ok(std::io::BufReader::new(File::open(path)?).lines())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A corpus of `0..count`, so a window is correct exactly when its ids
    /// count up by one.
    fn corpus(name: &str, count: u32) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("rusting_brain_{name}_{}.bin", std::process::id()));
        TokenFile::write(&path, &(0..count).collect::<Vec<u32>>()).unwrap();
        path
    }

    #[test]
    fn a_batch_holds_contiguous_windows_of_the_corpus() {
        let path = corpus("contiguous", 1_000);
        let mut file = TokenFile::open(&path, 7).unwrap();
        assert_eq!(file.tokens(), 1_000);

        let batch = file.batch(0, 4, 16).unwrap();
        assert_eq!(batch.batch(), 4);
        assert_eq!(batch.seq_len(), 16);
        assert!(!batch.is_padded());
        for window in batch.ids().chunks_exact(16) {
            assert!(window.windows(2).all(|pair| pair[1] == pair[0] + 1));
            assert!(window[15] < 1_000);
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn the_same_step_draws_the_same_windows_and_other_steps_do_not() {
        let path = corpus("steps", 10_000);
        let mut file = TokenFile::open(&path, 7).unwrap();

        assert_eq!(file.batch(3, 4, 16).unwrap(), file.batch(3, 4, 16).unwrap());
        assert_ne!(file.batch(3, 4, 16).unwrap(), file.batch(4, 4, 16).unwrap());

        let mut other = TokenFile::open(&path, 8).unwrap();
        assert_ne!(
            file.batch(3, 4, 16).unwrap(),
            other.batch(3, 4, 16).unwrap()
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn chunks_walk_the_corpus_once_in_order_and_then_stop() {
        let path = corpus("chunks", 50);
        let mut file = TokenFile::open(&path, 7).unwrap();

        // 50 tokens, windows of 8: six whole windows, two tokens dropped.
        let first = file.chunk(0, 4, 8).unwrap().unwrap();
        assert_eq!(first.batch(), 4);
        assert_eq!(&first.ids()[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(&first.ids()[24..], &[24, 25, 26, 27, 28, 29, 30, 31]);

        // The last batch is short rather than padded out.
        let second = file.chunk(1, 4, 8).unwrap().unwrap();
        assert_eq!(second.batch(), 2);
        assert!(!second.is_padded());
        assert_eq!(&second.ids()[..8], &[32, 33, 34, 35, 36, 37, 38, 39]);

        assert!(file.chunk(2, 4, 8).unwrap().is_none());
        // Deterministic, unlike `batch`.
        assert_eq!(file.chunk(0, 4, 8).unwrap(), Some(first));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_window_as_long_as_the_corpus_is_the_whole_corpus() {
        let path = corpus("exact", 8);
        let mut file = TokenFile::open(&path, 1).unwrap();

        let batch = file.batch(0, 2, 8).unwrap();
        assert_eq!(
            batch.ids(),
            &[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert!(file.batch(0, 2, 9).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_streamed_corpus_holds_the_same_ids_as_the_file_it_replaces() {
        let jsonl =
            std::env::temp_dir().join(format!("rusting_brain_stream_{}.jsonl", std::process::id()));
        std::fs::write(&jsonl, "{\"text\":\"abcd\"}\n\n{\"text\":\"efgh\"}\n").unwrap();
        let bin =
            std::env::temp_dir().join(format!("rusting_brain_stream_{}.bin", std::process::id()));

        // The pre-tokenized path, which the streamed one has to agree with.
        let tokenize = |text: &str| text.bytes().map(u32::from).collect::<Vec<u32>>();
        TokenFile::write_jsonl(&bin, &jsonl, "text", Some(0), tokenize).unwrap();
        let expected: Vec<u32> = std::fs::read(&bin)
            .unwrap()
            .chunks_exact(4)
            .map(|id| u32::from_le_bytes([id[0], id[1], id[2], id[3]]))
            .collect();
        assert_eq!(expected.len(), 10, "four bytes and a separator, twice");

        let mut corpus = TokenStream::jsonl(&jsonl, "text", Some(0), tokenize).unwrap();
        let mut streamed = Vec::new();
        let mut batches = 0;
        while let Some(batch) = corpus.batch(2, 4).unwrap() {
            // Read back window by window, because the last one is a short tail
            // the batch has padded.
            for (sequence, &length) in batch.lengths().iter().enumerate() {
                let start = sequence * batch.seq_len();
                streamed.extend_from_slice(&batch.ids()[start..start + length]);
            }
            batches += 1;
        }

        assert_eq!(streamed, expected);
        assert_eq!(batches, 2, "two full windows, then the two-token tail");

        // A second pass sees the corpus again, which is what an epoch is.
        corpus.restart().unwrap();
        assert!(corpus.batch(2, 4).unwrap().is_some());

        let missing = TokenStream::jsonl(&jsonl, "body", None, tokenize)
            .unwrap()
            .batch(1, 2)
            .unwrap_err()
            .to_string();
        assert!(
            missing.contains("no string field `body`"),
            "the error was {missing}"
        );

        std::fs::remove_file(jsonl).unwrap();
        std::fs::remove_file(bin).unwrap();
    }

    #[test]
    fn a_jsonl_corpus_is_tokenized_document_by_document() {
        let jsonl =
            std::env::temp_dir().join(format!("rusting_brain_{}.jsonl", std::process::id()));
        std::fs::write(&jsonl, "{\"text\":\"ab\",\"id\":1}\n\n{\"text\":\"c\"}\n").unwrap();
        let bin =
            std::env::temp_dir().join(format!("rusting_brain_jsonl_{}.bin", std::process::id()));

        let written = TokenFile::write_jsonl(&bin, &jsonl, "text", Some(255), |text| {
            text.bytes().map(u32::from).collect()
        })
        .unwrap();

        // Two documents, a blank line skipped, a separator after each.
        assert_eq!(written, 5);
        let mut file = TokenFile::open(&bin, 0).unwrap();
        assert_eq!(file.tokens(), 5);
        assert_eq!(file.batch(0, 1, 5).unwrap().ids(), &[97, 98, 255, 99, 255]);

        // A line without the field names the line rather than the file.
        assert!(matches!(
            TokenFile::write_jsonl(&bin, &jsonl, "body", None, |_| vec![1]),
            Err(NetworkError::InvalidDataset(message)) if message.contains("line 1")
        ));

        std::fs::remove_file(jsonl).unwrap();
        std::fs::remove_file(bin).unwrap();
    }

    #[test]
    fn a_file_that_is_not_whole_ids_is_rejected() {
        let path =
            std::env::temp_dir().join(format!("rusting_brain_ragged_{}.bin", std::process::id()));
        std::fs::write(&path, [1u8, 2, 3]).unwrap();

        assert!(matches!(
            TokenFile::open(&path, 0),
            Err(NetworkError::InvalidDataset(_))
        ));
        std::fs::remove_file(path).unwrap();
    }

    /// Two corpora told apart by their ids: the first holds `0..1000`, the
    /// second `1000..2000`.
    fn two_sources(name: &str, weights: [f64; 2]) -> (TokenMix, [std::path::PathBuf; 2]) {
        let low = corpus(&format!("{name}_low"), 1_000);
        let high = std::env::temp_dir().join(format!(
            "rusting_brain_{name}_high_{}.bin",
            std::process::id()
        ));
        TokenFile::write(&high, &(1_000..2_000).collect::<Vec<u32>>()).unwrap();
        let mix = TokenMix::new(
            vec![
                (TokenFile::open(&low, 0).unwrap(), weights[0]),
                (TokenFile::open(&high, 0).unwrap(), weights[1]),
            ],
            5,
        )
        .unwrap();
        (mix, [low, high])
    }

    #[test]
    fn a_mix_draws_each_source_in_proportion_to_its_weight() {
        let (mut mix, paths) = two_sources("mix_ratio", [3.0, 1.0]);
        let (mut low, mut total) = (0, 0);
        for step in 0..200 {
            let batch = mix.batch(step, 8, 16).unwrap();
            for window in batch.ids().chunks_exact(16) {
                // A window never straddles two sources.
                assert_eq!(window[0] < 1_000, window[15] < 1_000);
                assert!(window.windows(2).all(|pair| pair[1] == pair[0] + 1));
                low += usize::from(window[0] < 1_000);
                total += 1;
            }
        }
        paths
            .iter()
            .for_each(|path| std::fs::remove_file(path).unwrap());
        let share = low as f64 / total as f64;
        assert!((share - 0.75).abs() < 0.04, "share {share}");
    }

    #[test]
    fn a_mix_replays_the_same_batch_for_the_same_step() {
        let (mut mix, paths) = two_sources("mix_replay", [1.0, 1.0]);
        let first = mix.batch(9, 4, 16).unwrap().ids().to_vec();
        let (mut reopened, more) = two_sources("mix_replay_again", [1.0, 1.0]);
        let again = reopened.batch(9, 4, 16).unwrap().ids().to_vec();
        let other = mix.batch(10, 4, 16).unwrap().ids().to_vec();
        paths
            .iter()
            .chain(&more)
            .for_each(|path| std::fs::remove_file(path).unwrap());
        assert_eq!(first, again);
        assert_ne!(first, other);
    }

    #[test]
    fn a_mix_with_no_positive_weight_is_refused() {
        let (_, paths) = two_sources("mix_refused", [1.0, 1.0]);
        let open = |path| TokenFile::open(path, 0).unwrap();
        let zero = TokenMix::new(vec![(open(&paths[0]), 0.0), (open(&paths[1]), 0.0)], 0);
        let negative = TokenMix::new(vec![(open(&paths[0]), 1.0), (open(&paths[1]), -1.0)], 0);
        let empty = TokenMix::new(Vec::new(), 0);
        paths
            .iter()
            .for_each(|path| std::fs::remove_file(path).unwrap());
        assert!(zero.is_err() && negative.is_err() && empty.is_err());
    }
}
