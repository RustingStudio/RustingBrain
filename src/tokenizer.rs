//! Byte-level byte-pair encoding, read from a `tokenizer.json`.
//!
//! A text-to-image model conditions on a prompt, and the prompt has to become
//! token ids the way the text encoder was trained to see them. Every encoder in
//! use here — CLIP, Qwen3, and the GPT-2 line generally — tokenizes with the
//! same algorithm: map the text's bytes into a printable alphabet, split it
//! into words, then merge adjacent symbol pairs in the order a merge table
//! lists them.
//!
//! This is a reader for the file Hugging Face publishes that table in. No new
//! dependency: the file is JSON, and the crate already reads JSON.
//!
//! The same type also reads the sentencepiece flavour of BPE that LLaMA 2,
//! Mistral and TinyLlama ship: no byte alphabet, spaces written as `▁`, and
//! a character the vocabulary lacks spelled as `<0xNN>` byte tokens.
//!
//! [`Unigram`] beside it is the sentencepiece algorithm T5 tokenizes with,
//! which is a different model rather than a different table: the file lists a
//! score for every piece, and tokenizing is the highest-scoring way to cut the
//! text into pieces that are in the list.
//!
//! NOTE: the pre-tokenizer is the GPT-2 split rule written as a state
//! machine instead of the regular expression the reference uses, because the
//! crate has no regular-expression engine and this rule is small enough to
//! write out. Upgrade path if a checkpoint ever ships a different rule: read
//! `pre_tokenizer` out of the file and dispatch on it.

use crate::network::NetworkError;
use std::collections::HashMap;

/// A byte-level BPE tokenizer.
#[derive(Clone, Debug)]
pub struct Bpe {
    vocab: HashMap<String, u32>,
    tokens: Vec<String>,
    /// Merge priority: a lower rank is applied first.
    ranks: HashMap<(String, String), u32>,
    /// Tokens matched literally before anything else, longest first.
    specials: Vec<(String, u32)>,
    /// What CLIP appends to the last symbol of each word.
    end_of_word: Option<String>,
    lowercase: bool,
    /// Set for a sentencepiece-style file; `None` is byte-level BPE.
    metaspace: Option<Metaspace>,
    /// Spell a symbol the vocabulary lacks as `<0xNN>` tokens.
    byte_fallback: bool,
    byte_encoder: Vec<char>,
    byte_decoder: HashMap<char, u8>,
}

/// When a sentencepiece-style file puts a `▁` in front of the text.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Metaspace {
    /// The `Metaspace` pre-tokenizer: in front of the input's first piece of
    /// text only, and not when that piece already starts with `▁`.
    First,
    /// The same, in front of every piece of text between special tokens.
    Always,
    Never,
    /// The older `Prepend` normalizer: in front of every piece, always.
    Prepend,
}

impl Bpe {
    /// Reads a `tokenizer.json`.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, NetworkError> {
        let text = std::fs::read_to_string(path)?;
        Self::from_json(&text)
    }

    /// The same, from the file's contents.
    pub fn from_json(text: &str) -> Result<Self, NetworkError> {
        let json: serde_json::Value = serde_json::from_str(text)
            .map_err(|error| NetworkError::InvalidDataset(format!("tokenizer.json: {error}")))?;
        let model = json
            .get("model")
            .ok_or_else(|| NetworkError::InvalidDataset("tokenizer.json has no model".into()))?;

        let vocab: HashMap<String, u32> = model
            .get("vocab")
            .and_then(|vocab| vocab.as_object())
            .ok_or_else(|| NetworkError::InvalidDataset("the model has no vocabulary".into()))?
            .iter()
            .map(|(token, id)| {
                let id = id
                    .as_u64()
                    .ok_or_else(|| NetworkError::InvalidDataset(format!("{token} has no id")))?;
                Ok((token.clone(), id as u32))
            })
            .collect::<Result<_, NetworkError>>()?;

        let mut tokens =
            vec![String::new(); vocab.values().map(|id| *id as usize + 1).max().unwrap_or(0)];
        for (token, id) in &vocab {
            tokens[*id as usize] = token.clone();
        }

        // Merges are a list of "a b" strings in older files and of ["a", "b"]
        // pairs in newer ones.
        let mut ranks = HashMap::new();
        if let Some(merges) = model.get("merges").and_then(|merges| merges.as_array()) {
            for (rank, merge) in merges.iter().enumerate() {
                let pair = match merge {
                    serde_json::Value::String(merge) => {
                        let mut parts = merge.splitn(2, ' ');
                        match (parts.next(), parts.next()) {
                            (Some(first), Some(second)) => (first.to_string(), second.to_string()),
                            _ => continue,
                        }
                    }
                    serde_json::Value::Array(pair) if pair.len() == 2 => (
                        pair[0].as_str().unwrap_or_default().to_string(),
                        pair[1].as_str().unwrap_or_default().to_string(),
                    ),
                    _ => continue,
                };
                ranks.insert(pair, rank as u32);
            }
        }

        let mut specials: Vec<(String, u32)> = json
            .get("added_tokens")
            .and_then(|added| added.as_array())
            .map(|added| {
                added
                    .iter()
                    .filter_map(|token| {
                        Some((
                            token.get("content")?.as_str()?.to_string(),
                            token.get("id")?.as_u64()? as u32,
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default();
        // Longest first, so `<|im_start|>` wins over a prefix of itself.
        specials.sort_by_key(|(token, _)| std::cmp::Reverse(token.len()));

        let end_of_word = model
            .get("end_of_word_suffix")
            .and_then(|suffix| suffix.as_str())
            .map(str::to_string);
        let lowercase = json
            .get("normalizer")
            .map(|normalizer| normalizer.to_string().contains("Lowercase"))
            .unwrap_or(false);

        // A file lists its added tokens separately, but they are part of the
        // vocabulary: a caller asking for `<|endoftext|>` by name has to find
        // it, and decoding has to print it.
        let mut vocab = vocab;
        for (token, id) in &specials {
            if tokens.len() <= *id as usize {
                tokens.resize(*id as usize + 1, String::new());
            }
            tokens[*id as usize] = token.clone();
            vocab.entry(token.clone()).or_insert(*id);
        }

        let normalizer = json
            .get("normalizer")
            .map(|normalizer| normalizer.to_string())
            .unwrap_or_default();
        let metaspace = match pre_tokenizer(&json, "Metaspace") {
            Some(pre) => Some(
                match pre.get("prepend_scheme").and_then(|scheme| scheme.as_str()) {
                    Some("first") => Metaspace::First,
                    Some("never") => Metaspace::Never,
                    // Files older than `prepend_scheme` say `add_prefix_space`.
                    _ if pre.get("add_prefix_space") == Some(&serde_json::Value::Bool(false)) => {
                        Metaspace::Never
                    }
                    _ => Metaspace::Always,
                },
            ),
            None if normalizer.contains("\"Prepend\"") => Some(Metaspace::Prepend),
            None => None,
        };
        let byte_fallback = model
            .get("byte_fallback")
            .and_then(|flag| flag.as_bool())
            .unwrap_or(false);

        let byte_encoder = byte_alphabet();
        let byte_decoder = byte_encoder
            .iter()
            .enumerate()
            .map(|(byte, character)| (*character, byte as u8))
            .collect();

        Ok(Self {
            vocab,
            tokens,
            ranks,
            specials,
            end_of_word,
            lowercase,
            metaspace,
            byte_fallback,
            byte_encoder,
            byte_decoder,
        })
    }

    /// How many ids the vocabulary holds.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// The id of a token written out in full, which is how a caller finds a
    /// special token such as `<|endoftext|>`.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Tokenizes text.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, NetworkError> {
        let owned;
        let text = if self.lowercase {
            owned = text.to_lowercase();
            &owned
        } else {
            text
        };

        let mut ids = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // A special token is matched literally wherever it appears, and
            // never merged into its neighbours.
            let special = self
                .specials
                .iter()
                .filter_map(|(token, id)| rest.find(token.as_str()).map(|at| (at, token, *id)))
                .min_by_key(|(at, token, _)| (*at, usize::MAX - token.len()));
            let (head, tail) = match special {
                Some((at, token, id)) => {
                    let head = &rest[..at];
                    let tail = &rest[at + token.len()..];
                    self.encode_plain(head, rest.len() == text.len(), &mut ids)?;
                    ids.push(id);
                    ("", tail)
                }
                None => (rest, ""),
            };
            if !head.is_empty() {
                self.encode_plain(head, rest.len() == text.len(), &mut ids)?;
            }
            rest = tail;
        }
        Ok(ids)
    }

    /// Encodes text that holds no special token. `first` says whether it
    /// starts the input, which is where a sentencepiece file puts its `▁`.
    fn encode_plain(
        &self,
        text: &str,
        first: bool,
        ids: &mut Vec<u32>,
    ) -> Result<(), NetworkError> {
        if let Some(metaspace) = self.metaspace {
            return self.encode_metaspace(text, metaspace, first, ids);
        }
        // CLIP marks the end of a word with a suffix instead of carrying the
        // space that precedes it, so its rule throws whitespace away rather
        // than joining it to the next word, and cuts a run of digits into
        // single digits. The GPT-2 rule keeps both.
        let words: Vec<String> = match self.end_of_word.is_some() {
            true => pre_tokenize(text)
                .iter()
                .map(|word| word.trim())
                .filter(|word| !word.is_empty())
                .flat_map(|word| match word.chars().all(char::is_numeric) {
                    true => word.chars().map(|digit| digit.to_string()).collect(),
                    false => vec![word.to_string()],
                })
                .collect(),
            false => pre_tokenize(text),
        };

        for word in words {
            // Bytes into the printable alphabet, so every input is spellable in
            // the vocabulary whatever it held.
            let mut symbols: Vec<String> = word
                .bytes()
                .map(|byte| self.byte_encoder[byte as usize].to_string())
                .collect();
            if symbols.is_empty() {
                continue;
            }
            if let Some(suffix) = &self.end_of_word {
                symbols
                    .last_mut()
                    .expect("the word is not empty")
                    .push_str(suffix);
            }

            self.merge(&mut symbols);

            for symbol in symbols {
                let id = self.vocab.get(&symbol).copied().ok_or_else(|| {
                    NetworkError::InvalidDataset(format!(
                        "the vocabulary has no token for {symbol:?}, so this file's merges and \
                         its vocabulary disagree"
                    ))
                })?;
                ids.push(id);
            }
        }
        Ok(())
    }

    /// Sentencepiece-style BPE: the whole piece of text is one word, with
    /// spaces as `▁`, starting from characters rather than bytes.
    ///
    /// NOTE: one merge loop over the whole piece, quadratic in its length.
    /// Fine for prompts and training lines; split long documents on newlines
    /// first, or replace the loop with a priority queue, if it ever shows up.
    fn encode_metaspace(
        &self,
        text: &str,
        metaspace: Metaspace,
        first: bool,
        ids: &mut Vec<u32>,
    ) -> Result<(), NetworkError> {
        if text.is_empty() {
            return Ok(());
        }
        let mut spaced = text.replace(' ', "\u{2581}");
        let prepend = match metaspace {
            Metaspace::First => first && !spaced.starts_with('\u{2581}'),
            Metaspace::Always => !spaced.starts_with('\u{2581}'),
            Metaspace::Never => false,
            Metaspace::Prepend => true,
        };
        if prepend {
            spaced.insert(0, '\u{2581}');
        }
        let mut symbols: Vec<String> = spaced.chars().map(String::from).collect();
        self.merge(&mut symbols);
        for symbol in symbols {
            if let Some(id) = self.vocab.get(&symbol) {
                ids.push(*id);
                continue;
            }
            if !self.byte_fallback {
                return Err(NetworkError::InvalidDataset(format!(
                    "the vocabulary has no token for {symbol:?} and the file has no byte fallback"
                )));
            }
            for byte in symbol.bytes() {
                let name = format!("<0x{byte:02X}>");
                let id = self.vocab.get(&name).ok_or_else(|| {
                    NetworkError::InvalidDataset(format!("byte fallback, but no {name} token"))
                })?;
                ids.push(*id);
            }
        }
        Ok(())
    }

    /// Repeatedly joins the best-ranked adjacent pair, which is the whole of
    /// byte-pair encoding.
    fn merge(&self, symbols: &mut Vec<String>) {
        while symbols.len() > 1 {
            let best = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(index, pair)| {
                    let rank = self.ranks.get(&(pair[0].clone(), pair[1].clone()))?;
                    Some((*rank, index))
                })
                .min();
            let Some((_, index)) = best else {
                return;
            };
            let second = symbols.remove(index + 1);
            symbols[index].push_str(&second);
        }
    }

    /// Turns ids back into text, which is what a round-trip check needs.
    pub fn decode(&self, ids: &[u32]) -> String {
        if let Some(metaspace) = self.metaspace {
            return self.decode_metaspace(ids, metaspace);
        }
        let mut bytes = Vec::new();
        for id in ids {
            let Some(token) = self.tokens.get(*id as usize) else {
                continue;
            };
            let token = match &self.end_of_word {
                Some(suffix) => token.replace(suffix.as_str(), " "),
                None => token.clone(),
            };
            if self.specials.iter().any(|(special, _)| *special == token) {
                bytes.extend_from_slice(token.as_bytes());
                continue;
            }
            for character in token.chars() {
                match self.byte_decoder.get(&character) {
                    Some(byte) => bytes.push(*byte),
                    // A special token written straight into the vocabulary is
                    // not in the byte alphabet, so it passes through as text.
                    None => bytes.extend_from_slice(character.to_string().as_bytes()),
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

impl Bpe {
    /// `▁` back to a space, `<0xNN>` back to its byte, and the space the
    /// encoder put in front taken off again.
    fn decode_metaspace(&self, ids: &[u32], metaspace: Metaspace) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            let Some(token) = self.tokens.get(*id as usize) else {
                continue;
            };
            let byte = token
                .strip_prefix("<0x")
                .and_then(|hex| hex.strip_suffix('>'))
                .filter(|hex| self.byte_fallback && hex.len() == 2)
                .and_then(|hex| u8::from_str_radix(hex, 16).ok());
            match byte {
                Some(byte) => bytes.push(byte),
                None => bytes.extend_from_slice(token.replace('\u{2581}', " ").as_bytes()),
            }
        }
        let text = String::from_utf8_lossy(&bytes).into_owned();
        match metaspace {
            Metaspace::Never => text,
            _ => text.strip_prefix(' ').map(str::to_string).unwrap_or(text),
        }
    }
}

/// The GPT-2 byte alphabet: every byte gets a printable character, so text in
/// any encoding survives the trip through a vocabulary of strings.
fn byte_alphabet() -> Vec<char> {
    let printable: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut alphabet = vec!['\0'; 256];
    let mut next = 0u32;
    for byte in 0..256u32 {
        let character = if printable.contains(&byte) {
            byte
        } else {
            let character = 256 + next;
            next += 1;
            character
        };
        alphabet[byte as usize] = char::from_u32(character).expect("a valid code point");
    }
    alphabet
}

/// Splits text the way GPT-2's pre-tokenizer does: contractions, then runs of
/// letters, digits or punctuation, each keeping the space in front of it.
fn pre_tokenize(text: &str) -> Vec<String> {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
    let characters: Vec<char> = text.chars().collect();
    let mut words = Vec::new();
    let mut index = 0;

    while index < characters.len() {
        let rest: String = characters[index..].iter().collect();
        if let Some(contraction) = CONTRACTIONS
            .iter()
            .find(|contraction| rest.starts_with(**contraction))
        {
            words.push((*contraction).to_string());
            index += contraction.chars().count();
            continue;
        }

        let start = index;
        // A single leading space belongs to the word that follows it.
        let space = characters[index] == ' ' && index + 1 < characters.len();
        let first = if space { index + 1 } else { index };
        let kind = characters
            .get(first)
            .copied()
            .map(classify)
            .unwrap_or(Kind::Space);

        if space && kind != Kind::Space {
            index += 1;
        } else if space {
            // Runs of whitespace stand alone, except that the last space goes
            // to the word after it.
            while index < characters.len() && characters[index].is_whitespace() {
                index += 1;
            }
            let trailing = index < characters.len();
            let end = if trailing && index - start > 1 {
                index - 1
            } else {
                index
            };
            words.push(characters[start..end].iter().collect());
            index = end;
            continue;
        }

        let body = index;
        while index < characters.len() && classify(characters[index]) == kind {
            index += 1;
        }
        if index == body {
            index += 1;
        }
        words.push(characters[start..index].iter().collect());
    }

    words
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Letter,
    Number,
    Other,
    Space,
}

fn classify(character: char) -> Kind {
    if character.is_alphabetic() {
        Kind::Letter
    } else if character.is_numeric() {
        Kind::Number
    } else if character.is_whitespace() {
        Kind::Space
    } else {
        Kind::Other
    }
}

/// A sentencepiece unigram tokenizer, which is what the T5 family uses.
///
/// Where byte-pair encoding builds tokens up by merging, unigram cuts the text
/// down: every piece in the vocabulary carries a score, and the tokenization is
/// the segmentation whose scores sum highest. That is one pass of Viterbi over
/// the string.
///
/// NOTE: the normalizer is whitespace folding and the sentencepiece space
/// marker, not the precompiled character map the file also carries. That map is
/// a serialized trie of Unicode foldings, and prompts in this field are the
/// text a person typed. Upgrade path if it ever matters: read `precompiled_
/// charsmap` and apply it before the split.
#[derive(Clone, Debug)]
pub struct Unigram {
    pieces: HashMap<String, (u32, f32)>,
    tokens: Vec<String>,
    unk: u32,
    /// What an unspellable character scores, which is below every real piece so
    /// that it is only ever used when nothing else fits.
    unk_score: f32,
    /// The longest piece in bytes, which bounds how far back Viterbi looks.
    longest: usize,
    specials: Vec<(String, u32)>,
    /// The character standing in for a space, `▁` in every published file.
    space: char,
    prefix_space: bool,
}

impl Unigram {
    /// Reads a `tokenizer.json`.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, NetworkError> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    /// The same, from the file's contents.
    pub fn from_json(text: &str) -> Result<Self, NetworkError> {
        let json: serde_json::Value = serde_json::from_str(text)
            .map_err(|error| NetworkError::InvalidDataset(format!("tokenizer.json: {error}")))?;
        let model = json
            .get("model")
            .ok_or_else(|| NetworkError::InvalidDataset("tokenizer.json has no model".into()))?;
        if model.get("type").and_then(|kind| kind.as_str()) != Some("Unigram") {
            return Err(NetworkError::InvalidDataset(
                "this tokenizer.json is not a unigram model".into(),
            ));
        }

        let entries = model
            .get("vocab")
            .and_then(|vocab| vocab.as_array())
            .ok_or_else(|| NetworkError::InvalidDataset("the model has no vocabulary".into()))?;
        let mut pieces = HashMap::with_capacity(entries.len());
        let mut tokens = Vec::with_capacity(entries.len());
        let mut lowest = 0.0f32;
        for (id, entry) in entries.iter().enumerate() {
            // Each entry is a `[piece, score]` pair.
            let piece = entry
                .get(0)
                .and_then(|piece| piece.as_str())
                .ok_or_else(|| {
                    NetworkError::InvalidDataset(format!("vocabulary entry {id} has no piece"))
                })?
                .to_string();
            let score = entry.get(1).and_then(|score| score.as_f64()).unwrap_or(0.0) as f32;
            lowest = lowest.min(score);
            pieces.insert(piece.clone(), (id as u32, score));
            tokens.push(piece);
        }
        let longest = tokens.iter().map(String::len).max().unwrap_or(1);

        let mut specials: Vec<(String, u32)> = json
            .get("added_tokens")
            .and_then(|added| added.as_array())
            .map(|added| {
                added
                    .iter()
                    .filter_map(|token| {
                        Some((
                            token.get("content")?.as_str()?.to_string(),
                            token.get("id")?.as_u64()? as u32,
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default();
        specials.sort_by_key(|(token, _)| std::cmp::Reverse(token.len()));

        let metaspace = pre_tokenizer(&json, "Metaspace");
        Ok(Self {
            unk: model.get("unk_id").and_then(|id| id.as_u64()).unwrap_or(0) as u32,
            unk_score: lowest - 10.0,
            longest,
            specials,
            space: metaspace
                .as_ref()
                .and_then(|metaspace| metaspace.get("replacement"))
                .and_then(|replacement| replacement.as_str())
                .and_then(|replacement| replacement.chars().next())
                .unwrap_or('\u{2581}'),
            // Published files write this as a flag or as a "prepend_scheme",
            // and both mean the same thing.
            prefix_space: metaspace
                .as_ref()
                .map(|metaspace| {
                    metaspace
                        .get("add_prefix_space")
                        .and_then(|flag| flag.as_bool())
                        .unwrap_or_else(|| {
                            metaspace.get("prepend_scheme").and_then(|s| s.as_str())
                                != Some("never")
                        })
                })
                .unwrap_or(true),
            pieces,
            tokens,
        })
    }

    /// How many ids the vocabulary holds.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// The id of a piece written out in full, which is how a caller finds a
    /// special token such as `</s>`.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.pieces.get(token).map(|(id, _)| *id)
    }

    /// Tokenizes text.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, NetworkError> {
        let mut ids = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // A special token is matched literally, whatever the scores say.
            let special = self
                .specials
                .iter()
                .filter_map(|(token, id)| rest.find(token.as_str()).map(|at| (at, token, *id)))
                .min_by_key(|(at, token, _)| (*at, usize::MAX - token.len()));
            match special {
                Some((at, token, id)) => {
                    self.encode_plain(&rest[..at], &mut ids);
                    ids.push(id);
                    rest = &rest[at + token.len()..];
                }
                None => {
                    self.encode_plain(rest, &mut ids);
                    rest = "";
                }
            }
        }
        Ok(ids)
    }

    /// Turns ids back into text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let text: String = ids
            .iter()
            .filter_map(|id| self.tokens.get(*id as usize))
            .map(String::as_str)
            .collect::<String>()
            .replace(self.space, " ");
        match self.prefix_space {
            true => text.strip_prefix(' ').unwrap_or(&text).to_string(),
            false => text,
        }
    }

    /// Encodes text that holds no special token: normalize, then Viterbi.
    fn encode_plain(&self, text: &str, ids: &mut Vec<u32>) {
        let text = self.normalize(text);
        if text.is_empty() {
            return;
        }

        // `best[at]` is the score of the best way to reach byte `at`, and
        // `from[at]` is the piece that got there.
        let mut best = vec![f32::NEG_INFINITY; text.len() + 1];
        let mut from: Vec<(usize, u32)> = vec![(0, self.unk); text.len() + 1];
        best[0] = 0.0;
        for end in 1..=text.len() {
            if !text.is_char_boundary(end) {
                continue;
            }
            let earliest = end.saturating_sub(self.longest);
            for start in earliest..end {
                if best[start] == f32::NEG_INFINITY || !text.is_char_boundary(start) {
                    continue;
                }
                let piece = &text[start..end];
                // Anything not in the vocabulary is only allowed as a single
                // character, standing for the unknown token.
                let (id, score) = match self.pieces.get(piece) {
                    Some((id, score)) => (*id, *score),
                    None if piece.chars().count() == 1 => (self.unk, self.unk_score),
                    None => continue,
                };
                let total = best[start] + score;
                if total > best[end] {
                    best[end] = total;
                    from[end] = (start, id);
                }
            }
        }

        let mut path = Vec::new();
        let mut at = text.len();
        while at > 0 {
            let (start, id) = from[at];
            path.push(id);
            at = start;
        }
        path.reverse();
        ids.extend(path);
    }

    /// Whitespace folding and the space marker, which is the part of the
    /// published normalizer that changes a prompt.
    fn normalize(&self, text: &str) -> String {
        let mut normalized = String::with_capacity(text.len() + 1);
        if self.prefix_space {
            normalized.push(self.space);
        }
        let mut spaced = self.prefix_space;
        for character in text.trim().chars() {
            match character.is_whitespace() {
                // Runs of whitespace fold to one marker, and a run next to the
                // prefix marker disappears into it.
                true if spaced => {}
                true => {
                    normalized.push(self.space);
                    spaced = true;
                }
                false => {
                    normalized.push(character);
                    spaced = false;
                }
            }
        }
        normalized
    }
}

/// The pre-tokenizer of this kind, whether the file holds one or a sequence of
/// them.
fn pre_tokenizer(json: &serde_json::Value, kind: &str) -> Option<serde_json::Value> {
    let is_kind =
        |value: &serde_json::Value| value.get("type").and_then(|name| name.as_str()) == Some(kind);
    let pre = json.get("pre_tokenizer")?;
    if is_kind(pre) {
        return Some(pre.clone());
    }
    pre.get("pretokenizers")?
        .as_array()?
        .iter()
        .find(|value| is_kind(value))
        .cloned()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A vocabulary over the byte alphabet, plus whatever merges a test needs.
    pub(crate) fn tokenizer(merges: &[&str], extra: &[&str], suffix: Option<&str>) -> Bpe {
        Bpe::from_json(&tokenizer_json(merges, extra, suffix)).unwrap()
    }

    /// The same vocabulary, written where a loader will look for it.
    pub(crate) fn write(
        path: &std::path::Path,
        merges: &[&str],
        extra: &[&str],
        suffix: Option<&str>,
    ) {
        std::fs::write(path, tokenizer_json(merges, extra, suffix)).unwrap();
    }

    fn tokenizer_json(merges: &[&str], extra: &[&str], suffix: Option<&str>) -> String {
        let alphabet = byte_alphabet();
        let mut vocab = serde_json::Map::new();
        for (id, character) in alphabet.iter().enumerate() {
            vocab.insert(character.to_string(), serde_json::json!(id));
        }
        let mut next = alphabet.len();
        for token in extra {
            vocab.insert((*token).to_string(), serde_json::json!(next));
            next += 1;
        }
        let mut model = serde_json::json!({
            "vocab": vocab,
            "merges": merges.iter().map(|merge| serde_json::json!(merge)).collect::<Vec<_>>(),
        });
        if let Some(suffix) = suffix {
            model["end_of_word_suffix"] = serde_json::json!(suffix);
        }
        let json = serde_json::json!({
            "model": model,
            "added_tokens": [{"id": next, "content": "<|end|>"}],
        });
        json.to_string()
    }

    #[test]
    fn text_survives_the_round_trip_through_ids() {
        // No merges: every byte is its own token, which still has to decode
        // back to the original text.
        let bpe = tokenizer(&[], &[], None);
        for text in [
            "hello world",
            "a café, 3 times!",
            "  spaced\n\ttext",
            "日本語",
        ] {
            let ids = bpe.encode(text).unwrap();
            assert_eq!(bpe.decode(&ids), text, "{text:?}");
        }
    }

    #[test]
    fn merges_are_applied_in_rank_order() {
        // "lo" merges before "hel", so "hello" becomes he + l + lo.
        let bpe = tokenizer(&["l o", "h e", "he l"], &["lo", "he", "hel"], None);
        let ids = bpe.encode("hello").unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(bpe.decode(&ids), "hello");

        // A word with no merge available stays one token per byte.
        assert_eq!(bpe.encode("xyz").unwrap().len(), 3);
    }

    #[test]
    fn the_pre_tokenizer_splits_the_way_gpt_two_does() {
        assert_eq!(pre_tokenize("Hello world"), vec!["Hello", " world"]);
        assert_eq!(
            pre_tokenize("it's a test, really!"),
            vec!["it", "'s", " a", " test", ",", " really", "!"]
        );
        assert_eq!(pre_tokenize("x  y"), vec!["x", " ", " y"]);
        assert_eq!(pre_tokenize("value = 42"), vec!["value", " =", " 42"]);
        assert_eq!(pre_tokenize("trailing  "), vec!["trailing", "  "]);
    }

    #[test]
    fn a_special_token_is_matched_whole_and_never_merged() {
        let bpe = tokenizer(&["h i"], &["hi"], None);
        let end = bpe.token_id("<|end|>").unwrap();
        let ids = bpe.encode("hi<|end|>hi").unwrap();

        assert_eq!(ids.len(), 3);
        assert_eq!(ids[1], end);
        assert_eq!(bpe.decode(&ids), "hi<|end|>hi");
    }

    #[test]
    fn a_word_suffix_vocabulary_marks_the_end_of_each_word() {
        // What CLIP does: the last symbol of a word carries `</w>`.
        let bpe = tokenizer(&[], &["a</w>", "b</w>"], Some("</w>"));
        let ids = bpe.encode("ab").unwrap();
        assert_eq!(ids.len(), 2);
        // The suffix decodes back to the space it stands for.
        assert_eq!(bpe.decode(&ids), "ab ");
    }

    #[test]
    fn a_word_suffix_vocabulary_drops_the_space_between_words() {
        // CLIP's own rule throws whitespace away and cuts digits apart. A
        // space that survives as its own token reaches the text encoder as a
        // word, and the prompt is then not the prompt that was asked for.
        let bpe = tokenizer(&[], &["a</w>", "b</w>", "1</w>", "2</w>"], Some("</w>"));
        let space = bpe
            .token_id("\u{0120}")
            .expect("the byte alphabet holds a space");

        let ids = bpe.encode("a b 12").unwrap();
        assert!(!ids.contains(&space), "{ids:?} carries a space as a token");
        // One token per word, and one per digit.
        assert_eq!(ids.len(), 4);
    }

    /// A LLaMA 2 style file: `▁` for spaces, `<0xNN>` byte fallback, and
    /// either the `Metaspace` pre-tokenizer or the older `Prepend` normalizer.
    fn sentencepiece_bpe(legacy: bool) -> Bpe {
        let tokens = [
            "<unk>", "<s>", "</s>", "<0xF0>", "<0x9F>", "<0xA6>", "<0x80>", "▁", "h", "e", "l",
            "o", "w", "r", "d", "▁h", "ll", "▁he", "llo", "▁hello", "▁w", "or", "▁wor", "▁worl",
            "▁world", "é",
        ];
        let vocab: serde_json::Map<String, serde_json::Value> = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.to_string(), id.into()))
            .collect();
        let merges = [
            "▁ h",
            "l l",
            "▁h e",
            "ll o",
            "▁he llo",
            "▁ w",
            "o r",
            "▁w or",
            "▁wor l",
            "▁worl d",
        ];
        let added: Vec<_> = (0..3)
            .map(|id| serde_json::json!({"id": id, "content": tokens[id], "special": true}))
            .collect();
        let (pre, normalizer) = match legacy {
            false => (
                serde_json::json!({"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": false}),
                serde_json::Value::Null,
            ),
            true => (
                serde_json::Value::Null,
                serde_json::json!({"type": "Sequence", "normalizers": [
                    {"type": "Prepend", "prepend": "▁"},
                    {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
                ]}),
            ),
        };
        let json = serde_json::json!({
            "added_tokens": added,
            "normalizer": normalizer,
            "pre_tokenizer": pre,
            "model": {"type": "BPE", "byte_fallback": true, "vocab": vocab, "merges": merges},
        });
        Bpe::from_json(&json.to_string()).unwrap()
    }

    #[test]
    fn sentencepiece_bpe_matches_the_reference_tokenizer() {
        // Ids and decoded text from the Hugging Face `tokenizers` library on
        // the same two files. Checked as well against a real 32k LLaMA 2
        // vocabulary: 310 lines of this README, every id equal.
        let cases: [(bool, &str, &[u32], &str); 8] = [
            (false, "hello world", &[19, 24], "hello world"),
            (false, "é 🦀", &[7, 25, 7, 3, 4, 5, 6], "é 🦀"),
            (
                false,
                "<s>hello</s>world",
                &[1, 8, 9, 18, 2, 12, 21, 10, 14],
                "<s>hello</s>world",
            ),
            (false, " hello", &[19], "hello"),
            (true, "hello world", &[19, 24], "hello world"),
            (true, "é 🦀", &[7, 25, 7, 3, 4, 5, 6], "é 🦀"),
            (
                true,
                "<s>hello</s>world",
                &[1, 19, 2, 24],
                "<s> hello</s> world",
            ),
            (true, " hello", &[7, 19], " hello"),
        ];
        for (legacy, text, ids, decoded) in cases {
            let bpe = sentencepiece_bpe(legacy);
            assert_eq!(bpe.encode(text).unwrap(), ids, "{text:?}, legacy {legacy}");
            assert_eq!(bpe.decode(ids), decoded, "{text:?}, legacy {legacy}");
        }
    }

    #[test]
    fn a_file_that_is_not_a_tokenizer_is_refused() {
        assert!(Bpe::from_json("{}").is_err());
        assert!(Bpe::from_json("{\"model\": {}}").is_err());
        assert!(Bpe::from_json("not json").is_err());
    }
    /// A unigram vocabulary over a handful of pieces, scored so that the long
    /// ones win where they fit.
    pub(crate) fn unigram_json() -> String {
        let vocab: Vec<serde_json::Value> = [
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
            ("\u{2581}", -3.0),
            ("\u{2581}a", -1.5),
            ("\u{2581}ab", -0.6),
            ("\u{2581}abc", -0.4),
            ("b", -2.0),
            ("c", -2.0),
            ("\u{2581}c", -2.5),
        ]
        .iter()
        .map(|(piece, score)| serde_json::json!([piece, score]))
        .collect();
        serde_json::json!({
            "model": {"type": "Unigram", "unk_id": 2, "vocab": vocab},
            "pre_tokenizer": {"type": "Metaspace", "replacement": "\u{2581}",
                              "add_prefix_space": true},
            "added_tokens": [{"id": 1, "content": "</s>"}],
        })
        .to_string()
    }

    #[test]
    fn unigram_cuts_the_text_the_highest_scoring_way() {
        let tokenizer = Unigram::from_json(&unigram_json()).unwrap();

        // "▁abc" alone scores -0.4, which beats "▁ab" + "c" at -2.6.
        assert_eq!(tokenizer.encode("abc").unwrap(), vec![6]);
        // No piece spells "▁ac", so the best cut is "▁a" then "c".
        assert_eq!(tokenizer.encode("ac").unwrap(), vec![4, 8]);
        assert_eq!(tokenizer.decode(&[6]), "abc");
        assert_eq!(tokenizer.decode(&tokenizer.encode("a b").unwrap()), "a b");
    }

    #[test]
    fn unigram_folds_whitespace_and_keeps_special_tokens_whole() {
        let tokenizer = Unigram::from_json(&unigram_json()).unwrap();

        assert_eq!(
            tokenizer.encode("  ab \n c ").unwrap(),
            tokenizer.encode("ab c").unwrap()
        );
        let ids = tokenizer.encode("ab</s>").unwrap();
        assert_eq!(ids.last(), Some(&1));
        // A character no piece spells becomes the unknown token rather than
        // failing the prompt.
        assert_eq!(tokenizer.encode("\u{1F600}").unwrap(), vec![3, 2]);
    }

    #[test]
    fn unigram_refuses_a_file_that_is_not_one() {
        assert!(Unigram::from_json(&tokenizer_json(&[], &[], None)).is_err());
    }
}
