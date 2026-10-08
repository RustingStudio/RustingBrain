//! Reading and writing the weight format every published model ships in.
//!
//! A `.safetensors` file is an 8-byte little-endian header length, a JSON
//! header naming each tensor with its dtype, shape and byte range, and then the
//! tensor bytes back to back. Nothing in it is compressed and nothing in it is
//! executable, which is the whole point of the format.
//!
//! Tensors are read one at a time from an open file rather than by loading the
//! file into memory: a 4-billion-parameter model is eight gigabytes on disk and
//! is meant to be moved onto a device layer by layer.
//!
//! Every dtype a published checkpoint uses arrives as `f32` here, including the
//! two eight-bit float formats, so the rest of the crate never sees a dtype it
//! has no arithmetic for. What to do with the values afterwards - keep them,
//! round them back to `i8` with [`Quantized`](crate::quantized::Quantized), or
//! move them to a device - is the caller's decision.

use crate::matrix::Matrix;
use crate::network::NetworkError;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// The element types a `.safetensors` header can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F64,
    F32,
    F16,
    Bf16,
    /// `E4M3`, the eight-bit float weights are usually quantized to.
    F8E4M3,
    /// `E5M2`, the eight-bit float with the wider range and less precision.
    F8E5M2,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl Dtype {
    fn parse(name: &str) -> Result<Self, NetworkError> {
        Ok(match name {
            "F64" => Self::F64,
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E5M2" => Self::F8E5M2,
            "I64" => Self::I64,
            "I32" => Self::I32,
            "I16" => Self::I16,
            "I8" => Self::I8,
            "U8" => Self::U8,
            "BOOL" => Self::Bool,
            other => {
                return Err(NetworkError::InvalidDataset(format!(
                    "unknown safetensors dtype {other}"
                )));
            }
        })
    }

    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            Self::F64 | Self::I64 => 8,
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::Bf16 | Self::I16 => 2,
            Self::F8E4M3 | Self::F8E5M2 | Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    /// Reads one element from the front of `bytes` as `f32`.
    fn read(self, bytes: &[u8]) -> f32 {
        match self {
            Self::F64 => f64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as f32,
            Self::F32 => f32::from_le_bytes(bytes[..4].try_into().expect("4 bytes")),
            Self::F16 => f16_to_f32(u16::from_le_bytes(bytes[..2].try_into().expect("2 bytes"))),
            Self::Bf16 => f32::from_bits(
                (u16::from_le_bytes(bytes[..2].try_into().expect("2 bytes")) as u32) << 16,
            ),
            Self::F8E4M3 => f8_to_f32(bytes[0], 4, 3),
            Self::F8E5M2 => f8_to_f32(bytes[0], 5, 2),
            Self::I64 => i64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as f32,
            Self::I32 => i32::from_le_bytes(bytes[..4].try_into().expect("4 bytes")) as f32,
            Self::I16 => i16::from_le_bytes(bytes[..2].try_into().expect("2 bytes")) as f32,
            Self::I8 => bytes[0] as i8 as f32,
            Self::U8 | Self::Bool => bytes[0] as f32,
        }
    }
}

/// An IEEE half to a single, including the subnormals a naive shift loses.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits as u32 & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits as u32 & 0x3ff;

    let rest = match exponent {
        // Zero or subnormal: no implicit leading one, so normalize by hand.
        0 => {
            if mantissa == 0 {
                0
            } else {
                let shift = mantissa.leading_zeros() - 21;
                let exponent = 127 - 15 - shift;
                (exponent << 23) | ((mantissa << (shift + 1)) & 0x7f_ffff)
            }
        }
        // Infinity or NaN.
        0x1f => 0x7f80_0000 | (mantissa << 13),
        _ => ((exponent as u32 + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(sign | rest)
}

/// One of the eight-bit float formats, given its exponent and mantissa widths.
///
/// `E4M3` here is the `e4m3fn` variant the machine-learning world uses: no
/// infinities, and the all-ones exponent with a full mantissa is the only NaN.
fn f8_to_f32(bits: u8, exponent_bits: u32, mantissa_bits: u32) -> f32 {
    let sign = ((bits as u32) & 0x80) << 24;
    let exponent = ((bits as u32) >> mantissa_bits) & ((1 << exponent_bits) - 1);
    let mantissa = (bits as u32) & ((1 << mantissa_bits) - 1);
    let bias = (1 << (exponent_bits - 1)) - 1;
    let all_ones = (1 << exponent_bits) - 1;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: value is mantissa * 2^(1 - bias) / 2^mantissa_bits.
        let scale = (1.0f32 / (1 << mantissa_bits) as f32) * 2.0f32.powi(1 - bias as i32);
        let magnitude = mantissa as f32 * scale;
        return if sign == 0 { magnitude } else { -magnitude };
    }
    if exponent == all_ones && (exponent_bits == 5 || mantissa == (1 << mantissa_bits) - 1) {
        // E5M2 keeps IEEE infinities and NaNs; E4M3 has only the one NaN.
        return if mantissa == 0 && exponent_bits == 5 {
            f32::from_bits(sign | 0x7f80_0000)
        } else {
            f32::NAN
        };
    }

    let shifted_exponent = exponent + 127 - bias;
    f32::from_bits(sign | (shifted_exponent << 23) | (mantissa << (23 - mantissa_bits)))
}

/// Where one tensor lives in the file, and what shape it has.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Byte range within the data segment, which starts after the header.
    pub start: u64,
    pub end: u64,
}

impl TensorInfo {
    /// Elements, which is the product of the shape.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An open `.safetensors` file, read tensor by tensor.
///
/// ```no_run
/// # use rusting_brain::SafeTensors;
/// let mut weights = SafeTensors::open("model.safetensors")?;
/// println!("{} tensors", weights.names().count());
/// let embedding = weights.matrix("model.embed_tokens.weight")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct SafeTensors {
    file: BufReader<File>,
    path: PathBuf,
    tensors: BTreeMap<String, TensorInfo>,
    /// Where the tensor bytes begin: eight bytes of length plus the header.
    data_offset: u64,
    /// The `__metadata__` entry, which carries a model's own notes about
    /// itself and is what tells a loader which architecture it is holding.
    metadata: BTreeMap<String, String>,
}

impl SafeTensors {
    /// Reads the header and leaves the tensor bytes on disk.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, NetworkError> {
        let path = path.as_ref().to_path_buf();
        let mut file = BufReader::new(File::open(&path)?);

        let mut length = [0u8; 8];
        file.read_exact(&mut length)?;
        let length = u64::from_le_bytes(length);
        // A header is JSON, so a length past the file or past any plausible
        // header means this is not the format it claims to be.
        let size = file.get_ref().metadata()?.len();
        if length == 0 || length > size.saturating_sub(8) {
            return Err(NetworkError::InvalidDataset(format!(
                "{}: a {length}-byte safetensors header does not fit in a {size}-byte file",
                path.display()
            )));
        }

        let mut header = vec![0u8; length as usize];
        file.read_exact(&mut header)?;
        let header: serde_json::Value = serde_json::from_slice(&header).map_err(|error| {
            NetworkError::InvalidDataset(format!("{}: {error}", path.display()))
        })?;
        let header = header.as_object().ok_or_else(|| {
            NetworkError::InvalidDataset(format!("{}: the header is not an object", path.display()))
        })?;

        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (name, value) in header {
            if name == "__metadata__" {
                for (key, value) in value.as_object().into_iter().flatten() {
                    metadata.insert(key.clone(), value.as_str().unwrap_or_default().to_string());
                }
                continue;
            }
            tensors.insert(name.clone(), Self::parse_tensor(&path, name, value)?);
        }

        // The header can promise bytes the file does not have, which is what a
        // run killed mid-write leaves behind. Catching it here turns a
        // resumable preprocessing run's bad shard into a named error instead of
        // an `UnexpectedEof` thousands of reads later.
        let promised = tensors
            .values()
            .map(|tensor| tensor.end)
            .max()
            .unwrap_or_default();
        if 8 + length + promised > size {
            return Err(NetworkError::InvalidDataset(format!(
                "{}: the header promises {} bytes of tensor data and the file holds {}",
                path.display(),
                promised,
                size.saturating_sub(8 + length)
            )));
        }

        Ok(Self {
            file,
            path,
            tensors,
            data_offset: 8 + length,
            metadata,
        })
    }

    fn parse_tensor(
        path: &Path,
        name: &str,
        value: &serde_json::Value,
    ) -> Result<TensorInfo, NetworkError> {
        let bad = |what: &str| {
            NetworkError::InvalidDataset(format!("{}: {name} has {what}", path.display()))
        };

        let dtype = Dtype::parse(value["dtype"].as_str().ok_or_else(|| bad("no dtype"))?)?;
        let shape: Vec<usize> = value["shape"]
            .as_array()
            .ok_or_else(|| bad("no shape"))?
            .iter()
            .map(|dimension| dimension.as_u64().map(|size| size as usize))
            .collect::<Option<_>>()
            .ok_or_else(|| bad("a shape that is not a list of sizes"))?;
        let offsets = value["data_offsets"]
            .as_array()
            .ok_or_else(|| bad("no data_offsets"))?;
        let (start, end) = match offsets.as_slice() {
            [start, end] => (
                start.as_u64().ok_or_else(|| bad("a bad start offset"))?,
                end.as_u64().ok_or_else(|| bad("a bad end offset"))?,
            ),
            _ => return Err(bad("data_offsets that are not a pair")),
        };

        let info = TensorInfo {
            dtype,
            shape,
            start,
            end,
        };
        if end < start || end - start != (info.len() * dtype.size()) as u64 {
            return Err(bad("a byte range that does not match its shape and dtype"));
        }
        Ok(info)
    }

    /// Every tensor name in the file, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// The shape and dtype of one tensor, without reading it.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// The file's `__metadata__`, which is where a model records its format
    /// and sometimes its architecture.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Reads one tensor as `f32`, whatever it is stored as, with its shape.
    pub fn tensor(&mut self, name: &str) -> Result<(Vec<f32>, Vec<usize>), NetworkError> {
        let info = self.tensors.get(name).cloned().ok_or_else(|| {
            NetworkError::InvalidDataset(format!("{}: no tensor named {name}", self.path.display()))
        })?;

        let mut bytes = vec![0u8; (info.end - info.start) as usize];
        self.file
            .seek(SeekFrom::Start(self.data_offset + info.start))?;
        self.file.read_exact(&mut bytes)?;

        let size = info.dtype.size();
        let count = bytes.len() / size;
        // Widening a checkpoint to f32 is the whole cost of loading one: a
        // Stable Diffusion XL model is billions of elements, and one core
        // converting them takes longer than the disk takes to hand them over.
        // Small tensors stay on this thread, where a spawn would cost more
        // than the work.
        let lanes = match count >= 1 << 20 {
            true => std::thread::available_parallelism().map_or(1, |count| count.get()),
            false => 1,
        };
        if lanes == 1 {
            let values = bytes
                .chunks_exact(size)
                .map(|element| info.dtype.read(element))
                .collect();
            return Ok((values, info.shape));
        }
        let mut values = vec![0f32; count];
        let stride = count.div_ceil(lanes);
        let dtype = info.dtype;
        std::thread::scope(|scope| {
            for (slot, source) in values.chunks_mut(stride).zip(bytes.chunks(stride * size)) {
                scope.spawn(move || {
                    for (target, element) in slot.iter_mut().zip(source.chunks_exact(size)) {
                        *target = dtype.read(element);
                    }
                });
            }
        });
        Ok((values, info.shape))
    }

    /// Reads one tensor as a [`Matrix`].
    ///
    /// A one-dimensional tensor - a norm gain, a bias - comes back as a single
    /// row, and anything with more than two dimensions is an error here:
    /// convolution weights want their own reshaping, and silently flattening
    /// them would hide a mistake rather than report it.
    pub fn matrix(&mut self, name: &str) -> Result<Matrix, NetworkError> {
        let (values, shape) = self.tensor(name)?;
        match shape.as_slice() {
            [cols] => Ok(Matrix::from_vec(1, *cols, values)),
            [rows, cols] => Ok(Matrix::from_vec(*rows, *cols, values)),
            _ => Err(NetworkError::InvalidDataset(format!(
                "{}: {name} has shape {shape:?}, which is not a matrix",
                self.path.display()
            ))),
        }
    }
}

/// A model split over several `.safetensors` files, as anything past a couple
/// of billion parameters is.
///
/// The shards are named by `model.safetensors.index.json`, a
/// `{"weight_map": {tensor: file}}` next to them. Opening the index opens
/// nothing else: each shard is opened the first time a tensor is read from it,
/// and stays open afterwards.
#[derive(Debug)]
pub struct ShardedSafeTensors {
    directory: PathBuf,
    /// Tensor name to shard file name.
    map: BTreeMap<String, String>,
    shards: BTreeMap<String, SafeTensors>,
    /// What [`ShardedSafeTensors::add_lora`] adds to a weight as it is read.
    deltas: BTreeMap<String, Vec<f32>>,
}

impl ShardedSafeTensors {
    /// Opens the index at `path`, or a single-file model if `path` is a
    /// `.safetensors` file, so a caller does not have to know which it has.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, NetworkError> {
        let path = path.as_ref();
        let directory = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();

        if path
            .extension()
            .is_some_and(|extension| extension == "safetensors")
        {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_string();
            let shard = SafeTensors::open(path)?;
            let map = shard
                .names()
                .map(|tensor| (tensor.to_string(), name.clone()))
                .collect();
            return Ok(Self {
                directory,
                map,
                shards: BTreeMap::from([(name, shard)]),
                deltas: BTreeMap::new(),
            });
        }

        let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)
            .map_err(|error| {
                NetworkError::InvalidDataset(format!("{}: {error}", path.display()))
            })?;
        let weight_map = index["weight_map"].as_object().ok_or_else(|| {
            NetworkError::InvalidDataset(format!(
                "{}: an index needs a weight_map of tensor names to files",
                path.display()
            ))
        })?;

        let map = weight_map
            .iter()
            .map(|(tensor, file)| {
                file.as_str()
                    .map(|file| (tensor.clone(), file.to_string()))
                    .ok_or_else(|| {
                        NetworkError::InvalidDataset(format!(
                            "{}: {tensor} does not name a file",
                            path.display()
                        ))
                    })
            })
            .collect::<Result<_, NetworkError>>()?;

        Ok(Self {
            directory,
            map,
            shards: BTreeMap::new(),
            deltas: BTreeMap::new(),
        })
    }

    /// Every tensor name across every shard, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(String::as_str)
    }

    /// Merges a LoRA file into the weights it adapts, as they are read from
    /// now on, and says how many weights it changes.
    ///
    /// Both published spellings are read: PEFT and diffusers' `lora_A` and
    /// `lora_B`, and kohya's `lora_down`, `lora_up` and `alpha`, which name the
    /// weight with underscores for dots. Each adapted weight gains
    /// `scale * alpha / rank * up @ down`, with `alpha` the rank where the file
    /// has none.
    ///
    /// One file often adapts several parts of a model, each under its own
    /// prefix (`unet.`, `text_encoder.`, `lora_unet_`, `lora_te_`, ...). An
    /// adapter under a prefix not in `prefixes` belongs to another part and
    /// is skipped; `""` in `prefixes` takes names with none. An adapter for
    /// this part that names no weight here, or a kind of adapter other than
    /// LoRA, is an error, because merging only some of it is a different model.
    pub fn add_lora(
        &mut self,
        path: impl AsRef<Path>,
        prefixes: &[&str],
        scale: f32,
    ) -> Result<usize, NetworkError> {
        const PARTS: [&str; 12] = [
            "text_encoder_2",
            "text_encoder_3",
            "text_encoder",
            "lora_transformer",
            "transformer",
            "lora_unet",
            "unet",
            "lora_te1",
            "lora_te2",
            "lora_te3",
            "lora_te",
            "base_model_model",
        ];
        let flat = |name: &str| name.replace('.', "_");
        let weights: BTreeMap<String, String> = self
            .map
            .keys()
            .filter_map(|name| name.strip_suffix(".weight"))
            .map(|stem| (flat(stem), stem.to_string()))
            .collect();
        let mut file = SafeTensors::open(path)?;
        let names: Vec<String> = file.names().map(str::to_string).collect();
        let mut changed = 0;
        for name in &names {
            let (stem, up_name) = match name.strip_suffix(".lora_A.weight") {
                Some(stem) => (stem, format!("{stem}.lora_B.weight")),
                None => match name.strip_suffix(".lora_down.weight") {
                    Some(stem) => (stem, format!("{stem}.lora_up.weight")),
                    None if name.ends_with(".lora_B.weight")
                        || name.ends_with(".lora_up.weight")
                        || name.ends_with(".alpha") =>
                    {
                        continue;
                    }
                    None => {
                        return Err(NetworkError::InvalidDataset(format!(
                            "{name} is not part of a LoRA pair; other adapter kinds are not read"
                        )));
                    }
                },
            };
            // Peel off prefixes until what is left names a weight here, or a
            // prefix says it belongs elsewhere.
            let mut rest = flat(stem);
            let mut part = "";
            let target = loop {
                if let Some(target) = weights.get(&rest)
                    && prefixes.contains(&part)
                {
                    break Some(target);
                }
                let Some(prefix) = PARTS
                    .iter()
                    .find(|prefix| rest.starts_with(&format!("{prefix}_")))
                else {
                    break None;
                };
                rest = rest[prefix.len() + 1..].to_string();
                if *prefix != "base_model_model" {
                    part = prefix;
                }
            };
            let Some(target) = target else {
                if prefixes.contains(&part) {
                    return Err(NetworkError::InvalidDataset(format!(
                        "the LoRA adapts {stem}, which this checkpoint does not hold"
                    )));
                }
                continue;
            };

            let (down, down_shape) = file.tensor(name)?;
            let (up, up_shape) = file.tensor(&up_name)?;
            let rank = down_shape[0];
            let alpha = match file.tensor(&format!("{stem}.alpha")) {
                Ok((alpha, _)) => alpha[0],
                Err(_) => rank as f32,
            };
            let inputs = down.len() / rank.max(1);
            let outputs = up_shape[0];
            let base = self.map.get(&format!("{target}.weight")).cloned();
            let base_len = base
                .and_then(|file| self.size_of(&file, &format!("{target}.weight")))
                .unwrap_or(0);
            if up.len() != outputs * rank || outputs * inputs != base_len {
                return Err(NetworkError::InvalidDataset(format!(
                    "the LoRA for {target} is {up_shape:?} by {down_shape:?}, which does not \
                     make its {base_len} weights"
                )));
            }
            let up = Matrix::from_vec(outputs, rank, up);
            let down = Matrix::from_vec(rank, inputs, down);
            let mut product = Matrix::new(outputs, inputs);
            up.dot(&down, &mut product);
            let factor = scale * alpha / rank as f32;
            let delta = self
                .deltas
                .entry(format!("{target}.weight"))
                .or_insert_with(|| vec![0.0; base_len]);
            for (delta, value) in delta.iter_mut().zip(&product.data) {
                *delta += factor * value;
            }
            changed += 1;
        }
        Ok(changed)
    }

    /// How many values a tensor holds, from its shard's header.
    fn size_of(&mut self, file: &str, name: &str) -> Option<usize> {
        if !self.shards.contains_key(file) {
            let shard = SafeTensors::open(self.directory.join(file)).ok()?;
            self.shards.insert(file.to_string(), shard);
        }
        let info = self.shards.get(file)?.info(name)?;
        Some(info.shape.iter().product())
    }

    /// Reads one tensor as `f32`, opening its shard if this is the first
    /// tensor wanted from it.
    pub fn tensor(&mut self, name: &str) -> Result<(Vec<f32>, Vec<usize>), NetworkError> {
        let file = self
            .map
            .get(name)
            .ok_or_else(|| {
                NetworkError::InvalidDataset(format!("no tensor named {name} in the index"))
            })?
            .clone();
        if !self.shards.contains_key(&file) {
            let shard = SafeTensors::open(self.directory.join(&file))?;
            self.shards.insert(file.clone(), shard);
        }
        let (mut values, shape) = self
            .shards
            .get_mut(&file)
            .expect("the shard was just opened")
            .tensor(name)?;
        if let Some(delta) = self.deltas.get(name) {
            values
                .iter_mut()
                .zip(delta)
                .for_each(|(value, delta)| *value += delta);
        }
        Ok((values, shape))
    }

    /// Reads one tensor as a [`Matrix`].
    pub fn matrix(&mut self, name: &str) -> Result<Matrix, NetworkError> {
        let (values, shape) = self.tensor(name)?;
        match shape.as_slice() {
            [cols] => Ok(Matrix::from_vec(1, *cols, values)),
            [rows, cols] => Ok(Matrix::from_vec(*rows, *cols, values)),
            _ => Err(NetworkError::InvalidDataset(format!(
                "{name} has shape {shape:?}, which is not a matrix"
            ))),
        }
    }
}

/// Writes tensors to a `.safetensors` file.
///
/// The counterpart of [`SafeTensors::open`], and the reason the crate's own
/// checkpoints are readable from Python: this is the published format, not a
/// private one. Everything is written as `F32`.
///
/// `tensors` is `(name, shape, values)`, and a name may appear only once. The
/// values are streamed straight to the file, so a checkpoint costs the size of
/// its header in memory rather than the size of its weights.
///
/// `metadata` lands in the header's `__metadata__` entry, where a loader can
/// read it back without touching the tensor bytes. The format stores it as
/// strings; a caller with more to say puts JSON in one.
///
/// ```no_run
/// # use rusting_brain::safetensors;
/// # use std::collections::BTreeMap;
/// let weight = vec![1.0f32, 2.0, 3.0, 4.0];
/// safetensors::write(
///     "tiny.safetensors",
///     &[("layer.weight", &[2, 2][..], &weight[..])],
///     &BTreeMap::from([("format".into(), "rusting-brain".into())]),
/// )?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn write<P: AsRef<Path>>(
    path: P,
    tensors: &[(&str, &[usize], &[f32])],
    metadata: &BTreeMap<String, String>,
) -> Result<(), NetworkError> {
    write_as(path, tensors, metadata, Dtype::F32)
}

/// [`write()`], at a chosen width.
///
/// `F32` and `Bf16` are the two that are written. Bfloat16 halves a file for
/// three decimal digits of precision, which is the trade a cache of frozen
/// activations wants and a checkpoint of trainable weights does not: it is the
/// same width the mixed-precision training path already computes in.
///
/// Anything else is refused rather than silently widened, because a caller that
/// asked for `F8E4M3` and got `F32` files would find out from its disk.
pub fn write_as<P: AsRef<Path>>(
    path: P,
    tensors: &[(&str, &[usize], &[f32])],
    metadata: &BTreeMap<String, String>,
    dtype: Dtype,
) -> Result<(), NetworkError> {
    let path = path.as_ref();
    let width = match dtype {
        Dtype::F32 => 4,
        Dtype::Bf16 => 2,
        other => {
            return Err(NetworkError::InvalidSnapshot(format!(
                "{other:?} is read but not written"
            )));
        }
    };
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        header.insert(
            "__metadata__".into(),
            serde_json::Value::Object(
                metadata
                    .iter()
                    .map(|(key, value)| (key.clone(), serde_json::Value::from(value.clone())))
                    .collect(),
            ),
        );
    }

    let mut offset = 0u64;
    for (name, shape, values) in tensors {
        let expected: usize = shape.iter().product();
        if expected != values.len() {
            return Err(NetworkError::InvalidSnapshot(format!(
                "{name}: a {shape:?} tensor holds {expected} values, not {}",
                values.len()
            )));
        }
        let end = offset + (values.len() * width) as u64;
        let entry = serde_json::json!({
            "dtype": format!("{dtype:?}").to_uppercase(),
            "shape": shape,
            "data_offsets": [offset, end],
        });
        if header.insert((*name).to_string(), entry).is_some() {
            return Err(NetworkError::InvalidSnapshot(format!(
                "{name} appears twice in one checkpoint"
            )));
        }
        offset = end;
    }

    let mut header = serde_json::to_vec(&serde_json::Value::Object(header))?;
    // The format asks for the tensor bytes to start eight-byte aligned, and
    // the readers that enforce it accept trailing whitespace inside the JSON.
    header.resize(header.len().next_multiple_of(8), b' ');

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&(header.len() as u64).to_le_bytes())?;
    writer.write_all(&header)?;
    for (_, _, values) in tensors {
        // A value at a time rather than a `Vec<u8>` of the whole tensor, so a
        // 400 MB parameter does not need 400 MB of scratch to be written.
        for value in *values {
            match dtype {
                Dtype::Bf16 => writer.write_all(&to_bf16(*value).to_le_bytes())?,
                _ => writer.write_all(&value.to_le_bytes())?,
            }
        }
    }
    writer.flush()?;
    Ok(())
}

/// The top sixteen bits of an `f32`, rounded to nearest with ties to even.
///
/// Truncating instead would bias every value towards zero, which over a whole
/// tensor is a systematic shift rather than noise.
fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        // Keep it a NaN rather than letting the rounding carry it to infinity.
        return ((bits >> 16) as u16) | 0x0040;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// Writes a float checkpoint, for tests elsewhere in the crate that need a file
/// to read a model out of.
#[cfg(test)]
pub(crate) fn write_checkpoint(
    path: &std::path::Path,
    tensors: &BTreeMap<String, (Vec<usize>, Vec<f32>)>,
) {
    let flat: Vec<(&str, &[usize], &[f32])> = tensors
        .iter()
        .map(|(name, (shape, values))| (name.as_str(), &shape[..], &values[..]))
        .collect();
    write(path, &flat, &BTreeMap::new()).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn a_lora_adds_up_times_down_to_the_weight_it_names() {
        let dir = std::env::temp_dir().join("rusting_brain_lora_merge");
        std::fs::create_dir_all(&dir).unwrap();
        let file = |name: &str, tensors: &[(&str, Vec<usize>, Vec<f32>)]| {
            let path = dir.join(name);
            let tensors = tensors
                .iter()
                .map(|(name, shape, values)| (name.to_string(), (shape.clone(), values.clone())))
                .collect();
            write_checkpoint(&path, &tensors);
            path
        };
        let base = file(
            "base.safetensors",
            &[
                ("mid.to_q.weight", vec![2, 3], vec![1.0; 6]),
                ("mid.to_q.bias", vec![2], vec![0.0; 2]),
            ],
        );
        // Rank one: up [2, 1] times down [1, 3], at alpha 2 over rank 1.
        let pair = |stem: &str, down: &str, up: &str| {
            vec![
                (
                    format!("{stem}.{down}.weight"),
                    vec![1, 3],
                    vec![1.0, 2.0, 3.0],
                ),
                (format!("{stem}.{up}.weight"), vec![2, 1], vec![1.0, -1.0]),
                (format!("{stem}.alpha"), vec![1], vec![2.0]),
            ]
        };
        let lora = |name: &str, tensors: Vec<(String, Vec<usize>, Vec<f32>)>| {
            let tensors: Vec<_> = tensors
                .iter()
                .map(|(name, shape, values)| (name.as_str(), shape.clone(), values.clone()))
                .collect();
            file(name, &tensors)
        };
        let kohya = lora(
            "kohya.safetensors",
            [
                pair("lora_unet_mid_to_q", "lora_down", "lora_up"),
                pair("lora_te_text_model_fc1", "lora_down", "lora_up"),
            ]
            .concat(),
        );
        let peft = lora(
            "peft.safetensors",
            pair("unet.mid.to_q", "lora_A", "lora_B"),
        );
        let stray = lora(
            "stray.safetensors",
            pair("unet.mid.to_v", "lora_A", "lora_B"),
        );
        let other = lora(
            "other.safetensors",
            vec![("unet.mid.to_q.hada_w1_a".into(), vec![1], vec![0.0])],
        );

        for path in [&kohya, &peft] {
            let mut weights = ShardedSafeTensors::open(&base).unwrap();
            let prefixes = ["unet", "lora_unet", ""];
            assert_eq!(weights.add_lora(path, &prefixes, 0.5).unwrap(), 1);
            // 1 + 0.5 * 2 * up[row] * down[col].
            let (values, shape) = weights.tensor("mid.to_q.weight").unwrap();
            assert_eq!(shape, vec![2, 3]);
            assert_eq!(values, vec![2.0, 3.0, 4.0, 0.0, -1.0, -2.0]);
            assert_eq!(weights.matrix("mid.to_q.weight").unwrap().data, values);
            assert_eq!(weights.tensor("mid.to_q.bias").unwrap().0, vec![0.0; 2]);
        }
        let mut weights = ShardedSafeTensors::open(&base).unwrap();
        assert!(weights.add_lora(&stray, &["unet"], 1.0).is_err());
        assert!(weights.add_lora(&other, &["unet"], 1.0).is_err());
        assert!(
            weights
                .add_lora(&kohya, &["lora_te", "lora_unet"], 1.0)
                .is_err()
        );
        assert_eq!(weights.add_lora(&stray, &["transformer"], 1.0).unwrap(), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Writes a file in the format, which is also the shortest description of
    /// it: a length, a JSON header, and the bytes.
    fn write(path: &Path, header: &str, data: &[u8]) {
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.write_all(data).unwrap();
    }

    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("rb_{name}_{}", std::process::id()))
    }

    #[test]
    fn a_float_tensor_reads_back_with_its_shape() {
        let path = scratch("st_f32.safetensors");
        let values: Vec<f32> = vec![1.0, -2.0, 0.5, 4.25, -0.125, 8.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        write(
            &path,
            r#"{"__metadata__":{"format":"pt"},"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#,
            &data,
        );

        let mut file = SafeTensors::open(&path).unwrap();
        assert_eq!(file.names().collect::<Vec<_>>(), ["weight"]);
        assert_eq!(file.metadata()["format"], "pt");
        assert_eq!(file.info("weight").unwrap().shape, vec![2, 3]);

        let matrix = file.matrix("weight").unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!((matrix.rows, matrix.cols), (2, 3));
        assert_eq!(matrix.data, values);
    }

    #[test]
    fn half_precision_dtypes_arrive_as_f32() {
        let path = scratch("st_half.safetensors");
        // 1.0, -2.0 and the smallest normal half, then the same in bfloat16.
        let halves: [u16; 3] = [0x3c00, 0xc000, 0x0400];
        let bf16: [u16; 3] = [0x3f80, 0xc000, 0x3f00];
        let data: Vec<u8> = halves
            .iter()
            .chain(&bf16)
            .flat_map(|bits| bits.to_le_bytes())
            .collect();
        write(
            &path,
            r#"{"half":{"dtype":"F16","shape":[3],"data_offsets":[0,6]},"brain":{"dtype":"BF16","shape":[3],"data_offsets":[6,12]}}"#,
            &data,
        );

        let mut file = SafeTensors::open(&path).unwrap();
        let (half, _) = file.tensor("half").unwrap();
        let (brain, _) = file.tensor("brain").unwrap();
        std::fs::remove_file(&path).ok();

        // The smallest normal half, which is 2^-14 rather than f32::MIN_POSITIVE.
        assert_eq!(half, vec![1.0, -2.0, 2.0f32.powi(-14)]);
        assert_eq!(brain, vec![1.0, -2.0, 0.5]);
    }

    #[test]
    fn eight_bit_floats_and_integers_arrive_as_f32() {
        let path = scratch("st_f8.safetensors");
        // E4M3 has a bias of 7: 0x38 is 1.0, 0x40 is 2.0, 0xb8 is -1.0.
        // E5M2 has a bias of 15: 0x3c is 1.0, 0x40 is 2.0, 0xbc is -1.0.
        let data = vec![0x38u8, 0x40, 0xb8, 0x3c, 0x40, 0xbc, 0xff, 0x7f, 0x01];
        write(
            &path,
            r#"{"e4m3":{"dtype":"F8_E4M3","shape":[3],"data_offsets":[0,3]},"e5m2":{"dtype":"F8_E5M2","shape":[3],"data_offsets":[3,6]},"ints":{"dtype":"I8","shape":[3],"data_offsets":[6,9]}}"#,
            &data,
        );

        let mut file = SafeTensors::open(&path).unwrap();
        let (e4m3, _) = file.tensor("e4m3").unwrap();
        let (e5m2, _) = file.tensor("e5m2").unwrap();
        let (ints, _) = file.tensor("ints").unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(e4m3, vec![1.0, 2.0, -1.0]);
        // The smallest E4M3 subnormal, which a shift-only conversion loses.
        assert_eq!(f8_to_f32(0x01, 4, 3), 2.0f32.powi(-9));
        assert_eq!(f8_to_f32(0x81, 4, 3), -2.0f32.powi(-9));
        assert_eq!(e5m2, vec![1.0, 2.0, -1.0]);
        assert_eq!(ints, vec![-1.0, 127.0, 1.0]);
    }

    #[test]
    fn a_header_that_lies_about_a_tensor_is_refused() {
        let path = scratch("st_bad.safetensors");
        write(
            &path,
            r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,8]}}"#,
            &[0u8; 8],
        );
        let error = SafeTensors::open(&path).unwrap_err().to_string();
        assert!(error.contains("byte range"), "{error}");

        // A length that runs past the end of the file, which is what a
        // truncated download looks like.
        let mut file = File::create(&path).unwrap();
        file.write_all(&u64::MAX.to_le_bytes()).unwrap();
        file.write_all(b"{}").unwrap();
        drop(file);
        assert!(SafeTensors::open(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    /// A run killed while writing a shard leaves a valid header over a short
    /// data section. It has to be refused where it is opened, not where some
    /// later read walks off the end.
    #[test]
    fn a_file_cut_short_after_its_header_is_refused() {
        let path = scratch("st_truncated.safetensors");
        write(
            &path,
            r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#,
            &[0u8; 12],
        );
        let error = SafeTensors::open(&path).unwrap_err().to_string();
        assert!(error.contains("promises 24 bytes"), "{error}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_three_dimensional_tensor_is_not_a_matrix_but_still_reads() {
        let path = scratch("st_conv.safetensors");
        let data: Vec<u8> = (0..8).flat_map(|v| (v as f32).to_le_bytes()).collect();
        write(
            &path,
            r#"{"conv":{"dtype":"F32","shape":[2,2,2],"data_offsets":[0,32]}}"#,
            &data,
        );

        let mut file = SafeTensors::open(&path).unwrap();
        let (values, shape) = file.tensor("conv").unwrap();
        let error = file.matrix("conv").unwrap_err().to_string();
        std::fs::remove_file(&path).ok();

        assert_eq!(shape, vec![2, 2, 2]);
        assert_eq!(values.len(), 8);
        assert!(error.contains("not a matrix"), "{error}");
    }

    #[test]
    fn a_sharded_model_reads_through_its_index() {
        let directory = scratch("st_shards");
        std::fs::create_dir_all(&directory).unwrap();
        let first: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let second: Vec<u8> = [3.0f32, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        write(
            &directory.join("model-00001-of-00002.safetensors"),
            r#"{"a.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            &first,
        );
        write(
            &directory.join("model-00002-of-00002.safetensors"),
            r#"{"b.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]}}"#,
            &second,
        );
        std::fs::write(
            directory.join("model.safetensors.index.json"),
            r#"{"metadata":{"total_size":16},"weight_map":{"a.weight":"model-00001-of-00002.safetensors","b.weight":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();

        let mut model =
            ShardedSafeTensors::open(directory.join("model.safetensors.index.json")).unwrap();
        assert_eq!(model.names().collect::<Vec<_>>(), ["a.weight", "b.weight"]);
        assert_eq!(model.tensor("a.weight").unwrap().0, vec![1.0, 2.0]);
        assert_eq!(model.matrix("b.weight").unwrap().data, vec![3.0, 4.0]);

        // The same reader opens a single-file model, so a caller never has to
        // ask which kind it has.
        let mut single =
            ShardedSafeTensors::open(directory.join("model-00001-of-00002.safetensors")).unwrap();
        assert_eq!(single.tensor("a.weight").unwrap().0, vec![1.0, 2.0]);
        std::fs::remove_dir_all(&directory).ok();
    }
}

#[cfg(test)]
mod writer_tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "rusting-brain-{name}-{}.safetensors",
            std::process::id()
        ))
    }

    #[test]
    fn what_is_written_is_what_is_read_back() {
        let path = scratch("writer");
        let flat = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        write(
            &path,
            &[
                ("first", &[2, 3][..], &flat[..]),
                ("second", &[6][..], &flat[..]),
            ],
            &BTreeMap::from([("step".to_string(), "7".to_string())]),
        )
        .unwrap();

        // The format asks for the tensor bytes to start eight-byte aligned.
        let bytes = std::fs::read(&path).unwrap();
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        assert_eq!(header_len % 8, 0, "header padded to an eight-byte boundary");

        let mut file = SafeTensors::open(&path).unwrap();
        assert_eq!(file.metadata().get("step").map(String::as_str), Some("7"));
        assert_eq!(file.info("first").unwrap().shape, vec![2, 3]);
        assert_eq!(file.tensor("second").unwrap().0, flat);
        let matrix = file.matrix("first").unwrap();
        assert_eq!((matrix.rows, matrix.cols), (2, 3));
        assert_eq!(matrix.data, flat);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn bfloat16_halves_the_file_and_rounds_to_nearest() {
        let path = scratch("bf16");
        // Exactly halfway between the two bfloat16 values either side of 1.0,
        // so ties-to-even has to pick 1.0 rather than carrying upwards.
        let tie = f32::from_bits(0x3f80_8000);
        let values = [1.0f32, -2.5, 0.0, tie];
        write_as(
            &path,
            &[("w", &[4][..], &values[..])],
            &BTreeMap::new(),
            Dtype::Bf16,
        )
        .unwrap();

        let mut file = SafeTensors::open(&path).unwrap();
        assert_eq!(file.info("w").unwrap().dtype, Dtype::Bf16);
        assert_eq!(file.info("w").unwrap().end, 8, "two bytes per value");
        assert_eq!(file.tensor("w").unwrap().0, [1.0, -2.5, 0.0, 1.0]);

        assert!(write_as(&path, &[], &BTreeMap::new(), Dtype::I32).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_shape_that_does_not_match_its_values_is_refused() {
        let path = scratch("bad-shape");
        let error = write(
            &path,
            &[("w", &[2, 3][..], &[1.0f32][..])],
            &BTreeMap::new(),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("holds 6 values, not 1"),
            "{error}"
        );
    }

    #[test]
    fn a_repeated_name_is_refused() {
        let path = scratch("duplicate");
        let values = [1.0f32];
        let error = write(
            &path,
            &[("w", &[1][..], &values[..]), ("w", &[1][..], &values[..])],
            &BTreeMap::new(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("appears twice"), "{error}");
        std::fs::remove_file(&path).ok();
    }
}
