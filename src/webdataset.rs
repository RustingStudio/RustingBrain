//! WebDataset shards: tar files whose members are grouped into samples by
//! name, as in `000017.jpg`, `000017.txt`, `000017.json`. Everything up to
//! the first dot of the file name (directory included) is the sample's key,
//! and the rest is the extension under which that member is stored.
//! A sample's members must sit next to each other in the shard, which is how
//! `webdataset` writes them and what `tar` of a sorted directory gives.
//!
//! The reader streams: one sample in memory at a time, so a shard of any size
//! reads in constant memory. It understands ustar and old GNU headers, GNU
//! long names and pax `path` records, which covers what Python's `tarfile`,
//! GNU `tar` and the `webdataset` writer produce.
//!
//! NOTE: no sparse files and no base-256 sizes (members of 8 GiB or more);
//! both are refused, and a sample is never that large.

use crate::network::NetworkError;
use std::io::{BufReader, Read};
use std::path::Path;

const BLOCK: usize = 512;

/// One sample: every member that shares a key, in shard order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sample {
    pub key: String,
    /// `(extension, bytes)`, for example `("jpg", ...)` and `("txt", ...)`.
    pub files: Vec<(String, Vec<u8>)>,
}

impl Sample {
    /// The bytes stored under `extension`, if the sample has them.
    pub fn get(&self, extension: &str) -> Option<&[u8]> {
        self.files
            .iter()
            .find(|(ext, _)| ext == extension)
            .map(|(_, bytes)| bytes.as_slice())
    }
}

/// Reads [`Sample`]s from one shard, in order.
pub struct WebDataset<R: Read> {
    reader: R,
    /// The first member of the next sample, read while finishing this one.
    pending: Option<(String, String, Vec<u8>)>,
    done: bool,
}

impl WebDataset<BufReader<std::fs::File>> {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, NetworkError> {
        Ok(Self::new(BufReader::new(std::fs::File::open(path)?)))
    }
}

impl<R: Read> WebDataset<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            pending: None,
            done: false,
        }
    }

    /// The next member that belongs to a sample, as `(key, extension, bytes)`.
    /// Directories, links and files without an extension are skipped.
    fn next_member(&mut self) -> Result<Option<(String, String, Vec<u8>)>, NetworkError> {
        let mut long_name: Option<String> = None;
        loop {
            let mut header = [0u8; BLOCK];
            if !read_block(&mut self.reader, &mut header)? || header.iter().all(|&b| b == 0) {
                return Ok(None);
            }
            check_sum(&header)?;
            let size = octal(&header[124..136])?;
            let data = read_data(&mut self.reader, size)?;
            let name = match long_name.take() {
                Some(name) => name,
                None => header_name(&header),
            };
            match header[156] {
                b'0' | 0 | b'7' => {}
                b'L' => {
                    long_name = Some(c_string(&data));
                    continue;
                }
                b'x' => {
                    long_name = pax_path(&data)?;
                    continue;
                }
                b'S' => return Err(invalid(format!("{name} is a sparse file"))),
                _ => continue,
            }
            let (directory, file) = name.rsplit_once('/').unwrap_or(("", &name));
            let Some((stem, extension)) = file.split_once('.') else {
                continue;
            };
            let key = if directory.is_empty() {
                stem.to_string()
            } else {
                format!("{directory}/{stem}")
            };
            return Ok(Some((key, extension.to_string(), data)));
        }
    }

    fn next_sample(&mut self) -> Result<Option<Sample>, NetworkError> {
        let first = match self.pending.take() {
            Some(member) => Some(member),
            None => self.next_member()?,
        };
        let Some((key, extension, data)) = first else {
            return Ok(None);
        };
        let mut sample = Sample {
            key,
            files: vec![(extension, data)],
        };
        while let Some(member) = self.next_member()? {
            if member.0 != sample.key {
                self.pending = Some(member);
                break;
            }
            sample.files.push((member.1, member.2));
        }
        Ok(Some(sample))
    }
}

impl<R: Read> Iterator for WebDataset<R> {
    type Item = Result<Sample, NetworkError>;

    /// Stops after the first error: past a corrupt header the stream has no
    /// way back to a member boundary.
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let next = self.next_sample();
        self.done = !matches!(next, Ok(Some(_)));
        next.transpose()
    }
}

fn invalid(message: String) -> NetworkError {
    NetworkError::InvalidDataset(format!("tar: {message}"))
}

/// Fills `block`, or returns `false` at a clean end of the stream.
fn read_block(reader: &mut impl Read, block: &mut [u8; BLOCK]) -> Result<bool, NetworkError> {
    let mut filled = 0;
    while filled < BLOCK {
        match reader.read(&mut block[filled..])? {
            0 if filled == 0 => return Ok(false),
            0 => return Err(invalid("the file ends inside a header".into())),
            n => filled += n,
        }
    }
    Ok(true)
}

/// `size` bytes of member data and the padding to the next block. The buffer
/// grows as bytes arrive, so a made-up size cannot allocate past the file.
fn read_data(reader: &mut impl Read, size: u64) -> Result<Vec<u8>, NetworkError> {
    let padded = size.div_ceil(BLOCK as u64) * BLOCK as u64;
    let mut data = Vec::new();
    reader.take(padded).read_to_end(&mut data)?;
    if (data.len() as u64) < padded {
        return Err(invalid("the file ends inside a member".into()));
    }
    data.truncate(size as usize);
    Ok(data)
}

fn check_sum(header: &[u8; BLOCK]) -> Result<(), NetworkError> {
    let stored = octal(&header[148..156])?;
    let sum: u64 = header
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            if (148..156).contains(&i) {
                32
            } else {
                b as u64
            }
        })
        .sum();
    if stored != sum {
        return Err(invalid(format!(
            "header checksum {stored} does not match {sum}"
        )));
    }
    Ok(())
}

/// An octal field, padded with spaces or NULs.
fn octal(field: &[u8]) -> Result<u64, NetworkError> {
    if field[0] & 0x80 != 0 {
        return Err(invalid("base-256 numbers are not supported".into()));
    }
    let digits = c_string(field);
    let digits = digits.trim_matches(' ');
    if digits.is_empty() {
        return Ok(0);
    }
    u64::from_str_radix(digits, 8).map_err(|_| invalid(format!("bad octal field {digits:?}")))
}

fn c_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// The name, with the ustar prefix in front of it when there is one.
fn header_name(header: &[u8; BLOCK]) -> String {
    let name = c_string(&header[..100]);
    let prefix = if &header[257..262] == b"ustar" {
        c_string(&header[345..500])
    } else {
        String::new()
    };
    if prefix.is_empty() {
        name
    } else {
        format!("{prefix}/{name}")
    }
}

/// The `path` of a pax extended header: records of `"<len> <key>=<value>\n"`.
fn pax_path(data: &[u8]) -> Result<Option<String>, NetworkError> {
    let mut rest = data;
    let mut path = None;
    while !rest.is_empty() {
        let space = rest
            .iter()
            .position(|&b| b == b' ')
            .ok_or_else(|| invalid("bad pax record".into()))?;
        let length: usize = std::str::from_utf8(&rest[..space])
            .ok()
            .and_then(|digits| digits.parse().ok())
            .filter(|&length| length > space && length <= rest.len())
            .ok_or_else(|| invalid("bad pax record length".into()))?;
        let record = &rest[space + 1..length];
        if let Some(value) = record.strip_prefix(b"path=") {
            let value = value.strip_suffix(b"\n").unwrap_or(value);
            path = Some(String::from_utf8_lossy(value).into_owned());
        }
        rest = &rest[length..];
    }
    Ok(path)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A ustar member, its data and padding.
    fn member(out: &mut Vec<u8>, name: &str, kind: u8, data: &[u8]) {
        let mut header = [0u8; BLOCK];
        header[..name.len()].copy_from_slice(name.as_bytes());
        header[100..108].copy_from_slice(b"0000644\0");
        header[124..136].copy_from_slice(format!("{:011o}\0", data.len()).as_bytes());
        header[156] = kind;
        header[257..263].copy_from_slice(b"ustar\0");
        header[148..156].fill(b' ');
        let sum: u32 = header.iter().map(|&b| b as u32).sum();
        header[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(data);
        out.resize(out.len().div_ceil(BLOCK) * BLOCK, 0);
    }

    /// Two samples, a directory, a file without an extension, a GNU long name
    /// and a pax path.
    pub(crate) fn shard() -> Vec<u8> {
        let mut tar = Vec::new();
        member(&mut tar, "train/", b'5', b"");
        member(&mut tar, "train/000.jpg", b'0', &[0xFF, 0xD8, 0xFF]);
        member(&mut tar, "train/000.txt", b'0', b"a red cube");
        member(&mut tar, "README", b'0', b"not a sample");
        let long = format!("train/{}.txt", "x".repeat(120));
        member(
            &mut tar,
            "././@LongLink",
            b'L',
            format!("{long}\0").as_bytes(),
        );
        member(&mut tar, "truncated", b'0', b"long name");
        let record = "path=train/001.cls\n";
        let pax = format!("{} {record}", record.len() + 3);
        member(&mut tar, "PaxHeader", b'x', pax.as_bytes());
        member(&mut tar, "short", b'0', b"7");
        member(&mut tar, "train/001.json", b'0', b"{}");
        tar.resize(tar.len() + 2 * BLOCK, 0);
        tar
    }

    #[test]
    fn members_group_into_samples_by_key() {
        let samples: Vec<Sample> = WebDataset::new(shard().as_slice())
            .collect::<Result<_, _>>()
            .unwrap();
        let keys: Vec<&str> = samples.iter().map(|s| s.key.as_str()).collect();
        let long = format!("train/{}", "x".repeat(120));
        assert_eq!(keys, ["train/000", long.as_str(), "train/001"]);
        assert_eq!(samples[0].get("jpg"), Some(&[0xFF, 0xD8, 0xFF][..]));
        assert_eq!(samples[0].get("txt"), Some(&b"a red cube"[..]));
        assert_eq!(samples[1].get("txt"), Some(&b"long name"[..]));
        assert_eq!(samples[2].get("cls"), Some(&b"7"[..]));
        assert_eq!(samples[2].get("json"), Some(&b"{}"[..]));
    }

    #[test]
    fn a_truncated_or_corrupt_shard_is_an_error() {
        let tar = shard();
        let cut: Vec<_> = WebDataset::new(&tar[..BLOCK * 3 + 100]).collect();
        assert!(cut.last().unwrap().is_err());
        let mut bad = tar.clone();
        bad[BLOCK + 130] = b'7';
        let read: Vec<_> = WebDataset::new(bad.as_slice()).collect();
        assert!(read.last().unwrap().is_err());
        // A huge size is refused at the end of the file, not allocated.
        let mut huge = tar;
        huge[BLOCK + 124..BLOCK + 136].copy_from_slice(b"77777777777\0");
        huge[BLOCK + 148..BLOCK + 156].fill(b' ');
        let sum: u32 = huge[BLOCK..2 * BLOCK].iter().map(|&b| b as u32).sum();
        huge[BLOCK + 148..BLOCK + 156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
        let read: Vec<_> = WebDataset::new(huge.as_slice()).collect();
        assert!(read.last().unwrap().is_err());
    }
}
