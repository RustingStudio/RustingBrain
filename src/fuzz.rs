//! Every reader of untrusted bytes, fed a valid file and then thousands of
//! corruptions of it. A corrupt file must come back as an error: no panic, no
//! allocation sized by a field the file made up, no unbounded recursion.
//!
//! NOTE: seeded mutations of one valid file per format rather than
//! coverage-guided fuzzing, because `cargo fuzz` is a new tool and a new
//! dependency. It finds the shallow panics (bad lengths, truncation, overflow);
//! a coverage-guided run finds the deep ones, if this ever needs more.

use crate::dataset::Dataset;
use crate::ema::Ema;
use crate::gltf::Glb;
use crate::mesh::Mesh;
use crate::safetensors::SafeTensors;
use crate::tokenizer::{Bpe, Unigram};
use crate::transformer::{LoraConfig, Precision, TransformerLm};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

const MUTATIONS: usize = 2_000;

/// One corruption of `bytes`: a flipped bit, an overwritten run, a cut, a
/// splice, or a number in a text header replaced by a huge one.
fn mutate(bytes: &[u8], rng: &mut StdRng) -> Vec<u8> {
    let mut out = bytes.to_vec();
    for _ in 0..rng.gen_range(1..=4) {
        if out.is_empty() {
            break;
        }
        let at = rng.gen_range(0..out.len());
        match rng.gen_range(0..6) {
            0 => out[at] ^= 1 << rng.gen_range(0..8),
            1 => {
                let value = [0x00, 0xFF, 0x7F, 0x80][rng.gen_range(0..4)];
                let end = (at + rng.gen_range(1..=8)).min(out.len());
                out[at..end].fill(value);
            }
            2 => out.truncate(at),
            3 => {
                let end = (at + rng.gen_range(1..=32)).min(out.len());
                let piece = out[at..end].to_vec();
                let to = rng.gen_range(0..out.len());
                out.splice(to..to, piece);
            }
            4 => {
                if let Some(digit) = (at..out.len()).find(|i| out[*i].is_ascii_digit()) {
                    let huge = [
                        "99999999999999999999",
                        "18446744073709551615",
                        "4294967296",
                        "-1",
                    ][rng.gen_range(0..4)];
                    out.splice(digit..digit + 1, huge.bytes());
                }
            }
            _ => out[at] = rng.r#gen(),
        }
    }
    out
}

/// Runs `read` on the valid file and then on every mutation, and names the
/// first one that panicked.
fn survives(name: &str, valid: &[u8], read: impl Fn(&[u8])) {
    read(valid);
    let mut rng = StdRng::seed_from_u64(0);
    for index in 0..MUTATIONS {
        let bytes = mutate(valid, &mut rng);
        if catch_unwind(AssertUnwindSafe(|| read(&bytes))).is_err() {
            let path = std::env::temp_dir().join(format!("rusting_brain_fuzz_{name}_{index}"));
            std::fs::write(&path, &bytes).unwrap();
            panic!(
                "{name}: mutation {index} panicked; the input is at {}",
                path.display()
            );
        }
    }
}

/// A path holding `bytes`, for the readers that take a path.
fn file(name: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "rusting_brain_fuzz_{name}_{:?}",
        std::thread::current().id()
    ));
    std::fs::write(&path, bytes).unwrap();
    path
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap()
}

#[test]
fn safetensors_survives_corruption() {
    let valid = file("st_valid", &[]);
    crate::safetensors::write(
        &valid,
        &[("a", &[2, 3], &[1.0; 6]), ("b", &[4], &[2.0; 4])],
        &Default::default(),
    )
    .unwrap();
    survives("safetensors", &read(&valid), |bytes| {
        let path = file("st", bytes);
        if let Ok(mut st) = SafeTensors::open(&path) {
            let names: Vec<String> = st.names().map(str::to_string).collect();
            for name in names {
                let _ = st.tensor(&name);
            }
        }
    });
}

#[test]
fn glb_survives_corruption() {
    let mesh = Mesh {
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals: vec![[0.0, 0.0, 1.0]; 3],
        uvs: vec![[0.0, 0.0]; 3],
        indices: vec![[0, 1, 2]],
        ..Default::default()
    };
    let valid = mesh.to_glb().unwrap().to_bytes();
    survives("glb", &valid, |bytes| {
        if let Ok(glb) = Glb::parse(bytes.to_vec()) {
            let _ = glb.mesh();
        }
    });
}

#[test]
fn tokenizer_json_survives_corruption() {
    let bpe = crate::tokenizer::tests::tokenizer_json(&["a b", "ab c"], &["ab", "abc"], None);
    survives("bpe", bpe.as_bytes(), |bytes| {
        if let Ok(bpe) = Bpe::from_json(&String::from_utf8_lossy(bytes)) {
            if let Ok(ids) = bpe.encode("abc ab\u{e9} <|end|>") {
                bpe.decode(&ids);
            }
        }
    });
    let unigram = crate::tokenizer::tests::unigram_json();
    survives("unigram", unigram.as_bytes(), |bytes| {
        if let Ok(unigram) = Unigram::from_json(&String::from_utf8_lossy(bytes)) {
            if let Ok(ids) = unigram.encode("hello world") {
                unigram.decode(&ids);
            }
        }
    });
}

#[test]
fn npy_and_idx_survive_corruption() {
    let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
    let mut header = header.to_string();
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut npy = b"\x93NUMPY\x01\x00".to_vec();
    npy.extend((header.len() as u16).to_le_bytes());
    npy.extend(header.bytes());
    npy.extend([1.0f32; 6].iter().flat_map(|x| x.to_le_bytes()));
    let targets = file("npy_targets", &npy);
    survives("npy", &npy, |bytes| {
        let _ = Dataset::from_npy(file("npy", bytes), &targets);
    });

    let mut images = vec![0, 0, 0x08, 3];
    for dimension in [2u32, 2, 2] {
        images.extend(dimension.to_be_bytes());
    }
    images.extend([7u8; 8]);
    let mut labels = vec![0, 0, 0x08, 1];
    labels.extend(2u32.to_be_bytes());
    labels.extend([0u8, 1]);
    let labels = file("idx_labels", &labels);
    survives("idx", &images, |bytes| {
        let _ = Dataset::from_idx(file("idx", bytes), &labels);
    });
}

#[test]
fn weight_file_survives_corruption() {
    let mut model = TransformerLm::builder()
        .vocab_size(8)
        .d_model(4)
        .n_layers(1)
        .heads(1, 1, 4)
        .d_ff(8)
        .max_seq_len(4)
        .moe_layers([])
        .seed(1)
        .build()
        .unwrap();
    let valid = file("bin_valid", &[]);
    model.save_bin(&valid, Precision::F32).unwrap();
    survives("bin", &read(&valid), |bytes| {
        let _ = TransformerLm::load_bin(file("bin", bytes));
    });

    // The header's adapter rank sizes allocations the file does not hold.
    model.add_lora(LoraConfig::new(2)).unwrap();
    model.save_bin(&valid, Precision::F32).unwrap();
    let adapted = read(&valid);
    let length = u64::from_le_bytes(adapted[8..16].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&adapted[16..16 + length]).unwrap();
    let header = header.replacen("\"rank\":2", "\"rank\":99999999999", 1);
    let mut huge = adapted[..8].to_vec();
    huge.extend((header.len() as u64).to_le_bytes());
    huge.extend(header.bytes());
    huge.extend(&adapted[16 + length..]);
    assert_ne!(huge.len(), adapted.len());
    match TransformerLm::load_bin(file("bin_rank", &huge)) {
        Err(error) => assert!(error.to_string().contains("does not fit"), "{error}"),
        Ok(_) => panic!("a rank the file cannot hold loaded"),
    }

    let ema = file("ema_valid", &[]);
    Ema::new(&mut model.params_mut()).save(&ema).unwrap();
    survives("ema", &read(&ema), |bytes| {
        let _ = Ema::load(file("ema", bytes));
    });
}
