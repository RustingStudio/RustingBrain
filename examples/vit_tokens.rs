//! Precomputes CLIP image tokens for a directory of renders.
//!
//! The image tower is frozen, so running it inside the training loop would
//! recompute the same numbers every epoch. This walks a directory once, encodes
//! each image, and writes the tokens to bfloat16 `.safetensors` shards that the
//! training loop reads back with `SafeTensors::open`.
//!
//! ```bash
//! cargo run --release --features images --example vit_tokens -- \
//!     --checkpoint clip-vit-base-patch16/model.safetensors \
//!     --images renders/ --out tokens/ --shard 512
//! ```
//!
//! Tensors are named after the image's file stem, so a training example finds
//! its condition by the name it already has. An index file maps each stem to
//! the shard holding it.
//!
//! A corpus-sized run resumes: a shard already on disk is read for its names
//! and skipped, `index.json` is rewritten after every shard, and an image that
//! will not decode is reported and passed over rather than ending the run.
//! Resuming assumes the same `--images` directory and the same `--shard`,
//! because images are chunked by position in the sorted listing.

use rusting_brain::{
    Precision, VitEncoder, VitEncoderConfig, read_image,
    safetensors::{self, Dtype, SafeTensors},
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    checkpoint: PathBuf,
    images: PathBuf,
    out: PathBuf,
    shard: usize,
    prefix: String,
    heads: usize,
    background: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse()?;
    std::fs::create_dir_all(&args.out)?;

    let tower = VitEncoder::load(
        &args.checkpoint,
        &args.prefix,
        VitEncoderConfig {
            num_heads: args.heads,
            ..VitEncoderConfig::default()
        },
        Precision::F32,
    )?;
    let size = tower.image_size();
    println!(
        "tower: {}x{size}x{size} in, {} tokens of {} out",
        tower.channels(),
        tower.tokens(),
        tower.d_model()
    );

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&args.images)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension().is_some_and(|extension| {
                matches!(
                    extension.to_ascii_lowercase().to_str(),
                    Some("png" | "jpg" | "jpeg")
                )
            })
        })
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("no images in {}", args.images.display()).into());
    }
    println!("{} images, {} per shard", paths.len(), args.shard);

    let mut index: BTreeMap<String, String> = BTreeMap::new();
    let mut skipped = 0;
    let started = Instant::now();
    let shards = paths.len().div_ceil(args.shard);
    for (number, chunk) in paths.chunks(args.shard).enumerate() {
        let shard = format!("tokens-{number:05}.safetensors");

        // Already done by an earlier run: take its names and move on.
        let done = completed(&args.out.join(&shard));
        if !done.is_empty() {
            for stem in &done {
                index.insert(stem.clone(), shard.clone());
            }
            write_index(&args.out, &index)?;
            println!("{shard}: {} images, already written", done.len());
            continue;
        }
        // One shard at a time in memory: the shard size is the knob that keeps
        // this bounded, and the tokens are small next to the images.
        let mut encoded = Vec::new();
        for path in chunk {
            let stem = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or_else(|| format!("{} has no usable name", path.display()))?
                .to_string();
            // One unreadable render in a scraped corpus should not cost the
            // whole run, the same bargain `mesh_samples` makes.
            let image = match read_image(path, size, args.background) {
                Ok(image) => image,
                Err(error) => {
                    eprintln!("skipping {}: {error}", path.display());
                    skipped += 1;
                    continue;
                }
            };
            let tokens = tower.encode(&image)?;
            index.insert(stem.clone(), shard.clone());
            encoded.push((stem, [tokens.rows, tokens.cols], tokens.data));
        }
        if encoded.is_empty() {
            continue;
        }

        let borrowed: Vec<(&str, &[usize], &[f32])> = encoded
            .iter()
            .map(|(name, shape, data)| (name.as_str(), &shape[..], &data[..]))
            .collect();
        safetensors::write_as(
            args.out.join(&shard),
            &borrowed,
            &BTreeMap::from([("source".to_string(), "clip-vit".to_string())]),
            Dtype::Bf16,
        )?;
        write_index(&args.out, &index)?;
        println!(
            "{shard}: {} images, {}",
            encoded.len(),
            progress(started, number + 1, shards)
        );
    }

    write_index(&args.out, &index)?;
    println!(
        "wrote {} entries to index.json, skipped {skipped}",
        index.len()
    );
    Ok(())
}

/// The image names an already-written shard holds, or nothing if it is missing,
/// unreadable, or was cut short by a killed run.
fn completed(path: &Path) -> Vec<String> {
    let Ok(shard) = SafeTensors::open(path) else {
        return Vec::new();
    };
    shard.names().map(str::to_string).collect()
}

/// Rewritten after every shard, so a killed run leaves an index that describes
/// exactly the shards that are on disk.
fn write_index(
    out: &Path,
    index: &BTreeMap<String, String>,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::write(out.join("index.json"), serde_json::to_vec_pretty(index)?)?;
    Ok(())
}

/// Elapsed and remaining, estimated from the shards finished so far.
fn progress(started: Instant, done: usize, total: usize) -> String {
    let elapsed = started.elapsed().as_secs_f32();
    let remaining = elapsed / done.max(1) as f32 * total.saturating_sub(done) as f32;
    format!(
        "{done}/{total} shards, {} elapsed, {} left",
        clock(elapsed),
        clock(remaining)
    )
}

fn clock(seconds: f32) -> String {
    let seconds = seconds as u64;
    match seconds / 60 {
        0 => format!("{seconds}s"),
        minutes if minutes < 60 => format!("{minutes}m"),
        minutes => format!("{}h{:02}m", minutes / 60, minutes % 60),
    }
}

fn parse() -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        checkpoint: PathBuf::new(),
        images: PathBuf::new(),
        out: PathBuf::from("tokens"),
        shard: 512,
        prefix: String::new(),
        heads: 12,
        background: 1.0,
    };
    let mut rest = std::env::args().skip(1);
    while let Some(flag) = rest.next() {
        let mut value = || rest.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--checkpoint" => args.checkpoint = value()?.into(),
            "--images" => args.images = value()?.into(),
            "--out" => args.out = value()?.into(),
            "--shard" => args.shard = value()?.parse()?,
            "--prefix" => args.prefix = value()?,
            "--heads" => args.heads = value()?.parse()?,
            "--background" => args.background = value()?.parse()?,
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    if args.checkpoint.as_os_str().is_empty() || args.images.as_os_str().is_empty() {
        return Err("usage: --checkpoint <model.safetensors> --images <dir> [--out <dir>] [--shard n] [--prefix p] [--heads n] [--background v]".into());
    }
    if args.shard == 0 {
        return Err("--shard needs at least one image".into());
    }
    Ok(args)
}
