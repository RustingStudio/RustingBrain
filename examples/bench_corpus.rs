//! Times the shard loader on a corpus the size a real run has.
//!
//! The training loop only goes as fast as batches arrive, and at ten thousand
//! meshes the batches come from a directory that no longer fits in page cache.
//! This writes a synthetic corpus with the record sizes `mesh_samples` and
//! `vit_tokens` write, then streams batches off it and reports how long one
//! takes.
//!
//! ```bash
//! cargo run --release --example bench_corpus -- --meshes 128 --batches 40
//! ```
//!
//! Shuffled and sequential are both timed: a shuffled pass asks for examples
//! in an order that ignores which shard they are in, and the loader holds one
//! shard open at a time, so the gap between the two numbers is what that
//! choice costs.

use rand::{Rng, SeedableRng, rngs::StdRng};
use rusting_brain::safetensors::{self, Dtype};
use rusting_brain::{BatchConfig, Corpus};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    root: PathBuf,
    meshes: usize,
    shard: usize,
    queries: usize,
    surface: usize,
    tokens: usize,
    cond_dim: usize,
    batch: usize,
    batches: usize,
    keep: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse()?;
    let tokens_dir = args.root.join("tokens");
    let samples_dir = args.root.join("samples");
    if !samples_dir.join("index.json").exists() {
        write_corpus(&args, &tokens_dir, &samples_dir)?;
    }

    let megabytes = |path: &Path| -> f64 {
        std::fs::read_dir(path)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| entry.metadata().ok())
            .map(|data| data.len() as f64)
            .sum::<f64>()
            / (1024.0 * 1024.0)
    };
    println!(
        "{} meshes over {} shards, {:.0} MB of samples and {:.0} MB of tokens",
        args.meshes,
        args.meshes.div_ceil(args.shard),
        megabytes(&samples_dir),
        megabytes(&tokens_dir)
    );

    for shuffle in [false, true] {
        let corpus = Corpus::open(&tokens_dir, &samples_dir)?;
        let stream = corpus.stream(BatchConfig {
            batch: args.batch,
            surface_points: Some(2048),
            queries: Some(4096),
            epochs: None,
            shuffle,
            seed: 1,
        })?;

        // The first batch is decoded while the stream is being built, so it
        // measures startup rather than steady state.
        let mut stream = stream.skip(1);
        let started = Instant::now();
        let mut examples = 0;
        for _ in 0..args.batches {
            examples += stream.next().transpose()?.map_or(0, |batch| batch.len());
        }
        let elapsed = started.elapsed().as_secs_f64();
        println!(
            "{:<10} {:.1} ms/batch, {:.0} examples/s",
            match shuffle {
                true => "shuffled",
                false => "sequential",
            },
            1e3 * elapsed / args.batches as f64,
            examples as f64 / elapsed
        );
    }

    if !args.keep {
        std::fs::remove_dir_all(&args.root)?;
    } else {
        println!("kept {}", args.root.display());
    }
    Ok(())
}

/// Writes the corpus the two preprocessing binaries would have written, with
/// the same tensor names, dtype and shapes and with noise for contents.
fn write_corpus(
    args: &Args,
    tokens: &Path,
    samples: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(tokens)?;
    std::fs::create_dir_all(samples)?;
    let mut rng = StdRng::seed_from_u64(3);
    let mut noise =
        |count: usize| -> Vec<f32> { (0..count).map(|_| rng.gen_range(-1.0..1.0)).collect() };

    let names: Vec<String> = (0..args.meshes).map(|i| format!("mesh{i:06}")).collect();
    let mut token_index = BTreeMap::new();
    let mut sample_index = BTreeMap::new();
    for (number, chunk) in names.chunks(args.shard).enumerate() {
        let sample_shard = format!("samples-{number:05}.safetensors");
        let token_shard = format!("tokens-{number:05}.safetensors");

        let mut sampled: Vec<(String, [usize; 2], Vec<f32>)> = Vec::new();
        let mut encoded: Vec<(String, [usize; 2], Vec<f32>)> = Vec::new();
        for name in chunk {
            sampled.push((
                format!("{name}.surface"),
                [args.surface, 6],
                noise(args.surface * 6),
            ));
            sampled.push((
                format!("{name}.queries"),
                [args.queries, 4],
                noise(args.queries * 4),
            ));
            sampled.push((
                format!("{name}.transform"),
                [1, 4],
                vec![0.0, 0.0, 0.0, 1.0],
            ));
            encoded.push((
                name.clone(),
                [args.tokens, args.cond_dim],
                noise(args.tokens * args.cond_dim),
            ));
            sample_index.insert(name.clone(), sample_shard.clone());
            token_index.insert(name.clone(), token_shard.clone());
        }
        write_shard(&samples.join(&sample_shard), &sampled)?;
        write_shard(&tokens.join(&token_shard), &encoded)?;
    }
    std::fs::write(
        samples.join("index.json"),
        serde_json::to_vec_pretty(&sample_index)?,
    )?;
    std::fs::write(
        tokens.join("index.json"),
        serde_json::to_vec_pretty(&token_index)?,
    )?;
    Ok(())
}

fn write_shard(
    path: &Path,
    tensors: &[(String, [usize; 2], Vec<f32>)],
) -> Result<(), Box<dyn std::error::Error>> {
    let borrowed: Vec<(&str, &[usize], &[f32])> = tensors
        .iter()
        .map(|(name, shape, data)| (name.as_str(), &shape[..], &data[..]))
        .collect();
    safetensors::write_as(path, &borrowed, &BTreeMap::new(), Dtype::Bf16)?;
    Ok(())
}

fn parse() -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        root: std::env::temp_dir().join("rusting-brain-bench-corpus"),
        meshes: 128,
        shard: 64,
        queries: 200_000,
        surface: 8192,
        tokens: 197,
        cond_dim: 768,
        batch: 8,
        batches: 20,
        keep: false,
    };
    let mut rest = std::env::args().skip(1);
    while let Some(flag) = rest.next() {
        let mut value = || rest.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--root" => args.root = value()?.into(),
            "--meshes" => args.meshes = value()?.parse()?,
            "--shard" => args.shard = value()?.parse()?,
            "--queries" => args.queries = value()?.parse()?,
            "--surface" => args.surface = value()?.parse()?,
            "--batch" => args.batch = value()?.parse()?,
            "--batches" => args.batches = value()?.parse()?,
            "--keep" => args.keep = true,
            other => {
                return Err(format!(
                    "unknown flag {other}\nusage: bench_corpus [--root <dir>] [--meshes n] \
                 [--shard n] [--queries n] [--surface n] [--batch n] [--batches n] [--keep]"
                )
                .into());
            }
        }
    }
    Ok(args)
}
