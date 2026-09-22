//! Reports what a shape autoencoder's posterior costs, checkpoint by checkpoint.
//!
//! The decoder is only ever trained on draws from the posterior, so the mean is
//! off-distribution and reconstructing from it is a different, worse number.
//! How much worse depends on how wide the posterior is, which is what the KL
//! weight controls. This reads one or more checkpoints and reports, on the same
//! objects and the same query points for every checkpoint:
//!
//! - `l1 draw` — clamped L1 of the distances decoded from a draw, which is what
//!   training measured.
//! - `l1 mean` — the same objects decoded from the posterior mean, which is what
//!   a caller gets from [`ShapeVae::encode`] if they ignore the warning.
//! - `sigma` — the posterior's mean standard deviation, and `|mean|` the mean
//!   magnitude of its centre. Their ratio is why the two errors differ.
//! - `active` — the fraction of latent units whose KL is above 0.01 nats, which
//!   is the measurement a free-bits floor would be aimed at.
//!
//! ```bash
//! cargo run --release --features cuda --example kl_report -- \
//!     --tokens tokens/ --samples samples/ --cuda 0 \
//!     kl-1e-5.safetensors kl-1e-4.safetensors kl-1e-3.safetensors
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rusting_brain::{Corpus, Matrix, ShapeVae, losses};
use std::path::PathBuf;

struct Args {
    tokens: PathBuf,
    samples: PathBuf,
    objects: usize,
    surface: usize,
    queries: usize,
    clamp: f32,
    seed: u64,
    cuda: Option<usize>,
    checkpoints: Vec<PathBuf>,
}

/// One object's samples, drawn once and reused for every checkpoint.
struct Object {
    surface: Matrix,
    queries: Matrix,
    distances: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse()?;
    let mut rng = StdRng::seed_from_u64(args.seed);

    let mut corpus = Corpus::open(&args.tokens, &args.samples)?;
    if corpus.is_empty() {
        return Err("no render is paired with a mesh; check the two directories".into());
    }
    let objects: Vec<Object> = (0..args.objects.min(corpus.len()))
        .map(|index| {
            let example =
                corpus.example(index, Some(args.surface), Some(args.queries), &mut rng)?;
            Ok::<_, Box<dyn std::error::Error>>(Object {
                surface: example.surface,
                queries: example.queries,
                distances: example.distances,
            })
        })
        .collect::<Result<_, _>>()?;
    println!(
        "{} objects, {} surface points, {} query points, clamp {}",
        objects.len(),
        args.surface,
        args.queries,
        args.clamp
    );
    println!(
        "{:<28} {:>9} {:>9} {:>7} {:>7} {:>8} {:>7}",
        "checkpoint", "l1 draw", "l1 mean", "sigma", "|mean|", "kl/unit", "active"
    );

    for path in &args.checkpoints {
        // A fresh generator per checkpoint, so every model is asked for the
        // same draw of noise as well as the same objects.
        let mut rng = StdRng::seed_from_u64(args.seed ^ 0x5a1);
        // `mut` is only used by the device move below.
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let (mut model, _) = ShapeVae::load(path, &mut StdRng::seed_from_u64(args.seed))?;
        if let Some(device) = args.cuda {
            #[cfg(feature = "cuda")]
            model.to_cuda(device, 0)?;
            #[cfg(not(feature = "cuda"))]
            return Err(format!(
                "--cuda {device} needs the crate's `cuda` feature: rebuild with --features cuda"
            )
            .into());
        }

        let (mut draw_l1, mut mean_l1, mut sigma, mut magnitude, mut kl, mut active) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for object in &objects {
            let (mean, log_variance) = model.encode(&object.surface)?;
            let (latent, _) = ShapeVae::sample(&mean, &log_variance, &mut rng);

            let from_draw = model.decode(&latent, &object.queries, 4096)?;
            let from_mean = model.decode(&mean, &object.queries, 4096)?;
            draw_l1 += losses::clamped_l1(&from_draw, &object.distances, args.clamp).0;
            mean_l1 += losses::clamped_l1(&from_mean, &object.distances, args.clamp).0;

            let units = mean.data.len() as f32;
            magnitude += mean.data.iter().map(|value| value.abs()).sum::<f32>() / units;
            sigma += log_variance
                .data
                .iter()
                .map(|value| (0.5 * value).exp())
                .sum::<f32>()
                / units;
            // The per-unit KL against a unit Gaussian, which is what a
            // free-bits floor would be applied to one unit at a time.
            let mut per_unit = 0.0;
            let mut above = 0.0;
            for (mean, log_variance) in mean.data.iter().zip(&log_variance.data) {
                let nats = 0.5 * (mean * mean + log_variance.exp() - 1.0 - log_variance);
                per_unit += nats;
                if nats > 0.01 {
                    above += 1.0;
                }
            }
            kl += per_unit / units;
            active += above / units;
        }

        let count = objects.len() as f32;
        println!(
            "{:<28} {:>9.5} {:>9.5} {:>7.3} {:>7.3} {:>8.4} {:>6.1}%",
            path.file_name()
                .unwrap_or(path.as_os_str())
                .to_string_lossy(),
            draw_l1 / count,
            mean_l1 / count,
            sigma / count,
            magnitude / count,
            kl / count,
            100.0 * active / count
        );
    }
    Ok(())
}

fn parse() -> Result<Args, Box<dyn std::error::Error>> {
    let mut rest = std::env::args().skip(1);
    let mut args = Args {
        tokens: PathBuf::from("tokens"),
        samples: PathBuf::from("samples"),
        objects: 16,
        surface: 2048,
        queries: 4096,
        clamp: 0.1,
        seed: 0,
        cuda: None,
        checkpoints: Vec::new(),
    };
    while let Some(flag) = rest.next() {
        let mut value = || rest.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--tokens" => args.tokens = value()?.into(),
            "--samples" => args.samples = value()?.into(),
            "--objects" => args.objects = value()?.parse()?,
            "--surface" => args.surface = value()?.parse()?,
            "--queries" => args.queries = value()?.parse()?,
            "--clamp" => args.clamp = value()?.parse()?,
            "--seed" => args.seed = value()?.parse()?,
            "--cuda" => args.cuda = Some(value()?.parse()?),
            other if other.starts_with("--") => {
                return Err(format!("unknown flag {other}\n{}", usage()).into());
            }
            path => args.checkpoints.push(path.into()),
        }
    }
    if args.checkpoints.is_empty() || args.objects == 0 {
        return Err(usage().into());
    }
    Ok(args)
}

fn usage() -> String {
    "usage: kl_report [--tokens <dir>] [--samples <dir>] [--objects n] [--surface n] \
     [--queries n] [--clamp f] [--seed n] [--cuda device] <checkpoint>..."
        .to_string()
}
