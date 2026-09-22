//! Trains the two halves of the image-to-3D pipeline, one stage at a time.
//!
//! Both stages read the same corpus: the image tokens `vit_tokens` wrote and
//! the surface and query samples `mesh_samples` wrote, paired by name.
//!
//! ```bash
//! # Stage one: the shape autoencoder, which never looks at the images.
//! cargo run --release --example train_shape -- vae \
//!     --tokens tokens/ --samples samples/ --out vae.safetensors --steps 20000
//!
//! # Stage two: the flow transformer, conditioned on the tokens, writing into
//! # the latent space stage one just built.
//! cargo run --release --example train_shape -- flow \
//!     --tokens tokens/ --samples samples/ --vae vae.safetensors \
//!     --out flow.safetensors --steps 40000
//! ```
//!
//! `--cuda <device>` moves both stages onto a CUDA device, with the crate
//! built `--features cuda`: stage one's projections and attentions through
//! [`rusting_brain::gpu_shape`], stage two's transformer blocks through
//! `gpu_flow`. Stage one packs short equal-sized point clouds so a batch shares
//! projection, attention and synchronization launches; large query sets stay
//! shape-at-a-time to bound the attention workspace.
//!
//! The autoencoder is frozen during stage two: it is only asked to encode, and
//! its gradients are never touched. That is what makes the latent space a
//! fixed target rather than one that moves under the flow transformer.
//!
//! Stage one also trains the colour head, on the meshes whose samples carry
//! vertex colours: `--color` weights its error against the distance error and
//! `--color-points 0` turns it off. A corpus with no colours anywhere trains
//! exactly as it did before the head existed, since a batch without colour
//! targets skips it.
//!
//! `--latent-scale` multiplies the encoder's latent before the flow sees it.
//! Flow matching mixes the latent with unit-variance noise, so a latent whose
//! own spread is far from one trains badly. The first batch prints the spread
//! it measured, and the scale is written into the checkpoint so
//! `image_to_3d` divides by the same number.
//!
//! The CPU path remains one shape at a time inside a batch. CUDA batches are
//! uniform because the corpus sampler already selects the same point counts
//! for every shape, so no padding or mask is needed.

use rand::SeedableRng;
use rand::rngs::StdRng;
#[cfg(feature = "cuda")]
use rusting_brain::Batch;
use rusting_brain::{
    BatchConfig, BatchStream, Corpus, FlowConfig, FlowTransformer, Matrix, Optimizer, ShapeVae,
    ShapeVaeConfig, losses, optimizers,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq)]
enum Stage {
    Vae,
    Flow,
}

struct Args {
    stage: Stage,
    tokens: PathBuf,
    samples: PathBuf,
    vae: PathBuf,
    out: PathBuf,
    resume: Option<PathBuf>,
    config: Option<PathBuf>,
    steps: usize,
    batch: usize,
    surface: usize,
    queries: usize,
    learning_rate: f32,
    kl: f32,
    clamp: f32,
    /// How heavily the colour head's error counts against the distance error.
    color: f32,
    /// Surface points per example the colour head is trained on. 0 turns the
    /// colour head off.
    color_points: usize,
    dropout: f32,
    latent_scale: f32,
    report: usize,
    checkpoint_every: usize,
    seed: u64,
    /// Which CUDA device the flow transformer's blocks train on, if any.
    cuda: Option<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse()?;
    let mut rng = StdRng::seed_from_u64(args.seed);

    let mut corpus = Corpus::open(&args.tokens, &args.samples)?;
    if corpus.is_empty() {
        return Err("no render is paired with a mesh; check the two directories".into());
    }
    println!("{} paired examples", corpus.len());

    // The token shape is the condition's shape, and only the file knows it.
    // One example with a single point sampled is the cheapest way to ask.
    let condition = match args.stage {
        Stage::Vae => None,
        Stage::Flow => {
            let example = corpus.example(0, Some(1), Some(1), &mut rng)?;
            Some((example.tokens.rows, example.tokens.cols))
        }
    };

    let stream = corpus.stream(BatchConfig {
        batch: args.batch,
        surface_points: Some(args.surface),
        queries: Some(args.queries),
        epochs: None,
        shuffle: true,
        seed: args.seed,
    })?;

    match condition {
        None => train_vae(&args, stream, &mut rng),
        Some(condition) => train_flow(&args, condition, stream, &mut rng),
    }
}

fn train_vae(
    args: &Args,
    stream: BatchStream,
    rng: &mut StdRng,
) -> Result<(), Box<dyn std::error::Error>> {
    let (mut model, mut step) = match &args.resume {
        Some(path) => {
            let (model, metadata) = ShapeVae::load(path, rng)?;
            (model, started_at(&metadata))
        }
        None => (ShapeVae::new(shape_config(args)?, rng)?, 0),
    };
    if let Some(device) = args.cuda {
        #[cfg(feature = "cuda")]
        {
            // 0 as the budget: the caller picked the card, and the weights are
            // a small part of what a run of this size needs.
            model.to_cuda(device, 0)?;
            println!("shape autoencoder: training on CUDA device {device}");
        }
        #[cfg(not(feature = "cuda"))]
        return Err(format!(
            "--cuda {device} needs the crate's `cuda` feature: rebuild with --features cuda"
        )
        .into());
    }
    let optimizer = adam(args.learning_rate);
    println!(
        "shape autoencoder: {} parameters, from step {step}",
        model.num_parameters()
    );

    let (mut reconstruction, mut divergence, mut tint) = (0.0, 0.0, 0.0);
    for batch in stream {
        let batch = batch?;
        let count = batch.len();
        let points = batch.surface.rows / count;
        let queries = batch.queries.rows / count;

        // Packing helps the ordinary 4K-query training shape, but at 16K
        // queries the larger attention workspace is slower and batch eight can
        // exceed a 12 GiB card. Keep that workload shape-at-a-time.
        let cuda_batch = model.on_device() && queries <= 4096;
        #[cfg(feature = "cuda")]
        if cuda_batch {
            let (loss, kl, color) = train_vae_cuda_batch(args, &mut model, &batch, rng)?;
            // The packed losses are means across the batch; the reporting
            // accumulator below historically stores a sum over examples.
            reconstruction += loss * count as f32;
            divergence += kl * count as f32;
            tint += color * count as f32;
        }

        if !cuda_batch {
            for index in 0..count {
                let surface = slice(&batch.surface, index, points);
                let (mean, log_variance, cache) = model.encode_train(&surface)?;
                let (latent, noise) = ShapeVae::sample(&mean, &log_variance, rng);

                let (prediction, decoded) =
                    model.decode_train(&latent, &slice(&batch.queries, index, queries))?;
                let target = &batch.distances[index * queries..(index + 1) * queries];
                let (loss, grad) = losses::clamped_l1(&prediction, target, args.clamp);
                let (kl, grad_mean, grad_log_variance) =
                    losses::kl_divergence(&mean.data, &log_variance.data);
                reconstruction += loss;
                divergence += kl;

                let mut grad_latent = model.decode_backward(&decoded, &grad)?;

                // The colour targets sit on the surface points, not the distance
                // queries, so the colour head trains on a second short decode.
                // The loader's points are already in random order, so the first
                // `--color-points` of them are a random draw.
                if let Some(colors) = &batch.colors
                    && args.color > 0.0
                    && args.color_points > 0
                {
                    let wanted = args.color_points.min(points);
                    let positions = positions(&surface, wanted);
                    let targets = head_rows(&slice(colors, index, points), wanted);
                    let (_, decoded) = model.decode_train(&latent, &positions)?;

                    let predicted = model.colors(&decoded)?;
                    let mut grad = predicted.clone();
                    let scale = args.color / predicted.data.len() as f32;
                    for (slot, target) in grad.data.iter_mut().zip(&targets.data) {
                        let error = *slot - target;
                        tint += error * error / predicted.data.len() as f32;
                        *slot = 2.0 * scale * error;
                    }

                    let from_colors = model.decode_backward_colored(
                        &decoded,
                        &vec![0.0; positions.rows],
                        Some(&grad),
                    )?;
                    for (slot, value) in grad_latent.data.iter_mut().zip(&from_colors.data) {
                        *slot += value;
                    }
                }

                let (mut into_mean, mut into_log_variance) =
                    ShapeVae::sample_backward(&grad_latent, &log_variance, &noise);
                for (slot, value) in into_mean.data.iter_mut().zip(&grad_mean) {
                    *slot += args.kl * value;
                }
                for (slot, value) in into_log_variance.data.iter_mut().zip(&grad_log_variance) {
                    *slot += args.kl * value;
                }
                model.encode_backward(&cache, &into_mean, &into_log_variance)?;
            }
        }

        step += 1;
        // The gradients were summed over the batch, so the step is scaled by
        // its size rather than the loop dividing every gradient it makes.
        optimizers::step_clipped(
            &mut model.params_mut(),
            &optimizer,
            step,
            1.0 / count as f32,
            1.0,
        )?;
        optimizers::zero_grad(&mut model.params_mut());

        if step % args.report == 0 {
            let scale = (args.report * count) as f32;
            println!(
                "step {step}: l1 {:.5}, kl {:.5}, color {:.5}",
                reconstruction / scale,
                divergence / scale,
                tint / scale
            );
            reconstruction = 0.0;
            divergence = 0.0;
            tint = 0.0;
        }
        if step % args.checkpoint_every == 0 || step == args.steps {
            model.save(&args.out, &BTreeMap::from([step_at(step)]))?;
            println!("wrote {} at step {step}", args.out.display());
        }
        if step >= args.steps {
            break;
        }
    }
    Ok(())
}

/// One packed device step. Packing is important here: the old loop launched
/// and synchronized the whole encoder and decoder once per shape.
#[cfg(feature = "cuda")]
fn train_vae_cuda_batch(
    args: &Args,
    model: &mut ShapeVae,
    batch: &Batch,
    rng: &mut StdRng,
) -> Result<(f32, f32, f32), Box<dyn std::error::Error>> {
    let count = batch.len();
    let points = batch.surface.rows / count;
    let (mean, log_variance, encoder) = model.encode_train_batch(&batch.surface, count)?;
    let (latent, noise) = ShapeVae::sample(&mean, &log_variance, rng);

    let (prediction, decoded) = model.decode_train_batch(&latent, &batch.queries, count)?;
    let (reconstruction, mut grad) = losses::clamped_l1(&prediction, &batch.distances, args.clamp);
    let (divergence, mut grad_mean, mut grad_log_variance) =
        losses::kl_divergence(&mean.data, &log_variance.data);
    // The shape-at-a-time path accumulates one mean-loss gradient per example
    // and lets the optimizer divide by the batch. Re-expand this packed mean
    // so gradient clipping and the optimizer see the same summed gradient.
    let batch_scale = count as f32;
    grad.iter_mut().for_each(|value| *value *= batch_scale);
    grad_mean.iter_mut().for_each(|value| *value *= batch_scale);
    grad_log_variance
        .iter_mut()
        .for_each(|value| *value *= batch_scale);
    let mut grad_latent = model.decode_backward(&decoded, &grad)?;

    let mut tint = 0.0;
    if let Some(colors) = &batch.colors
        && args.color > 0.0
        && args.color_points > 0
    {
        let wanted = args.color_points.min(points);
        let positions = packed_prefix_rows(&batch.surface, count, points, wanted, 3);
        let targets = packed_prefix_rows(colors, count, points, wanted, 3);
        let (_, decoded) = model.decode_train_batch(&latent, &positions, count)?;
        let predicted = model.colors(&decoded)?;
        let mut grad = predicted.clone();
        let scale = args.color * batch_scale / predicted.data.len() as f32;
        for (slot, target) in grad.data.iter_mut().zip(&targets.data) {
            let error = *slot - target;
            tint += error * error / predicted.data.len() as f32;
            *slot = 2.0 * scale * error;
        }
        let from_colors =
            model.decode_backward_colored(&decoded, &vec![0.0; positions.rows], Some(&grad))?;
        for (slot, value) in grad_latent.data.iter_mut().zip(&from_colors.data) {
            *slot += value;
        }
    }

    let (mut into_mean, mut into_log_variance) =
        ShapeVae::sample_backward(&grad_latent, &log_variance, &noise);
    for (slot, value) in into_mean.data.iter_mut().zip(&grad_mean) {
        *slot += args.kl * value;
    }
    for (slot, value) in into_log_variance.data.iter_mut().zip(&grad_log_variance) {
        *slot += args.kl * value;
    }
    model.encode_backward(&encoder, &into_mean, &into_log_variance)?;
    Ok((reconstruction, divergence, tint))
}

fn train_flow(
    args: &Args,
    (cond_tokens, cond_dim): (usize, usize),
    stream: BatchStream,
    rng: &mut StdRng,
) -> Result<(), Box<dyn std::error::Error>> {
    // Frozen: the flow transformer is trained against a latent space that has
    // already stopped moving, so only `encode` is ever called here.
    let (vae, _) = ShapeVae::load(&args.vae, rng)?;
    let shape = *vae.config();

    let (mut model, mut step) = match &args.resume {
        Some(path) => {
            let (model, metadata) = FlowTransformer::load(path, rng)?;
            (model, started_at(&metadata))
        }
        None => {
            // The latent shape is the autoencoder's and the condition's shape
            // is the corpus's, whatever a supplied configuration says.
            let config = FlowConfig {
                latents: shape.latents,
                latent_dim: shape.latent_dim,
                cond_tokens,
                cond_dim,
                ..read_config(args, FlowConfig::default())?
            };
            (FlowTransformer::new(config, rng)?, 0)
        }
    };
    if model.config().cond_dim != cond_dim || model.config().cond_tokens != cond_tokens {
        return Err(format!(
            "the checkpoint is conditioned on [{}, {}] tokens and the corpus has [{cond_tokens}, {cond_dim}]",
            model.config().cond_tokens,
            model.config().cond_dim
        )
        .into());
    }
    model.set_optimizer(adam(args.learning_rate));
    if let Some(device) = args.cuda {
        #[cfg(feature = "cuda")]
        {
            // 0 as the budget: the caller picked the card, and the blocks'
            // weights are a small part of what a run of this size needs.
            model.to_cuda(device, 0)?;
            println!("flow transformer: training on CUDA device {device}");
        }
        #[cfg(not(feature = "cuda"))]
        return Err(format!(
            "--cuda {device} needs the crate's `cuda` feature: rebuild with --features cuda"
        )
        .into());
    }
    println!(
        "flow transformer: {} parameters, from step {step}",
        model.num_parameters()
    );

    let mut running = 0.0;
    for batch in stream {
        let batch = batch?;
        let count = batch.len();
        let points = batch.surface.rows / count;

        let mut clean = Matrix::new(count * shape.latents, shape.latent_dim);
        for index in 0..count {
            // A draw from the posterior, not its mean. Every autoencoder step
            // decoded a draw, so a draw is the only kind of latent the frozen
            // decoder has ever answered for. The posterior is wide enough for
            // that to matter — sigma about twice the magnitude of the mean —
            // and the mean reconstructs at clamped L1 0.029 against 0.0025 for
            // a draw, so a flow trained towards the mean would be aiming at the
            // one latent the decoder is worst at.
            let latent = vae.encode_sample(&slice(&batch.surface, index, points), rng)?;
            let width = shape.latents * shape.latent_dim;
            for (slot, value) in clean.data[index * width..(index + 1) * width]
                .iter_mut()
                .zip(&latent.data)
            {
                *slot = value * args.latent_scale;
            }
        }
        if step == 0 {
            println!("latent spread after scaling: {:.3}", spread(&clean));
        }

        running += model.train_step(&clean, &batch.tokens, args.dropout, rng)?;
        step += 1;
        model.step_clipped(1.0, 1.0)?;
        model.zero_grad();

        if step % args.report == 0 {
            println!(
                "step {step}: velocity mse {:.5}",
                running / args.report as f32
            );
            running = 0.0;
        }
        if step % args.checkpoint_every == 0 || step == args.steps {
            model.save(
                &args.out,
                &BTreeMap::from([
                    step_at(step),
                    ("latent_scale".to_string(), args.latent_scale.to_string()),
                ]),
            )?;
            println!("wrote {} at step {step}", args.out.display());
        }
        if step >= args.steps {
            break;
        }
    }
    Ok(())
}

/// The autoencoder's shape, from `--config` or the plan's default.
fn shape_config(args: &Args) -> Result<ShapeVaeConfig, Box<dyn std::error::Error>> {
    read_config(args, ShapeVaeConfig::default())
}

/// A configuration read from `--config`, or the default when there is none.
///
/// The configurations are `serde` types already, so a JSON file is the whole
/// of the knob: no per-field flag, and a run's shape is a file it can keep
/// beside its checkpoint.
fn read_config<T: serde::de::DeserializeOwned>(
    args: &Args,
    fallback: T,
) -> Result<T, Box<dyn std::error::Error>> {
    match &args.config {
        Some(path) => Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?),
        None => Ok(fallback),
    }
}

/// One example's rows out of a batch's stacked matrix.
/// The first `rows` rows of a matrix.
fn head_rows(matrix: &Matrix, rows: usize) -> Matrix {
    Matrix::from_vec(
        rows,
        matrix.cols,
        matrix.data[..rows * matrix.cols].to_vec(),
    )
}

/// The first `wanted` rows from every equally sized example in a packed batch.
/// `cols` may select a prefix, which drops normals from surface rows.
#[cfg(feature = "cuda")]
fn packed_prefix_rows(
    matrix: &Matrix,
    examples: usize,
    rows_per_example: usize,
    wanted: usize,
    cols: usize,
) -> Matrix {
    let mut data = Vec::with_capacity(examples * wanted * cols);
    for example in 0..examples {
        let first = example * rows_per_example;
        for row in first..first + wanted {
            data.extend_from_slice(&matrix.row(row)[..cols]);
        }
    }
    Matrix::from_vec(examples * wanted, cols, data)
}

/// The positions of the first `rows` surface points, dropping the normals.
fn positions(surface: &Matrix, rows: usize) -> Matrix {
    Matrix::from_vec(
        rows,
        3,
        surface
            .data
            .chunks_exact(surface.cols)
            .take(rows)
            .flat_map(|point| point[..3].iter().copied())
            .collect(),
    )
}

fn slice(stacked: &Matrix, index: usize, rows: usize) -> Matrix {
    let width = rows * stacked.cols;
    Matrix::from_vec(
        rows,
        stacked.cols,
        stacked.data[index * width..(index + 1) * width].to_vec(),
    )
}

/// The standard deviation of a matrix, which is what `--latent-scale` tunes.
fn spread(values: &Matrix) -> f32 {
    let count = values.data.len().max(1) as f32;
    let mean = values.data.iter().sum::<f32>() / count;
    (values
        .data
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f32>()
        / count)
        .sqrt()
}

fn adam(learning_rate: f32) -> Optimizer {
    Optimizer::Adam {
        learning_rate,
        beta1: 0.9,
        beta2: 0.95,
        epsilon: 1e-8,
        weight_decay: 0.0,
    }
}

fn step_at(step: usize) -> (String, String) {
    ("step".to_string(), step.to_string())
}

fn started_at(metadata: &BTreeMap<String, String>) -> usize {
    metadata
        .get("step")
        .and_then(|step| step.parse().ok())
        .unwrap_or(0)
}

fn parse() -> Result<Args, Box<dyn std::error::Error>> {
    let mut rest = std::env::args().skip(1);
    let stage = match rest.next().as_deref() {
        Some("vae") => Stage::Vae,
        Some("flow") => Stage::Flow,
        _ => return Err(usage().into()),
    };
    let mut args = Args {
        stage,
        tokens: PathBuf::from("tokens"),
        samples: PathBuf::from("samples"),
        vae: PathBuf::from("vae.safetensors"),
        out: PathBuf::from(match stage {
            Stage::Vae => "vae.safetensors",
            Stage::Flow => "flow.safetensors",
        }),
        resume: None,
        config: None,
        steps: 10_000,
        batch: 4,
        surface: 2048,
        queries: 4096,
        learning_rate: 1e-4,
        kl: 1e-4,
        clamp: 0.1,
        color: 0.1,
        color_points: 512,
        dropout: 0.1,
        latent_scale: 1.0,
        report: 50,
        checkpoint_every: 1000,
        seed: 0,
        cuda: None,
    };
    while let Some(flag) = rest.next() {
        let mut value = || rest.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--tokens" => args.tokens = value()?.into(),
            "--samples" => args.samples = value()?.into(),
            "--vae" => args.vae = value()?.into(),
            "--out" => args.out = value()?.into(),
            "--resume" => args.resume = Some(value()?.into()),
            "--config" => args.config = Some(value()?.into()),
            "--steps" => args.steps = value()?.parse()?,
            "--batch" => args.batch = value()?.parse()?,
            "--surface" => args.surface = value()?.parse()?,
            "--queries" => args.queries = value()?.parse()?,
            "--learning-rate" => args.learning_rate = value()?.parse()?,
            "--kl" => args.kl = value()?.parse()?,
            "--clamp" => args.clamp = value()?.parse()?,
            "--color" => args.color = value()?.parse()?,
            "--color-points" => args.color_points = value()?.parse()?,
            "--dropout" => args.dropout = value()?.parse()?,
            "--latent-scale" => args.latent_scale = value()?.parse()?,
            "--report" => args.report = value()?.parse()?,
            "--checkpoint-every" => args.checkpoint_every = value()?.parse()?,
            "--seed" => args.seed = value()?.parse()?,
            "--cuda" => args.cuda = Some(value()?.parse()?),
            other => return Err(format!("unknown flag {other}\n{}", usage()).into()),
        }
    }
    if args.batch == 0 || args.steps == 0 {
        return Err("--batch and --steps both need to be at least one".into());
    }
    if args.report == 0 || args.checkpoint_every == 0 {
        return Err("--report and --checkpoint-every are counted in steps, so at least one".into());
    }
    if !(0.0..=1.0).contains(&args.dropout) {
        return Err("--dropout is the chance of dropping a condition, so 0 to 1".into());
    }
    Ok(args)
}

fn usage() -> String {
    "usage: train_shape <vae|flow> [--tokens <dir>] [--samples <dir>] [--vae <file>] \
     [--out <file>] [--resume <file>] [--config <file.json>] [--steps n] [--batch n] [--surface n] [--queries n] \
     [--learning-rate f] [--kl f] [--clamp f] [--color f] [--color-points n] [--dropout f] \
     [--latent-scale f] [--report n] \
     [--checkpoint-every n] [--seed n] [--cuda device]"
        .to_string()
}
