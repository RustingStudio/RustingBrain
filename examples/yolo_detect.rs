//! Fine-tunes YOLO11s on a YOLO-format dataset the way ultralytics' trainer
//! does with its defaults at `optimizer=auto` (Gate 2 of the detection track):
//! mosaic, scale and translate, HSV and flip augmentation, mosaic closed for
//! the last 10 epochs; AdamW at 0.002 with decay 5e-4 on the convolution
//! weights only, three warmup epochs, a linear decay to 1%, gradients
//! accumulated to a nominal batch of 64 and clipped at 10; an EMA of the
//! weights (0.9999, ramped over 2000 updates) that is what gets validated.
//!
//! ```bash
//! cargo run --release --features cuda,images --example yolo_detect -- \
//!     --weights /home/vasylt/Code/games/cs2/fixtures/k13d_yolo11s.safetensors \
//!     --data /home/vasylt/Code/games/cs2/dataset --epochs 100 --batch 8
//! ```
//!
//! `--steps N` stops after N batches and prints the time per stage, for
//! sizing a run before starting it. `--warmup N` shortens the warmup to N
//! batches, so such a run can time the steady state, where the optimizer
//! steps once every 64 samples.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rusting_brain::detection::{DetectionImage, mosaic, read_yolo_split};
use rusting_brain::ema::Ema;
use rusting_brain::optimizers::{Optimizer, apply_step, grad_norm, zero_grad};
use rusting_brain::param::{CudaDevice, Param};
use rusting_brain::safetensors::SafeTensors;
use rusting_brain::yolo::{Target, Yolo11, Yolo11Scale, detection_loss, input_batch, validate};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

const SIZE: usize = 1280;
const NOMINAL_BATCH: usize = 64;

struct Args {
    weights: PathBuf,
    data: PathBuf,
    epochs: usize,
    batch: usize,
    close_mosaic: usize,
    val_every: usize,
    steps: Option<usize>,
    warmup: Option<usize>,
    seed: u64,
}

fn args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        weights: "/home/vasylt/Code/games/cs2/fixtures/k13d_yolo11s.safetensors".into(),
        data: "/home/vasylt/Code/games/cs2/dataset".into(),
        epochs: 100,
        batch: 8,
        close_mosaic: 10,
        val_every: 10,
        steps: None,
        warmup: None,
        seed: 0,
    };
    let raw: Vec<String> = std::env::args().skip(1).collect();
    for pair in raw.chunks(2) {
        let [flag, value] = pair else {
            return Err(format!("{} has no value", pair[0]).into());
        };
        match flag.as_str() {
            "--weights" => args.weights = value.into(),
            "--data" => args.data = value.into(),
            "--epochs" => args.epochs = value.parse()?,
            "--batch" => args.batch = value.parse()?,
            "--close-mosaic" => args.close_mosaic = value.parse()?,
            "--val-every" => args.val_every = value.parse()?,
            "--steps" => args.steps = Some(value.parse()?),
            "--warmup" => args.warmup = Some(value.parse()?),
            "--seed" => args.seed = value.parse()?,
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    Ok(args)
}

/// One augmented 1280 square, as ultralytics' train transforms make it.
fn sample(
    images: &[DetectionImage],
    index: usize,
    mosaic_on: bool,
    rng: &mut StdRng,
) -> DetectionImage {
    let base = match mosaic_on {
        true => {
            let others: [usize; 3] = std::array::from_fn(|_| rng.gen_range(0..images.len()));
            let centre = [0; 2].map(|_| rng.gen_range(SIZE / 2..=3 * SIZE / 2));
            mosaic(
                [
                    &images[index],
                    &images[others[0]],
                    &images[others[1]],
                    &images[others[2]],
                ],
                SIZE,
                centre,
            )
        }
        false => images[index].letterbox(SIZE, SIZE, true).0,
    };
    let scale = rng.gen_range(0.5..1.5);
    let centre = [0; 2].map(|_| rng.gen_range(0.4..0.6) * SIZE as f32);
    let mut image = base.affine(SIZE, SIZE, scale, centre);
    image.hsv([0.015, 0.7, 0.4].map(|gain| rng.gen_range(-1.0..1.0) * gain + 1.0));
    if rng.r#gen::<f32>() < 0.5 {
        image.flip_horizontal();
    }
    image
}

/// The parameters by optimizer group: convolution weights decay, norms and
/// biases do not.
fn groups(model: &mut Yolo11) -> (Vec<&mut Param>, Vec<&mut Param>) {
    let (decayed, plain): (Vec<_>, Vec<_>) = model
        .state_mut()
        .into_iter()
        .filter(|(name, _)| !name.ends_with(".running"))
        .partition(|(name, _)| name.ends_with(".weight") && !name.ends_with(".bn.weight"));
    (
        decayed.into_iter().map(|(_, p)| p).collect(),
        plain.into_iter().map(|(_, p)| p).collect(),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = args()?;
    let device = CudaDevice::with_mixed_precision(0, 11000)?;
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut model = Yolo11::new(Yolo11Scale::S, 1, &mut rng)?;
    let skipped = model.transfer_ultralytics(&mut SafeTensors::open(&args.weights)?)?;
    println!("transferred all but {skipped:?}");
    model.set_recompute_columns(true);
    model.to_cuda_on(&device)?;

    let load = |split| -> Result<Vec<DetectionImage>, Box<dyn std::error::Error>> {
        Ok(read_yolo_split(&args.data, split)?
            .into_par_iter()
            .map(|(path, labels)| DetectionImage::open(&path, &labels, SIZE))
            .collect::<Result<Vec<_>, _>>()?)
    };
    let started = Instant::now();
    let train = Arc::new(load("train")?);
    let val = load("val")?;
    println!(
        "{} train and {} val images cached in {:.1} s",
        train.len(),
        val.len(),
        started.elapsed().as_secs_f64()
    );

    let batches = train.len().div_ceil(args.batch);
    let warmup = args.warmup.unwrap_or((3 * batches).max(100)).max(1);
    let full_accumulate = (NOMINAL_BATCH as f32 / args.batch as f32).round().max(1.0);
    let decay = 5e-4 * args.batch as f32 * full_accumulate / NOMINAL_BATCH as f32;
    let lr0 = 0.002;
    let lf = |epoch: usize| (1.0 - epoch as f32 / args.epochs as f32).max(0.0) * 0.99 + 0.01;
    let mut ema = Ema::new(&mut state(&mut model));
    let (mut optimizer_steps, mut last_step, mut seen) = (0usize, 0usize, 0usize);
    let mut timings = [Duration::ZERO; 4]; // data, forward, loss, backward and step
    let run = Instant::now();

    for epoch in 0..args.epochs {
        let mosaic_on = epoch + args.close_mosaic < args.epochs;
        let mut order: Vec<usize> = (0..train.len()).collect();
        order.shuffle(&mut rng);
        let chunks: Vec<Vec<usize>> = order.chunks(args.batch).map(<[usize]>::to_vec).collect();
        let make = |chunk: Vec<usize>, seed: u64| {
            let train = Arc::clone(&train);
            std::thread::spawn(move || {
                let images = chunk
                    .par_iter()
                    .enumerate()
                    .map(|(k, &i)| {
                        let mut rng = StdRng::seed_from_u64(seed ^ (k as u64) << 32);
                        sample(&train, i, mosaic_on, &mut rng)
                    })
                    .collect::<Vec<_>>();
                let input = input_batch(&images.iter().collect::<Vec<_>>());
                (images, input)
            })
        };
        let mut next = Some(make(chunks[0].clone(), rng.r#gen()));
        let mut sums = [0.0f64; 3];
        for i in 0..chunks.len() {
            let ni = i + batches * epoch;
            let accumulate = match ni <= warmup {
                true => (1.0 + (full_accumulate - 1.0) * ni as f32 / warmup as f32)
                    .round()
                    .max(1.0) as usize,
                false => full_accumulate as usize,
            };
            let lr = match ni <= warmup {
                true => lr0 * lf(epoch) * ni as f32 / warmup as f32,
                false => lr0 * lf(epoch),
            };

            let t = Instant::now();
            let (images, input) = next
                .take()
                .expect("a batch is always queued")
                .join()
                .expect("augmentation");
            if i + 1 < chunks.len() {
                next = Some(make(chunks[i + 1].clone(), rng.r#gen()));
            }
            let targets: Vec<Vec<Target>> = images
                .iter()
                .map(|im| {
                    im.classes
                        .iter()
                        .copied()
                        .zip(im.boxes.iter().copied())
                        .collect()
                })
                .collect();
            let input = device.upload(&input)?;
            timings[0] += t.elapsed();

            let t = Instant::now();
            let tape = model.forward_device(&input, true)?;
            let maps = model
                .outputs_device(&tape)
                .map(|m| m.download())
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            timings[1] += t.elapsed();

            let t = Instant::now();
            let (loss, grads) = detection_loss([&maps[0], &maps[1], &maps[2]], 1, &targets);
            let grads = grads
                .iter()
                .map(|g| device.upload(g))
                .collect::<Result<Vec<_>, _>>()?;
            timings[2] += t.elapsed();

            let t = Instant::now();
            model.backward_device(tape, [&grads[0], &grads[1], &grads[2]])?;
            if ni - last_step >= accumulate {
                optimizer_steps += 1;
                let (mut decayed, mut plain) = groups(&mut model);
                let norm = grad_norm(&mut decayed)?.hypot(grad_norm(&mut plain)?);
                if norm.is_finite() {
                    let clip = (10.0 / norm).min(1.0);
                    apply_step(
                        &mut decayed,
                        &Optimizer::adam_with_weight_decay(lr, decay),
                        optimizer_steps,
                        clip,
                    );
                    apply_step(&mut plain, &Optimizer::adam(lr), optimizer_steps, clip);
                } else {
                    println!("skipped a step: gradient norm {norm}");
                }
                zero_grad(&mut decayed);
                zero_grad(&mut plain);
                let d = 0.9999 * (1.0 - (-((ema.updates() + 1) as f32) / 2000.0).exp());
                ema.update(&mut state(&mut model), d)?;
                last_step = ni;
            }
            timings[3] += t.elapsed();

            for (sum, term) in sums.iter_mut().zip(loss) {
                *sum += term as f64;
            }
            seen += images.len();
            if args.steps.is_some_and(|steps| ni + 1 >= steps) {
                let per = |d: Duration| d.as_secs_f64() * 1e3 / (ni + 1) as f64;
                println!(
                    "{} batches of {}: {:.1} samples/s; per batch data {:.0} ms, forward {:.0} ms, loss {:.0} ms, backward and step {:.0} ms; loss {:?}",
                    ni + 1,
                    args.batch,
                    seen as f64 / run.elapsed().as_secs_f64(),
                    per(timings[0]),
                    per(timings[1]),
                    per(timings[2]),
                    per(timings[3]),
                    sums.map(|s| s / (i + 1) as f64),
                );
                return Ok(());
            }
        }
        let n = chunks.len() as f64;
        print!(
            "epoch {} box {:.3} cls {:.3} dfl {:.3} lr {:.6} {:.1} samples/s {:.0} s",
            epoch + 1,
            sums[0] / n,
            sums[1] / n,
            sums[2] / n,
            lr0 * lf(epoch),
            seen as f64 / run.elapsed().as_secs_f64(),
            run.elapsed().as_secs_f64(),
        );
        if (epoch + 1) % args.val_every == 0 || epoch + 1 == args.epochs {
            ema.swap_in(&mut state(&mut model))?;
            let v = validate(&mut model, &device, &val, SIZE)?;
            ema.swap_out(&mut state(&mut model))?;
            print!(" | EMA val mAP50 {:.4} mAP50-95 {:.4}", v.map50, v.map50_95);
        }
        println!();
    }
    println!(
        "{} epochs in {:.3} h",
        args.epochs,
        run.elapsed().as_secs_f64() / 3600.0
    );
    Ok(())
}

/// Every float the EMA averages, running statistics included, as
/// ultralytics' `ModelEMA` does.
fn state(model: &mut Yolo11) -> Vec<&mut Param> {
    model.state_mut().into_iter().map(|(_, p)| p).collect()
}
