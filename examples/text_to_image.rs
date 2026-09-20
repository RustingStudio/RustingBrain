//! Hosts a published text-to-image model and writes a PNG.
//!
//! ```bash
//! cargo run --release --features images --example text_to_image -- \
//!     models/FLUX.2-klein-4B "a lighthouse in fog" out.png
//! ```
//!
//! The model directory is the one a Hugging Face repository has: a
//! `transformer`, a `vae`, a `text_encoder` and a `tokenizer` beside each
//! other. Nothing about the model is hard-coded here — the shapes come from
//! the files.
//!
//! Set `QUANTIZE=1` to hold the denoiser and the encoder at one byte per
//! weight, which is a quarter of the memory.
//!
//! `STEPS` and `GUIDANCE` override what the model asked for, which is what
//! lets a run be lined up against another implementation's. `NEGATIVE` is the
//! prompt a guided run pushes away from, empty by default.
//!
//! Set `CUDA=0` (or any device index) to run the UNet, the VAE decoder and the
//! CLIP towers on that device. A build without the `cuda` feature, or a
//! machine with no device, ignores it and runs on the CPU.
//!
//! The parts are dropped as the run passes them, which is what lets a model
//! larger than memory finish: the encoder goes once the prompt is encoded, and
//! the denoiser goes once the latent is sampled.

use rusting_brain::pipeline::ImagePipeline;
use rusting_brain::transformer::Precision;
use rusting_brain::vae::save_png;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let (directory, prompt, output) = match (
        arguments.next(),
        arguments.next(),
        arguments.next().unwrap_or_else(|| "image.png".into()),
    ) {
        (Some(directory), Some(prompt), output) => (directory, prompt, output),
        _ => {
            eprintln!("usage: text_to_image <model directory> <prompt> [output.png]");
            std::process::exit(2);
        }
    };
    let width: usize = std::env::var("WIDTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(512);
    let height: usize = std::env::var("HEIGHT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(512);
    let seed: u64 = std::env::var("SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);

    // A 4B model is 16 GB in f32 and 4 GB a byte per weight, so the switch is
    // what decides whether it fits at all.
    let precision = match std::env::var("QUANTIZE").is_ok() {
        true => Precision::Q8,
        false => Precision::F32,
    };

    println!("loading {directory}");
    let started = std::time::Instant::now();
    let mut pipeline = ImagePipeline::load_at(&directory, precision)?;
    println!("loaded in {:.1}s", started.elapsed().as_secs_f32());

    // Opt-in, and never fatal: a device that is not there leaves the run
    // exactly where it was.
    if let Some(device) = std::env::var("CUDA").ok().and_then(|v| v.parse().ok()) {
        match pipeline.try_cuda(device) {
            true => println!("running on CUDA device {device}"),
            false => println!("no usable CUDA device {device}, staying on the CPU"),
        }
    }

    // The defaults come from the model. These are here so a run can be lined
    // up against another implementation's, which is the only way the timing
    // below means anything.
    if let Some(steps) = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()) {
        pipeline.config.steps = steps;
    }
    if let Some(guidance) = std::env::var("GUIDANCE").ok().and_then(|v| v.parse().ok()) {
        pipeline.config.guidance = guidance;
    }

    let conditioning = pipeline.condition(&prompt, width, height)?;
    // Classifier-free guidance denoises a second prompt alongside the real one
    // and steps along the difference, so a guidance above one costs two
    // denoiser passes a step. A model that asked for guidance gets it here,
    // pushing away from `NEGATIVE` or from the empty prompt.
    let unconditional = match pipeline.config.guidance > 1.0 {
        true => {
            let negative = std::env::var("NEGATIVE").unwrap_or_default();
            Some(pipeline.condition(&negative, width, height)?)
        }
        false => None,
    };
    let steps = pipeline.config.steps;
    let started = std::time::Instant::now();
    let latent =
        pipeline.sample_latent_guided(conditioning, unconditional, Some(seed), |step, _| {
            print!("\rstep {}/{steps}", step + 1);
            let _ = std::io::Write::flush(&mut std::io::stdout());
            true
        })?;

    let (columns, rows) = pipeline.patch_grid(width, height)?;
    let image = pipeline.decode(&latent, columns, rows)?;
    println!(
        "\n{width}x{height} in {:.1}s",
        started.elapsed().as_secs_f32()
    );

    save_png(&image, &output)?;
    println!("wrote {output}");
    Ok(())
}
