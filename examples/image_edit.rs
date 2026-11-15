//! Edits a picture with FLUX.2-klein: the image is a reference the prompt
//! reworks, and the output is the reference's size.
//!
//! ```bash
//! CUDA=0 cargo run --release --features cuda,images --example image_edit -- \
//!     models/FLUX.2-klein-4B in.png "turn this into a watercolor painting" out.png
//! ```
//!
//! `RUNS` repeats the edit (default 1) and prints each stage's time, the
//! first being the cold one. `SIZE` is the side the reference is resized to
//! (default 512, a multiple of 16).

use rusting_brain::pipeline::ImagePipeline;
use rusting_brain::transformer::Precision;
use rusting_brain::vae::{from_image_bytes, save_png};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let [directory, input, prompt, output] = &arguments[..] else {
        eprintln!("usage: image_edit <model directory> <in.png> <prompt> <out.png>");
        std::process::exit(2);
    };
    let env = |name: &str| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
    };
    let (size, runs) = (env("SIZE").unwrap_or(512), env("RUNS").unwrap_or(1));

    let mut pipeline = ImagePipeline::load_at(directory, Precision::Q8)?;
    if let Some(device) = env("CUDA") {
        println!("device: {}", pipeline.try_cuda(device));
    }
    let reference = from_image_bytes(&std::fs::read(input)?, size, size)?;
    for run in 0..runs {
        let started = std::time::Instant::now();
        let lap = |what: &str, since: std::time::Instant| {
            println!("run {run}: {what} {:.3}s", since.elapsed().as_secs_f32())
        };
        let step = std::time::Instant::now();
        pipeline.set_references(std::slice::from_ref(&reference))?;
        lap("reference encode", step);
        let step = std::time::Instant::now();
        let conditioning = pipeline.condition(prompt, size, size)?;
        lap("prompt encode", step);
        let step = std::time::Instant::now();
        let latent = pipeline.sample_latent(conditioning, Some(0), |_, _| true)?;
        lap("sampling", step);
        let step = std::time::Instant::now();
        let (columns, rows) = pipeline.patch_grid(size, size)?;
        let image = pipeline.decode(&latent, columns, rows)?;
        lap("decode", step);
        lap("total", started);
        save_png(&image, output)?;
    }
    pipeline.set_references(&[])?;
    println!("wrote {output}");
    Ok(())
}
