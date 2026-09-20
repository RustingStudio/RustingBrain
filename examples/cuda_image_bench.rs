//! Times one UNet pass on the CPU and on a CUDA device, and says what the
//! device path cost in memory.
//!
//! ```bash
//! cargo run --release --features cuda --example cuda_image_bench -- \
//!     /home/vasylt/ml/models/pony-v6-xl
//! ```
//!
//! `SIZES` is a comma-separated list of square image sides, `512,1024` by
//! default. `QUANTIZE=1` holds the host copy at one byte per weight, which is
//! what makes an SDXL checkpoint fit in host memory; the device copy is BF16
//! either way. `CPU=0` skips the host timing, which at 1024x1024 is minutes.
//!
//! The pass being timed is the denoiser alone, without classifier-free
//! guidance. A sampling step with guidance runs it twice.

use rusting_brain::conv::FeatureMap;
use rusting_brain::cuda_image::memory_info;
use rusting_brain::pipeline::{ImageDenoiser, ImagePipeline};
use rusting_brain::transformer::Precision;

fn seconds(start: std::time::Instant) -> f32 {
    start.elapsed().as_secs_f32()
}

fn gigabytes(bytes: usize) -> f32 {
    bytes as f32 / (1024.0 * 1024.0 * 1024.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(directory) = std::env::args().nth(1) else {
        eprintln!("usage: cuda_image_bench <model directory>");
        std::process::exit(2);
    };
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "a lighthouse in fog, oil painting".to_string());
    let sizes: Vec<usize> = std::env::var("SIZES")
        .unwrap_or_else(|_| "512,1024".into())
        .split(',')
        .filter_map(|value| value.trim().parse().ok())
        .collect();
    let on_cpu = std::env::var("CPU").map_or(true, |value| value != "0");
    let precision = match std::env::var("QUANTIZE").is_ok() {
        true => Precision::Q8,
        false => Precision::F32,
    };

    let free_before = memory_info().map(|(free, _)| free);

    println!("loading {directory}");
    let started = std::time::Instant::now();
    let mut pipeline = ImagePipeline::load_at(&directory, precision)?;
    println!("loaded in {:.1}s", seconds(started));

    // The conditioning has to be built while the encoders are still on the
    // host, because the device timing below wants the same prompt.
    let mut conditioning = Vec::new();
    for size in &sizes {
        conditioning.push((*size, pipeline.condition(&prompt, *size, *size)?));
    }

    let latents: Vec<FeatureMap> = sizes
        .iter()
        .map(|size| {
            let side = size / pipeline.upscale();
            let channels = pipeline.denoiser.in_channels();
            FeatureMap::from_vec(
                channels,
                side,
                side,
                (0..channels * side * side)
                    .map(|index| ((index % 97) as f32 - 48.0) * 0.02)
                    .collect(),
            )
        })
        .collect::<Result<_, _>>()?;

    let mut host = Vec::new();
    if on_cpu {
        let ImageDenoiser::Unet(unet) = &pipeline.denoiser else {
            return Err("this model is not a UNet".into());
        };
        for ((size, conditioning), latent) in conditioning.iter().zip(&latents) {
            let started = std::time::Instant::now();
            let output = unet.forward(latent, 1.0, conditioning)?;
            println!(
                "cpu  {size}x{size}: {:.2}s ({} channels out)",
                seconds(started),
                output.channels
            );
            host.push(output);
        }
    }

    let started = std::time::Instant::now();
    pipeline.to_cuda(0)?;
    println!("uploaded in {:.1}s", seconds(started));
    if let (Some(before), Some((free, total))) = (free_before, memory_info()) {
        println!(
            "vram: {:.2} GB held, {:.2} GB free of {:.2} GB after the upload",
            gigabytes(before.saturating_sub(free)),
            gigabytes(free),
            gigabytes(total)
        );
    }

    let ImageDenoiser::Unet(unet) = &pipeline.denoiser else {
        return Err("this model is not a UNet".into());
    };
    let mut lowest = free_before.unwrap_or(0);
    for (index, ((size, conditioning), latent)) in conditioning.iter().zip(&latents).enumerate() {
        // The first pass on a device pays for cuBLAS picking its kernels, so
        // it is reported separately rather than averaged in.
        let started = std::time::Instant::now();
        unet.forward(latent, 1.0, conditioning)?;
        let warm_up = seconds(started);

        // The card's clock sags as it heats, so the fastest of a handful of
        // passes is a steadier number than any single one.
        let repeat: usize = std::env::var("REPEAT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1);
        let mut best = f32::MAX;
        let mut output = unet.forward(latent, 1.0, conditioning)?;
        for _ in 0..repeat {
            let started = std::time::Instant::now();
            output = unet.forward(latent, 1.0, conditioning)?;
            best = best.min(seconds(started));
        }
        println!(
            "cuda {size}x{size}: {best:.3}s (first pass {warm_up:.2}s, {} channels out)",
            output.channels
        );
        if let Some((free, _)) = memory_info() {
            lowest = lowest.min(free);
        }
        // The same latent and the same prompt through both paths. The bar is
        // relative to the largest value the host produced, because that is
        // what BF16's eight mantissa bits are relative to.
        if let Some(expected) = host.get(index) {
            let scale = expected
                .data
                .iter()
                .fold(0.0f32, |largest, value| largest.max(value.abs()))
                .max(1e-6);
            let worst = output
                .data
                .iter()
                .zip(&expected.data)
                .fold(0.0f32, |worst, (actual, expected)| {
                    worst.max((actual - expected).abs() / scale)
                });
            println!(
                "     {size}x{size}: {:.4} worst relative error against the CPU",
                worst
            );
        }
    }

    // The decoder runs once per image and at the image's own resolution, so
    // it is the one part whose scratch grows with the square of the side.
    for (size, latent) in sizes.iter().zip(&latents) {
        if let Some((free, _)) = memory_info() {
            println!("     {free} bytes free before the decode", free = free);
        }
        let _ = pipeline.decoder.decode(latent);
        let started = std::time::Instant::now();
        match pipeline.decoder.decode(latent) {
            Ok(image) => println!(
                "vae  {size}x{size}: {:.2}s ({}x{} out)",
                seconds(started),
                image.height,
                image.width
            ),
            Err(error) => println!("vae  {size}x{size}: {error}"),
        }
        if let Some((free, _)) = memory_info() {
            lowest = lowest.min(free);
        }
    }

    if let (Some(before), Some((_, total))) = (free_before, memory_info()) {
        println!(
            "peak vram: {:.2} GB of {:.2} GB",
            gigabytes(before.saturating_sub(lowest)),
            gigabytes(total)
        );
    }
    Ok(())
}
