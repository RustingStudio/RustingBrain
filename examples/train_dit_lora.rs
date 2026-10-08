//! Trains a LoRA on a FLUX-style DiT and writes it as a diffusers PEFT file.
//!
//! ```bash
//! cargo run --release --features cuda,images --example train_dit_lora -- \
//!     models/tiny-flux pictures/ lora.safetensors
//! ```
//!
//! `pictures/` holds images, each with a caption beside it under the same
//! name and a `.txt` extension. Every image is encoded to a latent and every
//! caption to prompt states once, before training, and the encoders are then
//! dropped; the training itself runs on the CUDA device.
//!
//! `STEPS` (500), `RANK` (16), `LR` (1e-4), `SIZE` (512, the square side
//! images are resized to) and `SEED` (0) override the defaults. The file
//! loads with `ImagePipeline::load_with_loras` and with diffusers'
//! `load_lora_weights`.
//!
//! The base weights go to the device as BF16, so the model must fit there:
//! FLUX.1 at 12B does not fit a 12 GB card.

use rand::{Rng, SeedableRng};
use rusting_brain::cuda_mmdit::DeviceDit;
use rusting_brain::dit_lora::DitLora;
use rusting_brain::gpu_transformer::GpuContext;
use rusting_brain::pipeline::{ImageDenoiser, ImagePipeline};
use rusting_brain::{Optimizer, read_image};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let [model, pictures, output] = arguments.as_slice() else {
        eprintln!("usage: train_dit_lora <model directory> <image folder> <output.safetensors>");
        std::process::exit(2);
    };
    let setting = |name: &str, default: f64| -> f64 {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    };
    let steps = setting("STEPS", 500.0) as usize;
    let rank = setting("RANK", 16.0) as usize;
    let learning_rate = setting("LR", 1e-4) as f32;
    let size = setting("SIZE", 512.0) as usize;
    let seed = setting("SEED", 0.0) as u64;

    let pipeline = ImagePipeline::load(model)?;
    let encoder = pipeline
        .image_encoder
        .as_ref()
        .ok_or("this checkpoint's autoencoder holds no encoder")?;
    let mut samples = Vec::new();
    let mut paths: Vec<_> = std::fs::read_dir(pictures)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<_, _>>()?;
    paths.sort();
    for path in paths {
        let caption = path.with_extension("txt");
        if path.extension().is_none_or(|extension| extension == "txt") || !caption.exists() {
            continue;
        }
        let mut image = read_image(&path, size, 1.0)?;
        // The autoencoder reads pixels in [-1, 1].
        image
            .data
            .iter_mut()
            .for_each(|value| *value = *value * 2.0 - 1.0);
        let latent = pipeline.pack(&encoder.encode(&image)?)?;
        let conditioning = pipeline.condition(&std::fs::read_to_string(&caption)?, size, size)?;
        samples.push((latent, conditioning));
    }
    if samples.is_empty() {
        return Err("no image in the folder has a caption beside it".into());
    }
    println!("{} captioned images encoded", samples.len());
    let ImageDenoiser::Transformer(dit) = pipeline.denoiser else {
        return Err("train_dit_lora trains a DiT, and this model has a UNet".into());
    };

    let context = GpuContext::new(0)?;
    let device = DeviceDit::upload(&context, &dit)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut lora = DitLora::new(&dit, rank, rank as f32, &mut rng);
    lora.move_to_cuda(&context)?;
    let optimizer = Optimizer::adam(learning_rate);
    let (mut recent, mut count) = (0.0, 0);
    for step in 1..=steps {
        let (latent, conditioning) = &samples[rng.gen_range(0..samples.len())];
        let noise = rusting_brain::diffusion::noise(latent.rows, latent.cols, Some(rng.r#gen()));
        // NOTE: uniform noise levels; FLUX's own recipe samples them
        // logit-normal, worth it once a run shows the extremes wasting steps.
        let sigma = rng.gen_range(0.0..1.0);
        let loss = device.flow_step(
            &dit,
            &mut lora,
            latent,
            &noise,
            sigma,
            conditioning,
            &optimizer,
            step,
        )?;
        recent += loss;
        count += 1;
        if step % 50 == 0 || step == steps {
            println!("step {step}: loss {:.4}", recent / count as f32);
            (recent, count) = (0.0, 0);
        }
    }
    lora.move_to_cpu()?;
    lora.write_peft(output, &dit)?;
    println!("wrote {output}");
    Ok(())
}
