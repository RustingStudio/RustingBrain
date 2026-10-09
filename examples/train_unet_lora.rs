//! Trains a LoRA on a Stable Diffusion UNet (1.x shaped) and writes it as a
//! diffusers PEFT file.
//!
//! ```bash
//! cargo run --release --features cuda,images --example train_unet_lora -- \
//!     models/stable-diffusion-v1-5 pictures/ lora.safetensors
//! ```
//!
//! `pictures/` holds images, each with a caption beside it under the same
//! name and a `.txt` extension. Every image is encoded to a latent and every
//! caption to prompt states once, before training; the training itself runs
//! on the CUDA device, with adapters on `to_q`, `to_k`, `to_v` and `to_out.0`
//! of every attention, as diffusers' `train_text_to_image_lora` puts them.
//! The loss is the noise prediction's mean squared error at a training
//! timestep drawn uniformly.
//!
//! `STEPS` (500), `RANK` (4), `LR` (1e-4), `SIZE` (512, the square side
//! images are resized to) and `SEED` (0) override the defaults. The file
//! loads with `UnetLora::read_peft` at scale 1 and with diffusers'
//! `load_lora_weights`.

use rand::{Rng, SeedableRng};
use rusting_brain::cuda_image::{DeviceUnet, ImageGpu};
use rusting_brain::diffusion::Scheduler;
use rusting_brain::pipeline::{ImageDenoiser, ImagePipeline};
use rusting_brain::unet_lora::UnetLora;
use rusting_brain::{FeatureMap, Optimizer, read_image};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let [model, pictures, output] = arguments.as_slice() else {
        eprintln!("usage: train_unet_lora <model directory> <image folder> <output.safetensors>");
        std::process::exit(2);
    };
    let setting = |name: &str, default: f64| -> f64 {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    };
    let steps = setting("STEPS", 500.0) as usize;
    let rank = setting("RANK", 4.0) as usize;
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
        let latent = encoder.encode(&image)?;
        let conditioning = pipeline.condition(&std::fs::read_to_string(&caption)?, size, size)?;
        samples.push((latent, conditioning));
    }
    if samples.is_empty() {
        return Err("no image in the folder has a caption beside it".into());
    }
    println!("{} captioned images encoded", samples.len());
    let ImageDenoiser::Unet(unet) = pipeline.denoiser else {
        return Err("train_unet_lora trains a UNet, and this model has a transformer".into());
    };

    // Every training timestep's noise level, the clean end left off.
    let mut levels = Scheduler::ddim().sigmas(1000)?;
    levels.pop();
    let gpu = ImageGpu::new(0)?;
    let device = DeviceUnet::upload(&gpu, &unet)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut lora = UnetLora::new(&unet, rank, rank as f32, &mut rng);
    let optimizer = Optimizer::adam(learning_rate);
    let (mut recent, mut count) = (0.0, 0);
    for step in 1..=steps {
        let (latent, conditioning) = &samples[rng.gen_range(0..samples.len())];
        let noise = rusting_brain::diffusion::noise(1, latent.data.len(), Some(rng.r#gen()));
        let noise = FeatureMap::from_vec(latent.channels, latent.height, latent.width, noise.data)?;
        let sigma = levels[rng.gen_range(0..levels.len())];
        let loss = device.noise_step(
            &unet,
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
    lora.write_peft(output)?;
    println!("wrote {output}");
    Ok(())
}
