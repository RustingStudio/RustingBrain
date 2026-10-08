//! Chapter 18: turn a prompt into a picture with a published diffusion model.
//!
//! Loads a Hugging Face diffusers directory (Stable Diffusion 1.x, 2.x, XL,
//! Stable Diffusion 3 or FLUX), moves it to the GPU if there is one, samples
//! an image with classifier-free guidance, writes it, and then repaints it
//! with a second prompt (image-to-image).
//!
//! Needs the `images` feature, and `cuda` for the GPU:
//!
//! ```bash
//! cargo run --release -- models/stable-diffusion-xl-base-1.0 "a lighthouse in fog"
//! ```

use rusting_brain::vae::save_png;
use rusting_brain::{ImagePipeline, Precision, Solver};

const SIZE: usize = 1024;
const SEED: u64 = 7;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let (Some(directory), Some(prompt)) = (arguments.next(), arguments.next()) else {
        eprintln!("usage: 18_text_to_image <model directory> <prompt>");
        std::process::exit(2);
    };

    // 18.2: one byte per weight for the big parts; the decoder stays f32.
    let started = std::time::Instant::now();
    let mut pipeline = ImagePipeline::load_at(&directory, Precision::Q8)?;
    println!("loaded in {:.1}s", started.elapsed().as_secs_f32());
    match pipeline.try_cuda(0) {
        true => println!("running on the GPU"),
        false => println!("no GPU, running on the CPU (slow)"),
    }

    // 18.3: the model's own settings, read from its scheduler file.
    println!(
        "model asks for {} steps, guidance {}, {:?}",
        pipeline.config.steps, pipeline.config.guidance, pipeline.config.solver
    );
    // A second-order solver gets there in half the steps at the same cost
    // per step.
    pipeline.config.solver = Solver::DpmPlusPlus2m;
    pipeline.config.steps = 25;

    // 18.4: the prompt and the one guidance pushes away from.
    let conditioning = pipeline.condition(&prompt, SIZE, SIZE)?;
    let unconditional = match pipeline.config.guidance > 1.0 {
        true => Some(pipeline.condition("blurry, low quality", SIZE, SIZE)?),
        false => None,
    };
    let steps = pipeline.config.steps;
    let started = std::time::Instant::now();
    let latent =
        pipeline.sample_latent_guided(conditioning, unconditional, Some(SEED), |step, _| {
            print!("\rstep {}/{steps}", step + 1);
            let _ = std::io::Write::flush(&mut std::io::stdout());
            true
        })?;
    let (columns, rows) = pipeline.patch_grid(SIZE, SIZE)?;
    let image = pipeline.decode(&latent, columns, rows)?;
    println!(
        "\n{SIZE}x{SIZE} in {:.1}s",
        started.elapsed().as_secs_f32()
    );
    save_png(&image, "image.png")?;
    println!("wrote image.png");

    // 18.5: image-to-image. Half the schedule is redone, so the layout stays
    // and the detail is repainted.
    let started = std::time::Instant::now();
    let repainted = pipeline.generate_from_image(
        &format!("{prompt}, oil painting"),
        &image,
        0.5,
        Some(SEED),
        |_, _| true,
    )?;
    println!(
        "repainted in {:.1}s",
        started.elapsed().as_secs_f32()
    );
    save_png(&repainted, "repainted.png")?;
    println!("wrote repainted.png");
    Ok(())
}
