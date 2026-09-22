//! One image in, one GLB out: the whole pipeline end to end.
//!
//! ```bash
//! cargo run --release --features images --example image_to_3d -- \
//!     --clip clip-vit-base-patch16/model.safetensors \
//!     --vae vae.safetensors --flow flow.safetensors \
//!     --image render.png --out shape.glb
//! ```
//!
//! The three checkpoints are the three stages: the frozen CLIP tower that
//! `vit_tokens` uses, and the two `train_shape` wrote. The image is encoded
//! once, the flow transformer samples a shape latent from noise conditioned on
//! those tokens, the autoencoder's decoder turns the latent into a signed
//! distance field, marching tetrahedra turns the field into triangles, and the
//! result is simplified and written as a GLB. The decoder's colour head is
//! asked for a colour at every surviving vertex, which the GLB carries as its
//! `COLOR_0` attribute — a texture stand-in, not a texture.
//!
//! The mesh comes out in the unit cube the autoencoder trained in. The
//! per-asset centre and scale that would put it back in a source asset's frame
//! is what `mesh_samples` wrote as `{stem}.transform`, and nothing here needs
//! it: a generated shape has no source asset.
//!
//! `--guidance` above 1.0 runs the model twice per step, once on the image and
//! once on the learned null condition, and pushes away from the second. It
//! costs what it says: two model reads per step instead of one.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rusting_brain::flow_transformer::scheduler;
use rusting_brain::{
    FlowDenoiser, FlowTransformer, Matrix, Precision, SamplingConfig, ShapeVae, Solver, VitEncoder,
    VitEncoderConfig, marching_tetrahedra, noise, read_image, sample,
};
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    clip: PathBuf,
    prefix: String,
    heads: usize,
    vae: PathBuf,
    flow: PathBuf,
    image: PathBuf,
    out: PathBuf,
    steps: usize,
    guidance: f32,
    shift: f32,
    resolution: usize,
    faces: usize,
    chunk: usize,
    extent: f32,
    background: f32,
    seed: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse()?;
    let mut rng = StdRng::seed_from_u64(args.seed);

    let tower = VitEncoder::load(
        &args.clip,
        &args.prefix,
        VitEncoderConfig {
            num_heads: args.heads,
            ..VitEncoderConfig::default()
        },
        Precision::F32,
    )?;
    let tokens = tower.encode(&read_image(
        &args.image,
        tower.image_size(),
        args.background,
    )?)?;
    println!(
        "{}: [{}, {}] tokens",
        args.image.display(),
        tokens.rows,
        tokens.cols
    );

    let (model, metadata) = FlowTransformer::load(&args.flow, &mut rng)?;
    let config = *model.config();
    if tokens.rows != config.cond_tokens || tokens.cols != config.cond_dim {
        return Err(format!(
            "the flow transformer wants [{}, {}] tokens and this tower gives [{}, {}]",
            config.cond_tokens, config.cond_dim, tokens.rows, tokens.cols
        )
        .into());
    }
    // The scale training multiplied the latent by, which sampling divides out.
    let latent_scale: f32 = metadata
        .get("latent_scale")
        .and_then(|value| value.parse().ok())
        .unwrap_or(1.0);

    let started = Instant::now();
    // The sampler borrows the model, so the borrow is scoped: the transformer
    // is the largest thing here and the decoder is loaded once it is gone.
    let mut latent = sample(
        &mut FlowDenoiser {
            model: &model,
            tokens: &tokens,
        },
        scheduler(args.shift),
        noise(config.latents, config.latent_dim, Some(args.seed)),
        &SamplingConfig {
            steps: args.steps,
            solver: Solver::Euler,
            guidance: args.guidance,
            seed: Some(args.seed),
        },
        |step, _| {
            print!("\rsampling {}/{}", step + 1, args.steps);
            let _ = std::io::Write::flush(&mut std::io::stdout());
            true
        },
    )?;
    println!("\rsampled in {:.1}s", started.elapsed().as_secs_f32());
    if latent_scale != 1.0 {
        for value in &mut latent.data {
            *value /= latent_scale;
        }
    }
    drop(model);

    let (vae, _) = ShapeVae::load(&args.vae, &mut rng)?;
    let started = Instant::now();
    let mut mesh = marching_tetrahedra(
        vae.field(&latent, args.chunk),
        args.resolution,
        ([-args.extent; 3], [args.extent; 3]),
        0.0,
    )?;
    println!(
        "field {0}x{0}x{0}: {1} vertices, {2} faces in {3:.1}s",
        args.resolution,
        mesh.positions.len(),
        mesh.indices.len(),
        started.elapsed().as_secs_f32()
    );
    if mesh.indices.is_empty() {
        return Err(
            "the field never crossed zero, so there is no surface: an undertrained model, or a \
             shape outside --extent"
                .into(),
        );
    }

    if args.faces > 0 && mesh.indices.len() > args.faces {
        let kept = mesh.simplify(args.faces);
        println!("simplified to {kept} faces");
    }
    if mesh.normals.is_empty() {
        mesh.recompute_normals();
    }

    // After the simplification, so the colours are asked for at the vertices
    // the file will actually hold.
    let vertices = Matrix::from_vec(mesh.positions.len(), 3, mesh.positions.concat());
    mesh.colors = vae.decode_colors(&latent, &vertices, args.chunk)?;

    mesh.write_glb(&args.out)?;
    println!("wrote {}", args.out.display());
    Ok(())
}

fn parse() -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        clip: PathBuf::new(),
        prefix: String::new(),
        heads: 12,
        vae: PathBuf::from("vae.safetensors"),
        flow: PathBuf::from("flow.safetensors"),
        image: PathBuf::new(),
        out: PathBuf::from("shape.glb"),
        steps: 28,
        guidance: 4.0,
        shift: 1.0,
        resolution: 128,
        faces: 50_000,
        chunk: 65_536,
        extent: 1.0,
        background: 1.0,
        seed: 0,
    };
    let mut rest = std::env::args().skip(1);
    while let Some(flag) = rest.next() {
        let mut value = || rest.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--clip" => args.clip = value()?.into(),
            "--prefix" => args.prefix = value()?,
            "--heads" => args.heads = value()?.parse()?,
            "--vae" => args.vae = value()?.into(),
            "--flow" => args.flow = value()?.into(),
            "--image" => args.image = value()?.into(),
            "--out" => args.out = value()?.into(),
            "--steps" => args.steps = value()?.parse()?,
            "--guidance" => args.guidance = value()?.parse()?,
            "--shift" => args.shift = value()?.parse()?,
            "--resolution" => args.resolution = value()?.parse()?,
            "--faces" => args.faces = value()?.parse()?,
            "--chunk" => args.chunk = value()?.parse()?,
            "--extent" => args.extent = value()?.parse()?,
            "--background" => args.background = value()?.parse()?,
            "--seed" => args.seed = value()?.parse()?,
            other => return Err(format!("unknown flag {other}\n{}", usage()).into()),
        }
    }
    if args.clip.as_os_str().is_empty() || args.image.as_os_str().is_empty() {
        return Err(usage().into());
    }
    if args.resolution < 2 {
        return Err("--resolution is the marching grid, so at least two".into());
    }
    if args.chunk == 0 || args.steps == 0 {
        return Err("--chunk and --steps both need to be at least one".into());
    }
    if args.extent <= 0.0 {
        return Err("--extent is the half-width of the cube the field is read over".into());
    }
    Ok(args)
}

fn usage() -> String {
    "usage: image_to_3d --clip <file> --image <file> [--prefix s] [--heads n] [--vae <file>] \
     [--flow <file>] [--out <file>] [--steps n] [--guidance f] [--shift f] [--resolution n] \
     [--faces n] [--chunk n] [--extent f] [--background f] [--seed n]"
        .to_string()
}
