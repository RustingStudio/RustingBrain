//! Short end-to-end timing probe for the shape VAE CUDA training path.

use rand::{Rng, SeedableRng, rngs::StdRng};
use rusting_brain::{Matrix, Optimizer, ShapeVae, ShapeVaeConfig, losses, optimizers};
use std::time::Instant;

fn random_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Matrix {
    Matrix::from_vec(
        rows,
        cols,
        (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect(),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let batch = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let surface_rows = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);
    let query_rows = std::env::args()
        .nth(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let steps = std::env::args()
        .nth(4)
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let mixed = !std::env::args().any(|v| v == "fp32");
    let packed = !std::env::args().any(|v| v == "sequential");

    let mut rng = StdRng::seed_from_u64(7);
    let mut model = ShapeVae::new(ShapeVaeConfig::default(), &mut rng)?;
    model.to_cuda_with_precision(0, 0, mixed)?;
    let surfaces: Vec<_> = (0..batch)
        .map(|_| random_matrix(surface_rows, 6, &mut rng))
        .collect();
    let queries: Vec<_> = (0..batch)
        .map(|_| random_matrix(query_rows, 3, &mut rng))
        .collect();
    let targets: Vec<Vec<f32>> = queries
        .iter()
        .map(|matrix| {
            matrix
                .data
                .chunks_exact(3)
                .map(|p| p.iter().map(|v| v * v).sum::<f32>().sqrt() - 0.5)
                .collect()
        })
        .collect();
    let surface_batch = Matrix::from_vec(
        batch * surface_rows,
        6,
        surfaces
            .iter()
            .flat_map(|m| m.data.iter().copied())
            .collect(),
    );
    let query_batch = Matrix::from_vec(
        batch * query_rows,
        3,
        queries
            .iter()
            .flat_map(|m| m.data.iter().copied())
            .collect(),
    );
    let target_batch: Vec<f32> = targets.iter().flatten().copied().collect();
    let optimizer = Optimizer::adam(1e-4);

    let step = |number: usize, model: &mut ShapeVae, rng: &mut StdRng| {
        if packed {
            let (mean, log_variance, encoder) = model.encode_train_batch(&surface_batch, batch)?;
            let (latent, noise) = ShapeVae::sample(&mean, &log_variance, rng);
            let (distances, decoder) = model.decode_train_batch(&latent, &query_batch, batch)?;
            let (_, grad) = losses::clamped_l1(&distances, &target_batch, 0.1);
            let grad_latent = model.decode_backward(&decoder, &grad)?;
            let (grad_mean, grad_log_variance) =
                ShapeVae::sample_backward(&grad_latent, &log_variance, &noise);
            model.encode_backward(&encoder, &grad_mean, &grad_log_variance)?;
        } else {
            for ((surface, queries), targets) in surfaces.iter().zip(&queries).zip(&targets) {
                let (mean, log_variance, encoder) = model.encode_train(surface)?;
                let (latent, noise) = ShapeVae::sample(&mean, &log_variance, rng);
                let (distances, decoder) = model.decode_train(&latent, queries)?;
                let (_, grad) = losses::clamped_l1(&distances, targets, 0.1);
                let grad_latent = model.decode_backward(&decoder, &grad)?;
                let (grad_mean, grad_log_variance) =
                    ShapeVae::sample_backward(&grad_latent, &log_variance, &noise);
                model.encode_backward(&encoder, &grad_mean, &grad_log_variance)?;
            }
        }
        optimizers::step_clipped(
            &mut model.params_mut(),
            &optimizer,
            number,
            if packed { 1.0 } else { 1.0 / batch as f32 },
            1.0,
        )?;
        optimizers::zero_grad(&mut model.params_mut());
        Ok::<_, Box<dyn std::error::Error>>(())
    };

    step(1, &mut model, &mut rng)?;
    let started = Instant::now();
    for index in 0..steps {
        step(index + 2, &mut model, &mut rng)?;
    }
    model.to_cpu()?;
    let seconds = started.elapsed().as_secs_f64() / steps as f64;
    println!(
        "{} {} batch {batch}, surface {surface_rows}, queries {query_rows}: {seconds:.3} s/step ({:.3} s/shape)",
        if mixed { "mixed" } else { "fp32" },
        if packed { "packed" } else { "sequential" },
        seconds / batch as f64,
    );
    Ok(())
}
