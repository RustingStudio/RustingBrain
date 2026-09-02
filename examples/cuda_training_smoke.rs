//! `cargo run --example cuda_training_smoke --features cuda`
use rusting_brain::{
    Activation, Dataset, Loss, Network, Optimizer, TrainConfig, TrainingBackend, cuda_doctor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let budget_mib = 8192;
    let doctor = cuda_doctor(0, budget_mib)?;
    println!(
        "CUDA device {}: {} (cc {})",
        doctor.device, doctor.name, doctor.compute_capability
    );
    println!(
        "memory: {} MiB total, {:?} MiB free; budget: {} MiB",
        doctor.total_memory_mib, doctor.free_memory_mib, doctor.requested_budget_mib
    );
    println!(
        "cuBLAS={}, kernels={}, allocation={}",
        doctor.cublas_available, doctor.kernel_available, doctor.allocation_test
    );

    let data = Dataset::new(
        (0..64).map(|i| vec![i as f32 / 64.0]).collect(),
        (0..64).map(|i| vec![2.0 * i as f32 / 64.0 + 1.0]).collect(),
    );
    let mut model = Network::builder()
        .input_size(1)
        .dense(8, Activation::Tanh)
        .dense(1, Activation::Linear)
        .loss(Loss::Mse)
        .optimizer(Optimizer::adam(0.03))
        .seed(7)
        .build();
    let history = model.fit_with_backend(
        &data,
        TrainConfig {
            epochs: 100,
            batch_size: 16,
            shuffle: true,
            seed: Some(7),
        },
        TrainingBackend::Cuda {
            device: 0,
            memory_budget_mib: budget_mib,
        },
    )?;
    println!(
        "CUDA training loss: {:.6} -> {:.6}; prediction at 0.5: {:?}",
        history.losses[0],
        history.losses.last().unwrap(),
        model.predict(&[0.5])?
    );
    Ok(())
}
