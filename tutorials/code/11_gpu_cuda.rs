//! Chapter 11 — the same network on the CPU and on the GPU, timed.
//!
//!     cargo run --release --features cuda --bin 11_gpu_cuda
//!
//! Without `--features cuda` this still builds and runs; it reports that the
//! GPU section was skipped rather than failing to compile.

use rusting_brain::{
    Activation, Dataset, Network, NetworkError, Optimizer, TrainConfig, TrainingBackend,
    TrainingSession, accelerator_doctor, estimate_tensor_memory_mib,
};
use std::time::Instant;

const BUDGET_MIB: usize = 8192;
const BATCH_SIZE: usize = 256;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = synthetic_dataset(8192, 64);
    let base = Network::builder()
        .input_size(64)
        .dense(256, Activation::Relu)
        .dense(256, Activation::Relu)
        .dense(1, Activation::Linear)
        .optimizer(Optimizer::adam(0.001))
        .seed(42)
        .build();
    let config = TrainConfig {
        epochs: 20,
        batch_size: BATCH_SIZE,
        shuffle: true,
        seed: Some(7),
    };

    println!(
        "tensors for batch {BATCH_SIZE}: {} MiB",
        estimate_tensor_memory_mib(&base, BATCH_SIZE)?
    );

    let backend = TrainingBackend::Cuda {
        device: 0,
        memory_budget_mib: BUDGET_MIB,
    };

    // 11.3 — ask the device before committing a run to it.
    match accelerator_doctor(backend) {
        Ok(Some(report)) => {
            println!(
                "{} ({}), {} MiB total, {:?} MiB free",
                report.name, report.revision, report.total_memory_mib, report.free_memory_mib
            );
            println!(
                "cuBLAS={}, kernels={}, allocation={}",
                report.gemm_available, report.kernel_available, report.allocation_test
            );
        }
        Ok(None) => unreachable!("Cuda backend never reports None"),
        Err(NetworkError::CudaFeatureDisabled) => {
            println!("built without `--features cuda`; running the CPU half only");
            let mut cpu = base.clone();
            let started = Instant::now();
            cpu.fit(&data, config)?;
            println!("cpu: {:?}", started.elapsed());
            return Ok(());
        }
        Err(other) => return Err(other.into()),
    }

    // 11.7 — the only honest way to decide: time both, in release mode.
    let mut cpu = base.clone();
    let cpu_started = Instant::now();
    cpu.fit(&data, config)?;
    let cpu_time = cpu_started.elapsed();

    let mut gpu = base.clone();
    let gpu_started = Instant::now();
    gpu.fit_with_backend(&data, config, backend)?;
    let gpu_time = gpu_started.elapsed();

    println!(
        "cpu {cpu_time:?}, gpu {gpu_time:?}, speedup {:.2}x",
        cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
    );

    // 11.6 — one warm session, checkpointing only when the loss improves.
    let one_epoch = TrainConfig { epochs: 1, ..config };
    let mut model = base.clone();
    let mut session = TrainingSession::new(&model, &data, one_epoch, backend)?
        .expect("the Cuda backend always opens a session");

    let mut best = f32::INFINITY;
    for epoch in 0..20 {
        let loss = session.train_epoch()?;
        if loss < best {
            best = loss;
            model.restore_cuda_checkpoint(session.checkpoint()?)?;
            model.save_json("best.json")?;
            println!("epoch {epoch}: loss {loss:.6} (checkpointed)");
        } else {
            println!("epoch {epoch}: loss {loss:.6}");
        }
    }

    let stats = session.stats();
    println!(
        "peak {} MiB, setup {:?}, training {:?}, checkpointing {:?}, dataset resident: {}",
        stats.peak_allocated_bytes / (1024 * 1024),
        stats.setup_time,
        stats.training_time,
        stats.checkpoint_time,
        stats.dataset_resident
    );
    Ok(())
}

/// A learnable but non-trivial regression target, big enough that the GPU has
/// something to do.
fn synthetic_dataset(rows: usize, features: usize) -> Dataset {
    let inputs: Vec<Vec<f32>> = (0..rows)
        .map(|r| {
            (0..features)
                .map(|c| ((r * 31 + c * 17) % 101) as f32 / 101.0)
                .collect()
        })
        .collect();
    let targets = inputs
        .iter()
        .map(|row| vec![row.iter().take(8).sum::<f32>() / 8.0])
        .collect();
    Dataset::new(inputs, targets)
}
