use rusting_brain::{
    Activation, CudaTrainingSession, Dataset, Network, Optimizer, TrainConfig, TrainingBackend,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = Dataset::new(
        (0..4096)
            .map(|r| {
                (0..16)
                    .map(|c| ((r * 17 + c * 11) % 101) as f32 / 101.0)
                    .collect()
            })
            .collect(),
        (0..4096).map(|r| vec![(r % 23) as f32 / 23.0]).collect(),
    );
    let base = Network::builder()
        .input_size(16)
        .dense(32, Activation::Relu)
        .dense(1, Activation::Linear)
        .optimizer(Optimizer::adam(0.001))
        .seed(42)
        .build();
    let one_epoch = TrainConfig {
        epochs: 1,
        batch_size: 256,
        shuffle: true,
        seed: Some(7),
    };
    let backend = TrainingBackend::Cuda {
        device: 0,
        memory_budget_mib: 8192,
    };

    let mut repeated = base.clone();
    let old_started = Instant::now();
    for _ in 0..10 {
        repeated.fit_with_backend(&data, one_epoch, backend)?;
    }
    let old_elapsed = old_started.elapsed();

    let mut session = CudaTrainingSession::new(&base, &data, one_epoch, 0, 8192)?;
    let new_started = Instant::now();
    for _ in 0..10 {
        session.train_epoch()?;
        let _checkpoint = session.checkpoint()?;
    }
    let new_elapsed = new_started.elapsed();
    println!("repeated fit setup: {old_elapsed:?}");
    println!("persistent session: {new_elapsed:?}");
    println!(
        "speedup: {:.2}x",
        old_elapsed.as_secs_f64() / new_elapsed.as_secs_f64()
    );
    println!("stats: {:?}", session.stats());
    Ok(())
}
