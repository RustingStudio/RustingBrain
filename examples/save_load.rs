use rusting_brain::{Activation, Dataset, Loss, Network, Optimizer, TrainConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = Dataset::new(
        vec![vec![0.0], vec![0.5], vec![1.0]],
        vec![vec![0.0], vec![1.0], vec![2.0]],
    );

    let mut model = Network::builder()
        .input_size(1)
        .dense(6, Activation::Relu)
        .dense(1, Activation::Linear)
        .loss(Loss::Mse)
        .optimizer(Optimizer::adam(0.01))
        .build();

    model.fit(
        &dataset,
        TrainConfig {
            epochs: 500,
            batch_size: 3,
            shuffle: true,
            seed: Some(5),
        },
    )?;

    let path = std::env::temp_dir().join("rusting_brain_example_model.json");
    model.save_json(&path)?;

    let loaded = Network::load_json(&path)?;
    println!("saved to {}", path.display());
    println!("[0.25] -> {:.4}", loaded.predict(&[0.25])?[0]);

    Ok(())
}
