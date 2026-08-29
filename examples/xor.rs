use rusting_brain::{Activation, Dataset, Loss, Network, Optimizer, TrainConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = Dataset::new(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]],
    );

    let mut model = Network::builder()
        .input_size(2)
        .dense(8, Activation::Tanh)
        .dense(1, Activation::Sigmoid)
        .loss(Loss::BinaryCrossEntropy)
        .optimizer(Optimizer::adam(0.05))
        .build();

    model.fit(
        &dataset,
        TrainConfig {
            epochs: 2_000,
            batch_size: 4,
            shuffle: true,
            seed: Some(11),
        },
    )?;

    for input in &dataset.inputs {
        println!("{input:?} -> {:.4}", model.predict(input)?[0]);
    }

    Ok(())
}
