use rusting_brain::{Activation, Dataset, Loss, Network, Optimizer, TrainConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = Dataset::new(
        vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
            vec![0.8, 0.8],
            vec![0.7, 0.9],
        ],
        vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0],
        ],
    );

    let mut model = Network::builder()
        .input_size(2)
        .dense(10, Activation::Relu)
        .dense(3, Activation::Softmax)
        .loss(Loss::CrossEntropy)
        .optimizer(Optimizer::adam(0.03))
        .build();

    model.fit(
        &dataset,
        TrainConfig {
            epochs: 1_000,
            batch_size: 6,
            shuffle: true,
            seed: Some(19),
        },
    )?;

    for input in &dataset.inputs {
        println!("{input:?} -> {:?}", model.predict(input)?);
    }

    Ok(())
}
