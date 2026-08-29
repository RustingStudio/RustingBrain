use rusting_brain::{Activation, Dataset, Loss, Network, Optimizer, TrainConfig};

fn main() {
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

    let history = model
        .fit(
            &dataset,
            TrainConfig {
                epochs: 2_000,
                batch_size: 4,
                shuffle: true,
                seed: Some(42),
            },
        )
        .expect("training failed");

    println!("Final loss: {:.6}", history.losses.last().unwrap());
    for input in &dataset.inputs {
        let prediction = model.predict(input).expect("prediction failed");
        println!("{input:?} -> {:.4}", prediction[0]);
    }
}
