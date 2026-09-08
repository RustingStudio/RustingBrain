use rusting_brain::{Activation, Dataset, Loss, Network, Optimizer, TrainConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = Dataset::new(
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
        .seed(42)
        .build();

    println!("Before training:");
    for input in &data.inputs {
        println!("  {:?} -> {:.4}", input, model.predict(input)?[0]);
    }

    let history = model.fit(
        &data,
        TrainConfig { epochs: 2_000, batch_size: 4, shuffle: true, seed: Some(11) },
    )?;

    println!("\nLoss during training:");
    for epoch in [0, 99, 499, 999, 1999] {
        println!("  epoch {:>4}: {:.6}", epoch + 1, history.losses[epoch]);
    }

    println!("\nAfter training:");
    for (input, target) in data.inputs.iter().zip(&data.targets) {
        let raw = model.predict(input)?[0];
        let decision = if raw > 0.5 { 1.0 } else { 0.0 };
        let mark = if (decision - target[0]).abs() < 0.001 { "OK" } else { "WRONG" };
        println!("  {:?} -> {:.4}  decision: {}  expected: {}  {}",
            input, raw, decision, target[0], mark);
    }

    // Same problem, no hidden layer.
    let mut flat = Network::builder()
        .input_size(2)
        .dense(1, Activation::Sigmoid)
        .loss(Loss::BinaryCrossEntropy)
        .optimizer(Optimizer::adam(0.05))
        .seed(42)
        .build();
    let flat_history = flat.fit(
        &data,
        TrainConfig { epochs: 2_000, batch_size: 4, shuffle: true, seed: Some(11) },
    )?;
    println!("\nNo hidden layer, same 2000 epochs:");
    println!("  final loss: {:.6}  (with hidden layer: {:.6})",
        flat_history.losses[1999], history.losses[1999]);
    for input in &data.inputs {
        println!("  {:?} -> {:.4}", input, flat.predict(input)?[0]);
    }

    Ok(())
}
