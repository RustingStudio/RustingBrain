use rusting_brain::{Activation, Dataset, Loss, Network, Optimizer, TrainConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for i in 0..20 {
        for j in 0..20 {
            let x0 = (i as f32) / 20.0;
            let x1 = (j as f32) / 20.0;
            inputs.push(vec![x0, x1]);
            targets.push(vec![x0 + x1]);
        }
    }

    let dataset = Dataset::new(inputs, targets);
    let mut model = Network::builder()
        .input_size(2)
        .dense(8, Activation::Relu)
        .dense(1, Activation::Linear)
        .loss(Loss::Mse)
        .optimizer(Optimizer::adam(0.01))
        .build();

    let initial = model.evaluate_loss(&dataset)?;
    model.fit(
        &dataset,
        TrainConfig {
            epochs: 500,
            batch_size: 16,
            shuffle: true,
            seed: Some(3),
        },
    )?;
    let final_loss = model.evaluate_loss(&dataset)?;

    println!("initial loss: {initial:.6}");
    println!("final loss:   {final_loss:.6}");
    println!("[0.25, 0.50] -> {:.4}", model.predict(&[0.25, 0.50])?[0]);

    Ok(())
}
