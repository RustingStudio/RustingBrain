use std::{thread::available_parallelism, time::Instant};

use crate::network::Network;

pub fn tensorflow_like_example() {
    let input_size = 512usize;
    let hidden_sizes = [1024usize, 1024usize, 512usize];
    let output_size = 10usize;
    let batch_size = 256usize;
    let samples = 20_000usize;
    let iterations = 50usize;

    let mut layers = Vec::new();
    layers.push(input_size);
    layers.extend_from_slice(&hidden_sizes);
    layers.push(output_size);

    let mut net = Network::new(layers, 0.01);

    let true_w = crate::matrix::Matrix::random(output_size, input_size);
    let target_scale = 0.01f32;

    let mut inputs = Vec::with_capacity(samples);
    let mut targets = Vec::with_capacity(samples);

    for _ in 0..samples {
        let mut x = Vec::with_capacity(input_size);
        for _ in 0..input_size {
            x.push(rand::random::<f32>());
        }

        let mut x_matrix = crate::matrix::Matrix::new(input_size, 1);
        x_matrix.copy_from_slice(&x);

        let mut y_matrix = crate::matrix::Matrix::new(output_size, 1);
        true_w.dot(&x_matrix, &mut y_matrix);

        for v in &mut y_matrix.data {
            *v *= target_scale;
        }

        inputs.push(x);
        targets.push(y_matrix.data.clone());
    }

    let num_threads = available_parallelism().map(|n| n.get()).unwrap_or(1);

    let start = Instant::now();

    for i in 0..iterations {
        println!("Iteration: {}", i + 1);
        let mut start_idx = 0usize;
        while start_idx < samples {
            let end_idx = (start_idx + batch_size).min(samples);
            let batch_inputs = &inputs[start_idx..end_idx];
            let batch_targets = &targets[start_idx..end_idx];

            net.train_batch_parallel(batch_inputs, batch_targets, num_threads);

            start_idx = end_idx;
        }
    }

    let duration = start.elapsed();

    let mut test_input = Vec::with_capacity(input_size);
    for _ in 0..input_size {
        test_input.push(rand::random::<f32>());
    }
    let mut test_x_matrix = crate::matrix::Matrix::new(input_size, 1);
    test_x_matrix.copy_from_slice(&test_input);
    let mut test_y_matrix = crate::matrix::Matrix::new(output_size, 1);
    true_w.dot(&test_x_matrix, &mut test_y_matrix);
    for v in &mut test_y_matrix.data {
        *v *= target_scale;
    }
    let expected = test_y_matrix.data.clone();
    let output = net.forward(&test_input);

    println!("Large model test (Rust NN):");
    println!("Input len: {}", test_input.len());
    println!("Output len: {}", output.len());
    println!("Time taken (seconds): {:.6}", duration.as_secs_f64());
    println!("Example test target (first 3):   {:?}", &expected[0..3]);
    println!("Example test predicted (first 3): {:?}", &output[0..3]);

    for i in 0..5 {
        let out = net.forward(&inputs[i]);
        println!("Target:  {:?}", &targets[i][0..3]);
        println!("Predicted: {:?}", &out[0..3]);
        println!();
    }
}

pub fn learning_sanity_test() {
    let input_size = 2usize;
    let hidden_sizes = [8usize];
    let output_size = 1usize;
    let batch_size = 32usize;
    let samples = 1024usize;
    let epochs = 50usize;

    let mut layers = Vec::new();
    layers.push(input_size);
    layers.extend_from_slice(&hidden_sizes);
    layers.push(output_size);

    let mut net = Network::new(layers, 0.05);

    let mut inputs = Vec::with_capacity(samples);
    let mut targets = Vec::with_capacity(samples);

    for _ in 0..samples {
        let x0 = rand::random::<f32>();
        let x1 = rand::random::<f32>();
        let y = x0 + x1;

        inputs.push(vec![x0, x1]);
        targets.push(vec![y]);
    }

    fn mse(net: &mut Network, inputs: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for (x, t) in inputs.iter().zip(targets.iter()) {
            let out = net.forward(x);
            for (o, &tt) in out.iter().zip(t.iter()) {
                let diff = tt - o;
                sum += diff * diff;
                count += 1;
            }
        }
        sum / (count as f32)
    }

    let mut start_idx = 0usize;
    let mut end_idx = batch_size.min(samples);
    let initial_loss = mse(&mut net, &inputs[start_idx..end_idx], &targets[start_idx..end_idx]);

    for _ in 0..epochs {
        let mut start_idx = 0usize;
        while start_idx < samples {
            let end_idx = (start_idx + batch_size).min(samples);
            let batch_inputs = &inputs[start_idx..end_idx];
            let batch_targets = &targets[start_idx..end_idx];
            net.train_batch_parallel(batch_inputs, batch_targets, 1);
            start_idx = end_idx;
        }
    }

    let final_loss = mse(&mut net, &inputs[start_idx..end_idx], &targets[start_idx..end_idx]);

    println!("Learning sanity test (y = x0 + x1):");
    println!("Initial MSE: {}", initial_loss);
    println!("Final   MSE: {}", final_loss);
}

pub fn large_model_learning_test() {
    let input_size = 512usize;
    let output_size = 10usize;
    let batch_size = 256usize;
    let samples = 40_000usize;
    let epochs = 200usize;

    let mut layers = Vec::new();
    layers.push(input_size);
    layers.push(output_size);

    let mut net = Network::new(layers, 0.01);

    let true_w = crate::matrix::Matrix::random(output_size, input_size);
    let target_scale = 0.01f32;

    let mut inputs = Vec::with_capacity(samples);
    let mut targets = Vec::with_capacity(samples);

    for _ in 0..samples {
        let mut x = Vec::with_capacity(input_size);
        for _ in 0..input_size {
            x.push(rand::random::<f32>());
        }

        let mut x_matrix = crate::matrix::Matrix::new(input_size, 1);
        x_matrix.copy_from_slice(&x);

        let mut y_matrix = crate::matrix::Matrix::new(output_size, 1);
        true_w.dot(&x_matrix, &mut y_matrix);

        for v in &mut y_matrix.data {
            *v *= target_scale;
        }

        inputs.push(x);
        targets.push(y_matrix.data.clone());
    }

    fn mse(net: &mut Network, inputs: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for (x, t) in inputs.iter().zip(targets.iter()) {
            let out = net.forward(x);
            for (o, &tt) in out.iter().zip(t.iter()) {
                let diff = tt - o;
                sum += diff * diff;
                count += 1;
            }
        }
        sum / (count as f32)
    }

    let initial_loss = mse(&mut net, &inputs[0..batch_size], &targets[0..batch_size]);

    for epoch in 0..epochs {
        let mut start = 0usize;
        while start < samples {
            let end = (start + batch_size).min(samples);
            net.train_batch_parallel(&inputs[start..end], &targets[start..end], 1);
            start = end;
        }

        let loss = mse(&mut net, &inputs[0..batch_size], &targets[0..batch_size]);
        println!("Epoch {} loss: {}", epoch + 1, loss);
    }

    let final_loss = mse(&mut net, &inputs[0..batch_size], &targets[0..batch_size]);

    println!("Large model linear mapping test:");
    println!("Initial MSE: {}", initial_loss);
    println!("Final   MSE: {}", final_loss);
    println!("\nExample predictions:");

    for i in 0..5 {
        let out = net.forward(&inputs[i]);
        println!("Target:  {:?}", &targets[i][0..3]);
        println!("Predicted: {:?}", &out[0..3]);
        println!();
    }

}
