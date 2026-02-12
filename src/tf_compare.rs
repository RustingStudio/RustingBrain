use std::{thread::available_parallelism, time::Instant};

use crate::network::Network;

pub fn tensorflow_like_example() {
    let input_size = 512usize;
    let hidden_sizes = [1024usize, 1024usize, 512usize];
    let output_size = 10usize;
    let batch_size = 256usize;
    let samples = 20_000usize;
    let iterations = 1usize;

    let mut layers = Vec::new();
    layers.push(input_size);
    layers.extend_from_slice(&hidden_sizes);
    layers.push(output_size);

    let mut net = Network::new(layers, 0.01);

    let mut inputs = Vec::with_capacity(samples);
    let mut targets = Vec::with_capacity(samples);

    for _ in 0..samples {
        let mut x = Vec::with_capacity(input_size);
        for _ in 0..input_size {
            x.push(rand::random::<f32>());
        }

        let mut y = Vec::with_capacity(output_size);
        for _ in 0..output_size {
            y.push(rand::random::<f32>());
        }

        inputs.push(x);
        targets.push(y);
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
    let mut expected = Vec::with_capacity(output_size);
    for _ in 0..output_size {
        expected.push(rand::random::<f32>());
    }
    let output = net.forward(&test_input);

    println!("Large model test (Rust NN):");
    println!("Input len: {}", test_input.len());
    println!("Output len: {}", output.len());
    println!("Time taken (seconds): {:.6}", duration.as_secs_f64());
}

