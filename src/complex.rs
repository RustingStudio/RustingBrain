use std::{thread::available_parallelism, time::Instant};

use crate::network::Network;

pub fn complex_example() {
    let start = Instant::now();

    let layers = vec![8, 16, 1];
    let mut net = Network::new(layers, 0.01);

    let samples = 10_000;
    let mut inputs = Vec::with_capacity(samples);
    let mut targets = Vec::with_capacity(samples);

    for _ in 0..samples {
        let mut x = Vec::with_capacity(8);
        for _ in 0..8 {
            let v = rand::random::<f32>();
            x.push(v);
        }
        let y = x.iter().sum::<f32>() / 8.0;
        inputs.push(x);
        targets.push(vec![y]);
    }

    let num_threads = available_parallelism().map(|n| n.get()).unwrap_or(1);

    println!("number of threads: {}", num_threads);
    for _ in 0..5_000 {
        net.train_batch_parallel(&inputs, &targets, num_threads);
    }

    let test_input: Vec<f32> = (0..8).map(|_| rand::random::<f32>()).collect();
    let expected = test_input.iter().sum::<f32>() / 8.0;
    let output = net.forward(&test_input);

    println!("Complex example (sum/8 regression):");
    println!("Input: {:?}", test_input);
    println!("Expected: {}", expected);
    println!("Network output: {:?}", output);

    let finish = start.elapsed();
    println!("Complex example time taken: {:?}", finish);
}

