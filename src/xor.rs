use std::time::Instant;

use crate::network::Network;

pub fn xor() {
    let start = Instant::now();

    let layers = vec![2, 3, 1];
    let mut net = Network::new(layers, 0.05); 

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    for _ in 0..100_000_000 {
        for i in 0..inputs.len() {
            net.train(&inputs[i], &targets[i]);
        }
    }

    println!("Testing XOR ReLU:");
    let input = vec![1.0, 0.0];
    let result = net.forward(&input);
    
    println!("Input: [1, 0] -> Output: {:?}", result); 
    
    let finish = start.elapsed();
    println!("Time taken: {:?}", finish);
}
