mod network;
mod matrix;
use crate::matrix::Matrix;
use crate::network::{Activation, Network};

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

fn main() {
    let sigmoid_activation = Activation {
        function: sigmoid,
        derivative: sigmoid_derivative,
    };

    let layers = vec![2, 3, 1];
    let mut net = Network::new(layers, sigmoid_activation, 0.5);

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

    for _ in 0..10_000 {
        for i in 0..inputs.len() {
            let input_matrix = Matrix::from(vec![inputs[i].clone()]).transpose();
            let target_matrix = Matrix::from(vec![targets[i].clone()]).transpose();
            
            net.back_propagate(input_matrix, target_matrix);
        }
    }

    println!("Testing XOR:");
    let input = Matrix::from(vec![vec![1.0, 0.0]]).transpose();
    let result = net.feed_forward(input);
    println!("Input: [1, 0] -> Output: {:?}", result.data);
}
