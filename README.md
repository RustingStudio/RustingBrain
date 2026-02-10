***

# 🦀 RustingBrain

**A lightweight, "from-scratch" Neural Network library written in pure Rust.**

RustingBrain is a foundational Deep Learning library designed to demystify the mathematics behind Artificial Intelligence. It implements Matrix operations, Feed Forward propagation, and Backpropagation (Gradient Descent) without relying on heavy external frameworks like TensorFlow or PyTorch.

It serves as an educational tool for understanding how tensors and neurons actually learn.

## ⚡ Features

*   **Matrix Engine**: Custom implementation of linear algebra operations (Dot Product, Transpose, Hadamard Product).
*   **Dynamic Architecture**: Create networks with any number of layers and neurons (e.g., `2 -> 3 -> 1`).
*   **Backpropagation**: Implements Stochastic Gradient Descent to adjust weights and biases.
*   **Custom Activation**: Support for injecting custom activation functions (Sigmoid, etc.) and their derivatives.
*   **Pure Rust**: Minimal dependencies (only uses `rand` for initial weight generation).

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/RustingBrain.git
cd RustingBrain
cargo run
```

## 🚀 Usage Example

Here is how to use the library to solve the classic **XOR** (Exclusive OR) problem. This demonstrates how to define activation functions, structure the network, and run the training loop.

```rust
use crate::network::{Network, Activation};
use crate::matrix::Matrix;

fn main() {
    // 1. Define the Activation Function (Sigmoid)
    // Squeezes numbers between 0.0 and 1.0
    let sigmoid = Activation {
        function: |x| 1.0 / (1.0 + (-x).exp()),
        derivative: |x| x * (1.0 - x),
    };

    // 2. Initialize the Network
    // 2 Input Neurons -> 3 Hidden Neurons -> 1 Output Neuron
    let layers = vec![2, 3, 1];
    let learning_rate = 0.5;
    let mut net = Network::new(layers, sigmoid, learning_rate);

    // 3. Define Training Data (XOR Logic)
    // Input: [0,0] -> Target: [0]
    // Input: [0,1] -> Target: [1]
    // ...
    let inputs = vec![
        vec![0.0, 0.0], vec![0.0, 1.0], 
        vec![1.0, 0.0], vec![1.0, 1.0],
    ];
    let targets = vec![
        vec![0.0], vec![1.0], 
        vec![1.0], vec![0.0],
    ];

    // 4. Train the Network (10,000 Epochs)
    println!("Training started...");
    for _ in 0..10_000 {
        for i in 0..inputs.len() {
            let input_matrix = Matrix::from(vec![inputs[i].clone()]);
            let target_matrix = Matrix::from(vec![targets[i].clone()]);
            
            // The library handles the math:
            // FeedForward -> Calculate Error -> BackPropagate -> Update Weights
            net.back_propagate(input_matrix, target_matrix);
        }
    }

    // 5. Test Prediction
    let test_input = Matrix::from(vec![vec![1.0, 0.0]]);
    let prediction = net.feed_forward(test_input);
    
    println!("Training complete!");
    println!("Input: [1, 0] -> Prediction: {:?}", prediction.data);
}
```

## 📂 Project Structure

*   **`src/matrix.rs`**: The math engine. Handles low-level data manipulation, including dot products, element-wise multiplication, and transposing.
*   **`src/network.rs`**: The brain. Manages layers, weights, biases, and the orchestration of data flowing forward and errors flowing backward.
*   **`src/main.rs`**: The implementation/entry point used for testing and training models.

## 🛣️ Roadmap

Future features planned for this library:

- [ ] Save and Load trained models (serialize weights to JSON/Binary).
- [ ] Implement additional activation functions (ReLU, Tanh, Softmax).
- [ ] Add support for Batch Training (Learning from multiple inputs at once).
- [ ] Implement Cost Functions (Mean Squared Error, Cross Entropy).

## 🤝 Contributing

Contributions are welcome! If you want to add a new activation function or optimize the Matrix math:

1.  Fork the project.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes.
4.  Open a Pull Request.

## 📄 License

Distributed under the MIT License.
