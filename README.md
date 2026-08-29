# RustingBrain

RustingBrain is a small neural-network library for learning how inference and
training work under the hood.

The crate keeps the core pieces visible: matrices, dense layers, activations,
loss functions, optimizers, batches, and model weights. It is not trying to
replace PyTorch or TensorFlow. It is for experiments, study, and small models
where readable Rust code matters more than having every deep-learning feature.

## Features

- Dense feedforward networks
- Regression, binary classification, multiclass classification, and XOR examples
- ReLU, sigmoid, tanh, softmax, and linear activations
- Mean squared error, binary cross entropy, and cross entropy losses
- SGD and Adam optimizers
- Mini-batch training with reproducible shuffling
- JSON save/load for RustingBrain models
- ONNX inference for models trained elsewhere
- Optional CUDA/cuBLAS matrix benchmark

## Install

Add the crate from crates.io:

```bash
cargo add rusting_brain
```

Or clone and run the examples:

```bash
git clone https://github.com/Vasyl-Trefilov/RustingBrain.git
cd RustingBrain
cargo test
cargo run --example xor
```

## Quick Example

```rust
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
        .build();

    model.fit(
        &data,
        TrainConfig {
            epochs: 2_000,
            batch_size: 4,
            shuffle: true,
            seed: Some(42),
        },
    )?;

    println!("{:?}", model.predict(&[1.0, 0.0])?);
    Ok(())
}
```

## Examples

```bash
cargo run --example xor
cargo run --example regression
cargo run --example classification
cargo run --example save_load
```

## Save And Load

```rust
model.save_json("model.json")?;
let loaded = rusting_brain::Network::load_json("model.json")?;
```

The saved JSON contains the network shape, activations, loss, weights, biases,
and file format version. Optimizer state is not stored yet, so saved models are
mainly for inference and reproducible examples.

## Import Models

RustingBrain can run ONNX models for inference:

```bash
cargo run --example onnx_inference --features onnx -- model.onnx
```

Some TensorFlow exports leave the input shape dynamic. In that case, pass the
shape and input values explicitly:

```bash
cargo run --example onnx_inference --features onnx -- xor.onnx 1,2 0,1
```

See [IMPORT_MODELS.md](IMPORT_MODELS.md) for the TensorFlow/Keras to ONNX flow.

## CUDA Benchmark

CUDA is optional. The normal crate build does not require CUDA, `nvcc`, or an
NVIDIA GPU.

To compile the CUDA benchmark support:

```bash
cargo check --features cuda
```

To run the benchmark on a CUDA machine:

```bash
cargo run --release --example cuda_benchmark --features cuda
```

See [INSTALL_CUDA.md](INSTALL_CUDA.md) for driver, toolkit, and cuBLAS setup.

## Scope

The current focus is dense neural networks and inference interop. Convolution
layers, transformer experiments, mixed precision, and full GPU training are
future work.
