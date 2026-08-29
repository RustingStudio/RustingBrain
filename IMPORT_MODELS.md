# Import Models

RustingBrain runs external models through ONNX. This is inference-only: you can
load a model trained in TensorFlow or Keras and run predictions from Rust, but
training continues to belong in the original framework.

## TensorFlow Or Keras

Export your trained model:

```python
model.export("saved_model")
```

Convert the SavedModel directory to ONNX:

```bash
python -m pip install tf2onnx
python -m tf2onnx.convert \
  --saved-model saved_model \
  --output model.onnx \
  --opset 13
```

Run it from RustingBrain:

```bash
cargo run --example onnx_inference --features onnx -- model.onnx
```

## Dynamic Input Shapes

Many TensorFlow exports keep the batch dimension dynamic. If loading fails with:

```text
Source node without a determined fact
```

pass the concrete input shape:

```bash
cargo run --example onnx_inference --features onnx -- xor.onnx 1,2 0,1
```

The arguments are:

- `xor.onnx`: model path
- `1,2`: input tensor shape, one sample with two features
- `0,1`: input values

The same thing in Rust:

```rust
use rusting_brain::onnx::OnnxModel;

let model = OnnxModel::load_with_input_shape("xor.onnx", &[1, 2])?;
let output = model.predict(&[0.0, 1.0])?;
```

## Fixed Input Shapes

If the ONNX file already includes a concrete input shape:

```rust
use rusting_brain::onnx::OnnxModel;

let model = OnnxModel::load("model.onnx")?;
let output = model.predict(&input_values)?;
```
