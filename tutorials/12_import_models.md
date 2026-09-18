# 12. Importing Models From TensorFlow and PyTorch

**You need:** chapters 1–10. Python, if you want to follow the export half.

**Time:** 45 minutes.

**Full code:** [`code/12_import_models.rs`](code/12_import_models.rs)

Sooner or later you will want to run a model you did not train. A colleague
trained it in Keras. You found one on Hugging Face. You prototyped in PyTorch
because the plotting was easier, and now the thing has to live inside a Rust
service.

That is what this chapter is for. It is a one-way street, and being clear about
which way it runs will save you an afternoon:

```
TensorFlow / PyTorch  →  ONNX file  →  RustingBrain  →  predictions
```

**Inference only.** You cannot import a model and keep training it here.
Training stays in the framework that started it.

---

## 12.1 What ONNX is

Every framework stores models in its own format, and none of them can read each
other's. ONNX (Open Neural Network Exchange) is the neutral one in the middle:
a file holding a computation graph — "multiply by this matrix, add this vector,
apply ReLU, multiply by that matrix" — plus the weights, in a format nothing in
particular owns.

Almost every framework exports to it. RustingBrain reads it, through the
[`tract`](https://github.com/sonos/tract) runtime.

The thing to understand about ONNX is that it is a graph of *operators*, not a
model description. There is no "Dense layer" in the file. There is a `MatMul`,
an `Add`, and a `Relu`. This is why the import direction works and the export
direction is hard: reading someone's graph and running it is a much smaller
problem than reconstructing your layers from one.

---

## 12.2 Turning on the feature

```bash
cargo add rusting_brain --features onnx
```

This pulls in `tract-onnx`, which is a real dependency with a real compile
time. That is why it is opt-in.

Without the feature the API still exists — `OnnxModel::load` compiles, and
returns `OnnxError::FeatureDisabled` at runtime. Code that conditionally uses
ONNX therefore builds in both configurations and fails with a sentence you can
read, rather than a missing symbol.

---

## 12.3 Exporting from Keras or TensorFlow

Train something small so you can check the numbers by hand:

```python
import numpy as np
import tensorflow as tf

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
y = np.array([[0], [1], [1], [0]], dtype="float32")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(8, activation="tanh"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.05), loss="binary_crossentropy")
model.fit(x, y, epochs=2000, verbose=0)

print(model.predict(x))      # write these four numbers down
model.export("saved_model")
```

Convert the SavedModel directory:

```bash
python -m pip install tf2onnx
python -m tf2onnx.convert --saved-model saved_model --output xor.onnx --opset 13
```

`--opset 13` is a good default. The opset is the version of the operator
vocabulary; a newer one may contain operators `tract` has not implemented, and
an older one may not contain the ones your model needs.

---

## 12.4 Exporting from PyTorch

PyTorch exports directly, no extra tool:

```python
import torch, torch.nn as nn

model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1), nn.Sigmoid())
# ... train ...
model.eval()

torch.onnx.export(
    model,
    torch.zeros(1, 2),            # a sample input; its shape is baked in
    "xor.onnx",
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
)
```

`model.eval()` matters. Dropout and batch-norm behave differently in training
mode, and whichever mode the model is in when you export is the one the graph
records. Exporting in training mode gives you a model that randomly drops
activations during inference, which looks exactly like a model that trained
badly.

---

## 12.5 Running it

```rust
use rusting_brain::onnx::OnnxModel;

let model = OnnxModel::load("xor.onnx")?;
let output = model.predict(&[0.0, 1.0])?;
println!("{output:?}");        // [0.9856]
```

From the command line, without writing anything:

```bash
cargo run --example onnx_inference --features onnx -- xor.onnx
```

Compare the four outputs against the numbers you wrote down in Python. They
should agree to about five decimal places. They will not agree exactly —
`tract` and TensorFlow sum floats in different orders, and float addition is
not associative. A disagreement in the sixth decimal is arithmetic. A
disagreement in the second is a bug, and it is nearly always the input shape.

---

## 12.6 The dynamic shape problem

This is the error you will actually hit:

```text
Source node without a determined fact
```

Keras exports usually leave the batch dimension symbolic — the graph says
`[None, 2]`, meaning "any number of rows, two columns". `tract` optimizes a
graph ahead of time, and it cannot plan memory for a dimension whose size is
unknown.

Tell it the shape:

```rust
let model = OnnxModel::load_with_input_shape("xor.onnx", &[1, 2])?;
let output = model.predict(&[0.0, 1.0])?;
```

or:

```bash
cargo run --example onnx_inference --features onnx -- xor.onnx 1,2 0,1
```

where `1,2` is the shape and `0,1` are the values.

The shape is `[batch, features]`. `[1, 2]` is one sample with two features.
`[4, 2]` is four samples at once, and then `predict` wants eight values —
row-major, so all of sample 0, then all of sample 1:

```rust
let model = OnnxModel::load_with_input_shape("xor.onnx", &[4, 2])?;
let all_four = model.predict(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])?;
// four outputs, one per row
```

Batching is worth doing. One `predict` over 64 rows is much faster than 64
calls over one row each, for the same reason mini-batching was faster in
chapter 7: one big matrix multiply beats 64 small ones.

`predict` checks the length for you:

```text
input length 2 does not match ONNX input shape [4, 2] (8 values)
```

---

## 12.7 What does not come across

**Training.** No gradients, no optimizer, no `fit`. An imported model is a
function from inputs to outputs.

**Preprocessing.** This is the one that bites. Chapter 4 taught you to
normalize inputs, and chapter 10 saved the normalization alongside the weights.
An ONNX graph almost never contains that normalization — it was `sklearn` code
sitting in front of the model in Python. If you feed raw values to a model that
was trained on standardized ones, you get confident nonsense and no error
message.

Ask for the preprocessing constants along with the file, and apply them
yourself:

```rust
let normalized: Vec<f32> = raw.iter()
    .zip(&means).zip(&std_devs)
    .map(|((v, m), s)| (v - m) / s)
    .collect();
let output = model.predict(&normalized)?;
```

**Unsupported operators.** `tract` implements a large part of the ONNX
vocabulary but not all of it. A model built out of custom layers may fail to
load. The error names the operator:

```text
Unsupported operator: GridSample
```

There is no way around that except simplifying the model on the export side.

---

## 12.8 Going the other way

RustingBrain does not export to ONNX. If you want a RustingBrain model in
another framework, the practical route is to read the JSON — it is a documented
format holding weights, biases, activations, and shapes — and rebuild the model
where you need it. For a dense network that is about thirty lines of Python.

For RustingBrain's own models, use the native formats: `save_json` for dense
networks (chapter 9), `save_bin` for transformers (chapter 14). They are
smaller, faster, lossless, and they carry things ONNX has no place for, such as
optimizer state.

---

## 12.9 If something went wrong

| Symptom | Cause | Fix |
|---|---|---|
| `ONNX support is disabled; rebuild with --features onnx` | Built without the feature | `--features onnx` |
| `Source node without a determined fact` | Symbolic batch dimension | `load_with_input_shape` |
| `input length N does not match ONNX input shape [..]` | Shape and data disagree | The value count is the product of the dims |
| `Unsupported operator: X` | `tract` has no implementation | Simplify or replace the layer before exporting |
| Outputs differ in the 6th decimal | Float summation order | Expected, ignore |
| Outputs differ in the 2nd decimal | Wrong shape, or missing preprocessing | Check both, in that order |
| Output is the same for every input | Model exported in training mode, or inputs not normalized | `model.eval()` before export; check preprocessing |

---

## Exercises

1. Export the XOR model above and run all four inputs through it from Rust.
   Compare against the Python numbers.
2. Export the *same* model with `--opset 9` and with `--opset 17`. Does
   `tract` load both?
3. Batch: load with shape `[4, 2]`, predict all four XOR rows in one call, and
   time it against four separate `[1, 2]` calls over 10,000 repetitions.
4. Train a model in Keras on standardized inputs, export it, and then feed it
   raw unstandardized values from Rust. Note how plausible the wrong answers
   look.

---

## Recap

- ONNX is the neutral format in the middle. Import works; export does not.
- The `onnx` feature is opt-in; without it the API returns `FeatureDisabled`
  rather than failing to compile.
- `tf2onnx` for TensorFlow and Keras, `torch.onnx.export` for PyTorch, opset 13
  as a default. `model.eval()` first.
- `Source node without a determined fact` means a symbolic batch dimension.
  `load_with_input_shape` fixes it.
- Shapes are `[batch, features]`, row-major, and batching one call over many
  rows is much faster than many calls.
- Preprocessing does not travel with the file. Getting the weights without the
  normalization gives you confident nonsense.

---

**Next:** [13. When Things Go Wrong](13_troubleshooting.md)
