//! Chapter 12 — loading an ONNX model and running a batch through it.
//!
//!     cargo run --release --features onnx --bin 12_import_models -- xor.onnx
//!
//! `xor.onnx` is whatever chapter 12.3 or 12.4 exported. Without a path the
//! program explains what it wants and exits.

use rusting_brain::onnx::{OnnxError, OnnxModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: 12_import_models <model.onnx>");
        eprintln!("export one first; see chapter 12.3 (Keras) or 12.4 (PyTorch)");
        return Ok(());
    };

    // The four XOR inputs, row-major: [batch, features] = [4, 2].
    let rows: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();

    // 12.6 — a Keras export usually leaves the batch dimension symbolic, so
    // the plain `load` fails on it and the shape has to be supplied. Try the
    // easy path first and fall back rather than always passing a shape: a
    // graph with a concrete shape does not need to be told its own.
    let model = match OnnxModel::load(&path) {
        Ok(model) => {
            println!("loaded with the shape baked into the file");
            model
        }
        Err(OnnxError::FeatureDisabled) => {
            eprintln!("rebuild with `--features onnx`");
            return Ok(());
        }
        Err(error) => {
            println!("{error}; retrying with an explicit [4, 2]");
            OnnxModel::load_with_input_shape(&path, &[4, 2])?
        }
    };

    // One call over four rows, not four calls over one row: the whole batch is
    // a single matrix multiply.
    let outputs = model.predict(&flat)?;
    for (row, output) in rows.iter().zip(&outputs) {
        println!("{row:?} -> {output:.4}");
    }

    // 12.6 again — the length check is on the value count, not the row count.
    match model.predict(&[0.0, 1.0]) {
        Ok(_) => println!("this model takes a single row"),
        Err(error) => println!("two values into a four-row model: {error}"),
    }

    Ok(())
}
