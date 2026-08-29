#[cfg(feature = "onnx")]
use rusting_brain::onnx::OnnxModel;

#[cfg(feature = "onnx")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or(
        "usage: cargo run --example onnx_inference --features onnx -- model.onnx [shape] [values]",
    )?;
    let shape = args.next().map(|arg| parse_usize_list(&arg)).transpose()?;
    let input = args
        .next()
        .map(|arg| parse_f32_list(&arg))
        .transpose()?
        .unwrap_or_else(|| vec![0.0]);

    let model = if let Some(shape) = shape {
        OnnxModel::load_with_input_shape(path, &shape)?
    } else {
        OnnxModel::load(path)?
    };
    let output = model.predict(&input)?;

    println!("{output:?}");
    Ok(())
}

#[cfg(feature = "onnx")]
fn parse_usize_list(value: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    value
        .split(',')
        .map(|part| Ok(part.trim().parse::<usize>()?))
        .collect()
}

#[cfg(feature = "onnx")]
fn parse_f32_list(value: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    value
        .split(',')
        .map(|part| Ok(part.trim().parse::<f32>()?))
        .collect()
}

#[cfg(not(feature = "onnx"))]
fn main() {
    println!(
        "Re-run with `--features onnx` and pass a model path, optional shape, and input values."
    );
}
