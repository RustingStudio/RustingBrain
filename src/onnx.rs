use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OnnxError {
    #[error("ONNX support is disabled; rebuild with `--features onnx`")]
    FeatureDisabled,
    #[error("input length {actual} does not match ONNX input shape {shape:?} ({expected} values)")]
    InvalidInput {
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },
    #[cfg(feature = "onnx")]
    #[error("ONNX runtime error: {0}")]
    Runtime(#[from] tract_onnx::prelude::TractError),
    #[cfg(feature = "onnx")]
    #[error("input shape contains unsupported dynamic dimensions")]
    DynamicInputShape,
}

#[cfg(not(feature = "onnx"))]
#[derive(Clone, Debug)]
pub struct OnnxModel;

#[cfg(not(feature = "onnx"))]
impl OnnxModel {
    pub fn load<P: AsRef<Path>>(_path: P) -> Result<Self, OnnxError> {
        Err(OnnxError::FeatureDisabled)
    }

    pub fn load_with_input_shape<P: AsRef<Path>>(
        _path: P,
        _input_shape: &[usize],
    ) -> Result<Self, OnnxError> {
        Err(OnnxError::FeatureDisabled)
    }

    pub fn predict(&self, _input: &[f32]) -> Result<Vec<f32>, OnnxError> {
        Err(OnnxError::FeatureDisabled)
    }
}

#[cfg(feature = "onnx")]
mod enabled {
    use super::{OnnxError, Path};
    use tract_onnx::prelude::*;
    use tract_onnx::tract_hir::internal::DimLike;

    type TractRunnable =
        RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

    pub struct OnnxModel {
        model: TractRunnable,
        input_shape: Vec<usize>,
    }

    impl OnnxModel {
        pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, OnnxError> {
            let model = tract_onnx::onnx().model_for_path(path)?.into_optimized()?;

            let input_fact = model.input_fact(0)?;
            let input_shape = input_fact
                .shape
                .dims()
                .iter()
                .map(|dim| dim.to_usize().map_err(OnnxError::Runtime))
                .collect::<Result<Vec<_>, _>>()?;
            let model = model.into_runnable()?;

            Ok(Self { model, input_shape })
        }

        pub fn load_with_input_shape<P: AsRef<Path>>(
            path: P,
            input_shape: &[usize],
        ) -> Result<Self, OnnxError> {
            let model = tract_onnx::onnx()
                .model_for_path(path)?
                .with_input_fact(0, f32::fact(input_shape).into())?
                .into_optimized()?
                .into_runnable()?;

            Ok(Self {
                model,
                input_shape: input_shape.to_vec(),
            })
        }

        pub fn predict(&self, input: &[f32]) -> Result<Vec<f32>, OnnxError> {
            let expected_len = self.input_shape.iter().product::<usize>();
            if input.len() != expected_len {
                return Err(OnnxError::InvalidInput {
                    shape: self.input_shape.clone(),
                    expected: expected_len,
                    actual: input.len(),
                });
            }

            let tensor = Tensor::from_shape(&self.input_shape, input)?;
            let outputs = self.model.run(tvec!(tensor.into()))?;
            let output = outputs[0].to_array_view::<f32>()?;

            Ok(output.iter().copied().collect())
        }
    }
}

#[cfg(feature = "onnx")]
pub use enabled::OnnxModel;
