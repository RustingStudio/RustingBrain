//! Saving and restoring a training run, across as many models as it has.
//!
//! [`TransformerLm::save_bin`](crate::transformer::TransformerLm::save_bin)
//! writes one model in a private format, positionally. A run that trains an
//! image encoder, a flow transformer and a VAE at once needs something else:
//! the parameters come from three trees, the trees change shape while the
//! architecture is still moving, and a mismatch has to name what it could not
//! find instead of failing on a count.
//!
//! So a checkpoint here is a `.safetensors` file keyed by name. The caller
//! supplies the names, which is where "multi-model" comes from: prefix each
//! tree and one file holds them all.
//!
//! ```no_run
//! # use rusting_brain::{checkpoint, Param, Matrix};
//! # use std::collections::BTreeMap;
//! # let mut weight = Param::new(Matrix::new(4, 4));
//! let mut params = [("flow.block0.weight".to_string(), &mut weight)];
//! checkpoint::save(
//!     "step-1200.safetensors",
//!     &mut params,
//!     &BTreeMap::from([("step".into(), "1200".into())]),
//! )?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Each parameter writes its weights under the given name and its Adam moments
//! under `name.moment1` and `name.moment2`, because a run that resumes without
//! them takes several oversized steps: the moments restart at zero while the
//! step counter carries on, so bias correction is correcting for a history that
//! is no longer there. A frozen parameter has released those buffers and writes
//! weights alone.

use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::param::Param;
use crate::safetensors::{self, SafeTensors};
use std::collections::BTreeMap;
use std::path::Path;

/// Writes every named parameter, with its Adam moments, to one file.
///
/// `metadata` is stored in the header, where
/// [`SafeTensors::metadata`](crate::safetensors::SafeTensors::metadata) reads
/// it back without touching the weights. Put the step counter and whatever
/// configuration the run needs to rebuild its models there; the format stores
/// strings, so a caller with structure to keep puts JSON in one.
///
/// Parameters resident on a device are read back from it here, so this is safe
/// to call mid-training without moving a model to the host first.
pub fn save<P: AsRef<Path>>(
    path: P,
    params: &mut [(String, &mut Param)],
    metadata: &BTreeMap<String, String>,
) -> Result<(), NetworkError> {
    for (_, param) in params.iter_mut() {
        // Both refresh the host copies that the borrows below read. The return
        // is those same matrices, but borrowed from the `&mut` held here.
        #[cfg(feature = "cuda")]
        param.sync_from_device()?;
        param.moments()?;
    }

    let mut entries: Vec<(String, [usize; 2], &[f32])> = Vec::new();
    for (name, param) in params.iter() {
        let value = &param.value;
        entries.push((name.clone(), [value.rows, value.cols], &value.data));
        if param.is_frozen() {
            continue;
        }
        let (first, second) = param.host_moments();
        entries.push((
            format!("{name}.moment1"),
            [first.rows, first.cols],
            &first.data,
        ));
        entries.push((
            format!("{name}.moment2"),
            [second.rows, second.cols],
            &second.data,
        ));
    }

    let borrowed: Vec<(&str, &[usize], &[f32])> = entries
        .iter()
        .map(|(name, shape, data)| (name.as_str(), &shape[..], *data))
        .collect();
    safetensors::write(path, &borrowed, metadata)
}

/// Restores the weights and moments [`save`] wrote, and returns the metadata.
///
/// The parameters have to be on the host: a device-resident parameter's weights
/// live on the device, and a write to the host copy would be overwritten by the
/// next thing that reads them back. Load first, then call `to_cuda`.
///
/// A name missing from the file is an error. Missing moments are not: a
/// checkpoint written from a frozen parameter has none, and one written by
/// another tool has none either, so the moments stay as they are.
pub fn load<P: AsRef<Path>>(
    path: P,
    params: &mut [(String, &mut Param)],
) -> Result<BTreeMap<String, String>, NetworkError> {
    let mut file = SafeTensors::open(path)?;
    for (name, param) in params.iter_mut() {
        if param.is_on_device() {
            return Err(NetworkError::InvalidSnapshot(format!(
                "{name} is resident on a device; load the checkpoint before moving the model there"
            )));
        }
        let value = file.matrix(name)?;
        let (rows, cols) = (param.value.rows, param.value.cols);
        if value.rows != rows || value.cols != cols {
            return Err(NetworkError::InvalidSnapshot(format!(
                "{name} is {}x{} in the checkpoint and {rows}x{cols} in the model",
                value.rows, value.cols
            )));
        }
        param.value = value;

        let moments: Vec<Matrix> = ["moment1", "moment2"]
            .iter()
            .filter_map(|suffix| file.matrix(&format!("{name}.{suffix}")).ok())
            .collect();
        if let [first, second] = &moments[..]
            && first.rows == rows
            && first.cols == cols
            && second.rows == rows
            && second.cols == cols
        {
            param.set_moments(first.clone(), second.clone())?;
        }
    }
    Ok(file.metadata().clone())
}

/// Names a model's parameters by their position, which is what the models in
/// this crate hand to [`save`] and [`load`].
///
/// A tree that grows a layer renames everything after it, so the name is the
/// index and the prefix says which model it belongs to. Two models in one file
/// only need two prefixes.
///
/// ```
/// # use rusting_brain::{checkpoint, Param, Matrix};
/// let mut weight = Param::new(Matrix::new(2, 2));
/// let named = checkpoint::positional(vec![&mut weight], "flow");
/// assert_eq!(named[0].0, "flow.0000");
/// ```
pub fn positional<'a>(params: Vec<&'a mut Param>, prefix: &str) -> Vec<(String, &'a mut Param)> {
    params
        .into_iter()
        .enumerate()
        .map(|(index, param)| (format!("{prefix}.{index:04}"), param))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::Optimizer;

    /// A parameter that has taken a step, so its moments are not zero and a
    /// checkpoint that drops them is visible.
    fn stepped(rows: usize, cols: usize, seed: f32) -> Param {
        let mut param = Param::new(Matrix::from_vec(
            rows,
            cols,
            (0..rows * cols).map(|i| seed + i as f32).collect(),
        ));
        param.grad = Matrix::from_vec(rows, cols, vec![0.5; rows * cols]);
        param.step(&Optimizer::adam(1e-2), 1, 1.0);
        param
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "rusting-brain-{name}-{}.safetensors",
            std::process::id()
        ))
    }

    #[test]
    fn a_round_trip_restores_weights_and_moments() {
        let path = scratch("round-trip");
        let mut first = stepped(3, 2, 1.0);
        let mut second = stepped(2, 4, 10.0);
        let mut saved = [
            ("vae.encoder.weight".to_string(), &mut first),
            ("flow.block0.weight".to_string(), &mut second),
        ];
        save(
            &path,
            &mut saved,
            &BTreeMap::from([("step".to_string(), "1200".to_string())]),
        )
        .unwrap();
        let expected: Vec<(Matrix, Matrix, Matrix)> = saved
            .iter_mut()
            .map(|(_, param)| {
                let value = param.value.clone();
                let (first, second) = param.moments().unwrap();
                (value, first.clone(), second.clone())
            })
            .collect();

        let mut blank_first = Param::new(Matrix::new(3, 2));
        let mut blank_second = Param::new(Matrix::new(2, 4));
        let mut restored = [
            ("vae.encoder.weight".to_string(), &mut blank_first),
            ("flow.block0.weight".to_string(), &mut blank_second),
        ];
        let metadata = load(&path, &mut restored).unwrap();
        assert_eq!(metadata.get("step").map(String::as_str), Some("1200"));

        for ((_, param), (value, first, second)) in restored.iter_mut().zip(&expected) {
            assert_eq!(param.value.data, value.data, "weights");
            let (one, two) = param.moments().unwrap();
            assert_eq!(&one.data, &first.data, "first moment");
            assert_eq!(&two.data, &second.data, "second moment");
            assert!(one.data.iter().any(|v| *v != 0.0), "a moment to compare");
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_frozen_parameter_saves_weights_alone() {
        let path = scratch("frozen");
        let mut param = stepped(2, 2, 1.0);
        param.freeze();
        save(
            &path,
            &mut [("tower.weight".to_string(), &mut param)],
            &BTreeMap::new(),
        )
        .unwrap();

        let file = SafeTensors::open(&path).unwrap();
        let names: Vec<&str> = file.names().collect();
        assert_eq!(names, ["tower.weight"], "no moments for a frozen parameter");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_shape_mismatch_names_the_parameter() {
        let path = scratch("mismatch");
        let mut param = stepped(3, 2, 1.0);
        save(
            &path,
            &mut [("w".to_string(), &mut param)],
            &BTreeMap::new(),
        )
        .unwrap();

        let mut wrong = Param::new(Matrix::new(2, 3));
        let error = load(&path, &mut [("w".to_string(), &mut wrong)]).unwrap_err();
        assert!(
            error.to_string().contains("3x2 in the checkpoint"),
            "{error}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_missing_parameter_is_an_error() {
        let path = scratch("missing");
        let mut param = stepped(2, 2, 1.0);
        save(
            &path,
            &mut [("w".to_string(), &mut param)],
            &BTreeMap::new(),
        )
        .unwrap();

        let mut other = Param::new(Matrix::new(2, 2));
        assert!(load(&path, &mut [("elsewhere".to_string(), &mut other)]).is_err());
        std::fs::remove_file(&path).ok();
    }
}
