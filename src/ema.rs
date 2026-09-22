//! An exponential moving average of the weights, kept beside the model.
//!
//! A training run reports the loss of the weights it is standing on, but the
//! weights that sample best are usually an average of the last few thousand
//! steps: the optimizer is still bouncing around a minimum it has already
//! found, and averaging cancels the bounce. This module keeps that average.
//!
//! # Why a separate type rather than a shadow copy inside `Param::step`
//!
//! [`Param::step`] runs on the device when the parameter is resident there, and
//! a shadow copy folded into it would have to be a second device buffer, a
//! second kernel, and another thing every device path has to keep consistent —
//! for something a run turns on near the end and a fine-tune never turns on at
//! all. An [`Ema`] holds host matrices and touches the model only through the
//! public parameter list, so it costs nothing until it is built, and a caller
//! who does not want one pays neither memory nor a kernel launch.
//!
//! The price is that [`Ema::update`] reads the weights back from the device
//! once per call. That is one download of the model per update, so a run
//! averages every few steps rather than every step; the average over the last
//! few thousand steps barely notices the difference.
//!
//! # Surviving a checkpoint
//!
//! An average built over twenty thousand steps and lost on the first resume is
//! worse than no average at all, so [`Ema::save`] and [`Ema::load`] write it to
//! its own file, in the same shape as
//! [`TransformerLm::save_optimizer_state`](crate::transformer::TransformerLm::save_optimizer_state):
//! weights live in the model snapshot, the state a run needs to continue lives
//! beside it.
//!
//! ```no_run
//! # use rusting_brain::{Ema, TransformerLm};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let mut model = TransformerLm::builder().build()?;
//! let mut ema = Ema::new(&mut model.params_mut());
//!
//! for _ in 0..1_000 {
//!     // model.train_step(&batch)?;
//!     ema.update(&mut model.params_mut(), 0.999)?;
//! }
//!
//! ema.save("ema.bin")?;
//!
//! ema.swap_in(&mut model.params_mut())?;      // sample with the average
//! // model.generate(..)?;
//! ema.swap_out(&mut model.params_mut())?;     // put the live weights back
//! # Ok(())
//! # }
//! ```

use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::param::Param;
use std::io::{Read, Write};
use std::path::Path;

const EMA_MAGIC: &[u8; 8] = b"RBEMA001";

/// An exponential moving average of a model's weights: one matrix per
/// parameter, updated towards the live weights and swapped in for sampling.
///
/// Built from a parameter list and driven with the same list; the order is the
/// order [`params_mut`](crate::transformer::TransformerLm::params_mut) returns,
/// and every call checks the count and the shapes, so a list from a differently
/// configured model is refused rather than averaged into nonsense.
#[derive(Clone, Debug)]
pub struct Ema {
    shadow: Vec<Matrix>,
    /// The live weights, parked while the average is swapped in. Empty
    /// otherwise, which is also how [`Ema::is_swapped_in`] answers.
    parked: Vec<Matrix>,
    updates: usize,
}

impl Ema {
    /// Starts an average at the current weights.
    ///
    /// Starting from the weights rather than from zero is what makes the
    /// average usable immediately: an average seeded with zeros spends the
    /// first `1 / (1 - decay)` steps being mostly zeros, which is why other
    /// implementations need a bias correction. This one does not.
    pub fn new(params: &mut [&mut Param]) -> Self {
        Self {
            shadow: params.iter_mut().map(|p| host_value(p).clone()).collect(),
            parked: Vec::new(),
            updates: 0,
        }
    }

    /// Moves the average `1 - decay` of the way towards the live weights.
    ///
    /// A `decay` of 0.999 averages over roughly the last thousand updates,
    /// 0.9999 over the last ten thousand. It has to lie in `[0, 1)`; a decay of
    /// exactly one would never move again.
    ///
    /// Call it after the optimizer step, not before, and refuse to call it
    /// while the average is swapped in — the live weights are parked then, and
    /// averaging towards the average is a no-op that silently stalls the run.
    pub fn update(&mut self, params: &mut [&mut Param], decay: f32) -> Result<(), NetworkError> {
        if !(0.0..1.0).contains(&decay) {
            return Err(NetworkError::InvalidConfig(format!(
                "an EMA decay has to lie in [0, 1), got {decay}"
            )));
        }
        if self.is_swapped_in() {
            return Err(NetworkError::InvalidConfig(
                "the average is swapped in; call swap_out before updating it".into(),
            ));
        }
        self.check(params)?;
        for (average, param) in self.shadow.iter_mut().zip(params.iter_mut()) {
            let value = host_value(param);
            for (a, &v) in average.data.iter_mut().zip(value.data.iter()) {
                *a = decay * *a + (1.0 - decay) * v;
            }
        }
        self.updates += 1;
        Ok(())
    }

    /// Whether the averaged weights are currently standing in the model.
    pub fn is_swapped_in(&self) -> bool {
        !self.parked.is_empty()
    }

    /// How many times [`Ema::update`] has run, which survives a save and a
    /// load. A caller that averages every few steps uses it to tell how much
    /// history the average actually holds.
    pub fn updates(&self) -> usize {
        self.updates
    }

    /// The averaged weights, in parameter order.
    pub fn weights(&self) -> &[Matrix] {
        &self.shadow
    }

    /// Parks the live weights and stands the average in their place, so a
    /// forward pass runs on the average.
    ///
    /// Reversed by [`Ema::swap_out`]. Calling it twice is refused rather than
    /// obeyed: the second call would park the average over the live weights and
    /// lose the run.
    pub fn swap_in(&mut self, params: &mut [&mut Param]) -> Result<(), NetworkError> {
        if self.is_swapped_in() {
            return Err(NetworkError::InvalidConfig(
                "the average is already swapped in".into(),
            ));
        }
        self.check(params)?;
        let mut parked = Vec::with_capacity(params.len());
        for (average, param) in self.shadow.iter().zip(params.iter_mut()) {
            parked.push(host_value(param).clone());
            param.set_value(average.clone())?;
        }
        self.parked = parked;
        Ok(())
    }

    /// Puts the live weights back. A no-op when the average is not swapped in.
    pub fn swap_out(&mut self, params: &mut [&mut Param]) -> Result<(), NetworkError> {
        if !self.is_swapped_in() {
            return Ok(());
        }
        self.check(params)?;
        for (live, param) in self.parked.drain(..).zip(params.iter_mut()) {
            param.set_value(live)?;
        }
        Ok(())
    }

    /// Writes the average and its update count.
    ///
    /// Refused while the average is swapped in, because the live weights are
    /// parked in memory only: writing then would save an average whose partner
    /// snapshot holds that same average, and the run would resume having lost
    /// its live weights.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), NetworkError> {
        if self.is_swapped_in() {
            return Err(NetworkError::InvalidConfig(
                "the average is swapped in; call swap_out before saving".into(),
            ));
        }
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        writer.write_all(EMA_MAGIC)?;
        writer.write_all(&(self.shadow.len() as u64).to_le_bytes())?;
        writer.write_all(&(self.updates as u64).to_le_bytes())?;
        for matrix in &self.shadow {
            writer.write_all(&(matrix.rows as u64).to_le_bytes())?;
            writer.write_all(&(matrix.cols as u64).to_le_bytes())?;
            let bytes: Vec<u8> = matrix.data.iter().flat_map(|v| v.to_le_bytes()).collect();
            writer.write_all(&bytes)?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Reads back a file written by [`Ema::save`].
    ///
    /// The result is not swapped in, whatever the run that wrote it was doing.
    /// Check it against the model with [`Ema::update`] or [`Ema::swap_in`],
    /// which both refuse a list that does not match.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, NetworkError> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != EMA_MAGIC {
            return Err(NetworkError::InvalidSnapshot(
                "not a RustingBrain weight-average file".into(),
            ));
        }
        let mut word = [0u8; 8];
        reader.read_exact(&mut word)?;
        let count = u64::from_le_bytes(word) as usize;
        reader.read_exact(&mut word)?;
        let updates = u64::from_le_bytes(word) as usize;
        let mut shadow = Vec::with_capacity(count);
        for _ in 0..count {
            reader.read_exact(&mut word)?;
            let rows = u64::from_le_bytes(word) as usize;
            reader.read_exact(&mut word)?;
            let cols = u64::from_le_bytes(word) as usize;
            let mut bytes = vec![0u8; rows * cols * 4];
            reader.read_exact(&mut bytes)?;
            let data = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            shadow.push(Matrix::from_vec(rows, cols, data));
        }
        Ok(Self {
            shadow,
            parked: Vec::new(),
            updates,
        })
    }

    /// Refuses a parameter list that is not the one this average was built
    /// from.
    fn check(&self, params: &[&mut Param]) -> Result<(), NetworkError> {
        if params.len() != self.shadow.len() {
            return Err(NetworkError::InvalidConfig(format!(
                "the average holds {} parameters, this model has {}",
                self.shadow.len(),
                params.len()
            )));
        }
        for (index, (average, param)) in self.shadow.iter().zip(params.iter()).enumerate() {
            if average.rows != param.value.rows || average.cols != param.value.cols {
                return Err(NetworkError::InvalidConfig(format!(
                    "parameter {index} is {}x{} but the average holds {}x{}",
                    param.value.rows, param.value.cols, average.rows, average.cols
                )));
            }
        }
        Ok(())
    }
}

/// The host copy of a parameter's weights, refreshed from the device first when
/// the parameter is resident there.
fn host_value(param: &mut Param) -> &Matrix {
    #[cfg(feature = "cuda")]
    if param.is_on_device() {
        // A download that fails leaves the previous host copy in place, which
        // averages a step-old weight rather than losing the run; the next
        // device call reports the failure properly.
        let _ = param.sync_from_device();
    }
    &param.value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::Optimizer;

    fn params(values: &[f32]) -> Vec<Param> {
        values
            .iter()
            .map(|&v| Param::filled(1, 2, v))
            .collect::<Vec<_>>()
    }

    fn borrow(params: &mut [Param]) -> Vec<&mut Param> {
        params.iter_mut().collect()
    }

    #[test]
    fn an_average_starts_at_the_weights_and_trails_them_afterwards() {
        let mut model = params(&[1.0]);
        let mut ema = Ema::new(&mut borrow(&mut model));
        assert_eq!(ema.weights()[0].data, vec![1.0, 1.0]);

        model[0].value.data = vec![2.0, 2.0];
        ema.update(&mut borrow(&mut model), 0.9).unwrap();
        // 0.9 * 1.0 + 0.1 * 2.0
        assert!((ema.weights()[0].data[0] - 1.1).abs() < 1e-6);
        assert_eq!(ema.updates(), 1);
    }

    #[test]
    fn an_average_is_smoother_than_the_weights_it_follows() {
        let mut model = params(&[0.0]);
        let mut ema = Ema::new(&mut borrow(&mut model));
        // Weights that bounce either side of 1.0 without settling.
        for step in 0..200 {
            let bounce = if step % 2 == 0 { 1.5 } else { 0.5 };
            model[0].value.data = vec![bounce; 2];
            ema.update(&mut borrow(&mut model), 0.9).unwrap();
        }
        let averaged = ema.weights()[0].data[0];
        assert!(
            (averaged - 1.0).abs() < 0.05,
            "the average should sit near the centre of the bounce, got {averaged}"
        );
    }

    #[test]
    fn swapping_in_stands_the_average_in_the_models_place_and_swapping_out_undoes_it() {
        let mut model = params(&[1.0, -1.0]);
        let mut ema = Ema::new(&mut borrow(&mut model));
        for param in &mut model {
            param.value.data = vec![5.0; 2];
        }

        ema.swap_in(&mut borrow(&mut model)).unwrap();
        assert!(ema.is_swapped_in());
        assert_eq!(model[0].value.data, vec![1.0, 1.0]);
        assert_eq!(model[1].value.data, vec![-1.0, -1.0]);

        ema.swap_out(&mut borrow(&mut model)).unwrap();
        assert!(!ema.is_swapped_in());
        assert_eq!(model[0].value.data, vec![5.0, 5.0]);
        assert_eq!(model[1].value.data, vec![5.0, 5.0]);
    }

    #[test]
    fn swapping_in_twice_is_refused_rather_than_losing_the_live_weights() {
        let mut model = params(&[1.0]);
        let mut ema = Ema::new(&mut borrow(&mut model));
        ema.swap_in(&mut borrow(&mut model)).unwrap();
        assert!(ema.swap_in(&mut borrow(&mut model)).is_err());
        assert!(ema.update(&mut borrow(&mut model), 0.9).is_err());
        assert!(ema.save("/dev/null").is_err());
    }

    #[test]
    fn an_average_survives_a_save_and_a_load() {
        let mut model = params(&[1.0, 2.0]);
        let mut ema = Ema::new(&mut borrow(&mut model));
        for param in &mut model {
            param.value.data = vec![0.0; 2];
        }
        for _ in 0..10 {
            ema.update(&mut borrow(&mut model), 0.8).unwrap();
        }

        let path = std::env::temp_dir().join("rusting_brain_ema_roundtrip.bin");
        ema.save(&path).unwrap();
        let restored = Ema::load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(restored.updates(), ema.updates());
        for (a, b) in restored.weights().iter().zip(ema.weights()) {
            assert_eq!(a.rows, b.rows);
            assert_eq!(a.cols, b.cols);
            assert_eq!(a.data, b.data);
        }
        // And it keeps averaging from where it left off.
        let mut resumed = restored;
        resumed.update(&mut borrow(&mut model), 0.8).unwrap();
        assert_eq!(resumed.updates(), 11);
    }

    #[test]
    fn an_average_from_a_different_model_is_refused() {
        let mut small = params(&[1.0]);
        let mut ema = Ema::new(&mut borrow(&mut small));
        let mut large = params(&[1.0, 1.0]);
        assert!(ema.update(&mut borrow(&mut large), 0.9).is_err());
        assert!(ema.swap_in(&mut borrow(&mut large)).is_err());
    }

    #[test]
    fn a_decay_outside_the_unit_interval_is_refused() {
        let mut model = params(&[1.0]);
        let mut ema = Ema::new(&mut borrow(&mut model));
        assert!(ema.update(&mut borrow(&mut model), 1.0).is_err());
        assert!(ema.update(&mut borrow(&mut model), -0.1).is_err());
    }

    #[cfg(feature = "cuda")]
    fn cuda_or_skip(budget_mib: usize) -> Option<crate::param::CudaDevice> {
        match crate::cuda_training::cuda_doctor(0, budget_mib) {
            Ok(_) => {
                Some(crate::param::CudaDevice::new(0, budget_mib).expect("the CUDA doctor passed"))
            }
            Err(NetworkError::Cuda(message))
                if message.contains("NO_DEVICE") || message.contains("no CUDA-capable device") =>
            {
                None
            }
            Err(error) => panic!("CUDA is present but the CUDA doctor failed: {error}"),
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn an_average_reads_and_writes_a_parameter_that_lives_on_the_device() {
        let Some(device) = cuda_or_skip(64) else {
            return;
        };
        let mut param = Param::filled(4, 4, 1.0);
        param.to_cuda_on(&device).expect("the parameter moved");

        let mut ema = Ema::new(&mut [&mut param]);
        assert_eq!(ema.weights()[0].data, vec![1.0; 16]);

        // Move the resident weights, then average towards them.
        param
            .set_value(Matrix::from_vec(4, 4, vec![3.0; 16]))
            .unwrap();
        ema.update(&mut [&mut param], 0.5).unwrap();
        assert!((ema.weights()[0].data[0] - 2.0).abs() < 1e-6);

        // Swapping in has to reach the device, not just the stale host copy.
        ema.swap_in(&mut [&mut param]).unwrap();
        param.to_cpu().expect("the parameter came back");
        assert!((param.value.data[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn an_average_of_a_trained_parameter_lags_the_optimizer() {
        // One parameter descending towards 3.0 under Adam, averaged as it goes.
        let mut param = Param::filled(1, 1, 0.0);
        let optimizer = Optimizer::adam(0.1);
        let mut ema = Ema::new(&mut [&mut param]);
        for step in 1..=200 {
            param.grad.data[0] = param.value.data[0] - 3.0;
            param.step(&optimizer, step, 1.0);
            ema.update(&mut [&mut param], 0.99).unwrap();
        }
        let live = param.value.data[0];
        let averaged = ema.weights()[0].data[0];
        assert!(
            (live - 3.0).abs() < 0.05,
            "the live weight converged: {live}"
        );
        assert!(
            averaged < live,
            "the average trails a weight that rose: live {live}, average {averaged}"
        );
    }
}
