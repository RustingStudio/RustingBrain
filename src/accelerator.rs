//! Backend-agnostic accelerator training.
//!
//! `cuda_training` and `metal_training` are two implementations of the same
//! contract: a persistent session that owns device buffers, trains one epoch at
//! a time, and can hand back a host checkpoint. This module is the part of that
//! contract that does not depend on either API, so it compiles on every
//! platform and in every feature combination. Applications should drive
//! training through [`TrainingSession`] and read [`AcceleratorDoctorReport`]
//! rather than reaching for a vendor module directly.

use crate::{
    CudaTrainingCheckpoint, Dataset, Network, NetworkError, TrainConfig, TrainingBackend,
};
use std::time::Duration;

pub(crate) const MIB: usize = 1024 * 1024;

/// Measurements for the lifetime of a persistent accelerator training session.
#[derive(Clone, Debug, Default)]
pub struct AcceleratorStats {
    /// Bytes owned by device allocations at the high-water mark (driver overhead excluded).
    pub peak_allocated_bytes: usize,
    pub setup_time: Duration,
    pub training_time: Duration,
    pub checkpoint_time: Duration,
    pub epochs: usize,
    pub batches: usize,
    pub host_to_device_bytes: usize,
    pub device_to_host_bytes: usize,
    pub dataset_resident: bool,
}

/// What a device offers, gathered before a campaign commits to it.
#[derive(Clone, Debug)]
pub struct AcceleratorDoctorReport {
    /// `"cuda"` or `"metal"`.
    pub backend: &'static str,
    pub device: usize,
    pub name: String,
    /// CUDA compute capability, or the Metal GPU family.
    pub revision: String,
    pub total_memory_mib: usize,
    pub free_memory_mib: Option<usize>,
    pub requested_budget_mib: usize,
    pub allocation_test: bool,
    /// cuBLAS on CUDA; the bundled tiled GEMM kernels on Metal.
    pub gemm_available: bool,
    pub kernel_available: bool,
}

/// Probes the device behind `backend`, or `Ok(None)` for [`TrainingBackend::Cpu`].
///
/// Fails rather than degrading: a backend that was not compiled in reports the
/// missing feature instead of quietly answering for the CPU.
pub fn accelerator_doctor(
    backend: TrainingBackend,
) -> Result<Option<AcceleratorDoctorReport>, NetworkError> {
    match backend {
        TrainingBackend::Cpu => Ok(None),
        TrainingBackend::Cuda {
            device,
            memory_budget_mib,
        } => {
            #[cfg(feature = "cuda")]
            {
                let r = crate::cuda_training::cuda_doctor(device, memory_budget_mib)?;
                Ok(Some(AcceleratorDoctorReport {
                    backend: "cuda",
                    device: r.device,
                    name: r.name,
                    revision: r.compute_capability,
                    total_memory_mib: r.total_memory_mib,
                    free_memory_mib: r.free_memory_mib,
                    requested_budget_mib: r.requested_budget_mib,
                    allocation_test: r.allocation_test,
                    gemm_available: r.cublas_available,
                    kernel_available: r.kernel_available,
                }))
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (device, memory_budget_mib);
                Err(NetworkError::CudaFeatureDisabled)
            }
        }
        TrainingBackend::Metal {
            device,
            memory_budget_mib,
        } => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                Ok(Some(crate::metal_training::metal_doctor(
                    device,
                    memory_budget_mib,
                )?))
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                let _ = (device, memory_budget_mib);
                Err(NetworkError::MetalFeatureDisabled)
            }
        }
    }
}

/// Conservative tensor allocation estimate (MiB, rounded up). This is public so
/// applications can reject an unsuitable configuration before touching a device.
pub fn estimate_tensor_memory_mib(
    network: &Network,
    batch_size: usize,
) -> Result<usize, NetworkError> {
    let overflow = || NetworkError::Accelerator("tensor size overflow".into());
    if batch_size == 0 {
        return Err(NetworkError::Accelerator(
            "batch size must be non-zero".into(),
        ));
    }
    let mut floats = batch_size
        .checked_mul(network.input_size())
        .ok_or_else(overflow)?;
    for l in network.layers() {
        let p = l.weights.data.len() + l.biases.data.len();
        // parameter, gradient, and both Adam moments (also allocated for SGD,
        // so the estimate is a stable upper bound for either optimizer)
        floats = floats
            .checked_add(p.checked_mul(4).ok_or_else(overflow)?)
            .ok_or_else(overflow)?;
        let a = batch_size
            .checked_mul(l.weights.rows)
            .ok_or_else(overflow)?;
        // activation and delta
        floats = floats
            .checked_add(a.checked_mul(2).ok_or_else(overflow)?)
            .ok_or_else(overflow)?;
    }
    // the epoch loss accumulator
    floats = floats.checked_add(1).ok_or_else(overflow)?;
    Ok(floats
        .checked_mul(4)
        .ok_or_else(overflow)?
        .div_ceil(MIB))
}

/// A persistent device session, whichever backend is behind it.
///
/// Epoch-at-a-time training exists because a campaign checkpoints and
/// early-stops after every epoch. Calling `fit_with_backend` once per epoch
/// would rebuild the device context, the compiled kernels, and every buffer
/// each time; a session keeps all of that, and the training set, resident.
pub enum TrainingSession {
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda_training::CudaTrainingSession),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(crate::metal_training::MetalTrainingSession),
}

impl TrainingSession {
    /// Opens a session for `backend`, or `Ok(None)` when the request is
    /// [`TrainingBackend::Cpu`] and the caller should use [`Network::fit`].
    pub fn new(
        network: &Network,
        dataset: &Dataset,
        config: TrainConfig,
        backend: TrainingBackend,
    ) -> Result<Option<Self>, NetworkError> {
        match backend {
            TrainingBackend::Cpu => Ok(None),
            TrainingBackend::Cuda {
                device,
                memory_budget_mib,
            } => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Some(Self::Cuda(
                        crate::cuda_training::CudaTrainingSession::new(
                            network,
                            dataset,
                            config,
                            device,
                            memory_budget_mib,
                        )?,
                    )))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (network, dataset, config, device, memory_budget_mib);
                    Err(NetworkError::CudaFeatureDisabled)
                }
            }
            TrainingBackend::Metal {
                device,
                memory_budget_mib,
            } => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    Ok(Some(Self::Metal(
                        crate::metal_training::MetalTrainingSession::new(
                            network,
                            dataset,
                            config,
                            device,
                            memory_budget_mib,
                        )?,
                    )))
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    let _ = (network, dataset, config, device, memory_budget_mib);
                    Err(NetworkError::MetalFeatureDisabled)
                }
            }
        }
    }

    /// Resumes a session from a host checkpoint, replaying the shuffle so the
    /// resumed run sees the same batches it would have seen without the crash.
    pub fn from_checkpoint(
        checkpoint: CudaTrainingCheckpoint,
        dataset: &Dataset,
        config: TrainConfig,
        backend: TrainingBackend,
    ) -> Result<Option<Self>, NetworkError> {
        match backend {
            TrainingBackend::Cpu => Ok(None),
            TrainingBackend::Cuda {
                device,
                memory_budget_mib,
            } => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Some(Self::Cuda(
                        crate::cuda_training::CudaTrainingSession::from_checkpoint(
                            checkpoint,
                            dataset,
                            config,
                            device,
                            memory_budget_mib,
                        )?,
                    )))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (checkpoint, dataset, config, device, memory_budget_mib);
                    Err(NetworkError::CudaFeatureDisabled)
                }
            }
            TrainingBackend::Metal {
                device,
                memory_budget_mib,
            } => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    Ok(Some(Self::Metal(
                        crate::metal_training::MetalTrainingSession::from_checkpoint(
                            checkpoint,
                            dataset,
                            config,
                            device,
                            memory_budget_mib,
                        )?,
                    )))
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    let _ = (checkpoint, dataset, config, device, memory_budget_mib);
                    Err(NetworkError::MetalFeatureDisabled)
                }
            }
        }
    }

    pub fn train_epoch(&mut self) -> Result<f32, NetworkError> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(session) => session.train_epoch(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(session) => session.train_epoch(),
            #[allow(unreachable_patterns)]
            _ => Err(NetworkError::Accelerator(
                "no accelerator backend is compiled in".into(),
            )),
        }
    }

    pub fn checkpoint(&mut self) -> Result<CudaTrainingCheckpoint, NetworkError> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(session) => session.checkpoint(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(session) => session.checkpoint(),
            #[allow(unreachable_patterns)]
            _ => Err(NetworkError::Accelerator(
                "no accelerator backend is compiled in".into(),
            )),
        }
    }

    /// Copies device state back into `network`, so host-side validation and
    /// campaign checkpoints see the weights this epoch produced.
    pub fn synchronize_network(&mut self, network: &mut Network) -> Result<(), NetworkError> {
        network.restore_cuda_checkpoint(self.checkpoint()?)
    }

    pub fn stats(&self) -> &AcceleratorStats {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(session) => session.stats(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(session) => session.stats(),
            #[allow(unreachable_patterns)]
            _ => unreachable!("TrainingSession has no variant without a backend feature"),
        }
    }

    pub fn epoch(&self) -> usize {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(session) => session.epoch(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(session) => session.epoch(),
            #[allow(unreachable_patterns)]
            _ => 0,
        }
    }

    /// True only for backends with a device-side validation path (CUDA
    /// today). Callers should keep using the host `evaluate_loss`
    /// round-trip on any backend this returns `false` for, Metal included:
    /// it has no `prepare_validation`/`validate_epoch` implementation.
    pub fn supports_device_validation(&self) -> bool {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => true,
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// Uploads `dataset` once so `validate_epoch` can score it on-device.
    /// Only call when `supports_device_validation()` is true. Returns
    /// `Ok(false)` (not an error) when it would not fit the device budget
    /// alongside what training already uses; the caller should fall back to
    /// host-side validation for the whole session in that case.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    pub fn prepare_validation(
        &mut self,
        dataset: &Dataset,
        budget_mib: usize,
    ) -> Result<bool, NetworkError> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(session) => session.prepare_validation(dataset, budget_mib),
            #[allow(unreachable_patterns)]
            _ => Err(NetworkError::Accelerator(
                "device-side validation is not implemented for this backend".into(),
            )),
        }
    }

    /// Device-side validation loss over the set uploaded by
    /// `prepare_validation`, using the weights currently resident on the
    /// device. No host copy of the model is needed to compute it.
    pub fn validate_epoch(&mut self) -> Result<f32, NetworkError> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(session) => session.validate_epoch(),
            #[allow(unreachable_patterns)]
            _ => Err(NetworkError::Accelerator(
                "device-side validation is not implemented for this backend".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Activation, Optimizer};

    fn tiny() -> Network {
        Network::builder()
            .input_size(2)
            .dense(3, Activation::Relu)
            .dense(1, Activation::Linear)
            .optimizer(Optimizer::sgd(0.1))
            .build()
    }

    #[test]
    fn cpu_backend_has_no_session_and_no_doctor_report() {
        let network = tiny();
        let data = Dataset::new(vec![vec![0.0, 0.0]], vec![vec![0.0]]);
        let config = TrainConfig {
            epochs: 1,
            batch_size: 1,
            shuffle: false,
            seed: Some(1),
        };
        assert!(
            TrainingSession::new(&network, &data, config, TrainingBackend::Cpu)
                .unwrap()
                .is_none()
        );
        assert!(accelerator_doctor(TrainingBackend::Cpu).unwrap().is_none());
    }

    #[test]
    fn memory_estimate_is_nonzero_and_rejects_a_zero_batch() {
        assert!(estimate_tensor_memory_mib(&tiny(), 4).unwrap() >= 1);
        assert!(estimate_tensor_memory_mib(&tiny(), 0).is_err());
    }

    #[test]
    fn metal_is_rejected_when_it_is_not_compiled_in() {
        let backend = TrainingBackend::Metal {
            device: 0,
            memory_budget_mib: 1024,
        };
        assert_eq!(backend.name(), "metal");
        assert!(backend.is_accelerated());
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        assert!(matches!(
            accelerator_doctor(backend),
            Err(NetworkError::MetalFeatureDisabled)
        ));
    }
}
