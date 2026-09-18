//! A device-resident matrix and the cuBLAS call that multiplies two of them.
//!
//! This exists for the matmul benchmark in [`gpu_test`](crate::gpu_test) and
//! nothing else. Both trained paths own their device buffers directly: dense
//! networks through [`Param`](crate::param::Param), transformers through
//! [`gpu_model`](crate::gpu_model).

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t};
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

/// Errors carry a message rather than a type because the single caller is a
/// diagnostic that prints them.
pub(crate) struct GpuMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: CudaSlice<f32>,
}

impl GpuMatrix {
    pub fn zeros(stream: &Arc<CudaStream>, rows: usize, cols: usize) -> Result<Self, String> {
        let data = stream
            .alloc_zeros::<f32>(rows * cols)
            .map_err(|e| format!("device allocation of {rows}x{cols} floats failed: {e}"))?;
        Ok(Self { rows, cols, data })
    }

    pub fn from_cpu(
        stream: &Arc<CudaStream>,
        cpu_matrix: &crate::matrix::Matrix,
    ) -> Result<Self, String> {
        let data = stream
            .clone_htod(&cpu_matrix.data)
            .map_err(|e| format!("host-to-device copy failed: {e}"))?;
        Ok(Self {
            rows: cpu_matrix.rows,
            cols: cpu_matrix.cols,
            data,
        })
    }
}

/// `c = a * b`, row-major.
///
/// cuBLAS is column-major, so the operands are passed in the other order: a
/// row-major product is the column-major product of the transposes, and
/// swapping the arguments gets that for free instead of transposing anything.
pub(crate) fn gpu_dot(
    blas: &CudaBlas,
    a: &GpuMatrix,
    b: &GpuMatrix,
    c: &mut GpuMatrix,
) -> Result<(), String> {
    debug_assert_eq!(a.cols, b.rows);
    debug_assert_eq!(c.rows, a.rows);
    debug_assert_eq!(c.cols, b.cols);

    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: b.cols as i32,
        n: a.rows as i32,
        k: a.cols as i32,
        alpha: 1.0f32,
        lda: b.cols as i32,
        ldb: a.cols as i32,
        beta: 0.0f32,
        ldc: c.cols as i32,
    };

    unsafe { blas.gemm(cfg, &b.data, &a.data, &mut c.data) }
        .map_err(|e| format!("cuBLAS sgemm failed: {e}"))
}
