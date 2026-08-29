use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::sync::Arc;

pub struct GpuMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: CudaSlice<f32>,
    pub stream: Arc<CudaStream>,
    pub context: Arc<CudaContext>,
}

impl GpuMatrix {
    pub fn new(
        stream: Arc<CudaStream>,
        context: Arc<CudaContext>,
        rows: usize,
        cols: usize,
    ) -> Self {
        let data = stream.alloc_zeros::<f32>(rows * cols).unwrap();
        Self {
            rows,
            cols,
            data,
            stream,
            context,
        }
    }

    pub fn from_cpu(
        stream: Arc<CudaStream>,
        context: Arc<CudaContext>,
        cpu_matrix: &crate::matrix::Matrix,
    ) -> Self {
        let data = stream.clone_htod(&cpu_matrix.data).unwrap();
        Self {
            rows: cpu_matrix.rows,
            cols: cpu_matrix.cols,
            data,
            stream,
            context,
        }
    }

    pub fn to_cpu(&self, target: &mut crate::matrix::Matrix) {
        self.stream
            .memcpy_dtoh(&self.data, &mut target.data)
            .unwrap();
    }
}

pub fn gpu_dot(blas: &CudaBlas, a: &GpuMatrix, b: &GpuMatrix, c: &mut GpuMatrix) {
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

    unsafe {
        blas.gemm(cfg, &b.data, &a.data, &mut c.data).unwrap();
    }
}
