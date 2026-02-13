use rand::Rng;

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(0.0..1.0)).collect();
        Self { rows, cols, data }
    }

    pub fn zeros(&mut self) {
        self.data.fill(0.0);
    }

    pub fn dot(&self, other: &Matrix, target: &mut Matrix) {
        debug_assert_eq!(self.cols, other.rows);
        debug_assert_eq!(target.rows, self.rows);
        debug_assert_eq!(target.cols, other.cols);

        unsafe {
            matrixmultiply::sgemm(
                self.rows,
                self.cols,
                other.cols,
                1.0,
                self.data.as_ptr(),
                self.cols as isize,
                1,
                other.data.as_ptr(),
                other.cols as isize,
                1,
                0.0,
                target.data.as_mut_ptr(),
                target.cols as isize,
                1,
            );
        }
    }

    pub fn dot_rhs_transposed(&self, other: &Matrix, target: &mut Matrix) {
        debug_assert_eq!(self.cols, other.cols);
        debug_assert_eq!(target.rows, self.rows);
        debug_assert_eq!(target.cols, other.rows);

        unsafe {
            matrixmultiply::sgemm(
                self.rows,
                self.cols,
                other.rows,
                1.0,
                self.data.as_ptr(),
                self.cols as isize,
                1,
                other.data.as_ptr(),
                1,
                other.cols as isize,
                0.0,
                target.data.as_mut_ptr(),
                target.cols as isize,
                1,
            );
        }
    }

    pub fn dot_self_transposed(&self, other: &Matrix, target: &mut Matrix) {
        debug_assert_eq!(self.rows, other.rows);
        debug_assert_eq!(target.rows, self.cols);
        debug_assert_eq!(target.cols, other.cols);

        unsafe {
            matrixmultiply::sgemm(
                self.cols,
                self.rows,
                other.cols,
                1.0,
                self.data.as_ptr(),
                1,
                self.cols as isize,
                other.data.as_ptr(),
                other.cols as isize,
                1,
                0.0,
                target.data.as_mut_ptr(),
                target.cols as isize,
                1,
            );
        }
    }

    pub fn outer_product(&self, input: &Matrix, target: &mut Matrix) {
        debug_assert_eq!(input.cols, 1);
        debug_assert_eq!(target.rows, self.rows);
        debug_assert_eq!(target.cols, input.rows);

        unsafe {
            matrixmultiply::sgemm(
                self.rows,
                1,
                input.rows,
                1.0,
                self.data.as_ptr(),
                self.cols as isize,
                1,
                input.data.as_ptr(),
                1,
                input.cols as isize,
                0.0,
                target.data.as_mut_ptr(),
                target.cols as isize,
                1,
            );
        }
    }

    pub fn dot_transpose_self(&self, error: &Matrix, target: &mut Matrix) {
        debug_assert_eq!(self.rows, error.rows);
        debug_assert_eq!(target.rows, self.cols);
        debug_assert_eq!(target.cols, 1);

        unsafe {
            matrixmultiply::sgemm(
                self.cols,
                self.rows,
                1,
                1.0,
                self.data.as_ptr(),
                1,
                self.cols as isize,
                error.data.as_ptr(),
                error.cols as isize,
                1,
                0.0,
                target.data.as_mut_ptr(),
                target.cols as isize,
                1,
            );
        }
    }

    pub fn copy_from_slice(&mut self, source: &[f32]) {
        self.data.copy_from_slice(source);
    }
}   
