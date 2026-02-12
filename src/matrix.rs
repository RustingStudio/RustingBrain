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
        for x in self.data.iter_mut() {
            *x = 0.0;
        }
    }

    #[inline(always)]
    pub fn dot(&self, other: &Matrix, target: &mut Matrix) {
        debug_assert_eq!(self.cols, other.rows);
        debug_assert_eq!(target.rows, self.rows);
        debug_assert_eq!(target.cols, other.cols);

        target.zeros();

        let m = self.rows;
        let n = self.cols;
        let p = other.cols;

        let a = &self.data;
        let b = &other.data;
        let c = &mut target.data;

        for i in 0..m {
            let a_row = &a[i * n..(i + 1) * n];
            let c_row = &mut c[i * p..(i + 1) * p];

            for k in 0..n {
                let r = a_row[k];
                if r == 0.0 {
                    continue;
                }
                let b_row = &b[k * p..(k + 1) * p];

                for j in 0..p {
                    c_row[j] += r * b_row[j];
                }
            }
        }
    }

    #[inline(always)]
    pub fn outer_product(&self, input: &Matrix, target: &mut Matrix) {
        debug_assert_eq!(input.cols, 1);
        debug_assert_eq!(target.rows, self.rows);
        debug_assert_eq!(target.cols, input.rows);

        let rows = self.rows;
        let cols = input.rows;

        let err = &self.data;
        let inp = &input.data;
        let out = &mut target.data;

        for i in 0..rows {
            let error_val = err[i];
            let row = &mut out[i * cols..(i + 1) * cols];

            for j in 0..cols {
                row[j] = error_val * inp[j];
            }
        }
    }

    #[inline(always)]
    pub fn dot_transpose_self(&self, error: &Matrix, target: &mut Matrix) {
        debug_assert_eq!(self.rows, error.rows);
        debug_assert_eq!(target.rows, self.cols);
        debug_assert_eq!(target.cols, 1);

        target.zeros();

        let rows = self.rows;
        let cols = self.cols;

        let a = &self.data;
        let e = &error.data;
        let t = &mut target.data;

        for i in 0..rows {
            let error_val = e[i];
            if error_val == 0.0 {
                continue;
            }

            let row = &a[i * cols..(i + 1) * cols];
            for j in 0..cols {
                t[j] += row[j] * error_val;
            }
        }
    }

    pub fn copy_from_slice(&mut self, source: &[f32]) {
        self.data.copy_from_slice(source);
    }
}
