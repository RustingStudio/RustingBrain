use rand::Rng;

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
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
        let data: Vec<f64> = (0..rows * cols).map(|_| rng.gen_range(0.0..1.0)).collect();
        Self { rows, cols, data }
    }

    pub fn zeros(&mut self) {
        for x in self.data.iter_mut() {
            *x = 0.0;
        }
    }

    pub fn dot(&self, other: &Matrix, target: &mut Matrix) {
        target.zeros();
        
        for i in 0..self.rows {
            for k in 0..self.cols {
                let r = self.data[i * self.cols + k];
                if r == 0.0 { continue; }
                
                let other_row_offset = k * other.cols;
                let target_row_offset = i * other.cols;
                
                for j in 0..other.cols {
                    target.data[target_row_offset + j] += r * other.data[other_row_offset + j];
                }
            }
        }
    }

    pub fn outer_product(&self, input: &Matrix, target: &mut Matrix) {
        for i in 0..self.rows {
            let error_val = self.data[i];
            let target_row_start = i * target.cols;
            
            for j in 0..input.rows { 
                target.data[target_row_start + j] = error_val * input.data[j];
            }
        }
    }

    pub fn dot_transpose_self(&self, error: &Matrix, target: &mut Matrix) {
        target.zeros();
        
        for i in 0..self.rows {
             let error_val = error.data[i];
             if error_val == 0.0 { continue; }

             for j in 0..self.cols {
                 target.data[j] += self.data[i * self.cols + j] * error_val;
             }
        }
    }

    pub fn copy_from_slice(&mut self, source: &[f64]) {
        self.data.copy_from_slice(source);
    }
}
