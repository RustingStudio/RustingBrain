use rand::Rng;

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
   
    pub fn map<F>(&self, f: F) -> Matrix
    where
        F: Fn(f64) -> f64,
    {
        let data = self.data.iter()
            .map(|&x| f(x))
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut buffer = Vec::with_capacity(rows * cols);
        let mut rng = rand::thread_rng();

        for _ in 0..rows * cols {
            buffer.push(rng.gen_range(0.0..1.0));
        }

        Matrix { rows, cols, data: buffer }
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Can`t add Matrices with different dimensions :<");
        }

        let mut buffer: Vec<f64> = Vec::<f64>::with_capacity(self.rows * self.cols);

        for i in 0..self.data.len() {

            let result: f64 = self.data[i] + other.data[i];

            buffer.push(result);

        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: buffer
        }
    }
    
    pub fn subtract(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Can't subtract Matrices with different dimensions (o_0)");
        }
        
        let mut buffer: Vec<f64> = Vec::<f64>::with_capacity(self.rows * self.cols);

        for i in 0..self.data.len() {

            let result: f64 = self.data[i] - other.data[i];

            buffer.push(result);

        }
    
    Matrix {
        rows: self.rows,
        cols: self.cols,
        data: buffer
    }
    }
    
   
    pub fn dot_multiply(&self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!("Can't multiply Matrices with different dimensions (>_<)");
        }

        let mut data = vec![0.0; self.rows * other.cols];

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k]
                        * other.data[k * other.cols + j];
                }
                data[i * other.cols + j] = sum;
            }
        }

        Matrix {
            rows: self.rows,
            cols: other.cols,
            data,
        }
    }

    pub fn transpose(&self) -> Matrix {
        let mut data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Matrix {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    pub fn multiply_elementwise(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrix dimensions must match for element-wise multiplication.");
        }
        let data = self.data.iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn multiply_scalar(&self, scalar: f64) -> Matrix {
        let data = self.data.iter().map(|val| val * scalar).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random() {
        let matrix = Matrix::random(3, 4);
        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.cols, 4);
        assert_eq!(matrix.data.len(), 12);
        
        for &value in &matrix.data {
            assert!(value >= 0.0 && value < 1.0);
        }
    }

    #[test]
    fn test_add_same_dimensions() {
        let m1 = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let m2 = Matrix {
            rows: 2,
            cols: 2,
            data: vec![5.0, 6.0, 7.0, 8.0],
        };
        
        let result = m1.add(&m2);
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    #[should_panic(expected = "Can`t add Matrices with different dimensions")]
    fn test_add_different_dimensions() {
        let m1 = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let m2 = Matrix {
            rows: 3,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        
        let _ = m1.add(&m2);
    }

    #[test]
    fn test_subtract_same_dimensions() {
        let m1 = Matrix {
            rows: 2,
            cols: 2,
            data: vec![5.0, 6.0, 7.0, 8.0],
        };
        let m2 = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        
        let result = m1.subtract(&m2);
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
        assert_eq!(result.data, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "Can't subtract Matrices with different dimensions")]
    fn test_subtract_different_dimensions() {
        let m1 = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let m2 = Matrix {
            rows: 3,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        
        let _ = m1.subtract(&m2);
    }

    #[test]
    fn test_dot_product() {
        let m1 = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let m2 = Matrix {
            rows: 3,
            cols: 2,
            data: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        };
        
        let result = m1.dot_multiply(&m2);
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
        // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12]
        // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]
        // [58, 64, 139, 154]
        assert_eq!(result.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    #[should_panic(expected = "Can't multiply Matrices with different dimensions")]
    fn test_dot_product_invalid_dimensions() {
        let m1 = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let m2 = Matrix {
            rows: 3,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        
        let _ = m1.dot_multiply(&m2);
    }
}

impl From<Vec<Vec<f64>>> for Matrix {
    fn from(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = data[0].len();
        let mut flat_data = Vec::with_capacity(rows * cols);

        for row in data {
            assert_eq!(row.len(), cols, "All rows must have the same number of columns");
            flat_data.extend(row);
        }

        Matrix {
            rows,
            cols,
            data: flat_data,
        }
    }
}
