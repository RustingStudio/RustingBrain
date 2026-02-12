use crate::matrix::Matrix;

pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    learning_rate: f64,
    
    activations: Vec<Matrix>,
    weighted_sums: Vec<Matrix>,
    errors: Vec<Matrix>,     
    gradients: Vec<Matrix>,  
}

impl Network {
    pub fn new(layers: Vec<usize>, learning_rate: f64) -> Self {
        let mut weights = vec![];
        let mut biases = vec![];
        
        let mut activations = vec![];
        let mut weighted_sums = vec![];
        let mut errors = vec![];
        let mut gradients = vec![];

        activations.push(Matrix::new(layers[0], 1));
        weighted_sums.push(Matrix::new(layers[0], 1)); 
        errors.push(Matrix::new(layers[0], 1));

        for i in 0..layers.len() - 1 {
            let rows = layers[i + 1];
            let cols = layers[i];
            
            weights.push(Matrix::random(rows, cols));
            biases.push(Matrix::random(rows, 1));
            
            activations.push(Matrix::new(rows, 1));
            weighted_sums.push(Matrix::new(rows, 1));
            errors.push(Matrix::new(rows, 1));
            gradients.push(Matrix::new(rows, cols));
        }

        Network {
            layers,
            weights,
            biases,
            learning_rate,
            activations,
            weighted_sums,
            errors,
            gradients,
        }
    }

    #[inline(always)]
    fn relu(x: f64) -> f64 {
        if x > 0.0 { x } else { 0.0 }
    }

    #[inline(always)]
    fn relu_derivative(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        self.activations[0].copy_from_slice(input);

        for i in 0..self.weights.len() {
            let (prev_a_slice, _current_z_slice) = self.activations.split_at(i + 1);
            let (_prev_w_sum_slice, current_w_sum_slice) = self.weighted_sums.split_at_mut(i + 1);
            
            let prev_a = &prev_a_slice[i];
            let current_z = &mut current_w_sum_slice[0];
            
            self.weights[i].dot(prev_a, current_z);
            
            let current_a = &mut self.activations[i+1];
            
            for j in 0..current_z.data.len() {
                current_z.data[j] += self.biases[i].data[j];
                current_a.data[j] = Self::relu(current_z.data[j]);
            }
        }

        self.activations.last().unwrap().data.clone()
    }

    pub fn train(&mut self, input: &[f64], target: &[f64]) {
        self.activations[0].copy_from_slice(input);

        for i in 0..self.weights.len() {
            let prev_a = &self.activations[i];
            let current_z = &mut self.weighted_sums[i+1];
            
            self.weights[i].dot(prev_a, current_z);
            
            let current_a = &mut self.activations[i+1];
            let bias = &self.biases[i];
            
            for j in 0..current_z.data.len() {
                let z = current_z.data[j] + bias.data[j];
                current_z.data[j] = z;
                current_a.data[j] = Self::relu(z);
            }
        }

        let last_idx = self.layers.len() - 1;
        let output_z = &self.weighted_sums[last_idx]; 
        let output_a = &self.activations[last_idx];
        let error = &mut self.errors[last_idx];

        for j in 0..error.data.len() {
            let e = target[j] - output_a.data[j];
            error.data[j] = e * Self::relu_derivative(output_z.data[j]);
        }

        for i in (0..self.weights.len()).rev() {
            
            let curr_error = &self.errors[i+1];
            let prev_activation = &self.activations[i];
            let gradient = &mut self.gradients[i];
            
            curr_error.outer_product(prev_activation, gradient);

            if i > 0 {
                let weight = &self.weights[i];
                let prev_z = &self.weighted_sums[i];
                
                let (prev_errs, curr_errs) = self.errors.split_at_mut(i + 1);
                let target_prev_error = &mut prev_errs[i];
                let source_curr_error = &curr_errs[0];

                weight.dot_transpose_self(source_curr_error, target_prev_error);
                
                for j in 0..target_prev_error.data.len() {
                    target_prev_error.data[j] *= Self::relu_derivative(prev_z.data[j]);
                }
            }

            let weight_matrix = &mut self.weights[i];
            let bias_matrix = &mut self.biases[i];
            let grad_matrix = &self.gradients[i];
            let err_vector = &self.errors[i+1];

            let lr = self.learning_rate;

            for j in 0..weight_matrix.data.len() {
                weight_matrix.data[j] += grad_matrix.data[j] * lr;
            }

            for j in 0..bias_matrix.data.len() {
                bias_matrix.data[j] += err_vector.data[j] * lr;
            }
        }
    }
}
