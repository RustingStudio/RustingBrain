use crate::matrix::Matrix;
use std::thread;

pub struct Gradients {
    pub d_weights: Vec<Matrix>,
    pub d_biases: Vec<Matrix>,
}

impl Gradients {
    pub fn new(layers: &[usize]) -> Self {
        let mut d_weights = Vec::new();
        let mut d_biases = Vec::new();

        for i in 0..layers.len() - 1 {
            let rows = layers[i + 1];
            let cols = layers[i];
            d_weights.push(Matrix::new(rows, cols));
            d_biases.push(Matrix::new(rows, 1));
        }

        Gradients { d_weights, d_biases }
    }

    pub fn zero(&mut self) {
        for m in &mut self.d_weights {
            m.zeros();
        }
        for m in &mut self.d_biases {
            m.zeros();
        }
    }

    pub fn add(&mut self, other: &Gradients) {
        for (a, b) in self.d_weights.iter_mut().zip(&other.d_weights) {
            for (x, y) in a.data.iter_mut().zip(&b.data) {
                *x += y;
            }
        }
        for (a, b) in self.d_biases.iter_mut().zip(&other.d_biases) {
            for (x, y) in a.data.iter_mut().zip(&b.data) {
                *x += y;
            }
        }
    }

    pub fn scale(&mut self, factor: f32) {
        for m in &mut self.d_weights {
            for x in &mut m.data {
                *x *= factor;
            }
        }
        for m in &mut self.d_biases {
            for x in &mut m.data {
                *x *= factor;
            }
        }
    }
}

pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    learning_rate: f32,
    
    activations: Vec<Matrix>,
    weighted_sums: Vec<Matrix>,
    errors: Vec<Matrix>,     
    gradients: Vec<Matrix>,  
}

impl Network {
    pub fn new(layers: Vec<usize>, learning_rate: f32) -> Self {
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
    fn relu(x: f32) -> f32 {
        if x > 0.0 { x } else { 0.0 }
    }

    #[inline(always)]
    fn relu_derivative(x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }

    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        self.activations[0].copy_from_slice(input);

        for i in 0..self.weights.len() {
            let (prev_a_slice, _current_z_slice) = self.activations.split_at(i + 1);
            let (_prev_w_sum_slice, current_w_sum_slice) = self.weighted_sums.split_at_mut(i + 1);
            
            let prev_a = &prev_a_slice[i];
            let current_z = &mut current_w_sum_slice[0];
            
            self.weights[i].dot(prev_a, current_z);
            
            let current_a = &mut self.activations[i+1];
            let is_last = i == self.weights.len() - 1;

            for j in 0..current_z.data.len() {
                let z = current_z.data[j] + self.biases[i].data[j];
                current_z.data[j] = z;
                current_a.data[j] = if is_last { z } else { Self::relu(z) };
            }
        }

        self.activations.last().unwrap().data.clone()
    }

    pub fn compute_gradients_single(
        &mut self,
        input: &[f32],
        target: &[f32],
        grads: &mut Gradients,
    ) {
        self.activations[0].copy_from_slice(input);

        for i in 0..self.weights.len() {
            let prev_a = &self.activations[i];
            let current_z = &mut self.weighted_sums[i + 1];

            self.weights[i].dot(prev_a, current_z);

            let current_a = &mut self.activations[i + 1];
            let bias = &self.biases[i];

            let is_last = i == self.weights.len() - 1;

            for j in 0..current_z.data.len() {
                let z = current_z.data[j] + bias.data[j];
                current_z.data[j] = z;
                current_a.data[j] = if is_last { z } else { Self::relu(z) };
            }
        }

        let last_idx = self.layers.len() - 1;
        let output_z = &self.weighted_sums[last_idx];
        let output_a = &self.activations[last_idx];
        let error = &mut self.errors[last_idx];

        for j in 0..error.data.len() {
            let e = target[j] - output_a.data[j];
            error.data[j] = e;
        }

        for i in (0..self.weights.len()).rev() {
            let curr_error = &self.errors[i + 1];
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

            let grad_matrix = &self.gradients[i];
            let err_vector = &self.errors[i + 1];

            let grad_w = &mut grads.d_weights[i];
            for j in 0..grad_w.data.len() {
                grad_w.data[j] += grad_matrix.data[j];
            }

            let grad_b = &mut grads.d_biases[i];
            for j in 0..grad_b.data.len() {
                grad_b.data[j] += err_vector.data[j];
            }
        }
    }

    pub fn apply_gradients(&mut self, grads: &Gradients, scale: f32) {
        let lr = self.learning_rate * scale;

        for (weight_matrix, grad_matrix) in self.weights.iter_mut().zip(&grads.d_weights) {
            for (w, g) in weight_matrix.data.iter_mut().zip(&grad_matrix.data) {
                *w += g * lr;
            }
        }

        for (bias_matrix, grad_b) in self.biases.iter_mut().zip(&grads.d_biases) {
            for (b, g) in bias_matrix.data.iter_mut().zip(&grad_b.data) {
                *b += g * lr;
            }
        }
    }

    pub fn train(&mut self, input: &[f32], target: &[f32]) {
        let mut grads = Gradients::new(&self.layers);
        grads.zero();
        self.compute_gradients_single(input, target, &mut grads);
        self.apply_gradients(&grads, 1.0);
    }

    pub fn train_batch_parallel(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        num_threads: usize,
    ) {
        assert_eq!(inputs.len(), targets.len());
        let batch_size = inputs.len();
        if batch_size == 0 {
            return;
        }

        let threads = num_threads.max(1).min(batch_size);
        let chunk_size = (batch_size + threads - 1) / threads;

        let layers = self.layers.clone();
        let weights = self.weights.clone();
        let biases = self.biases.clone();
        let learning_rate = self.learning_rate;

        let activations = self.activations.clone();
        let weighted_sums = self.weighted_sums.clone();
        let errors = self.errors.clone();
        let gradients = self.gradients.clone();

        let mut handles = Vec::new();

        for t in 0..threads {
            let start = t * chunk_size;
            if start >= batch_size {
                break;
            }
            let end = ((t + 1) * chunk_size).min(batch_size);

            let inputs_slice = inputs[start..end].to_vec();
            let targets_slice = targets[start..end].to_vec();

            let layers_clone = layers.clone();
            let weights_clone = weights.clone();
            let biases_clone = biases.clone();
            let activations_clone = activations.clone();
            let weighted_sums_clone = weighted_sums.clone();
            let errors_clone = errors.clone();
            let gradients_clone = gradients.clone();

            handles.push(thread::spawn(move || {
                let mut local_net = Network {
                    layers: layers_clone.clone(),
                    weights: weights_clone,
                    biases: biases_clone,
                    learning_rate,
                    activations: activations_clone,
                    weighted_sums: weighted_sums_clone,
                    errors: errors_clone,
                    gradients: gradients_clone,
                };

                let mut local_grads = Gradients::new(&layers_clone);
                local_grads.zero();

                for (x, y) in inputs_slice.iter().zip(targets_slice.iter()) {
                    local_net.compute_gradients_single(x, y, &mut local_grads);
                }

                local_grads
            }));
        }

        let mut total_grads = Gradients::new(&self.layers);
        total_grads.zero();

        for h in handles {
            let g = h.join().expect("thread panicked");
            total_grads.add(&g);
        }

        let scale = 1.0 / (batch_size as f32);
        total_grads.scale(scale);

        self.apply_gradients(&total_grads, 1.0);
    }
}
