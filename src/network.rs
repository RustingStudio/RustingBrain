use crate::matrix::Matrix;


pub struct Activation {
    pub function: fn(f64) -> f64,
    pub derivative: fn(f64) -> f64,
}

pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    activation: Activation,
    learning_rate: f64,
}

impl Network {
    
    pub fn new(layers: Vec<usize>, activation: Activation, learning_rate: f64) -> Self {
        
        let mut weights = vec![];

        let mut biases = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i+1], layers[i]));
            biases.push(Matrix::random(layers[i+1], 1));
        }

        Network {
            layers,
            weights,
            biases,
            data: vec![],
            activation,
            learning_rate
        }

    }

    pub fn feed_forward(&mut self, inputs: Matrix) -> Matrix {
        
        assert!(self.layers[0] == inputs.data.len(), "Invalid number of Inputs");

        let mut current: Matrix = inputs;

        self.data = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .dot_multiply(&current)
                .add(&self.biases[i]).map(self.activation.function);

            self.data.push(current.clone());
        }

        current

    }

    pub fn back_propagate(&mut self, inputs: Matrix, targets: Matrix) {
        let outputs = self.feed_forward(inputs);

        let mut error = targets.subtract(&outputs);

        for i in (0..self.layers.len() - 1).rev() {
            
            let outputs = &self.data[i + 1];
            
            let gradients = outputs.map(self.activation.derivative);
            
            let mut gradients = gradients.multiply_elementwise(&error);
            
            gradients = gradients.multiply_scalar(self.learning_rate);

            let inputs_t = self.data[i].transpose();
            let weight_deltas = gradients.dot_multiply(&inputs_t);

            self.weights[i] = self.weights[i].add(&weight_deltas);
            self.biases[i] = self.biases[i].add(&gradients);

            let weights_t = self.weights[i].transpose();
            error = weights_t.dot_multiply(&error);
        }
    }

}
