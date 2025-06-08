use crate::{
    activations::functions::{ActivationFunction, ActivationKind},
    linalg::matrix::Matrix,
};
use anyhow::{anyhow, Result};

use super::config::LayerConfig;

#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: Matrix,
    pub biases: Matrix,
    pub activator: ActivationKind, // TODO: Implement way to pass custum functions too that implement ActivationFunction Trait

    input_cache: Option<Matrix>, // Input to this layer (a from previous layer)
    a_cache: Option<Matrix>,     // Output of this layer (after activation: activation_fn(z))
}
impl Layer {
    pub fn new(input_neurons: usize, config: &LayerConfig) -> Self {
        // Shape = (batch_size,features)
        let weights = Matrix::random(input_neurons, config.neurons, 0.0, 1.0);
        // Shape = (1,neurons)
        let biases = Matrix::zeros(1, config.neurons); // Batch Size
        Self {
            weights,
            biases,
            activator: config.activator,
            input_cache: None,
            a_cache: None,
        }
    }

    pub fn forward(&mut self, prev_input: &Matrix) -> Result<Matrix> {
        if prev_input.cols != self.weights.rows {
            return Err(anyhow!(
                "Input cols {} doesn't match layer's weight matrix row size {}",
                prev_input.rows,
                self.weights.cols
            ));
        }

        self.input_cache = Some(prev_input.clone());
        let z = prev_input
            .mul(&self.weights)?
            .add_bias_vector(&self.biases)?;
        let a = self.activator.activate(&z);
        self.a_cache = Some(a.clone());
        Ok(a)
    }
    // Takes
    // 1. dL/da_current
    // Returns
    // 1. grad_weights :- dL/dW
    // 2. grad_biases :- dL/db
    // 3. grad_input :- dL/da_prev
    pub fn backward(&mut self, d_output: &Matrix) -> (Matrix, Matrix, Matrix) {
        let a_current = self
            .a_cache
            .as_ref()
            .expect("a_cache not set in forward pass");
        let input_prev_layer_a = self
            .input_cache
            .as_ref()
            .expect("input_cache not set in forward pass");

        // Calculates dz_current
        let d_activation = self.activator.derivative(a_current);
        let d_z = d_output
            .hadamard(&d_activation)
            .expect("Hadamard for dz failed");

        // Calculates dL/dW_current = dL/dz  * dz/dw
        // dz/dw = prev_input.T
        let grad_weights = input_prev_layer_a
            .transpose()
            .mul(&d_z)
            .expect("Dot product for grad_weights failed");

        // Calculates dL/db_current = dL/dz * dz/db
        // dz/db = 1
        // Sum acrros all rows

        let grad_biases = d_z.sum_cols(); // Result is (1, neurons_in_this_layer)
                                          // dL/da_prev = dL/dz * dz/da_prev
                                          // dz = da_prev =W_current.T
        let grad_input = d_z
            .mul(&self.weights.transpose())
            .expect("Dot product for grad_input failed");

        (grad_weights, grad_biases, grad_input)
    }

    pub fn update_parameters(
        &mut self,
        grad_weights: &Matrix,
        grad_biases: &Matrix,
        learning_rate: f64,
    ) {
        self.weights = self
            .weights
            .sub(&grad_weights.scale(learning_rate))
            .expect("Failed to update weights");
        self.biases = self
            .biases
            .sub(&grad_biases.scale(learning_rate))
            .expect("Failed to update biases");
    }
}
