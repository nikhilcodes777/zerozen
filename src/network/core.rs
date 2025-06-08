use std::collections::VecDeque;

use anyhow::{Ok, Result};

use crate::{
    layer::{config::LayerConfig, core::Layer},
    linalg::matrix::Matrix,
    loss::functions::{LossFunction, LossKind},
};
#[derive(Debug, Clone)]
pub struct Network {
    layers: Vec<Layer>,
    learning_rate: f64,
    loss: LossKind,
}
impl Network {
    pub fn new(
        layer_configs: &[LayerConfig],
        no_of_features: usize,
        learning_rate: f64,
        loss: LossKind,
    ) -> Self {
        let mut layers = Vec::new();
        let mut current_input_size = no_of_features;

        for config in layer_configs {
            let layer = Layer::new(current_input_size, config);
            current_input_size = config.neurons;
            layers.push(layer);
        }
        Network {
            layers,
            learning_rate,
            loss,
        }
    }

    pub fn forward(&mut self, input: &Matrix) -> Result<Matrix> {
        let mut current_output = input.clone();
        for layer in self.layers.iter_mut() {
            current_output = layer.forward(&current_output)?;
        }
        Ok(current_output)
    }

    pub fn backward(&mut self, predictions: &Matrix, targets: &Matrix) {
        let mut d_output = self
            .loss
            .gradient(predictions, targets)
            .expect("Failed Calculating loss gradient");

        let mut layer_gradients = VecDeque::new();

        for layer in self.layers.iter_mut().rev() {
            let (grad_weights, grad_biases, grad_input) = layer.backward(&d_output);
            layer_gradients.push_front((grad_weights, grad_biases));
            d_output = grad_input;
        }

        for (layer, (grad_weights, grad_biases)) in
            self.layers.iter_mut().zip(layer_gradients.iter())
        {
            layer.update_parameters(grad_weights, grad_biases, self.learning_rate);
        }
    }
}
