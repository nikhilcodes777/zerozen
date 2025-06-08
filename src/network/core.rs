use core::fmt;
use std::collections::VecDeque;

use anyhow::{Ok, Result};

use crate::{
    layer::core::Layer,
    linalg::matrix::Matrix,
    loss::functions::{LossFunction, LossKind},
    network::builder::NetworkBuilder,
};
#[derive(Debug, Clone)]
pub struct Network {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
    pub loss: LossKind,
    pub loss_history: Vec<(usize, f64)>,
    pub epochs: usize,
    pub logging: bool,
    pub log_level: usize,
}

impl Network {
    pub fn builder() -> NetworkBuilder {
        NetworkBuilder::new()
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

    pub fn train(&mut self, input: &Matrix, targets: &Matrix) -> Result<()> {
        for i in 0..self.epochs {
            let predictions = self.forward(&input)?;
            if i % self.log_level == 0 {
                let current_loss = self.loss.loss(&predictions, &targets)?;
                self.loss_history.push((i, current_loss));
                if self.logging {
                    println!("Epoch: {}, Loss: {:.7}", i, current_loss);
                }
            }
            self.backward(&predictions, &targets);
        }

        return Ok(());
    }
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{:-<65}", "")?;
        writeln!(f, "{:^65}", "Neural Network Summary")?;
        writeln!(f, "{:=<65}", "")?;
        writeln!(
            f,
            "{:<10} | {:<15} | {:<15} | {:<15}",
            "Layer", "Input Shape", "Output Shape", "Activation"
        )?;
        writeln!(f, "{:-<65}", "")?;

        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(
                f,
                "{:<10} | {:<15} | {:<15} | {:<15?}",
                format!("Dense {}", i),
                layer.weights.rows,
                layer.weights.cols,
                layer.activator,
            )?;
        }

        writeln!(f, "{:-<65}", "")?;
        writeln!(f, "Loss Function: {:?}", self.loss)?;
        writeln!(f, "Learning Rate: {}", self.learning_rate)?;
        writeln!(f, "Epochs: {}", self.epochs)?;
        writeln!(f, "{:-<65}", "")
    }
}
