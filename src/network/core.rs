use core::fmt;
use std::collections::VecDeque;

use anyhow::{Ok, Result};
use rand::seq::SliceRandom;

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
    pub batch_size: Option<usize>,
    pub shuffle: bool,
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

            let current_loss = self.loss.loss(&predictions, &targets)?;
            self.loss_history.push((i, current_loss));
            if i % self.log_level == 0 || i == self.epochs - 1 {
                if self.logging {
                    println!("Epoch: {}, Loss: {:.7}", i, current_loss);
                }
            }
            self.backward(&predictions, &targets);
        }

        return Ok(());
    }

    pub fn train_sgd(&mut self, input: &Matrix, targets: &Matrix) -> Result<()> {
        let num_samples = input.rows;
        let num_batches = (num_samples + self.batch_size.unwrap() - 1) / self.batch_size.unwrap();

        let mut indices: Vec<usize> = (0..num_samples).collect();

        let mut rng = rand::rng();

        for epoch in 0..self.epochs {
            if self.shuffle {
                indices.shuffle(&mut rng);
            }

            let mut epoch_loss = 0.0;
            let mut batch_count = 0;

            for batch_idx in 0..num_batches {
                let batch_start = batch_idx * self.batch_size.unwrap();
                let batch_end = std::cmp::min(batch_start + self.batch_size.unwrap(), num_samples);
                let current_batch_size = batch_end - batch_start;

                let batch_input = self.extract_batch(input, &indices, batch_start, batch_end)?;
                let batch_targets =
                    self.extract_batch(targets, &indices, batch_start, batch_end)?;

                let predictions = self.forward(&batch_input)?;

                let batch_loss = self.loss.loss(&predictions, &batch_targets)?;
                epoch_loss += batch_loss * current_batch_size as f64;
                batch_count += current_batch_size;
                self.backward(&predictions, &batch_targets);
            }

            epoch_loss /= batch_count as f64;

            if epoch % self.log_level == 0 || epoch == self.epochs - 1 {
                self.loss_history.push((epoch, epoch_loss));

                if self.logging {
                    println!(
                        "Epoch: {}, Loss: {:.7}, Batches: {}",
                        epoch, epoch_loss, num_batches
                    );
                }
            }
        }

        return Ok(());
    }

    fn extract_batch(
        &self,
        data: &Matrix,
        indices: &[usize],
        start: usize,
        end: usize,
    ) -> Result<Matrix> {
        let batch_size = end - start;
        let mut batch_data = Vec::with_capacity(batch_size * data.cols);

        for i in start..end {
            let row_idx = indices[i];
            let row_start = row_idx * data.cols;
            let row_end = row_start + data.cols;
            batch_data.extend_from_slice(&data.data[row_start..row_end]);
        }

        Ok(Matrix {
            rows: batch_size,
            cols: data.cols,
            data: batch_data,
        })
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
