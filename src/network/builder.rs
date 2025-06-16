use anyhow::{anyhow, Ok, Result};

use crate::{
    layer::{config::LayerConfig, core::Layer},
    loss::functions::LossKind,
    network::core::Network,
};
pub struct NetworkBuilder {
    layer_configs: Vec<LayerConfig>,
    learning_rate: f64,
    loss: LossKind,
    epochs: usize,
    batch_size: Option<usize>,
    shuffle: bool,
    logging: bool,
    log_level: usize,
}
impl NetworkBuilder {
    pub fn new() -> Self {
        Self {
            layer_configs: Vec::new(),
            learning_rate: 0.01,
            loss: LossKind::MeanSquaredError,
            epochs: 10 * 1000,
            batch_size: None,
            shuffle: true,
            logging: true,
            log_level: 200,
        }
    }

    pub fn layer(mut self, config: LayerConfig) -> Self {
        self.layer_configs.push(config);
        self
    }

    pub fn learning_rate(mut self, rate: f64) -> Self {
        self.learning_rate = rate;
        self
    }

    pub fn loss(mut self, loss: LossKind) -> Self {
        self.loss = loss;
        self
    }
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    pub fn with_logging(mut self, logging: bool, log_level: usize) -> Self {
        self.logging = logging;
        self.log_level = log_level;
        self
    }

    pub fn build(self, no_of_features: usize) -> Result<Network> {
        if self.layer_configs.is_empty() {
            return Err(anyhow!("Cannot build a network with no layers."));
        }

        let mut layers = Vec::new();
        let mut current_input_size = no_of_features;

        for config in &self.layer_configs {
            let layer = Layer::new(current_input_size, config);
            current_input_size = config.neurons;
            layers.push(layer);
        }

        Ok(Network {
            layers,
            learning_rate: self.learning_rate,
            loss: self.loss,
            loss_history: Vec::new(),
            epochs: self.epochs,
            batch_size: self.batch_size,
            shuffle: self.shuffle,
            logging: self.logging,
            log_level: self.log_level,
        })
    }
}
