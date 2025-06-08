use crate::activations::functions::ActivationKind;

#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub neurons: usize,
    pub activator: ActivationKind,
}
