use anyhow::{anyhow, Ok, Result};

use crate::linalg::matrix::Matrix;

#[derive(Debug, Clone)]
pub enum LossKind {
    MeanSquaredError,
    CrossEntropy,
}
pub trait LossFunction {
    fn loss(&self, predictions: &Matrix, targets: &Matrix) -> Result<f64>;
    fn gradient(&self, predictions: &Matrix, targets: &Matrix) -> Result<Matrix>;
}
impl LossFunction for LossKind {
    fn loss(&self, predictions: &Matrix, targets: &Matrix) -> Result<f64> {
        if predictions.shape() != targets.shape() {
            return Err(anyhow!("Predictions and targets shape mismatch"));
        }

        match self {
            LossKind::MeanSquaredError => {
                let diff = predictions.sub(targets).unwrap();
                let square_err = diff.hadamard(&diff).unwrap();
                Ok(square_err.data.iter().sum::<f64>()
                    / ((predictions.rows as f64) * (predictions.cols as f64)))
            }
            LossKind::CrossEntropy => todo!(),
        }
    }

    fn gradient(&self, predictions: &Matrix, targets: &Matrix) -> Result<Matrix> {
        if predictions.shape() != targets.shape() {
            return Err(anyhow!("Predictions and targets shape mismatch"));
        }
        match self {
            LossKind::MeanSquaredError => {
                let n_samples = predictions.rows as f64;

                Ok(predictions.sub(targets).unwrap().scale(2.0 / n_samples))
            }
            LossKind::CrossEntropy => todo!(),
        }
    }
}
