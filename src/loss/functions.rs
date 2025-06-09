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
            LossKind::CrossEntropy => {
                let n_samples = predictions.rows as f64;
                let mut total_loss = 0.0;
                // Small epsilon to prevent log(0)
                let epsilon = 1e-15;

                for i in 0..predictions.rows {
                    for j in 0..predictions.cols {
                        let pred_idx = i * predictions.cols + j;
                        let target_idx = i * targets.cols + j;

                        let pred = predictions.data[pred_idx];
                        let target = targets.data[target_idx];

                        // Clamp prediction to prevent log(0)
                        let clamped_pred = pred.max(epsilon).min(1.0 - epsilon);

                        // Cross entropy: -sum(y_true * log(y_pred))
                        total_loss -= target * clamped_pred.ln();
                    }
                }

                // Return average loss across all samples
                Ok(total_loss / n_samples)
            }
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
            LossKind::CrossEntropy => {
                let n_samples = predictions.rows as f64;
                // For cross-entropy with softmax, the gradient simplifies to:
                // dL/dy_pred = (y_pred - y_true) / n_samples
                let gradient = predictions.sub(targets)?;
                Ok(gradient.scale(1.0 / n_samples))
            }
        }
    }
}

pub fn labels_to_one_hot(labels: &Matrix, num_classes: usize) -> Result<Matrix> {
    if labels.cols != 1 {
        return Err(anyhow!(
            "Labels matrix must have exactly 1 column, got {}",
            labels.cols
        ));
    }

    let n_samples = labels.rows;
    let mut one_hot = Matrix::zeros(n_samples, num_classes);

    for i in 0..n_samples {
        let class_idx = labels.data[i] as usize;
        if class_idx >= num_classes {
            return Err(anyhow!(
                "Class index {} exceeds number of classes {}",
                class_idx,
                num_classes
            ));
        }
        one_hot.data[i * num_classes + class_idx] = 1.0;
    }

    Ok(one_hot)
}
