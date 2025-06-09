use core::panic;

use crate::linalg::matrix::Matrix;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationKind {
    Identity,
    Sigmoid,
    ReLU,
    Tanh,
    LeakyReLU(f64),
    SoftMax,
}
pub trait ActivationFunction {
    fn activate(&self, mat: &Matrix) -> Matrix;
    fn derivative(&self, mat: &Matrix) -> Matrix;
}

impl ActivationFunction for ActivationKind {
    fn activate(&self, mat: &Matrix) -> Matrix {
        match self {
            ActivationKind::Identity => mat.clone(),
            ActivationKind::Sigmoid => mat.mapelements(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationKind::ReLU => mat.mapelements(|x| x.max(0.0)),
            ActivationKind::Tanh => mat.mapelements(|x| x.tanh()),
            ActivationKind::LeakyReLU(f) => mat.mapelements(|x| x.max(f * x)),
            ActivationKind::SoftMax => {
                let mut result = Matrix::zeros(mat.rows, mat.cols);
                for i in 0..mat.rows {
                    // Find max value in the row for numerical stability
                    let mut max_val = mat.data[i * mat.cols];
                    for j in 1..mat.cols {
                        let val = mat.data[i * mat.cols + j];
                        if val > max_val {
                            max_val = val;
                        }
                    }

                    // Compute exp(x - max) and sum
                    let mut exp_sum = 0.0;
                    let mut exp_vals = vec![0.0; mat.cols];
                    for j in 0..mat.cols {
                        let exp_val = (mat.data[i * mat.cols + j] - max_val).exp();
                        exp_vals[j] = exp_val;
                        exp_sum += exp_val;
                    }

                    // Normalize by sum
                    for j in 0..mat.cols {
                        result.data[i * mat.cols + j] = exp_vals[j] / exp_sum;
                    }
                }
                result
            }
        }
    }

    fn derivative(&self, mat: &Matrix) -> Matrix {
        match self {
            ActivationKind::Identity => Matrix::ones(mat.rows, mat.cols),
            ActivationKind::Sigmoid => mat.mapelements(|x| x * (1.0 - x)),
            ActivationKind::ReLU => mat.mapelements(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            ActivationKind::Tanh => mat.mapelements(|x| 1.0 - x.powi(2)),
            ActivationKind::LeakyReLU(f) => mat.mapelements(|x| if x > 0.0 { 1.0 } else { *f }),
            ActivationKind::SoftMax => panic!("Softmax derivative is directly called!"), //SoftMax's derivative isnt that straightforward
        }
    }
}
