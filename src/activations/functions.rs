use crate::linalg::matrix::Matrix;

#[derive(Debug, Clone, Copy)]
pub enum ActivationKind {
    Identity,
    Sigmoid,
    ReLU,
    Tanh,
    LeakyReLU,
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
            ActivationKind::LeakyReLU => mat.mapelements(|x| x.max(0.01 * x)),
        }
    }

    fn derivative(&self, mat: &Matrix) -> Matrix {
        match self {
            ActivationKind::Identity => Matrix::ones(mat.rows, mat.cols),
            ActivationKind::Sigmoid => mat.mapelements(|x| x * (1.0 - x)),
            ActivationKind::ReLU => mat.mapelements(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            ActivationKind::Tanh => mat.mapelements(|x| 1.0 - x.powi(2)),
            ActivationKind::LeakyReLU => mat.mapelements(|x| if x > 0.0 { 1.0 } else { 0.01 }),
        }
    }
}
