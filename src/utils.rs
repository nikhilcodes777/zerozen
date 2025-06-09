use std::fs;

use crate::linalg::matrix::Matrix;
use anyhow::Ok;
use anyhow::Result;
pub fn load_csv(path: &str, cols: usize) -> Result<Matrix> {
    let file = fs::read_to_string(path)?;

    let mut data = Vec::new();
    let mut row_count = 0;

    for line_result in file.lines().skip(1) {
        let line = line_result;
        let values = line
            .trim()
            .split(',')
            .map(|v| v.parse::<f64>().unwrap_or(0.0))
            .collect::<Vec<f64>>();

        if !values.is_empty() {
            data.extend(values);
            row_count += 1;
        }
    }

    Ok(Matrix::new(row_count, cols, data)?)
}

pub fn one_hot_to_labels(one_hot: &Matrix) -> Result<Vec<usize>> {
    if one_hot.cols <= 1 {
        return Err(anyhow::anyhow!("Matrix is not one-hot encoded."));
    }

    let mut labels = Vec::with_capacity(one_hot.rows);

    for row in 0..one_hot.rows {
        let start = row * one_hot.cols;
        let end = start + one_hot.cols;

        let row_slice = &one_hot.data[start..end];

        // Find the index of the maximum element
        let (max_index, _) = row_slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        labels.push(max_index);
    }

    Ok(labels)
}

pub fn accuracy(predictions: &Matrix, targets: &Matrix) -> Result<f64> {
    let predicted_labels = one_hot_to_labels(predictions)?;
    let target_labels = one_hot_to_labels(targets)?;

    if predicted_labels.len() != target_labels.len() {
        return Err(anyhow::anyhow!(
            "Prediction and target lengths do not match."
        ));
    }

    let correct = predicted_labels
        .iter()
        .zip(target_labels.iter())
        .filter(|(pred, target)| pred == target)
        .count();

    let accuracy = correct as f64 / predicted_labels.len() as f64 * 100.0;

    Ok(accuracy)
}
