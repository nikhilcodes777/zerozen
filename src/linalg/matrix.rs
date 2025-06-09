use ::anyhow::{anyhow, Result};
use rand::Rng;
use std::fmt;
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}
impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if rows * cols != data.len() {
            return Err(anyhow!(
                "Data length {} doesn't match matrix dimensions {}x{} (expected {})",
                data.len(),
                rows,
                cols,
                rows * cols
            ));
        }
        Ok(Self { rows, cols, data })
    }

    pub fn get(&self, i: usize, j: usize) -> Result<f64> {
        if i >= self.rows {
            return Err(anyhow!(
                "Row index {} out of bounds for matrix with {} rows",
                i,
                self.rows
            ));
        }
        if j >= self.cols {
            return Err(anyhow!(
                "Column index {} out of bounds for matrix with {} columns",
                j,
                self.cols
            ));
        }
        Ok(self.data[j + i * self.cols])
    }

    pub fn set(&mut self, i: usize, j: usize, value: f64) -> Result<()> {
        if i >= self.rows {
            return Err(anyhow!(
                "Row index {} out of bounds for matrix with {} rows",
                i,
                self.rows
            ));
        }
        if j >= self.cols {
            return Err(anyhow!(
                "Column index {} out of bounds for matrix with {} columns",
                j,
                self.cols
            ));
        }
        self.data[i * self.cols + j] = value;
        Ok(())
    }
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![1.0; rows * cols],
        }
    }

    pub fn fill(rows: usize, cols: usize, value: f64) -> Self {
        Self {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }
    pub fn identity(size: usize) -> Self {
        let mut data = vec![0.0; size * size];
        unsafe {
            for i in 0..size {
                *data.get_unchecked_mut(i * size + i) = 1.0;
            }
        }
        Self {
            rows: size,
            cols: size,
            data,
        }
    }
    pub fn random(rows: usize, cols: usize, lowerbound: f64, upperbound: f64) -> Self {
        let mut rng = rand::rng();
        let data = (0..rows * cols)
            .map(|_| rng.random_range(lowerbound..upperbound))
            .collect();
        Self { rows, cols, data }
    }

    pub fn mapelements(&self, func: impl Fn(f64) -> f64) -> Self {
        let data = self.data.iter().map(|&x| func(x)).collect();
        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn scale(&self, scalar: f64) -> Self {
        let data = self.data.iter().map(|&x| x * scalar).collect();
        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
    pub fn add(&self, other: &Matrix) -> Result<Self> {
        if self.cols != other.cols || self.rows != other.rows {
            return Err(anyhow!(
                "Matrix dimensions don't match for addition: {}x{} vs {}x{}",
                self.rows,
                self.cols,
                other.rows,
                other.cols
            ));
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }
    pub fn sub(&self, other: &Matrix) -> Result<Self> {
        if self.cols != other.cols || self.rows != other.rows {
            return Err(anyhow!(
                "Matrix dimensions don't match for subtraction: {}x{} vs {}x{}",
                self.rows,
                self.cols,
                other.rows,
                other.cols
            ));
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }

    pub fn hadamard(&self, other: &Matrix) -> Result<Self> {
        if self.cols != other.cols || self.rows != other.rows {
            return Err(anyhow!(
                "Matrix dimensions don't match for hadmard product: {}x{} vs {}x{}",
                self.rows,
                self.cols,
                other.rows,
                other.cols
            ));
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }

    pub fn mul(&self, other: &Matrix) -> Result<Self> {
        if self.cols != other.rows {
            return Err(anyhow!(
                "Matrix dimensions not compatible for matrix product: (A: {}x{}, B: {}x{})",
                self.rows,
                self.cols,
                other.rows,
                other.cols
            ));
        }

        let mut result_data = vec![0.0; self.rows * other.cols];
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k)? * other.get(k, j)?;
                }
                result_data[i * other.cols + j] = sum;
            }
        }
        Ok(Self {
            rows: self.rows,
            cols: other.cols,
            data: result_data,
        })
    }

    pub fn dot(&self, other: &Matrix) -> Result<f64> {
        let is_self_vector = self.rows == 1 || self.cols == 1;
        let is_other_vector = other.rows == 1 || other.cols == 1;

        if !is_self_vector || !is_other_vector {
            return Err(anyhow!(
                "Dot product requires both matrices to be vectors (1xN or Nx1)."
            ));
        }

        if self.data.len() != other.data.len() {
            return Err(anyhow!(
                "Vector lengths do not match for dot product: {} vs {}",
                self.data.len(),
                other.data.len()
            ));
        }

        let mut sum = 0.0;
        for i in 0..self.data.len() {
            sum += self.data[i] * other.data[i];
        }
        Ok(sum)
    }
    pub fn transpose(&self) -> Self {
        let mut transposed_data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed_data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            data: transposed_data,
        }
    }

    pub fn split_matrix(&self, n: usize) -> Result<(Matrix, Matrix)> {
        assert!(n < self.cols, "Column index out of bounds");

        let mut selected_col = Vec::with_capacity(self.rows);
        let mut remaining_data = Vec::with_capacity(self.rows * (self.cols - 1));

        for row in 0..self.rows {
            let start = row * self.cols;
            let end = start + self.cols;
            let row_slice = &self.data[start..end];

            selected_col.push(row_slice[n]);

            for (i, val) in row_slice.iter().enumerate() {
                if i != n {
                    remaining_data.push(*val);
                }
            }
        }

        let col_matrix = Matrix::new(self.rows, 1, selected_col)?;
        let remaining_matrix = Matrix::new(self.rows, self.cols - 1, remaining_data)?;

        Ok((col_matrix, remaining_matrix))
    }
    pub fn add_scalar(&self, scalar: f64) -> Self {
        let data = self.data.iter().map(|&x| x + scalar).collect();
        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn sub_scalar(&self, scalar: f64) -> Self {
        let data = self.data.iter().map(|&x| x - scalar).collect();
        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn div_scalar(&self, scalar: f64) -> Result<Self> {
        if scalar == 0.0 {
            return Err(anyhow!("Division by zero is not allowed"));
        }
        let data = self.data.iter().map(|&x| x / scalar).collect();
        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }

    pub fn sum_rows(&self) -> Self {
        let mut result_data = vec![0.0; self.rows];
        for i in 0..self.rows {
            let mut row_sum = 0.0;
            for j in 0..self.cols {
                row_sum += self.data[i * self.cols + j];
            }
            result_data[i] = row_sum;
        }
        Self {
            rows: self.rows,
            cols: 1,
            data: result_data,
        }
    }

    pub fn sum_cols(&self) -> Self {
        let mut result_data = vec![0.0; self.cols];
        for j in 0..self.cols {
            let mut col_sum = 0.0;
            for i in 0..self.rows {
                col_sum += self.data[i * self.cols + j];
            }
            result_data[j] = col_sum;
        }
        Self {
            rows: 1,
            cols: self.cols,
            data: result_data,
        }
    }

    pub fn sum_all(&self) -> f64 {
        self.data.iter().sum()
    }

    pub fn add_bias_vector(&self, bias_vector: &Matrix) -> Result<Self> {
        if bias_vector.rows != 1 || self.cols != bias_vector.cols {
            return Err(anyhow!("Bias vector dimensions are incompatible"));
        }
        let mut result_data = self.data.clone();
        for i in 0..self.rows {
            for j in 0..self.cols {
                result_data[i * self.cols + j] += bias_vector.data[j];
            }
        }
        Matrix::new(self.rows, self.cols, result_data)
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[")?;
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{:8.3} ", self.get(i, j).unwrap())?;
            }
            writeln!(f)?;
        }

        write!(f, "]")?;
        Ok(())
    }
}
#[macro_export]
macro_rules! matrix {
    // Handle single row case
    ($($x:expr),+ $(,)?) => {
        {
            let data = vec![$($x as f64),+];
            let cols = data.len();
            Matrix::new(1, cols, data)?
        }
    };

    // Handle multiple rows separated by semicolons
    ($($($x:expr),+ $(,)?);+ $(;)?) => {
        {
            let mut all_data = Vec::new();
            let mut row_count = 0;
            let mut col_count = 0;

            $(
                let row_data = vec![$($x as f64),+];
                if row_count == 0 {
                    col_count = row_data.len();
                } else {
                    assert_eq!(col_count, row_data.len(), "All rows must have the same number of columns");
                }
                all_data.extend(row_data);
                row_count += 1;
            )+

            Matrix::new(row_count, col_count, all_data)?
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to compare f64 with a tolerance
    fn assert_matrix_approx_eq(matrix1: &Matrix, matrix2: &Matrix, epsilon: f64) {
        assert_eq!(matrix1.rows, matrix2.rows, "Rows differ");
        assert_eq!(matrix1.cols, matrix2.cols, "Cols differ");
        for i in 0..matrix1.data.len() {
            assert!(
                (matrix1.data[i] - matrix2.data[i]).abs() < epsilon,
                "Elements at index {} differ: {} vs {}",
                i,
                matrix1.data[i],
                matrix2.data[i]
            );
        }
    }

    #[test]
    fn test_new() {
        let matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);
        assert_eq!(matrix.data, vec![1.0, 2.0, 3.0, 4.0]);

        let err = Matrix::new(2, 2, vec![1.0, 2.0, 3.0]);
        assert!(err.is_err());
    }

    #[test]
    fn test_get_set() {
        let mut matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(matrix.get(0, 0).unwrap(), 1.0);
        assert_eq!(matrix.get(1, 1).unwrap(), 4.0);

        matrix.set(0, 0, 10.0).unwrap();
        assert_eq!(matrix.get(0, 0).unwrap(), 10.0);

        assert!(matrix.get(2, 0).is_err());
        assert!(matrix.set(0, 2, 5.0).is_err());
    }

    #[test]
    fn test_shape() {
        let matrix = Matrix::new(3, 2, vec![1.0; 6]).unwrap();
        assert_eq!(matrix.shape(), (3, 2));
    }

    #[test]
    fn test_zeros() {
        let matrix = Matrix::zeros(2, 3);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 3);
        assert_eq!(matrix.data, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ones() {
        let matrix = Matrix::ones(2, 3);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 3);
        assert_eq!(matrix.data, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_fill() {
        let matrix = Matrix::fill(2, 2, 7.0);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);
        assert_eq!(matrix.data, vec![7.0, 7.0, 7.0, 7.0]);
    }

    #[test]
    fn test_identity() {
        let matrix = Matrix::identity(3);
        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.cols, 3);
        assert_eq!(
            matrix.data,
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn test_random() {
        let matrix = Matrix::random(2, 2, 0.0, 1.0);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);
        assert_eq!(matrix.data.len(), 4);
        for &val in &matrix.data {
            assert!((0.0..1.0).contains(&val));
        }
    }

    #[test]
    fn test_mapelements() {
        let matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mapped_matrix = matrix.mapelements(|x| x * 2.0);
        assert_eq!(mapped_matrix.data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_add() {
        let m1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let m2 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let result = m1.add(&m2).unwrap();
        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);

        let m3 = Matrix::new(2, 1, vec![1.0, 2.0]).unwrap();
        assert!(m1.add(&m3).is_err());
    }

    #[test]
    fn test_sub() {
        let m1 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let m2 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = m1.sub(&m2).unwrap();
        assert_eq!(result.data, vec![4.0, 4.0, 4.0, 4.0]);

        let m3 = Matrix::new(1, 2, vec![1.0, 2.0]).unwrap();
        assert!(m1.sub(&m3).is_err());
    }

    #[test]
    fn test_hadmard() {
        let m1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let m2 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let result = m1.hadamard(&m2).unwrap();
        assert_eq!(result.data, vec![5.0, 12.0, 21.0, 32.0]);

        let m3 = Matrix::new(2, 1, vec![1.0, 2.0]).unwrap();
        assert!(m1.hadamard(&m3).is_err());
    }

    #[test]
    fn test_mul() {
        let m1 = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let m2 = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let result = m1.mul(&m2).unwrap();
        let expected_data = vec![
            (1.0 * 7.0 + 2.0 * 9.0 + 3.0 * 11.0),  // 7 + 18 + 33 = 58
            (1.0 * 8.0 + 2.0 * 10.0 + 3.0 * 12.0), // 8 + 20 + 36 = 64
            (4.0 * 7.0 + 5.0 * 9.0 + 6.0 * 11.0),  // 28 + 45 + 66 = 139
            (4.0 * 8.0 + 5.0 * 10.0 + 6.0 * 12.0), // 32 + 50 + 72 = 154
        ];
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
        assert_matrix_approx_eq(&result, &Matrix::new(2, 2, expected_data).unwrap(), 1e-9);

        let m3 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(m1.mul(&m3).is_err()); // Incompatible dimensions
    }

    #[test]
    fn test_matrix_product() {
        let m1 = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let m2 = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let result = m1.mul(&m2).unwrap();
        let expected_data = vec![
            (1.0 * 7.0 + 2.0 * 9.0 + 3.0 * 11.0),  // 7 + 18 + 33 = 58
            (1.0 * 8.0 + 2.0 * 10.0 + 3.0 * 12.0), // 8 + 20 + 36 = 64
            (4.0 * 7.0 + 5.0 * 9.0 + 6.0 * 11.0),  // 28 + 45 + 66 = 139
            (4.0 * 8.0 + 5.0 * 10.0 + 6.0 * 12.0), // 32 + 50 + 72 = 154
        ];
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
        assert_matrix_approx_eq(&result, &Matrix::new(2, 2, expected_data).unwrap(), 1e-9);

        let m3 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(m1.mul(&m3).is_err()); // Incompatible dimensions
    }
    #[test]
    fn test_transpose() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let transposed_m = m.transpose();
        assert_eq!(transposed_m.rows, 3);
        assert_eq!(transposed_m.cols, 2);
        assert_eq!(transposed_m.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        let m_square = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let transposed_m_square = m_square.transpose();
        assert_eq!(transposed_m_square.rows, 2);
        assert_eq!(transposed_m_square.cols, 2);
        assert_eq!(transposed_m_square.data, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_scale() {
        let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let scaled_m = m.scale(2.0);
        assert_eq!(scaled_m.data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_add_scalar() {
        let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let added_m = m.add_scalar(10.0);
        assert_eq!(added_m.data, vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_sub_scalar() {
        let m = Matrix::new(2, 2, vec![10.0, 11.0, 12.0, 13.0]).unwrap();
        let subtracted_m = m.sub_scalar(5.0);
        assert_eq!(subtracted_m.data, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_div_scalar() {
        let m = Matrix::new(2, 2, vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let divided_m = m.div_scalar(10.0).unwrap();
        assert_eq!(divided_m.data, vec![1.0, 2.0, 3.0, 4.0]);

        let m_div_by_zero = Matrix::new(1, 1, vec![1.0]).unwrap();
        assert!(m_div_by_zero.div_scalar(0.0).is_err());
    }

    #[test]
    fn test_sum_rows() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let sum_rows_matrix = m.sum_rows();
        assert_eq!(sum_rows_matrix.rows, 2);
        assert_eq!(sum_rows_matrix.cols, 1);
        assert_eq!(sum_rows_matrix.data, vec![1.0 + 2.0 + 3.0, 4.0 + 5.0 + 6.0]);
        // 6.0, 15.0
    }

    #[test]
    fn test_sum_cols() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let sum_cols_matrix = m.sum_cols();
        assert_eq!(sum_cols_matrix.rows, 1);
        assert_eq!(sum_cols_matrix.cols, 3);
        assert_eq!(sum_cols_matrix.data, vec![1.0 + 4.0, 2.0 + 5.0, 3.0 + 6.0]);
        // 5.0, 7.0, 9.0
    }

    #[test]
    fn test_sum_all() {
        let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(m.sum_all(), 1.0 + 2.0 + 3.0 + 4.0); // 10.0
    }
}
