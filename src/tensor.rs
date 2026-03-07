//tensor.rs
pub trait Tensor: Clone {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;

    fn new(rows: usize, cols: usize) -> Self;
    fn random(rows: usize, cols: usize) -> Self;

    fn zeros(&mut self);

    fn dot(&self, other: &Self, target: &mut Self);
    fn dot_rhs_transposed(&self, other: &Self, target: &mut Self);
    fn dot_self_transposed(&self, other: &Self, target: &mut Self);
    fn outer_product(&self, input: &Self, target: &mut Self);
    fn dot_transpose_self(&self, error: &Self, target: &mut Self);

    fn data(&self) -> &[f32];
    fn data_mut(&mut self) -> &mut [f32];

    fn copy_from_slice(&mut self, source: &[f32]);
}