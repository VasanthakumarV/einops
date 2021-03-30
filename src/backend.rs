#[cfg(feature = "torch")]
mod torch;

use crate::Operation;

pub trait Backend {
    fn shape(&self) -> Vec<usize>;
    fn reshape(&self, shape: &[usize]) -> Self;
    fn transpose(&self, axes: &[usize]) -> Self;
    fn reduce(&self, operation: Operation, axes: &[usize]) -> Self;
    fn add_axes(&self, naxes: usize, pos2len: &[(usize, usize)]) -> Self;
}
