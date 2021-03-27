//#[cfg(feature = "torch")]
mod torch;

pub enum Operation {
    Min,
    Max,
    Sum,
    Mean,
    Prod,
}

pub trait Backend {
    fn shape(&self) -> Vec<isize>;
    fn reshape(&self, shape: &[isize]) -> Self;
    fn transpose(&self, axes: &[isize]) -> Self;
    fn reduce(&self, operation: Operation, axes: &[isize]) -> Self;
    fn add_axes(&self, naxes: usize, pos2len: &[(usize, isize)]) -> Self;
}
