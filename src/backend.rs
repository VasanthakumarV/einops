#[cfg(feature = "tch-bindings")]
mod torch;

use crate::{EinopsError, Operation, Rearrange, Reduce, Repeat};

pub trait Backend {
    fn shape(&self) -> Vec<usize>;
    fn reshape(&self, shape: &[usize]) -> Self;
    fn transpose(&self, axes: &[usize]) -> Self;
    fn reduce_axes(&self, operation: Operation, axes: &[usize]) -> Self;
    fn reduce_axes_v2(&self, axes_operation: &mut [(usize, Operation)]) -> Self;
    fn add_axes(&self, naxes: usize, pos2len: &[(usize, usize)]) -> Self;
}

/// Trait that allows calling the rearrange operation directly on the tensor
pub trait RearrangeFn {
    /// Apply rearrange operation using the pattern
    fn rearrange(&self, pattern: &str) -> Result<Self, EinopsError>
    where
        Self: Sized + Backend,
    {
        Rearrange::new(pattern)?.apply(self)
    }

    /// Apply rearrange operation using the pattern and additional axes lengths
    /// attribute
    fn rearrange_with_lengths(
        &self,
        pattern: &str,
        axes_lengths: &[(&str, usize)],
    ) -> Result<Self, EinopsError>
    where
        Self: Sized + Backend,
    {
        Rearrange::with_lengths(pattern, axes_lengths)?.apply(self)
    }
}

/// Trait that allows calling the reduce operation directly on the tensor
pub trait ReduceFn {
    /// Apply reduce operation using the pattern and [`Operation`]
    fn reduce(&self, pattern: &str, operation: Operation) -> Result<Self, EinopsError>
    where
        Self: Sized + Backend,
    {
        Reduce::new(pattern, operation)?.apply(self)
    }

    /// Apply rearrange operation using the pattern, [`Operation`], and additional axes
    /// lengths attribute
    fn reduce_with_lengths(
        &self,
        pattern: &str,
        operation: Operation,
        axes_lengths: &[(&str, usize)],
    ) -> Result<Self, EinopsError>
    where
        Self: Sized + Backend,
    {
        Reduce::with_lengths(pattern, operation, axes_lengths)?.apply(self)
    }
}

/// Trait that allows calling the repeat operation directly on the tensor
pub trait RepeatFn {
    /// Apply repeat operation using the pattern
    fn repeat(&self, pattern: &str) -> Result<Self, EinopsError>
    where
        Self: Sized + Backend,
    {
        Repeat::new(pattern)?.apply(self)
    }

    /// Apply repeat operation using the pattern and additional axes lengths attribute
    fn repeat_with_lengths(
        &self,
        pattern: &str,
        axes_lengths: &[(&str, usize)],
    ) -> Result<Self, EinopsError>
    where
        Self: Sized + Backend,
    {
        Repeat::with_lengths(pattern, axes_lengths)?.apply(self)
    }
}
