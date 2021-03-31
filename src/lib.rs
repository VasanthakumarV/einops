mod backend;
pub mod error;
mod recipe;

use backend::Backend;
use error::EinopsError;
use recipe::{Function, TransformRecipe};

#[derive(Copy, Clone, Debug)]
pub enum Operation {
    Min,
    Max,
    Sum,
    Mean,
    Prod,
}

#[derive(Debug)]
pub struct Rearrange {
    recipe: TransformRecipe,
}

impl Rearrange {
    pub fn new(
        pattern: &str,
        axes_lengths: Option<&[(&str, usize)]>,
    ) -> Result<Self, EinopsError> {
        let recipe = TransformRecipe::new(pattern, Function::Rearrange, axes_lengths)?;

        Ok(Self { recipe })
    }

    pub fn apply<T: Backend>(&self, tensor: &T) -> Result<T, EinopsError> {
        self.recipe.apply(tensor)
    }
}

#[derive(Debug)]
pub struct Reduce {
    recipe: TransformRecipe,
}

impl Reduce {
    pub fn new(
        pattern: &str,
        operation: Operation,
        axes_lengths: Option<&[(&str, usize)]>,
    ) -> Result<Self, EinopsError> {
        let recipe = TransformRecipe::new(pattern, Function::Reduce(operation), axes_lengths)?;

        Ok(Self { recipe })
    }

    pub fn apply<T: Backend>(&self, tensor: &T) -> Result<T, EinopsError> {
        self.recipe.apply(tensor)
    }
}

#[derive(Debug)]
pub struct Repeat {
    recipe: TransformRecipe,
}

impl Repeat {
    pub fn new(
        pattern: &str,
        axes_lengths: Option<&[(&str, usize)]>,
    ) -> Result<Self, EinopsError> {
        let recipe = TransformRecipe::new(pattern, Function::Repeat, axes_lengths)?;

        Ok(Self { recipe })
    }

    pub fn apply<T: Backend>(&self, tensor: &T) -> Result<T, EinopsError> {
        self.recipe.apply(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn rearrange_test() -> Result<(), EinopsError> {
        let a = Tensor::arange(10 * 20 * 30 * 40, (Kind::Float, Device::Cpu))
            .reshape(&[10, 20, 30, 40]);
        let b = Rearrange::new(
            "b (c h1 w1) h w -> b c (h h1) (w w1)",
            Some(&[("h1", 2), ("w1", 2)]),
        )?
        .apply(&a)?;

        assert_eq!(b.shape(), vec![10, 5, 30 * 2, 40 * 2]);

        Ok(())
    }
}
