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
        let b = Reduce::new("b c h w -> b c () ()", Operation::Max, None)?.apply(&a)?;
        assert_eq!(b.shape(), vec![10, 20, 1, 1]);

        let c = Rearrange::new("b c () () -> c b", None)?.apply(&b)?;
        assert_eq!(c.shape(), vec![20, 10]);

        Ok(())
    }
}
