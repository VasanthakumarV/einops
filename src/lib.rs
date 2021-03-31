mod backend;
pub mod error;
mod recipe;

use backend::Backend;
use error::EinopsError;
use recipe::{Function, TransformRecipe};

#[derive(Copy, Clone)]
pub enum Operation {
    Min,
    Max,
    Sum,
    Mean,
    Prod,
}

pub struct Rearrange {
    recipe: TransformRecipe,
}

impl Rearrange {
    pub fn new<'a, L>(pattern: &str, axes_lengths: L) -> Result<Self, EinopsError>
    where
        L: Into<Option<&'a [(&'a str, usize)]>>,
    {
        let recipe = TransformRecipe::new(pattern, Function::Rearrange, axes_lengths.into())?;

        Ok(Self { recipe })
    }

    pub fn apply<T: Backend>(&self, tensor: T) -> Result<T, EinopsError> {
        self.recipe.apply(tensor)
    }
}

pub struct Reduce {
    recipe: TransformRecipe,
}

impl Reduce {
    pub fn new<'a, L>(
        pattern: &str,
        operation: Operation,
        axes_lengths: L,
    ) -> Result<Self, EinopsError>
    where
        L: Into<Option<&'a [(&'a str, usize)]>>,
    {
        let recipe =
            TransformRecipe::new(pattern, Function::Reduce(operation), axes_lengths.into())?;

        Ok(Self { recipe })
    }

    pub fn apply<T: Backend>(&self, tensor: T) -> Result<T, EinopsError> {
        self.recipe.apply(tensor)
    }
}

pub struct Repeat {
    recipe: TransformRecipe,
}

impl Repeat {
    pub fn new<'a, L>(pattern: &str, axes_lengths: L) -> Result<Self, EinopsError>
    where
        L: Into<Option<&'a [(&'a str, usize)]>>,
    {
        let recipe = TransformRecipe::new(pattern, Function::Repeat, axes_lengths.into())?;

        Ok(Self { recipe })
    }

    pub fn apply<T: Backend>(&self, tensor: T) -> Result<T, EinopsError> {
        self.recipe.apply(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn rearrange_test() -> Result<(), EinopsError> {
        let a = Tensor::arange(2 * 3 * 4 * 2, (Kind::Float, Device::Cpu)).reshape(&[2, 3, 4, 2]);
        let b = Rearrange::new("b c h w -> b h w c", None)?.apply(a)?;
        dbg!(b.shape());

        assert!(true);

        Ok(())
    }
}
