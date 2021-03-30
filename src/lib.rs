pub(crate) mod backend;
pub mod error;
pub(crate) mod parse;
mod recipe;

use backend::Backend;
use error::EinopsError;
use recipe::TransformRecipe;

#[derive(Copy, Clone)]
pub enum Operation {
    Min,
    Max,
    Sum,
    Mean,
    Prod,
}

enum Function {
    Rearrange,
    Repeat,
    Reduce(Operation),
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
