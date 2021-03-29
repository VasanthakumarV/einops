pub(crate) mod backend;
pub mod error;
pub(crate) mod parse;
mod recipe;

use backend::{Backend, Operation};
use error::EinopsError;
use recipe::TransformRecipe;

enum Function {
    Rearrange,
    Repeat,
    Reduce(Operation),
}

fn rearrange<T: Backend>(
    tensor: T,
    pattern: &str,
    axes_lengths: Option<&[(&str, usize)]>,
) -> Result<T, EinopsError> {
    let recipe = TransformRecipe::new(pattern, Function::Rearrange, axes_lengths)?;
    recipe.apply(tensor)
}

fn reduce<T: Backend>(
    tensor: T,
    pattern: &str,
    operation: Operation,
    axes_lengths: Option<&[(&str, usize)]>,
) -> Result<T, EinopsError> {
    let recipe = TransformRecipe::new(pattern, Function::Reduce(operation), axes_lengths)?;
    recipe.apply(tensor)
}

fn repeat<T: Backend>(
    tensor: T,
    pattern: &str,
    axes_lengths: Option<&[(&str, usize)]>,
) -> Result<T, EinopsError> {
    let recipe = TransformRecipe::new(pattern, Function::Repeat, axes_lengths)?;
    recipe.apply(tensor)
}
