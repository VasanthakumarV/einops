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
    pub fn new(pattern: &str) -> Result<Self, EinopsError> {
        let recipe = TransformRecipe::new(pattern, Function::Rearrange, None)?;

        Ok(Self { recipe })
    }

    pub fn with_lengths(
        pattern: &str,
        axes_lengths: &[(&str, usize)],
    ) -> Result<Self, EinopsError> {
        let recipe = TransformRecipe::new(pattern, Function::Rearrange, Some(axes_lengths))?;

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
    pub fn new(pattern: &str, operation: Operation) -> Result<Self, EinopsError> {
        let recipe = TransformRecipe::new(pattern, Function::Reduce(operation), None)?;

        Ok(Self { recipe })
    }

    pub fn with_lengths(
        pattern: &str,
        operation: Operation,
        axes_lengths: &[(&str, usize)],
    ) -> Result<Self, EinopsError> {
        let recipe =
            TransformRecipe::new(pattern, Function::Reduce(operation), Some(axes_lengths))?;

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
    pub fn new(pattern: &str) -> Result<Self, EinopsError> {
        let recipe = TransformRecipe::new(pattern, Function::Repeat, None)?;

        Ok(Self { recipe })
    }

    pub fn with_lengths(
        pattern: &str,
        axes_lengths: &[(&str, usize)],
    ) -> Result<Self, EinopsError> {
        let recipe = TransformRecipe::new(pattern, Function::Repeat, Some(axes_lengths))?;

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
    fn identity_patterns() -> Result<(), EinopsError> {
        let patterns = &[
            "... -> ...",
            "a b c d e -> a b c d e",
            "a b c d e ... -> ... a b c d e",
            "a b c d e ... -> a ... b c d e",
            "... a b c d e -> ... a b c d e",
            "a ... e -> a ... e",
            "a ... -> a ...",
            "a ... c d e -> a (...) c d e",
        ];

        //let input = Tensor::zeros[&[1, 1, 1, 1, 1]];
        let input = Tensor::arange(2 * 3 * 4 * 5 * 6, (Kind::Float, Device::Cpu))
            .reshape(&[2, 3, 4, 5, 6]);

        for pattern in patterns {
            assert_eq!(
                input,
                Rearrange::new(pattern)?.apply(&input)?,
                "{} failed",
                pattern
            );
        }

        Ok(())
    }

    #[test]
    fn equivalent_rearrange_patterns() -> Result<(), EinopsError> {
        let patterns = &[
            ("a b c d e -> (a b) c d e", "a b ... -> (a b) ..."),
            ("a b c d e -> a b (c d) e", "... c d e -> ... (c d) e"),
            ("a b c d e -> a b c d e", "... -> ..."),
            ("a b c d e -> (a b c d e)", "... -> (...)"),
            ("a b c d e -> b (c d e) a", "a b ... -> b (...) a"),
            ("a b c d e -> b (a c d) e", "a b ... e -> b (a ...) e"),
        ];

        let input = Tensor::arange(2 * 3 * 4 * 5 * 6, (Kind::Float, Device::Cpu))
            .reshape(&[2, 3, 4, 5, 6]);

        for (pattern1, pattern2) in patterns {
            let output1 = Rearrange::new(pattern1)?.apply(&input)?;
            let output2 = Rearrange::new(pattern2)?.apply(&input)?;

            assert_eq!(output1, output2);
        }

        Ok(())
    }

    #[test]
    fn rearrange_test() -> Result<(), EinopsError> {
        let a = Tensor::arange(10 * 20 * 30 * 40, (Kind::Float, Device::Cpu))
            .reshape(&[10, 20, 30, 40]);
        let b = Reduce::new("b ... -> b (...)", Operation::Max)?.apply(&a)?;
        dbg!(b.shape());
        //assert_eq!(b.shape(), vec![10, 20 * 30 * 40]);

        //let c = Rearrange::new("b c () () -> c b")?.apply(&b)?;
        //assert_eq!(c.shape(), vec![20, 10]);

        Ok(())
    }
}
