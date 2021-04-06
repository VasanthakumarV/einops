#![warn(missing_docs)]
//! # Einops
//!
//! This is a rust port of the incredible [einops](https://github.com/arogozhnikov/einops) library.
//! Almost all the operations specified in its tutorial should be available, if you find any
//! inconsistencies please raise a github issue.
//!
//! _Unlike its python counterpart, caching the parsed expression has not been implemented yet. So
//! when applying the same pattern multiple times, prefer_ `Rearrange::new(...)` _or_ `Rearrange::with_lengths(...)`
//! _api, over the methods available through `RearrangeFn` like traits_
//!
//! Flexible and powerful tensor operations for readable and reliable code.
//! Currently only supports [tch](https://github.com/LaurentMazare/tch-rs).
//!
//! ## Getting started
//!
//! Add the following to your `Cargo.toml` file,
//!
//! ```ignore
//! [dependencies]
//! einops = { version: "0.1", features: ["tch-bindings"] }
//! ```
//!
//! ## Examples
//!
//! Einops provies three operations, they cover stacking, reshape, transposition,
//! squeeze/unsqueeze, repeat, tile, concatenate and numerous reductions
//!
//! ```ignore
//! // Tch specific imports
//! use tch::{Tensor, Kind, Device};
//! // Structs that provide constructor like api
//! use einops::{Rearrange, Repeat, Reduce, Operation, Backend};
//! // Traits required to call functions directly on the tensors
//! use einops::{ReduceFn, RearrangeFn, RepeatFn};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // We create a random tensor as input
//! let input = Tensor::randn(&[100, 32, 64], (Kind::Float, Device::Cpu));
//!
//! // ------------------------------------------------------------------------
//! // Rearrange operation
//! let output = Rearrange::new("t b c -> b c t")?.apply(&input)?;
//! assert_eq!(output.size(), vec![32, 64, 100]);
//!
//! // Apply rearrange operation directly on the tensor using `RearrangeFn` trait
//! let output = input.rearrange("t b c -> b c t")?;
//! assert_eq!(output.size(), vec![32, 64, 100]);
//!
//! // ------------------------------------------------------------------------
//! // Perform reduction on first axis
//! let output = Reduce::new("t b c -> b c", Operation::Max)?.apply(&input)?;
//! assert_eq!(output.size(), vec![32, 64]);
//!
//! // Same reduction done directly on the tensor using `ReduceFn` trait
//! let output = input.reduce("t b c -> b c", Operation::Max)?;
//! assert_eq!(output.size(), vec![32, 64]);
//!
//! // ------------------------------------------------------------------------
//! // We repeat the third axis
//! let output = Repeat::new("t b c -> t b c 3")?.apply(&input)?;
//! assert_eq!(output.size(), vec![100, 32, 64, 3]);
//!
//! // Same as above, but using `RepeatFn` trait and we specify the repeating
//! // axis with a name along with its size and pass it in a slice
//! let output = input.repeat_with_lengths("t b c -> t b c repeat", &[("repeat", 3)])?;
//! assert_eq!(output.size(), vec![100, 32, 64, 3]);
//! # Ok(())
//! # }
//! ```

mod backend;
mod error;
mod recipe;

use backend::Backend;
pub use backend::{RearrangeFn, ReduceFn, RepeatFn};
pub use error::EinopsError;
use recipe::{Function, TransformRecipe};

/// Specifies the operation used to reduce an axis
#[derive(Copy, Clone, Debug)]
pub enum Operation {
    /// Take the minimum value
    Min,
    /// Take the maximum value
    Max,
    /// Add all elements
    Sum,
    /// Take the average
    Mean,
    /// Multiply all elements
    Prod,
}

/// Reader-friendly reordering of tensors. Includes the functionality of transpose
/// (axes permutation), reshape, squeeze, unsqueeze and other operations
#[derive(Debug)]
pub struct Rearrange {
    recipe: TransformRecipe,
}

impl Rearrange {
    /// Initialize with pattern
    pub fn new(pattern: &str) -> Result<Self, EinopsError> {
        let recipe = TransformRecipe::new(pattern, Function::Rearrange, None)?;

        Ok(Self { recipe })
    }

    /// Initialize with pattern and lengths of axes
    pub fn with_lengths(
        pattern: &str,
        axes_lengths: &[(&str, usize)],
    ) -> Result<Self, EinopsError> {
        let recipe = TransformRecipe::new(pattern, Function::Rearrange, Some(axes_lengths))?;

        Ok(Self { recipe })
    }

    /// Perform rearrange operation on the tensor
    pub fn apply<T: Backend>(&self, tensor: &T) -> Result<T, EinopsError> {
        self.recipe.apply(tensor)
    }
}

/// Provides combination of reordering and reduction using reader-friendly notation
#[derive(Debug)]
pub struct Reduce {
    recipe: TransformRecipe,
}

impl Reduce {
    /// Initialize using the pattern and [`Operation`] (the mode of reduction)
    pub fn new(pattern: &str, operation: Operation) -> Result<Self, EinopsError> {
        let recipe = TransformRecipe::new(pattern, Function::Reduce(operation), None)?;

        Ok(Self { recipe })
    }

    /// Initialize with pattern, [`Operation`], and a slice of tuples indicating the sizes
    /// of different axes
    pub fn with_lengths(
        pattern: &str,
        operation: Operation,
        axes_lengths: &[(&str, usize)],
    ) -> Result<Self, EinopsError> {
        let recipe =
            TransformRecipe::new(pattern, Function::Reduce(operation), Some(axes_lengths))?;

        Ok(Self { recipe })
    }

    /// Perform the reduction transformation on the supplied tensor
    pub fn apply<T: Backend>(&self, tensor: &T) -> Result<T, EinopsError> {
        self.recipe.apply(tensor)
    }
}

/// Repeat allows reordering elements and repeating them in arbitrary combinations. This
/// includes functionality of repeat, tile, and broadcast functions
#[derive(Debug)]
pub struct Repeat {
    recipe: TransformRecipe,
}

impl Repeat {
    /// Initialize with pattern
    pub fn new(pattern: &str) -> Result<Self, EinopsError> {
        let recipe = TransformRecipe::new(pattern, Function::Repeat, None)?;

        Ok(Self { recipe })
    }

    /// Initialize with pattern and lengths of axes
    pub fn with_lengths(
        pattern: &str,
        axes_lengths: &[(&str, usize)],
    ) -> Result<Self, EinopsError> {
        let recipe = TransformRecipe::new(pattern, Function::Repeat, Some(axes_lengths))?;

        Ok(Self { recipe })
    }

    /// Perform operation on the input tensor
    pub fn apply<T: Backend>(&self, tensor: &T) -> Result<T, EinopsError> {
        self.recipe.apply(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, IndexOp, Kind, Tensor};

    #[test]
    fn collapsed_ellipsis_error() {
        let patterns = &["a b c d (...) -> a b c ... d", "(...) -> (...)"];

        for pattern in patterns {
            assert!(Rearrange::new(pattern).is_err());
        }
    }

    #[test]
    fn rearrange_consistency() -> Result<(), EinopsError> {
        let input = Tensor::arange(1 * 2 * 3 * 5 * 7 * 11, (Kind::Float, Device::Cpu))
            .reshape(&[1, 2, 3, 5, 7, 11]);

        let output = Rearrange::new("a b c d e f -> a (b) (c d e) f")?.apply(&input)?;
        assert_eq!(
            input.flatten(0, input.size().len() as i64 - 1),
            output.flatten(0, output.size().len() as i64 - 1)
        );

        let output1 = Rearrange::new("a b c d e f -> f e d c b a")?.apply(&input)?;
        let output2 = Rearrange::new("f e d c b a -> a b c d e f")?.apply(&input)?;
        assert_eq!(output1, output2);

        let rearrange1 = Rearrange::new("a b c d e f -> (f d) c (e b) a")?;
        let rearrange2 =
            Rearrange::with_lengths("(f d) c (e b) a -> a b c d e f", &[("b", 2), ("d", 5)])?;
        let output = rearrange2.apply(&rearrange1.apply(&input)?)?;
        assert_eq!(output, input);

        let input = Tensor::arange(2 * 3 * 4, (Kind::Float, Device::Cpu)).reshape(&[2, 3, 4]);
        let output = Rearrange::new("a b c -> b c a")?.apply(&input)?;
        assert_eq!(input.i((1, 2, 3)), output.i((2, 3, 1)));
        assert_eq!(input.i((0, 1, 2)), output.i((1, 2, 0)));

        Ok(())
    }

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

        let input =
            Tensor::arange(2 * 3 * 4 * 5 * 6, (Kind::Float, Device::Cpu)).reshape(&[2, 3, 4, 5, 6]);

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

        let input =
            Tensor::arange(2 * 3 * 4 * 5 * 6, (Kind::Float, Device::Cpu)).reshape(&[2, 3, 4, 5, 6]);

        for (pattern1, pattern2) in patterns {
            let output1 = Rearrange::new(pattern1)?.apply(&input)?;
            let output2 = Rearrange::new(pattern2)?.apply(&input)?;

            assert_eq!(output1, output2);
        }

        Ok(())
    }

    #[test]
    fn equivalent_reduction_patterns() -> Result<(), EinopsError> {
        let patterns = &[
            ("a b c d e -> ", "... -> "),
            ("a b c d e -> (e a)", "a ... e -> (e a)"),
            ("a b c d e -> d (a e)", "a b c d e ... -> d (a e)"),
            ("a b c d e -> (a b)", "... c d e -> (...)"),
        ];
        let operations = &[Operation::Sum, Operation::Min, Operation::Max];

        let input =
            Tensor::arange(2 * 3 * 4 * 5 * 6, (Kind::Float, Device::Cpu)).reshape(&[2, 3, 4, 5, 6]);

        for operation in operations {
            for (pattern1, pattern2) in patterns {
                let output1 = Reduce::new(pattern1, *operation)?.apply(&input)?;
                let output2 = Reduce::new(pattern2, *operation)?.apply(&input)?;

                assert_eq!(output1, output2);
            }
        }

        Ok(())
    }

    #[test]
    fn repeat_anonymous_patterns() -> Result<(), EinopsError> {
        let tests = &[
            ("a b c d -> (c 2 d a b)", vec![("a", 1), ("c", 4), ("d", 6)]),
            ("1 b c d -> (d copy 1) 3 b c", vec![("copy", 3)]),
            (
                "() ... d -> 1 (copy1 d copy2) ...",
                vec![("copy1", 2), ("copy2", 3)],
            ),
            ("1 ... -> 3 ...", vec![]),
            ("1 b c d -> (1 1) (1 b) 2 c 3 d (1 1)", vec![]),
        ];

        let input =
            Tensor::arange(1 * 2 * 4 * 6, (Kind::Float, Device::Cpu)).reshape(&[1, 2, 4, 6]);

        for (pattern, lengths) in tests {
            let output = Repeat::with_lengths(pattern, lengths.as_slice())?.apply(&input)?;

            let mut pattern = pattern.split("->").collect::<Vec<_>>();
            pattern.reverse();
            let pattern = pattern.join("->");
            let expected_min =
                Reduce::with_lengths(pattern.as_str(), Operation::Min, lengths.as_slice())?
                    .apply(&output)?;
            let expected_max =
                Reduce::with_lengths(pattern.as_str(), Operation::Max, lengths.as_slice())?
                    .apply(&output)?;

            assert_eq!(input, expected_min);
            assert_eq!(input, expected_max);
        }

        Ok(())
    }

    #[test]
    fn repeat_patterns() -> Result<(), EinopsError> {
        let tests = &[
            ("a b c -> c a b", vec![]),
            (
                "a b c -> (c copy a b)",
                vec![("copy", 2), ("a", 2), ("b", 3), ("c", 5)],
            ),
            ("a b c -> (a copy) b c", vec![("copy", 1)]),
            (
                "a b c -> (c a) (copy1 b copy2)",
                vec![("a", 2), ("copy1", 1), ("copy2", 2)],
            ),
            ("a ... -> a ... copy", vec![("copy", 4)]),
            (
                "... c -> ... (copy1 c copy2)",
                vec![("copy1", 1), ("copy2", 2)],
            ),
            ("... -> copy1 ... copy2", vec![("copy1", 2), ("copy2", 3)]),
            ("... -> ...", vec![]),
            (
                "a b c -> copy1 a copy2 b c ()",
                vec![("copy1", 2), ("copy2", 1)],
            ),
        ];

        let input = Tensor::arange(2 * 3 * 5, (Kind::Float, Device::Cpu)).reshape(&[2, 3, 5]);

        for (pattern, lengths) in tests {
            let output = Repeat::with_lengths(pattern, lengths.as_slice())?.apply(&input)?;

            let mut pattern = pattern.split("->").collect::<Vec<_>>();
            pattern.reverse();
            let pattern = pattern.join("->");
            let expected_min =
                Reduce::with_lengths(pattern.as_str(), Operation::Min, lengths.as_slice())?
                    .apply(&output)?;
            let expected_max =
                Reduce::with_lengths(pattern.as_str(), Operation::Max, lengths.as_slice())?
                    .apply(&output)?;

            assert_eq!(input, expected_min);
            assert_eq!(input, expected_max);
        }

        Ok(())
    }

    #[test]
    fn rearrange_samples() -> Result<(), EinopsError> {
        let tests: &[(&str, Option<&[(&str, usize)]>, Vec<usize>)] = &[
            ("b c h w -> b h w c", None, vec![10, 30, 40, 20]),
            ("b c h w -> b (c h w)", None, vec![10, 20 * 30 * 40]),
            (
                "b (c h1 w1) h w -> b c (h h1) (w w1)",
                Some(&[("h1", 2), ("w1", 2)]),
                vec![10, 5, 30 * 2, 40 * 2],
            ),
            (
                "b c (h h1) (w w1) -> b (h1 w1 c) h w",
                Some(&[("h1", 2), ("w1", 2)]),
                vec![10, 20 * 4, 30 / 2, 40 / 2],
            ),
            (
                "b1 sound b2 letter -> b1 b2 sound letter",
                None,
                vec![10, 30, 20, 40],
            ),
        ];

        let input = Tensor::arange(10 * 20 * 30 * 40, (Kind::Float, Device::Cpu))
            .reshape(&[10, 20, 30, 40]);

        for (pattern, axes_lengths, expected) in tests {
            let output = if let Some(lengths) = axes_lengths {
                input.rearrange_with_lengths(pattern, lengths)?
            } else {
                input.rearrange(pattern)?
            };

            assert_eq!(&output.shape(), expected);
        }

        Ok(())
    }

    #[test]
    fn reduce_samples() -> Result<(), EinopsError> {
        let tests: &[(&str, Operation, Option<&[(&str, usize)]>, Vec<usize>)] = &[
            (
                "b c (h h1) (w w1) -> b c h w",
                Operation::Max,
                Some(&[("h1", 2), ("w1", 2)]),
                vec![10, 20, 30 / 2, 40 / 2],
            ),
            (
                "b c h w -> b c () ()",
                Operation::Max,
                None,
                vec![10, 20, 1, 1],
            ),
        ];

        let input = Tensor::arange(10 * 20 * 30 * 40, (Kind::Float, Device::Cpu))
            .reshape(&[10, 20, 30, 40]);

        for (pattern, operation, axes_lengths, expected) in tests {
            let output = if let Some(lengths) = axes_lengths {
                input.reduce_with_lengths(pattern, *operation, lengths)?
            } else {
                input.reduce(pattern, *operation)?
            };

            assert_eq!(&output.shape(), expected);
        }

        Ok(())
    }
}
