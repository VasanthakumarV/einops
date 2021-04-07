![einops](https://github.com/VasanthakumarV/einops/workflows/CI/badge.svg)
[![crates](https://img.shields.io/crates/v/einops)](https://crates.io/crates/einops)
[![docs](https://img.shields.io/docsrs/einops)](https://docs.rs/einops)

# Einops

This is a rust port of the incredible [einops](https://github.com/arogozhnikov/einops) library.
Almost all the operations specified in its tutorial should be available, if you find any
inconsistencies please raise a github issue.

_Unlike its python counterpart, caching the parsed expression has not been implemented yet. So
when applying the same pattern multiple times, prefer_ `Rearrange::new(...)` _or_ `Rearrange::with_lengths(...)`
_api, over the methods available through `RearrangeFn` like traits_

Flexible and powerful tensor operations for readable and reliable code.
Currently only supports [tch](https://github.com/LaurentMazare/tch-rs).

## Getting started

Add the following to your `Cargo.toml` file,

```
[dependencies]
einops = { version: "0.1", features: ["tch-bindings"] }
```

## Examples

Einops provies three operations, they cover stacking, reshape, transposition,
squeeze/unsqueeze, repeat, tile, concatenate and numerous reductions

```rust
// Tch specific imports
use tch::{Tensor, Kind, Device};
// Structs that provide constructor like api
use einops::{Rearrange, Repeat, Reduce, Operation};
// Traits required to call functions directly on the tensors
use einops::{ReduceFn, RearrangeFn, RepeatFn};

// We create a random tensor as input
let input = Tensor::randn(&[100, 32, 64], (Kind::Float, Device::Cpu));

// ------------------------------------------------------------------------
// Rearrange operation
let output = Rearrange::new("t b c -> b c t")?.apply(&input)?;
assert_eq!(output.size(), vec![32, 64, 100]);

// Apply rearrange operation directly on the tensor using `RearrangeFn` trait
let output = input.rearrange("t b c -> b c t")?
assert_eq!(output.size(), vec![32, 64, 100]);

// ------------------------------------------------------------------------
// Perform reduction on first axis
let output = Reduce::new("t b c -> b c", Operation::Max)?.apply(&input)?;
assert_eq!(output.size(), vec![32, 64]);

// Same reduction done directly on the tensor using `ReduceFn` trait
let output = input.reduce("t b c -> b c", Operation::Max)?;
assert_eq!(output.size(), vec![32, 64]);

// ------------------------------------------------------------------------
// We repeat the third axis
let output = Repeat::with_lengths("t b c -> t b c repeat", &[("repeat", 3)])?.apply(&input);
assert_eq!(output.size(), vec![100, 32, 64, 3]);

// Same as above using `RepeatFn` trait and directly specifying the `repeat` size
// in the pattern
let output = input.repeat("t b c -> t b c 3");
assert_eq!(output.size(), vec![100, 32, 64, 3]);
```

