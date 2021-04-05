use thiserror::Error;

/// Different categories of error that can be encountered
#[derive(Error, Debug)]
pub enum EinopsError {
    /// Error when parsing the pattern/expression provided
    #[error("expression parse error: {0}")]
    Parse(String),

    /// Error when a pattern violates rules set by einops
    #[error("pattern rules violated: {0}")]
    Pattern(String),

    /// Error because of invalid/missing identifiers of axes, or their sizes
    #[error("invalid input found: {0}")]
    InvalidInput(String),
}
