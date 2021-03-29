use thiserror::Error;

#[derive(Error, Debug)]
pub enum EinopsError {
    #[error("expression parse error: {0}")]
    Parse(String),
    #[error("pattern rules violated: {0}")]
    Pattern(String),
    #[error("invalid input found: {0}")]
    InvalidInput(String),
}
