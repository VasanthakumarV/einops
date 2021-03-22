use thiserror::Error;

#[derive(Error, Debug)]
pub enum EinopsError {
    #[error("expression parse error: {0}")]
    Parse(String),
}
