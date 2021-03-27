pub(crate) mod backend;
pub mod error;
mod parse;

use backend::Backend;

fn rearrange<T: Backend>(tensor: T, expression: &str, axes_lengths: &[(&str, isize)]) {
    println!("{}", expression);
    println!("{:?}", axes_lengths);
}
