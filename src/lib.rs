pub(crate) mod backend;
pub mod error;
mod parse;

use backend::Backend;

fn rearrange<T: Backend>(tensor: T, expression: &str, axes_lengths: &[(&str, isize)]) {
    println!("{}", expression);
    println!("{:?}", axes_lengths);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        rearrange("a -> b c", &[("b", 2), ("a", 10)]);

        assert!(true);
    }
}
