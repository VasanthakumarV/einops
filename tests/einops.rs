use einops::einops;
use tch::{Device, IndexOp, Kind, Tensor};

#[test]
fn consistency_checks() {
    let input = Tensor::arange(1 * 2 * 3 * 5 * 7 * 11, (Kind::Float, Device::Cpu))
        .reshape(&[1, 2, 3, 5, 7, 11]);

    let output = einops!("a b c d e f -> a (b) (c d e) f", input);
    assert_eq!(
        input.flatten(0, input.size().len() as i64 - 1),
        output.flatten(0, output.size().len() as i64 - 1)
    );

    let output1 = einops!("a b c d e f -> f e d c b a", input);
    let output2 = einops!("f e d c b a -> a b c d e f", input);
    assert_eq!(output1, output2);

    let intermediate = einops!("a b c d e f -> (f d) c (e b) a", input);
    let output = einops!("(f d:5) c (e b:2) a -> a b c d e f", intermediate);
    assert_eq!(output, input);

    let input = Tensor::arange(2 * 3 * 4, (Kind::Float, Device::Cpu)).reshape(&[2, 3, 4]);
    let output = einops!("a b c -> b c a", input);
    assert_eq!(input.i((1, 2, 3)), output.i((2, 3, 1)));
    assert_eq!(input.i((0, 1, 2)), output.i((1, 2, 0)));
}

//#[test]
//fn identity_patterns() {
//macro_rules! test {
//($pattern:literal, $tensor:ident) => {
//let output = einops!($pattern, $tensor);
//assert_eq!($tensor, output, "{} failed", $pattern);
//};
//}

//let input =
//Tensor::arange(2 * 3 * 4 * 5 * 6, (Kind::Float, Device::Cpu)).reshape(&[2, 3, 4, 5, 6]);
//test!(".. -> ..", input);
//}
