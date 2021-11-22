use einops::einops;
use tch::{Device, Kind, Tensor};

fn main() {
    let input = Tensor::arange(6 * 3 * 2 * 4 * 5 * 12, (Kind::Float, Device::Cpu))
        .reshape(&[18, 2, 4, 5, 12]);
    //let output = rearrange!("(a bb:3 min(e:3)) max(f) .. g -> (..) 3 (bb a) g", input);
    let output = einops!("(a f:2) a b c d e -> a f b c d e", input);
    dbg!(&output);
}
