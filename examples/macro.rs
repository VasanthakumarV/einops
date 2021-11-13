use einops::rearrange;
use tch::{Device, Kind, Tensor};

fn main() {
    let input = Tensor::arange(2 * 3 * 4, (Kind::Float, Device::Cpu)).reshape(&[2, 3, 4]);
    let output = rearrange!("a b c -> b (c a)", input);
    dbg!(&output);
}
