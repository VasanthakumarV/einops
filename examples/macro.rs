use einops::rearrange;
use tch::{Device, Kind, Tensor};

fn main() {
    let input = Tensor::arange(2 * 3 * 4 * 5, (Kind::Float, Device::Cpu)).reshape(&[6, 4, 5]);
    let output = rearrange!("(a bb:3) c d -> bb (c a) d", input);
    dbg!(&output);
}
