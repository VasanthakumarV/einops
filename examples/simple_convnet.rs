// To execute this example, you will have to unzip the [MNIST data files](http://yann.lecun.com/exdb/mnist/) in `data/`
use einops::EinopsError;
use einops::Rearrange;
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device};

fn simple_convnet(vs: &nn::Path) -> Result<impl ModuleT, EinopsError> {
    Ok(nn::seq_t()
        // Input images will be of size (batch, 28*28), we explode that
        // into the required shape (batch, 1, 28, 28)
        .add(Rearrange::with_lengths(
            "b (1 w h) -> b 1 w h",
            &[("w", 28), ("h", 28)],
        )?)
        .add(nn::conv2d(vs, 1, 32, 5, Default::default()))
        .add_fn(|xs| xs.max_pool2d_default(2))
        .add(nn::conv2d(vs, 32, 64, 5, Default::default()))
        .add_fn(|xs| xs.max_pool2d_default(2))
        // We flatten the tensor before the linear layer
        .add(Rearrange::new("b c h w -> b (c h w)")?)
        .add(nn::linear(vs, 1024, 1024, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn_t(|xs, train| xs.dropout(0.5, train))
        .add(nn::linear(vs, 1024, 10, Default::default())))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let m = tch::vision::mnist::load_dir("data")?;

    let vs = nn::VarStore::new(Device::cuda_if_available());

    let net = simple_convnet(&vs.root())?;

    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;

    for epoch in 1..100 {
        for (bimages, blabels) in m.train_iter(256).shuffle().to_device(vs.device()) {
            let loss = net
                .forward_t(&bimages, true)
                .cross_entropy_for_logits(&blabels);

            opt.backward_step(&loss);
        }

        let test_accuracy =
            net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 1024);

        println!(
            "Epoch: {:4}, Test acc: {:5.2}%",
            epoch,
            100. * test_accuracy
        );
    }

    Ok(())
}
