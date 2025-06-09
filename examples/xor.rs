use anyhow::Ok;
use anyhow::Result;
use zerozen::activations::functions::ActivationKind;
use zerozen::loss::functions::LossKind;
use zerozen::matrix;
use zerozen::network::core::Network;
use zerozen::{layer::config::LayerConfig, linalg::matrix::Matrix};
fn main() -> Result<()> {
    let mut net = Network::builder()
        .layer(LayerConfig {
            neurons: 3,
            activator: ActivationKind::LeakyReLU(0.01),
        })
        .layer(LayerConfig {
            neurons: 1,
            activator: ActivationKind::Sigmoid,
        })
        .loss(LossKind::MeanSquaredError)
        .learning_rate(0.5)
        .epochs(20 * 1000)
        .with_logging(true, 500)
        .build(2)?;

    let input = matrix!(0.0,0.0;
                        1.0,0.0;
                        0.0,1.0;
                        1.0,1.0;

    );
    let targets = matrix!(0.0;
        1.0;
        1.0;
        0.0;
    );

    println!("Training Started");
    net.train(&input, &targets)?;
    println!("Training Completed");
    let predictions = net.forward(&input)?;
    println!("Inputs\n{input}\nTargets\n{targets}\nPredictions\n{predictions}");

    println!("{net}");
    Ok(())
}
