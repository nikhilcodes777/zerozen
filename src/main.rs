use anyhow::Ok;
use anyhow::Result;
use neurorust::loss::functions::LossFunction;
use neurorust::loss::functions::LossKind;
use neurorust::matrix;
use neurorust::network::core::Network;
use neurorust::{layer::config::LayerConfig, linalg::matrix::Matrix};
fn basicxor() -> Result<()> {
    let mut net = Network::new(
        &[
            LayerConfig {
                neurons: 3,
                activator: neurorust::activations::functions::ActivationKind::Tanh,
            },
            LayerConfig {
                neurons: 1,
                activator: neurorust::activations::functions::ActivationKind::Sigmoid,
            },
        ],
        2,
        0.05,
        LossKind::MeanSquaredError,
    );
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
    for i in 0..20 * 1000 {
        let predictions = net.forward(&input)?;

        if i % 100 == 0 {
            let loss = LossKind::MeanSquaredError.loss(&predictions, &targets)?;
            println!("Loss :- {loss}");
        }
        net.backward(&predictions, &targets);
    }
    let predictions = net.forward(&input)?;
    println!("Training Completed");
    println!("Inputs\n{input}Targets\n{targets}Predictions\n{predictions}");
    Ok(())
}
fn main() -> Result<()> {
    basicxor()?;
    Ok(())
}
