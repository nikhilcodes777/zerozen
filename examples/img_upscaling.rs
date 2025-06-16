use anyhow::{Ok, Result};
use image::{ImageBuffer, Luma};
use zerozen::activations::functions::ActivationKind;
use zerozen::{
    layer::config::LayerConfig, linalg::matrix::Matrix, loss::functions::LossKind,
    network::core::Network,
};

fn main() -> Result<()> {
    let mut network = Network::builder()
        .layer(LayerConfig {
            neurons: 7,
            activator: ActivationKind::LeakyReLU(0.01),
        })
        .layer(LayerConfig {
            neurons: 5,
            activator: ActivationKind::LeakyReLU(0.01),
        })
        .layer(LayerConfig {
            neurons: 2,
            activator: ActivationKind::LeakyReLU(0.01),
        })
        .layer(LayerConfig {
            neurons: 1,
            activator: ActivationKind::Sigmoid,
        })
        .loss(LossKind::MeanSquaredError)
        .batch_size(200)
        .epochs(10000 * 15)
        .with_logging(true, 100)
        .learning_rate(0.4)
        .build(2)?;
    let training_height = 28;
    let training_width = 28;

    let raw_img = vec![
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 189, 190, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 143, 247, 153, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136, 247, 242, 86, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 192, 252, 187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 62, 185, 18, 0, 0, 0, 0, 89, 236, 217, 47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 216, 253, 60, 0, 0, 0, 0, 212, 255, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 206, 252, 68, 0, 0, 0, 48, 242, 253, 89, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 131, 251, 212, 21, 0, 0, 11, 167, 252, 197, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 29, 232, 247, 63, 0, 0, 0, 153, 252, 226, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 45, 219, 252, 143, 0, 0, 0, 116, 249, 252, 103, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 4, 96, 253, 255, 253, 200, 122, 7, 25, 201, 250, 158, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 92, 252, 252, 253, 217, 252, 252, 200, 227, 252, 231,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 87, 251, 247, 231, 65, 48, 189, 252, 252,
        253, 252, 251, 227, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 190, 221, 98, 0, 0, 0,
        42, 196, 252, 253, 252, 252, 162, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 111, 29, 0,
        0, 0, 0, 62, 239, 252, 86, 42, 42, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 15, 148, 253, 218, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 121, 252, 231, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 31, 221, 251, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 218, 252, 160, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 122, 252, 82, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    println!("Training image");
    let mut train_image = ImageBuffer::new(training_width as u32, training_height as u32);
    let train_pixels = raw_img.clone();

    for y in 0..training_height {
        for x in 0..training_width {
            let pixel_index = (y * training_width + x) as usize;
            let brightness = (train_pixels[pixel_index]) as u8;
            let pixel = Luma([brightness]);
            train_image.put_pixel(x as u32, y as u32, pixel);
        }
    }

    train_image.save("og_image.png").unwrap();
    println!("Built training image");

    let mut training_inputs = Vec::new();
    let mut training_targets = Vec::new();

    for y in 0..training_height {
        for x in 0..training_width {
            let norm_x = x as f64 / (training_width - 1) as f64;
            let norm_y = y as f64 / (training_height - 1) as f64;

            let brightness = raw_img[y * training_width + x];
            training_inputs.extend_from_slice(&[norm_x, norm_y]);
            training_targets.push(brightness as f64 / 255.0);
        }
    }

    let input_rows = training_width * training_height;
    let input = Matrix::new(input_rows, 2, training_inputs).unwrap();
    let targets = Matrix::new(input_rows, 1, training_targets).unwrap();

    println!("training started");
    network.train_sgd(&input, &targets)?;
    println!("training completed");

    println!("Upscaling to 512x512...");

    let upscale_resolution: u32 = 256;
    let mut upscale_coords = Vec::new();

    for y in 0..upscale_resolution {
        for x in 0..upscale_resolution {
            let norm_x = x as f64 / (upscale_resolution - 1) as f64;
            let norm_y = y as f64 / (upscale_resolution - 1) as f64;
            upscale_coords.extend_from_slice(&[norm_x, norm_y]);
        }
    }

    let upscale_input = Matrix::new(
        upscale_resolution as usize * upscale_resolution as usize,
        2,
        upscale_coords,
    )?;

    let predictions = network.forward(&upscale_input)?;

    println!("Upscaling complete. You can now reconstruct the image from the predictions matrix.");

    println!("Making image");
    let mut final_image = ImageBuffer::new(upscale_resolution, upscale_resolution);
    let predicted_pixels = predictions.data;

    for y in 0..upscale_resolution {
        for x in 0..upscale_resolution {
            let pixel_index = (y * upscale_resolution + x) as usize;
            let brightness = (predicted_pixels[pixel_index] * 255.0) as u8;
            let pixel = Luma([brightness]);
            final_image.put_pixel(x, y, pixel);
        }
    }

    final_image.save("upscaled_image.png").unwrap();
    println!("Saved upscaled image as upscaled_image.png");
    Ok(())
}
