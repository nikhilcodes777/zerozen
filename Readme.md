# ZeroZen

*A neural network from scratch in Rust. No frameworks, no magic — made with just raw math, gradients, and Zen-like focus.*

![ZeroZen Logo](./assets/logo.png)

---

## Why Make This? 🤔

Ever wondered what goes on **inside the black box**?
I did. So I built **zerozen** — a neural network from scratch to deeply understand how it works under the hood.

Also, it beautifully combines:

* My love for **mathematics** (linear algebra and calculus)
* My love for **computer science**
* My curiosity to peek behind the curtain

### Why Rust? 🦀

Because... why not?
I briefly considered C, but for reasons only my past self knows, I gracefully switched to Rust.

### Why Not Python? 🐍

I wanted to refresh my Rust skills and **keep dependencies minimal** — yes, *not even numpy*.
We’re going full monk mode here: no distractions, just raw vectors and pure determination.

---

## Features ✨

* [x] Zero dependencies (except `rand` because I like surprises)
* [x] Minimal yet functional neural network core
* [x] Logging during training
* [x] Loss function history tracking
* [x] Fully modular and customizable layers
* [x] Written in pure Rust 🦀

---

## Running ZeroZen 🏃‍♂️

### The Zen Nix Way 🌿

If you’re a NixOS user (or have Nix+flakes setup):

```bash
# Enter a development shell with all dependencies
nix develop

# Build the executable
nix build

# Run the binary
nix run github:nikhilcodes777/zerozen
```

### The Cargo Way 📦

```bash
cargo run --example xor
```

---

## Usage Example 🚀
### Basic XOR
```rust
fn main() -> Result<()> {
    let mut net = Network::builder() // Builder pattern
        .layer(LayerConfig {
            neurons: 3,
            activator: ActivationKind::LeakyReLU(0.01), // Leaky ReLU with customizable leak
        })
        .layer(LayerConfig {
            neurons: 1,
            activator: ActivationKind::Sigmoid,
        })
        .loss(LossKind::MeanSquaredError) // Customizable loss function
        .learning_rate(0.5)
        .epochs(20 * 1000)
        .with_logging(true, 500) // Log every 500 epochs
        .build(2)?; // Builds a network with 2 input features

    let input = matrix!(0.0, 0.0;
                        1.0, 0.0;
                        0.0, 1.0;
                        1.0, 1.0;);

    let targets = matrix!(0.0;
                          1.0;
                          1.0;
                          0.0;);

    println!("Training Started");
    net.train(&input, &targets)?;
    println!("Training Completed");

    let predictions = net.forward(&input)?;
    println!("Inputs\n{input}\nTargets\n{targets}\nPredictions\n{predictions}");

    println!("{net}");
    Ok(())
}
```
 ### [MNIST image upscaler using coordinate-based learning](./docs/image_upscaling.md)
---

## License 📄

This project is licensed under the [MIT License](LICENSE).

---

*Built with ❤️, lots of ☕, and a healthy dose of "how hard could it be?" attitude.*


