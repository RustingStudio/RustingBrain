//! Chapter 15: reading a mixture-of-experts layer.
//!
//! Two things you cannot see from a loss curve alone:
//!
//! 1. What `parameter_counts()` is actually reporting, and which knob moves
//!    total without moving active.
//! 2. What the load-balancing loss looks like when routing is healthy and when
//!    it has collapsed — so you recognise the second one in a real run.
//!
//! Run with `cargo run --release`.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rusting_brain::{Layout, Matrix, MoeConfig, MoeLayer, TransformerConfig};

const D_MODEL: usize = 64;
const TOKENS: usize = 512;
const EXPERTS: usize = 8;

/// Total and active parameters as the expert count changes.
///
/// `TransformerConfig::parameter_counts` works from the configuration alone, so
/// this costs nothing — no model is built.
fn parameter_table() {
    println!("experts   total    active    ratio");

    for num_experts in [1, 2, 4, 8, 16, 32] {
        let config = TransformerConfig {
            num_experts,
            experts_per_token: 2.min(num_experts),
            moe_layers: (2..8).collect(),
            ..TransformerConfig::default()
        };
        let counts = config.parameter_counts();
        println!(
            "{num_experts:>7}  {:>5.1}M   {:>5.1}M    {:>4.2}x",
            counts.total as f32 / 1e6,
            counts.active as f32 / 1e6,
            counts.total as f32 / counts.active as f32,
        );
    }
}

fn random_tokens(rows: usize, seed: u64) -> Matrix {
    let mut rng = StdRng::seed_from_u64(seed);
    let data = (0..rows * D_MODEL).map(|_| rng.gen_range(-1.0..1.0)).collect();
    Matrix::from_vec(rows, D_MODEL, data)
}

/// Every token identical, so the router has nothing to tell them apart by and
/// sends all of them to the same two experts. This is what collapse looks like
/// from the outside.
fn identical_tokens(rows: usize, seed: u64) -> Matrix {
    let single = random_tokens(1, seed);
    Matrix::from_vec(rows, D_MODEL, single.data.repeat(rows))
}

fn report(label: &str, layer: &MoeLayer, tokens: &Matrix) -> Result<(), Box<dyn std::error::Error>> {
    let (_output, cache) = layer.forward_train(tokens, Layout::default())?;
    let loads = cache.load_fractions();

    let busiest = loads.iter().copied().fold(0.0_f32, f32::max);
    let dead = loads.iter().filter(|&&share| share < 0.01).count();

    println!("{label}");
    print!("  routing ");
    for share in &loads {
        print!("{:>6.1}%", share * 100.0);
    }
    println!();
    println!(
        "  busiest {:.1}%   dead experts {dead}/{EXPERTS}   aux {:.4}   z {:.4}",
        busiest * 100.0,
        cache.aux_loss(),
        cache.z_loss(),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("== Total vs active, as experts are added ==\n");
    parameter_table();
    println!(
        "\nActive barely moves: top-2 of 32 experts costs the same per token as\n\
         top-2 of 4. Total is what grows.\n"
    );

    println!("== What the load-balancing loss measures ==\n");

    let mut rng = StdRng::seed_from_u64(7);
    let config = MoeConfig::new(EXPERTS, 2, 32);
    let layer = MoeLayer::new(D_MODEL, config, &mut rng)?;

    report("varied tokens, balanced routing:", &layer, &random_tokens(TOKENS, 1))?;
    println!();
    report("identical tokens, collapsed routing:", &layer, &identical_tokens(TOKENS, 1))?;

    println!(
        "\nThe second aux loss is the number to watch in a real run. Climbing\n\
         toward num_experts while lm_loss falls means the router is collapsing\n\
         and most of the checkpoint is about to stop training."
    );

    Ok(())
}
