//! Reports device memory after each stage of bringing a model up, so a large
//! fixed cost can be told apart from a per-batch one.
//!
//! `cargo run --release --features cuda --example vram_probe`

#[cfg(feature = "cuda")]
fn main() {
    use rusting_brain::{Optimizer, TransformerLm};

    fn used(stage: &str) {
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            .output()
            .unwrap();
        let mib = String::from_utf8_lossy(&output.stdout).trim().to_string();
        println!("{stage:<34} {mib:>6} MiB");
    }

    used("before anything");
    let context = rusting_brain::GpuContext::new(0).unwrap();
    let _ = &context;
    used("after GpuContext::new");
    let mut model = TransformerLm::builder()
        .vocab_size(16_384)
        .n_layers(1)
        .moe_layers(0..0)
        .max_seq_len(256)
        .optimizer(Optimizer::adam(1e-4))
        .seed(7)
        .build()
        .unwrap();
    used("after building on the host");
    model.to_cuda(0, 11_000).unwrap();
    used("after to_cuda");

    let tokens: Vec<Vec<u32>> = vec![(0..256).map(|t| (t * 37 % 16_384) as u32).collect()];
    model.train_step(&tokens).unwrap();
    used("after one train_step");
    model.train_step(&tokens).unwrap();
    used("after two train_steps");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("rebuild with --features cuda");
}
