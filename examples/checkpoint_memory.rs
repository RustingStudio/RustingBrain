//! Peak device memory and time of one training step on the default model, with every
//! block's activations kept and with every block checkpointed.
//!
//! `cargo run --release --features cuda --example checkpoint_memory -- <batch> <seq_len>`
//!
//! A third argument `lora` compares a rank-16 LoRA run on an FP32 base with
//! one on an int8 base (`quantize_lora_base`) instead, neither checkpointed.

#[cfg(feature = "cuda")]
fn main() {
    use cudarc::driver::sys;
    use rusting_brain::{LoraConfig, Optimizer, TransformerLm};

    let arg = |n: usize, default: usize| {
        std::env::args()
            .nth(n)
            .and_then(|a| a.parse().ok())
            .unwrap_or(default)
    };
    let (batch_size, seq_len) = (arg(1, 4), arg(2, 512));
    let batch: Vec<Vec<u32>> = (0..batch_size)
        .map(|s| {
            (0..seq_len)
                .map(|t| ((s * 131 + t * 17) % 32000) as u32)
                .collect()
        })
        .collect();

    // Buffers come from the device's default stream-ordered pool, whose
    // high-water mark is the peak this reports.
    let pool_high = |reset: bool| unsafe {
        let mut pool = std::ptr::null_mut();
        assert_eq!(
            sys::cuDeviceGetDefaultMemPool(&mut pool, 0),
            sys::CUresult::CUDA_SUCCESS
        );
        let mut value = 0u64;
        let attribute = sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH;
        let pointer = (&mut value as *mut u64).cast();
        if reset {
            sys::cuMemPoolSetAttribute(pool, attribute, pointer);
        } else {
            sys::cuMemPoolGetAttribute(pool, attribute, pointer);
        }
        value as f64 / (1 << 20) as f64
    };

    let lora = std::env::args().nth(3).is_some_and(|a| a == "lora");
    for variant in [false, true] {
        let checkpoint = variant && !lora;
        let mut model = TransformerLm::builder()
            .optimizer(Optimizer::adam(1e-4))
            .max_seq_len(seq_len.max(256))
            .checkpoint(0..if checkpoint { 64 } else { 0 })
            .seed(7)
            .build()
            .unwrap();
        if lora {
            model.add_lora(LoraConfig::new(16)).unwrap();
            if variant {
                model.quantize_lora_base().unwrap();
            }
        }
        model.to_cuda(0, 16384).unwrap();
        model.train_step(&batch).unwrap();
        pool_high(true);
        model.train_step(&batch).unwrap();
        let peak = pool_high(false);
        let start = std::time::Instant::now();
        for _ in 0..3 {
            model.train_step(&batch).unwrap();
        }
        println!(
            "batch {batch_size} x seq {seq_len}, {}: peak {peak:.0} MiB, {:.1} ms/step",
            match (lora, variant) {
                (false, _) => format!("checkpoint {checkpoint}"),
                (true, false) => "LoRA on FP32 base".into(),
                (true, true) => "LoRA on int8 base".into(),
            },
            start.elapsed().as_secs_f64() * 1e3 / 3.0
        );
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("rebuild with --features cuda");
}
