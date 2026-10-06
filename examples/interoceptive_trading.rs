//! Trains the interoceptive trading agent and checks that stepping the
//! simulator allocates nothing.
//!
//! ```bash
//! cargo run --release --example interoceptive_trading            # synthetic candles
//! cargo run --release --example interoceptive_trading -- ohlcv.csv
//! ```

use rusting_brain::interoceptive::{
    Action, AgentConfig, MarketEnvironment, SimConfig, Trainer, TrainerConfig, synthetic_ticks,
    ticks_from_csv,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ticks = match std::env::args().nth(1) {
        Some(path) => ticks_from_csv(&std::fs::read_to_string(path)?)?,
        None => synthetic_ticks(20_000, 42),
    };
    let data = Arc::new(ticks);
    let sim = SimConfig::default();

    let mut env = MarketEnvironment::new(data.clone(), sim.clone());
    let mut obs = vec![0.0; sim.obs_dim()];
    env.reset_into(&mut obs);
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for i in 0..100_000 {
        let info = env.step_into(Action::from_index(i % Action::COUNT), &mut obs);
        if info.done {
            env.reset_into(&mut obs);
        }
    }
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - before;
    println!("100000 simulator steps, {allocations} heap allocations");
    assert_eq!(allocations, 0);

    let agent = AgentConfig::new(sim.obs_dim());
    let mut trainer = Trainer::new(data, sim, agent, TrainerConfig::default());
    let start = std::time::Instant::now();
    let report = trainer.run();
    println!("{report:#?}");
    println!("trained in {:.1?}", start.elapsed());
    Ok(())
}
