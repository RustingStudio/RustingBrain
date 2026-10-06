//! `docs/interoceptive_spec.md` §5.3: stepping the simulation and cycling the
//! replay buffer must not touch the heap. The Q-network's forward and backward
//! passes are out of scope; they work on `Matrix` values and allocate by design.
//!
//! This file is its own test binary, so the counting allocator below sees only
//! this test.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rand::SeedableRng;
use rand::rngs::StdRng;
use rusting_brain::interoceptive::{
    Action, BatchData, MarketEnvironment, ReplayBuffer, SimConfig, synthetic_ticks,
};

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocations() -> usize {
    ALLOCATIONS.load(Ordering::Relaxed)
}

#[test]
fn stepping_and_replaying_never_allocate() {
    // The counter itself has to see an allocation, or a pass proves nothing.
    let before = allocations();
    std::hint::black_box(vec![0u8; 16]);
    assert!(allocations() > before, "counting allocator is not installed");

    let sim = SimConfig {
        lookback: 8,
        trend_horizons: vec![4, 16],
        episode_len: 32,
        ..SimConfig::default()
    };
    let dim = sim.obs_dim();
    let mut env = MarketEnvironment::new(Arc::new(synthetic_ticks(500, 1)), sim);
    let mut buffer = ReplayBuffer::new(256, dim);
    let mut batch = BatchData::new(32, dim);
    let mut rng = StdRng::seed_from_u64(7);
    let (mut obs, mut next_obs) = (vec![0.0f32; dim], vec![0.0f32; dim]);
    let mut body = env.reset_into(&mut obs);

    let before = allocations();
    // Several episodes, so resets and shock draws are inside the window too.
    for step in 0..1000 {
        let action = Action::from_index(step % Action::COUNT);
        let info = env.step_into(action, &mut next_obs);
        buffer.push(&obs, &body, action as usize, info.reward, &next_obs, &info.body, info.done);
        body = info.body;
        if info.done {
            body = env.reset_into(&mut next_obs);
        }
        std::mem::swap(&mut obs, &mut next_obs);
        buffer.sample_into(&mut batch, &mut rng);
    }
    assert_eq!(allocations() - before, 0, "the step loop allocated");
}
