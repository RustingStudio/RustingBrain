//! Stress injection: some episodes get a flash crash or a liquidity void.

use rand::Rng;
use rand::rngs::StdRng;

pub const FLASH_CRASH_STEPS: usize = 20;
pub const FLASH_CRASH_FACTOR: f32 = 3.0;
pub const LIQUIDITY_VOID_STEPS: usize = 50;
pub const LIQUIDITY_VOID_FACTOR: f32 = 10.0;

/// One episode's shock. `start` counts environment steps from the reset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shock {
    None,
    /// Every falling tick in the window falls three times as far.
    FlashCrash {
        start: usize,
    },
    /// Slippage costs ten times as much for the window.
    LiquidityVoid {
        start: usize,
    },
}

impl Shock {
    /// With `probability`, one of the two shocks at a uniform start; half each.
    pub fn draw(rng: &mut StdRng, probability: f32, episode_len: usize) -> Self {
        if rng.gen_range(0.0..1.0f32) >= probability {
            return Shock::None;
        }
        if rng.gen_bool(0.5) {
            let start = rng.gen_range(0..episode_len.saturating_sub(FLASH_CRASH_STEPS).max(1));
            Shock::FlashCrash { start }
        } else {
            let start = rng.gen_range(0..episode_len.saturating_sub(LIQUIDITY_VOID_STEPS).max(1));
            Shock::LiquidityVoid { start }
        }
    }

    /// The return the price makes on its move into step `step + 1`'s tick,
    /// given the return `ret` the data made.
    pub fn shocked_return(&self, step: usize, ret: f32) -> f32 {
        match *self {
            Shock::FlashCrash { start }
                if ret < 0.0 && (start..start + FLASH_CRASH_STEPS).contains(&step) =>
            {
                // A price cannot fall past zero; keep 5% of it at worst.
                (ret * FLASH_CRASH_FACTOR).max(-0.95)
            }
            _ => ret,
        }
    }

    pub fn kappa_multiplier(&self, step: usize) -> f32 {
        match *self {
            Shock::LiquidityVoid { start }
                if (start..start + LIQUIDITY_VOID_STEPS).contains(&step) =>
            {
                LIQUIDITY_VOID_FACTOR
            }
            _ => 1.0,
        }
    }
}
