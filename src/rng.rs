//! A random number generator a training run can checkpoint.
//!
//! A run that stops at step 24,000 and resumes reseeds its generator, and then
//! draws the same noise, the same timesteps and the same shuffle it drew in the
//! first hundred steps of training. The loss curve does not show it, but the
//! second half of the run is training on a replay of the first.
//!
//! # Why a counter and not the generator's own state
//!
//! The obvious fix is to write [`StdRng`]'s internal state into the checkpoint
//! and put it back on resume. `rand` 0.8 does not allow it: `StdRng` is
//! deliberately opaque — no accessor for its stream position, no `Serialize`,
//! and no promise that the algorithm behind it stays the same between releases.
//! Anything that pickled those bytes would be writing a number whose meaning
//! the next `cargo update` is free to change.
//!
//! So a [`RunRng`] stores what it can reproduce from: a seed and a step
//! counter. Each step gets its own [`StdRng`], seeded by mixing the two, which
//! is the trick [`TokenFile`](crate::token_file::TokenFile) already uses to
//! shuffle a corpus the same way every epoch. Two `u64`s go in the checkpoint,
//! and a resumed run continues the sequence exactly rather than replaying it.
//!
//! The one rule this asks of a caller: draw from the generator [`RunRng::next_step`]
//! hands out, and let it go at the end of the step. A generator kept across
//! steps is back to having a state nobody can write down.
//!
//! ```no_run
//! # use rusting_brain::RunRng;
//! # use std::collections::BTreeMap;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let metadata = BTreeMap::new();
//! // Resuming: the checkpoint's counter if it has one, seed 42 if it does not.
//! let mut run = RunRng::from_metadata(&metadata, 42);
//!
//! for _ in 0..1_000 {
//!     let mut rng = run.next_step();          // this step's generator
//!     // model.train_step(&batch, &mut rng)?;
//! }
//!
//! let mut metadata = BTreeMap::new();
//! run.to_metadata(&mut metadata);        // goes in the checkpoint header
//! # Ok(())
//! # }
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The metadata key holding the seed a run started from.
pub const SEED_KEY: &str = "rng_seed";
/// The metadata key holding how many steps have drawn from it.
pub const STEP_KEY: &str = "rng_step";

/// A seed and a position in the sequence of per-step generators it defines.
///
/// Cheap to copy and to store: a checkpoint carries two decimal numbers, and
/// [`RunRng::from_metadata`] turns them back into a run that carries on where
/// it left off.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunRng {
    seed: u64,
    step: u64,
}

impl RunRng {
    /// Starts a run at the beginning of `seed`'s sequence.
    pub fn new(seed: u64) -> Self {
        Self { seed, step: 0 }
    }

    /// Starts a run from a seed drawn from the operating system, for a run that
    /// wants a different sample every time and still wants to resume exactly.
    /// The seed it picked is in [`RunRng::seed`], and in the checkpoint.
    pub fn from_entropy() -> Self {
        Self::new(rand::random())
    }

    /// The seed this run started from.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// How many steps have drawn a generator, which is also the step
    /// [`RunRng::next_step`] will hand out.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// The generator for the next step, advancing the counter past it.
    pub fn next_step(&mut self) -> StdRng {
        let rng = self.at(self.step);
        self.step += 1;
        rng
    }

    /// The generator for a given step, without moving the counter. Two calls
    /// with the same step give two generators that draw the same numbers.
    pub fn at(&self, step: u64) -> StdRng {
        // Mixed rather than added: `StdRng` gives unrelated streams for
        // consecutive seeds, but a run whose seed is a small number and whose
        // step is a small number would otherwise collide with another one at a
        // different step. This is the mix `TokenFile` shuffles with.
        StdRng::seed_from_u64(self.seed ^ step.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    /// Moves the counter, for a caller that restores a step count from
    /// somewhere else in the checkpoint and wants the generator to agree.
    pub fn set_step(&mut self, step: u64) {
        self.step = step;
    }

    /// Writes the seed and the counter into checkpoint metadata.
    pub fn to_metadata(&self, metadata: &mut BTreeMap<String, String>) {
        metadata.insert(SEED_KEY.to_string(), self.seed.to_string());
        metadata.insert(STEP_KEY.to_string(), self.step.to_string());
    }

    /// Reads back what [`RunRng::to_metadata`] wrote.
    ///
    /// A checkpoint written before this existed carries neither key, and there
    /// is nothing to recover: the run falls back to `seed` at step zero, which
    /// is what that run was already doing. A checkpoint that carries a seed but
    /// no counter — hand-written metadata, or a header the run built itself —
    /// starts that seed at zero rather than refusing.
    pub fn from_metadata(metadata: &BTreeMap<String, String>, seed: u64) -> Self {
        let stored = metadata.get(SEED_KEY).and_then(|s| s.parse().ok());
        Self {
            seed: stored.unwrap_or(seed),
            step: metadata
                .get(STEP_KEY)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn draws(rng: &mut StdRng, count: usize) -> Vec<u32> {
        (0..count).map(|_| rng.r#gen()).collect()
    }

    #[test]
    fn a_resumed_run_carries_on_the_sequence_rather_than_replaying_it() {
        let mut uninterrupted = RunRng::new(42);
        let whole: Vec<Vec<u32>> = (0..20)
            .map(|_| draws(&mut uninterrupted.next_step(), 4))
            .collect();

        // The same run, stopped after ten steps and resumed from metadata.
        let mut first_half = RunRng::new(42);
        let opening: Vec<Vec<u32>> = (0..10)
            .map(|_| draws(&mut first_half.next_step(), 4))
            .collect();
        let mut metadata = BTreeMap::new();
        first_half.to_metadata(&mut metadata);

        let mut resumed = RunRng::from_metadata(&metadata, 0);
        assert_eq!(resumed.step(), 10);
        let rest: Vec<Vec<u32>> = (0..10)
            .map(|_| draws(&mut resumed.next_step(), 4))
            .collect();

        let joined: Vec<Vec<u32>> = opening.into_iter().chain(rest).collect();
        assert_eq!(joined, whole);
    }

    #[test]
    fn a_run_that_reseeded_instead_would_have_replayed_its_first_steps() {
        // The bug this type exists for: without the counter the resumed run
        // draws the opening steps again.
        let mut run = RunRng::new(42);
        let opening = draws(&mut run.next_step(), 4);
        for _ in 0..9 {
            run.next_step();
        }
        let reseeded = draws(&mut RunRng::new(42).next_step(), 4);
        assert_eq!(reseeded, opening);
        assert_ne!(draws(&mut run.next_step(), 4), opening);
    }

    #[test]
    fn consecutive_steps_draw_unrelated_numbers() {
        let run = RunRng::new(7);
        let first = draws(&mut run.at(0), 8);
        let second = draws(&mut run.at(1), 8);
        assert_ne!(first, second);
        assert_eq!(first, draws(&mut run.at(0), 8), "a step is reproducible");
    }

    #[test]
    fn a_checkpoint_from_before_this_existed_falls_back_to_the_seed() {
        let older = BTreeMap::from([("optimizer_step".to_string(), "1200".to_string())]);
        let run = RunRng::from_metadata(&older, 42);
        assert_eq!(run, RunRng::new(42));

        // A seed with no counter is a run that has not stepped yet.
        let partial = BTreeMap::from([(SEED_KEY.to_string(), "9".to_string())]);
        let run = RunRng::from_metadata(&partial, 42);
        assert_eq!(run.seed(), 9);
        assert_eq!(run.step(), 0);
    }

    #[test]
    fn metadata_survives_a_round_trip_through_json() {
        let mut run = RunRng::from_entropy();
        for _ in 0..5 {
            run.next_step();
        }
        let restored: RunRng = serde_json::from_str(&serde_json::to_string(&run).unwrap()).unwrap();
        assert_eq!(restored, run);
    }
}
