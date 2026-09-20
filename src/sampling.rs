//! Turning a row of logits into the next token id.
//!
//! The model produces logits; choosing from them is a separate decision with
//! its own knobs, and every one of them is a way of throwing away part of the
//! distribution. Greedy decoding keeps only the argmax, temperature reshapes
//! what is left, and the two truncations cut the tail that a model with a
//! large vocabulary spreads real probability mass across.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// How the next token is drawn from a row of logits.
///
/// Built by [`Sampler::greedy`] or [`Sampler::temperature`] and narrowed with
/// the chained methods:
///
/// ```
/// # use rusting_brain::Sampler;
/// let mut sampler = Sampler::temperature(0.8, Some(42))
///     .top_k(40)
///     .top_p(0.95)
///     .repetition_penalty(1.1);
/// let next = sampler.pick(&[0.1, 3.0, 0.2], &[]);
/// ```
#[derive(Clone, Debug)]
pub struct Sampler {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    rng: StdRng,
}

impl Sampler {
    /// Always takes the highest-scoring token. Reproducible by construction,
    /// and the right default for anything being measured rather than read.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            rng: StdRng::seed_from_u64(0),
        }
    }

    /// Draws from the distribution after dividing the logits by `temperature`:
    /// below one sharpens it, above one flattens it, and zero is
    /// [`Sampler::greedy`].
    ///
    /// `seed` of `None` takes one from the operating system, so two runs of the
    /// same prompt differ.
    pub fn temperature(temperature: f32, seed: Option<u64>) -> Self {
        Self {
            temperature,
            rng: seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64),
            ..Self::greedy()
        }
    }

    /// Keeps only the `k` highest-scoring tokens. Zero keeps all of them.
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Keeps the smallest set of tokens whose probabilities reach `p`, so the
    /// number kept follows how confident the model is. One keeps all of them.
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    /// Divides the logit of every token already in the history by `penalty`,
    /// which is how a small model is stopped from repeating one phrase for as
    /// long as it is allowed to run. One leaves the logits alone.
    pub fn repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// The next token id, given one row of logits and the ids generated so far.
    ///
    /// `history` is read only by [`Sampler::repetition_penalty`] and can be
    /// empty otherwise.
    ///
    /// # Panics
    ///
    /// If `logits` is empty: there is no token to return.
    pub fn pick(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        assert!(!logits.is_empty(), "cannot sample from an empty vocabulary");

        let mut scores: Vec<f32> = logits.to_vec();
        if self.repetition_penalty != 1.0 {
            for &id in history {
                if let Some(score) = scores.get_mut(id as usize) {
                    // A negative logit is made *more* negative, so the penalty
                    // pushes a token down whichever side of zero it sits on.
                    *score = if *score > 0.0 {
                        *score / self.repetition_penalty
                    } else {
                        *score * self.repetition_penalty
                    };
                }
            }
        }

        let mut ranked: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        // `total_cmp` rather than `partial_cmp`: a NaN logit means the model has
        // diverged, and an ordering that panics or silently shuffles makes that
        // harder to see than a token chosen from a defined order.
        let descending = |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1);

        if self.temperature <= 0.0 {
            let mut best = 0;
            for (index, &(_, score)) in ranked.iter().enumerate() {
                if score.total_cmp(&ranked[best].1).is_gt() {
                    best = index;
                }
            }
            return ranked[best].0 as u32;
        }

        // Sorting a 32000-token vocabulary costs more than some models spend
        // producing it. With a top-k, only those k entries have to be in order,
        // and a linear partition finds them.
        if self.top_k > 0 && self.top_k < ranked.len() {
            ranked.select_nth_unstable_by(self.top_k - 1, descending);
            ranked.truncate(self.top_k);
        }
        ranked.sort_unstable_by(descending);

        // Subtracting the maximum before `exp` is not a nicety: logits reach 30
        // or more, `exp(30)` overflows to infinity, and every weight becomes
        // NaN.
        let max = ranked[0].1;
        let mut weights: Vec<f32> = ranked
            .iter()
            .map(|&(_, score)| ((score - max) / self.temperature).exp())
            .collect();
        let total: f32 = weights.iter().sum();

        if self.top_p < 1.0 && total > 0.0 {
            let mut cumulative = 0.0;
            // The token that crosses the threshold is kept, so a distribution
            // whose first token already exceeds `top_p` still leaves one
            // candidate rather than none.
            let kept = weights
                .iter()
                .position(|weight| {
                    cumulative += weight / total;
                    cumulative >= self.top_p
                })
                .map_or(weights.len(), |index| index + 1);
            weights.truncate(kept);
        }

        let total: f32 = weights.iter().sum();
        // NaN as well as zero: a diverged model gives `gen_range` an empty or
        // undefined range, which panics.
        if total <= 0.0 || !total.is_finite() {
            return ranked[0].0 as u32;
        }

        let mut threshold = self.rng.gen_range(0.0..total);
        for (&(id, _), &weight) in ranked.iter().zip(&weights) {
            threshold -= weight;
            if threshold <= 0.0 {
                return id as u32;
            }
        }
        ranked[0].0 as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOGITS: [f32; 5] = [0.5, 3.0, 1.0, -2.0, 0.25];

    #[test]
    fn greedy_takes_the_argmax() {
        assert_eq!(Sampler::greedy().pick(&LOGITS, &[]), 1);
    }

    #[test]
    fn top_k_of_one_is_greedy() {
        let mut sampler = Sampler::temperature(5.0, Some(7)).top_k(1);
        for _ in 0..32 {
            assert_eq!(sampler.pick(&LOGITS, &[]), 1);
        }
    }

    #[test]
    fn top_p_keeps_the_token_that_crosses_the_threshold() {
        // Any `top_p` at or below the leading token's own probability has to
        // leave exactly that token, never an empty candidate set.
        let mut sampler = Sampler::temperature(1.0, Some(7)).top_p(0.01);
        for _ in 0..32 {
            assert_eq!(sampler.pick(&LOGITS, &[]), 1);
        }
    }

    #[test]
    fn the_same_seed_draws_the_same_tokens() {
        let draw = || {
            let mut sampler = Sampler::temperature(1.5, Some(99));
            (0..64)
                .map(|_| sampler.pick(&LOGITS, &[]))
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(), draw());
    }

    #[test]
    fn temperature_widens_the_spread() {
        let distinct = |temperature: f32| {
            let mut sampler = Sampler::temperature(temperature, Some(5));
            let mut seen = (0..256)
                .map(|_| sampler.pick(&LOGITS, &[]))
                .collect::<Vec<_>>();
            seen.sort_unstable();
            seen.dedup();
            seen.len()
        };
        assert!(distinct(0.1) < distinct(5.0));
    }

    #[test]
    fn the_repetition_penalty_pushes_a_repeated_token_down() {
        // The penalty has to beat a 2.0 logit gap, so greedy moves off token 1.
        let mut sampler = Sampler::greedy().repetition_penalty(4.0);
        assert_eq!(sampler.pick(&LOGITS, &[1]), 2);

        // A negative logit is pushed further down, never up past its peers.
        let mut sampler = Sampler::greedy().repetition_penalty(4.0);
        assert_eq!(sampler.pick(&LOGITS, &[3]), 1);
    }

    #[test]
    fn a_diverged_row_returns_a_token_rather_than_panicking() {
        let mut sampler = Sampler::temperature(1.0, Some(3)).top_k(2);
        let id = sampler.pick(&[f32::NAN, 1.0, f32::NAN], &[]);
        assert!(id < 3);
    }

    #[test]
    fn an_out_of_range_history_id_is_ignored() {
        let mut sampler = Sampler::greedy().repetition_penalty(4.0);
        assert_eq!(sampler.pick(&LOGITS, &[900]), 1);
    }
}
