//! Out-of-sample evaluation: run the greedy policy once over a span of real
//! candles, with fees and slippage but no shocks, and score the equity curve
//! against holding the asset.

use super::agent::InteroceptiveAgent;
use super::market_sim::{MarketEnvironment, MarketTick, SimConfig};
use std::ops::Range;
use std::sync::Arc;

/// Summary of one equity curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EquityStats {
    /// `last / first - 1`.
    pub total_return: f32,
    /// Mean over standard deviation of the per-step returns. Not annualised:
    /// multiply by `sqrt(steps per year)` for the candle size in use.
    pub sharpe: f32,
    /// Largest fall from a running peak, as a fraction of that peak.
    pub max_drawdown: f32,
}

impl EquityStats {
    /// Stats of `curve`, which needs at least two points.
    pub fn from_curve(curve: &[f32]) -> Self {
        assert!(curve.len() >= 2, "an equity curve needs two points");
        let returns: Vec<f64> = curve
            .windows(2)
            .map(|w| (w[1] / w[0].max(f32::EPSILON) - 1.0) as f64)
            .collect();
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let sharpe = if var > 0.0 { mean / var.sqrt() } else { 0.0 };

        let mut peak = curve[0];
        let mut max_drawdown = 0.0f32;
        for &e in curve {
            peak = peak.max(e);
            max_drawdown = max_drawdown.max((peak - e) / peak.max(f32::EPSILON));
        }
        Self {
            total_return: curve[curve.len() - 1] / curve[0] - 1.0,
            sharpe: sharpe as f32,
            max_drawdown,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BacktestReport {
    pub agent: EquityStats,
    /// Holding one unlevered unit of cash in the asset over the same candles,
    /// with no fees.
    pub buy_and_hold: EquityStats,
    /// The agent's equity after every step, starting with the initial cash.
    pub equity: Vec<f32>,
    /// The run ends at a liquidation, so this is the cause of an early end.
    pub liquidated: bool,
}

/// Runs `agent` greedily over all of `ticks`. `config` supplies the costs and
/// the observation shape; its `episode_len`, `shock_probability` and `seed`
/// are overridden so the single episode covers every candle unshocked.
pub fn backtest(
    agent: &InteroceptiveAgent,
    ticks: Arc<Vec<MarketTick>>,
    config: &SimConfig,
) -> BacktestReport {
    let history = config.history();
    assert!(
        ticks.len() > history + 1,
        "need more than {} candles, got {}",
        history + 1,
        ticks.len()
    );
    let config = SimConfig {
        episode_len: ticks.len() - history - 1,
        shock_probability: 0.0,
        ..config.clone()
    };
    // The environment acts from candle `history - 1` and needs one candle
    // past its last step, so the final candle is never acted on.
    let closes: Vec<f32> = ticks[history - 1..ticks.len() - 1]
        .iter()
        .map(|t| t.close)
        .collect();
    let mut env = MarketEnvironment::new(ticks, config);
    let (mut obs, mut body) = env.reset();
    let mut equity = vec![env.equity()];
    loop {
        let action = agent.select_action(&obs, &body, 0.0);
        let step = env.step(action);
        equity.push(env.equity());
        if step.done {
            break;
        }
        (obs, body) = (step.next_obs, step.next_body);
    }
    // A liquidation is the only way to finish with candles left.
    let liquidated = equity.len() < closes.len();
    let hold: Vec<f32> = closes.iter().map(|c| c / closes[0]).collect();
    BacktestReport {
        agent: EquityStats::from_curve(&equity),
        buy_and_hold: EquityStats::from_curve(&hold),
        equity,
        liquidated,
    }
}

/// Walk-forward windows over `n` candles: train on `train` candles, test on
/// the `test` that follow, then slide both forward by `test`. Windows that
/// would run past `n` are dropped.
pub fn walk_forward(n: usize, train: usize, test: usize) -> Vec<(Range<usize>, Range<usize>)> {
    assert!(train > 0 && test > 0);
    (0..)
        .map(|k| k * test)
        .take_while(|start| start + train + test <= n)
        .map(|start| (start..start + train, start + train..start + train + test))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interoceptive::{AgentConfig, synthetic_ticks};

    #[test]
    fn stats_of_a_known_curve() {
        let s = EquityStats::from_curve(&[100.0, 120.0, 90.0, 99.0]);
        assert!((s.total_return - -0.01).abs() < 1e-6);
        assert!((s.max_drawdown - 0.25).abs() < 1e-6);
        // Returns 0.2, -0.25, 0.1: mean 0.01667, population std 0.19293.
        assert!((s.sharpe - 0.08639).abs() < 1e-4, "{}", s.sharpe);
        assert_eq!(EquityStats::from_curve(&[5.0, 5.0]).sharpe, 0.0);
    }

    #[test]
    fn windows_slide_by_the_test_span_and_stay_inside() {
        assert_eq!(walk_forward(10, 4, 3), vec![(0..4, 4..7), (3..7, 7..10)]);
        assert!(walk_forward(5, 4, 3).is_empty());
    }

    #[test]
    fn a_backtest_covers_every_candle_and_scores_buy_and_hold() {
        let sim = SimConfig {
            lookback: 8,
            trend_horizons: vec![4, 16],
            ..SimConfig::default()
        };
        let ticks = Arc::new(synthetic_ticks(200, 3));
        let agent = InteroceptiveAgent::new(AgentConfig {
            hidden: 16,
            ..AgentConfig::new(sim.obs_dim())
        });
        let report = backtest(&agent, ticks.clone(), &sim);

        let first = ticks[sim.history() - 1].close;
        let last = ticks[ticks.len() - 2].close;
        assert!((report.buy_and_hold.total_return - (last / first - 1.0)).abs() < 1e-5);
        assert_eq!(report.equity[0], sim.initial_cash);
        assert!(report.agent.total_return.is_finite());
        if !report.liquidated {
            assert_eq!(report.equity.len(), ticks.len() - sim.history());
        }
    }
}
