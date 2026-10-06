//! Candle replay with fees, volume-dependent slippage, and a margin account
//! that can be liquidated.

use super::Tensor;
use super::loss::homeostatic_reward;
use super::shock::Shock;
use super::state::AccountBody;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

/// Features per candle in the observation window.
pub const TICK_FEATURES: usize = 5;
/// Account features appended after the window: signed leverage and
/// unrealized return. Position, not physiology: the body stays out of the
/// observation, but a Q-network cannot price `BuySmall` without knowing
/// whether it is already long.
pub const POSITION_FEATURES: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct MarketTick {
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
    pub volume: f32,
}

impl MarketTick {
    fn scaled(&self, factor: f32) -> Self {
        Self {
            open: self.open * factor,
            high: self.high * factor,
            low: self.low * factor,
            close: self.close * factor,
            volume: self.volume,
        }
    }
}

/// Candles from a flat `[open, high, low, close, volume, open, ...]` buffer.
pub fn ticks_from_flat(flat: &[f32]) -> Vec<MarketTick> {
    assert_eq!(
        flat.len() % TICK_FEATURES,
        0,
        "flat buffer is not whole candles"
    );
    flat.chunks_exact(TICK_FEATURES)
        .map(|c| MarketTick {
            open: c[0],
            high: c[1],
            low: c[2],
            close: c[3],
            volume: c[4],
        })
        .collect()
}

/// Candles from `open,high,low,close,volume` lines. A first line that does
/// not parse is taken as a header.
pub fn ticks_from_csv(text: &str) -> Result<Vec<MarketTick>, String> {
    let mut flat = Vec::new();
    for (number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Result<Vec<f32>, _> = line.split(',').map(|f| f.trim().parse()).collect();
        match fields {
            Ok(fields) if fields.len() == TICK_FEATURES => flat.extend(fields),
            Err(_) if number == 0 => continue,
            _ => {
                return Err(format!(
                    "line {}: expected {TICK_FEATURES} numbers",
                    number + 1
                ));
            }
        }
    }
    Ok(ticks_from_flat(&flat))
}

/// A geometric random walk with volatility clustering, for tests and demos.
pub fn synthetic_ticks(n: usize, seed: u64) -> Vec<MarketTick> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut normal = || {
        // Box-Muller; `rand_distr` is not a dependency.
        let u: f32 = rng.gen_range(f32::EPSILON..1.0);
        let v: f32 = rng.gen_range(0.0..1.0);
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    };
    let (mut close, mut vol) = (100.0f32, 0.005f32);
    (0..n)
        .map(|_| {
            vol = (0.9 * vol + 0.1 * 0.005 * (1.0 + normal().abs())).clamp(0.001, 0.05);
            let open = close;
            close = open * (1.0 + vol * normal()).max(0.5);
            let wick = open.max(close) * vol * normal().abs() * 0.5;
            MarketTick {
                open,
                high: open.max(close) + wick,
                low: (open.min(close) - wick).max(0.01),
                close,
                volume: 5000.0 * (0.3 * normal()).exp(),
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Hold = 0,
    BuySmall = 1,
    SellSmall = 2,
    DeRiskHalf = 3,
    EmergencyFlatten = 4,
}

impl Action {
    pub const COUNT: usize = 5;

    pub fn from_index(index: usize) -> Self {
        match index {
            0 => Action::Hold,
            1 => Action::BuySmall,
            2 => Action::SellSmall,
            3 => Action::DeRiskHalf,
            4 => Action::EmergencyFlatten,
            _ => panic!("action index {index} out of range"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SimConfig {
    /// Candles in the observation window.
    pub lookback: usize,
    /// Candles over which the observation also reports the scaled log
    /// return, `100 * ln(close_t / close_t-h) / sqrt(h)`: trend context far
    /// past the window without widening the trunk's input by `h` candles.
    pub trend_horizons: Vec<usize>,
    pub episode_len: usize,
    pub initial_cash: f32,
    /// Notional of one small order, as a fraction of equity.
    pub order_fraction: f32,
    /// Position notional is capped at this multiple of equity.
    pub max_leverage: f32,
    /// Taker fee as a fraction of notional.
    pub commission: f32,
    /// Slippage coefficient: `kappa * size / volume * price` per unit.
    pub kappa: f32,
    /// Maintenance margin as a fraction of position notional.
    pub maintenance_margin_rate: f32,
    /// Leak rate per body signal.
    pub alpha: [f32; 4],
    /// Body increase per unit of equity lost, as a fraction of initial cash.
    pub drawdown_gain: f32,
    /// Body increase per unit of slippage, as a fraction of price.
    pub slippage_gain: f32,
    /// Body increase per step per unit of leverage held.
    pub exposure_gain: f32,
    pub drive_weights: [f32; 4],
    /// Weight of PnL (as a fraction of initial cash) in the reward.
    pub pnl_lambda: f32,
    pub terminal_penalty: f32,
    /// Reward cost per unit of equity traded. Fees already reach the reward
    /// through PnL, but at a scale the drive terms drown out; this makes
    /// churning visibly expensive.
    pub trade_penalty: f32,
    /// Fraction of episodes that get a [`Shock`].
    pub shock_probability: f32,
    pub seed: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            lookback: 32,
            // One hour to about ten days of 15-minute candles.
            trend_horizons: vec![4, 16, 64, 256, 1024],
            episode_len: 256,
            initial_cash: 10_000.0,
            order_fraction: 0.5,
            max_leverage: 10.0,
            commission: 0.0004,
            kappa: 0.1,
            maintenance_margin_rate: 0.05,
            alpha: [0.2, 0.1, 0.3, 0.02],
            drawdown_gain: 20.0,
            slippage_gain: 100.0,
            exposure_gain: 0.01,
            drive_weights: [1.0; 4],
            pnl_lambda: 10.0,
            terminal_penalty: 1.0,
            trade_penalty: 0.05,
            shock_probability: 0.15,
            seed: 0,
        }
    }
}

impl SimConfig {
    pub fn obs_dim(&self) -> usize {
        self.lookback * TICK_FEATURES + POSITION_FEATURES + self.trend_horizons.len()
    }

    /// Candles an episode needs before its first step: the window, or the
    /// longest trend horizon plus one.
    pub fn history(&self) -> usize {
        let longest = self.trend_horizons.iter().max().map_or(0, |h| h + 1);
        self.lookback.max(longest)
    }
}

/// [`MarketEnvironment::step_into`]'s result: everything but the observation,
/// which went into the caller's buffer.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepInfo {
    pub body: AccountBody,
    pub reward: f32,
    pub done: bool,
    pub liquidated: bool,
}

pub struct StepResult {
    /// `[1, obs_dim]`.
    pub next_obs: Tensor,
    pub next_body: AccountBody,
    pub reward: f32,
    pub done: bool,
}

pub struct MarketEnvironment {
    data: Arc<Vec<MarketTick>>,
    config: SimConfig,
    rng: StdRng,
    /// This episode's candles, shock applied. Capacity is fixed at `new`, so a
    /// reset refills it without allocating.
    ticks: Vec<MarketTick>,
    shock: Shock,
    /// Index into `ticks` of the candle the agent is acting on.
    t: usize,
    step: usize,
    cash: f32,
    /// Signed units.
    position: f32,
    entry_price: f32,
    body: AccountBody,
}

impl MarketEnvironment {
    pub fn new(price_data: Arc<Vec<MarketTick>>, config: SimConfig) -> Self {
        let span = config.history() + config.episode_len + 1;
        assert!(config.lookback > 0 && config.episode_len > 0);
        assert!(
            price_data.len() >= span,
            "need {span} candles for one episode, got {}",
            price_data.len()
        );
        Self {
            data: price_data,
            rng: StdRng::seed_from_u64(config.seed),
            ticks: Vec::with_capacity(span),
            shock: Shock::None,
            t: 0,
            step: 0,
            cash: config.initial_cash,
            position: 0.0,
            entry_price: 0.0,
            body: AccountBody::default(),
            config,
        }
    }

    pub fn config(&self) -> &SimConfig {
        &self.config
    }

    pub fn shock(&self) -> Shock {
        self.shock
    }

    pub fn position(&self) -> f32 {
        self.position
    }

    pub fn price(&self) -> f32 {
        self.ticks[self.t].close
    }

    pub fn equity(&self) -> f32 {
        self.cash + self.position * (self.price() - self.entry_price)
    }

    /// Maintenance margin over equity; one or more is liquidation.
    pub fn margin_ratio(&self) -> f32 {
        let equity = self.equity();
        if equity <= 0.0 {
            return f32::INFINITY;
        }
        self.position.abs() * self.price() * self.config.maintenance_margin_rate / equity
    }

    pub fn reset(&mut self) -> (Tensor, AccountBody) {
        let mut obs = Tensor::new(1, self.config.obs_dim());
        let body = self.reset_into(&mut obs.data);
        (obs, body)
    }

    /// [`Self::reset`] writing the observation into `obs`.
    pub fn reset_into(&mut self, obs: &mut [f32]) -> AccountBody {
        let span = self.ticks.capacity();
        let start = self.rng.gen_range(0..=self.data.len() - span);
        self.shock = Shock::draw(
            &mut self.rng,
            self.config.shock_probability,
            self.config.episode_len,
        );
        self.fill_ticks(start);
        self.t = self.config.history() - 1;
        self.step = 0;
        self.cash = self.config.initial_cash;
        self.position = 0.0;
        self.entry_price = 0.0;
        self.body = AccountBody::default();
        self.write_obs(obs);
        self.body
    }

    /// Copies the episode's candles, bending prices through the shock.
    fn fill_ticks(&mut self, start: usize) {
        let source = &self.data[start..start + self.ticks.capacity()];
        let lookback = self.config.history();
        self.ticks.clear();
        self.ticks.push(source[0]);
        let mut scale = 1.0f32;
        for i in 1..source.len() {
            if i >= lookback {
                let ret = source[i].close / source[i - 1].close - 1.0;
                scale *= (1.0 + self.shock.shocked_return(i - lookback, ret)) / (1.0 + ret);
            }
            self.ticks.push(source[i].scaled(scale));
        }
    }

    pub fn step(&mut self, action: Action) -> StepResult {
        let mut next_obs = Tensor::new(1, self.config.obs_dim());
        let info = self.step_into(action, &mut next_obs.data);
        StepResult {
            next_obs,
            next_body: info.body,
            reward: info.reward,
            done: info.done,
        }
    }

    /// [`Self::step`] writing the observation into `obs`. Allocates nothing.
    pub fn step_into(&mut self, action: Action, obs: &mut [f32]) -> StepInfo {
        let cfg = &self.config;
        let price = self.price();
        let equity_before = self.equity();

        let cap = cfg.max_leverage * equity_before.max(0.0) / price;
        let small = cfg.order_fraction * equity_before.max(0.0) / price;
        let quantity = match action {
            Action::Hold => 0.0,
            Action::BuySmall => ((self.position + small).min(cap) - self.position).max(0.0),
            Action::SellSmall => ((self.position - small).max(-cap) - self.position).min(0.0),
            Action::DeRiskHalf => -self.position / 2.0,
            Action::EmergencyFlatten => -self.position,
        };
        let kappa = cfg.kappa * self.shock.kappa_multiplier(self.step);
        let volume = self.ticks[self.t].volume;
        let slippage = self.execute(quantity, price, volume, kappa);

        self.t += 1;
        self.step += 1;
        let cfg = &self.config;
        let mut equity = self.equity();
        let margin_ratio = self.margin_ratio();
        let liquidated = margin_ratio >= 1.0;
        if liquidated {
            equity = equity.max(0.0);
            self.cash = equity;
            self.position = 0.0;
            self.entry_price = 0.0;
        }

        let prev = self.body;
        // Margin stress rises to the measured ratio at once and relaxes at its
        // leak rate: fast attack, slow release.
        let leaked_margin =
            prev.margin_stress - cfg.alpha[0] * (prev.margin_stress - super::SETPOINT[0]);
        let leverage = self.position.abs() * self.price() / equity.max(f32::EPSILON);
        let delta = [
            (margin_ratio.min(1.0) - leaked_margin).max(0.0),
            cfg.drawdown_gain * ((equity_before - equity) / cfg.initial_cash).max(0.0),
            cfg.slippage_gain * slippage,
            cfg.exposure_gain * leverage,
        ];
        self.body = prev.evolve(cfg.alpha, delta);

        let pnl = (equity - equity_before) / cfg.initial_cash;
        let reward = homeostatic_reward(
            &prev,
            &self.body,
            pnl,
            liquidated,
            cfg.drive_weights,
            cfg.pnl_lambda,
            cfg.terminal_penalty,
        ) - cfg.trade_penalty * quantity.abs() * price
            / equity_before.max(f32::EPSILON);
        let done = liquidated || self.step >= cfg.episode_len;
        self.write_obs(obs);
        StepInfo {
            body: self.body,
            reward,
            done,
            liquidated,
        }
    }

    /// Fills `quantity` units (signed) against `price`, charging fee and
    /// slippage. Returns the slippage as a fraction of `price`.
    fn execute(&mut self, quantity: f32, price: f32, volume: f32, kappa: f32) -> f32 {
        if quantity == 0.0 {
            return 0.0;
        }
        let size = quantity.abs();
        let slip = kappa * size / volume.max(1.0) * price;
        let fill = price + quantity.signum() * slip;
        self.cash -= self.config.commission * size * fill;

        if self.position == 0.0 || self.position.signum() == quantity.signum() {
            let held = self.position.abs();
            self.entry_price = (self.entry_price * held + fill * size) / (held + size);
            self.position += quantity;
        } else {
            let closed = size.min(self.position.abs());
            self.cash += closed * (fill - self.entry_price) * self.position.signum();
            let flips = size > self.position.abs();
            self.position += quantity;
            if flips {
                self.entry_price = fill;
            } else if self.position.abs() < 1e-9 {
                self.position = 0.0;
                self.entry_price = 0.0;
            }
        }
        slip / price
    }

    /// Window of candles relative to the current close, then position.
    fn write_obs(&self, obs: &mut [f32]) {
        let lookback = self.config.lookback;
        debug_assert_eq!(obs.len(), self.config.obs_dim());
        let window = &self.ticks[self.t + 1 - lookback..=self.t];
        let close = window[lookback - 1].close;
        let mean_volume = window.iter().map(|c| c.volume).sum::<f32>() / lookback as f32;
        for (slot, c) in obs.chunks_exact_mut(TICK_FEATURES).zip(window) {
            slot[0] = (c.open / close - 1.0) * 100.0;
            slot[1] = (c.high / close - 1.0) * 100.0;
            slot[2] = (c.low / close - 1.0) * 100.0;
            slot[3] = (c.close / close - 1.0) * 100.0;
            slot[4] = c.volume / mean_volume.max(f32::EPSILON) - 1.0;
        }
        let equity = self.equity().max(f32::EPSILON);
        let tail = &mut obs[lookback * TICK_FEATURES..];
        tail[0] = self.position * close / equity / self.config.max_leverage;
        tail[1] = self.position * (close - self.entry_price) / equity * 10.0;
        for (slot, &h) in tail[POSITION_FEATURES..]
            .iter_mut()
            .zip(&self.config.trend_horizons)
        {
            let past = self.ticks[self.t - h].close;
            *slot = 100.0 * (close / past).ln() / (h as f32).sqrt();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_then_falling(config: &SimConfig, fall: f32) -> Arc<Vec<MarketTick>> {
        let n = config.lookback + config.episode_len + 1;
        let mut price = 100.0;
        let ticks = (0..n)
            .map(|i| {
                if i >= config.lookback + 5 {
                    price *= 1.0 - fall;
                }
                MarketTick {
                    open: price,
                    high: price,
                    low: price,
                    close: price,
                    volume: 1e6,
                }
            })
            .collect();
        Arc::new(ticks)
    }

    fn quiet_config() -> SimConfig {
        SimConfig {
            lookback: 4,
            trend_horizons: vec![],
            episode_len: 40,
            shock_probability: 0.0,
            ..SimConfig::default()
        }
    }

    #[test]
    fn a_levered_long_is_liquidated_in_a_crash_and_stress_spikes() {
        let config = quiet_config();
        let mut env = MarketEnvironment::new(flat_then_falling(&config, 0.02), config);
        env.reset();
        let mut last = None;
        for _ in 0..40 {
            let result = env.step(Action::BuySmall);
            last = Some((result.done, result.reward, result.next_body));
            if result.done {
                break;
            }
        }
        let (done, reward, body) = last.unwrap();
        assert!(done);
        assert!(
            reward < -0.5,
            "liquidation should cost the terminal penalty"
        );
        assert_eq!(env.position(), 0.0);
        assert!(body.margin_stress > 0.9);
    }

    #[test]
    fn flatten_closes_everything_and_fees_and_slippage_cost_money() {
        let config = SimConfig {
            kappa: 1000.0,
            ..quiet_config()
        };
        let mut env = MarketEnvironment::new(flat_then_falling(&config, 0.0), config);
        env.reset();
        let result = env.step(Action::BuySmall);
        assert!(env.position() > 0.0);
        assert!(result.next_body.slippage_pain > super::super::SETPOINT[2]);
        env.step(Action::EmergencyFlatten);
        assert_eq!(env.position(), 0.0);
        // Flat prices: only frictions moved equity.
        assert!(env.equity() < env.config().initial_cash);
    }

    #[test]
    fn csv_and_flat_buffers_parse_to_the_same_candles() {
        let csv = "open,high,low,close,volume\n1,2,0.5,1.5,10\n1.5,1.6,1.4,1.5,12\n";
        let flat = [1.0, 2.0, 0.5, 1.5, 10.0, 1.5, 1.6, 1.4, 1.5, 12.0];
        assert_eq!(ticks_from_csv(csv).unwrap(), ticks_from_flat(&flat));
        assert!(ticks_from_csv("1,2,3\n").is_err());
    }

    #[test]
    fn a_flash_crash_triples_the_falls_inside_its_window() {
        let config = quiet_config();
        let mut env = MarketEnvironment::new(flat_then_falling(&config, 0.01), config.clone());
        env.shock = Shock::FlashCrash { start: 10 };
        env.fill_ticks(0);
        let close = |i: usize| env.ticks[config.lookback + i].close;
        let fall = |i: usize| 1.0 - close(i + 1) / close(i);
        assert!((fall(5) - 0.01).abs() < 1e-4);
        assert!((fall(12) - 0.03).abs() < 1e-4);
        assert!((fall(35) - 0.01).abs() < 1e-4);
    }
}
