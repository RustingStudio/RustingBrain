//! An interoceptive trading agent: a Q-network whose market encoder is steered
//! by a small model of the account's own "body" rather than fed it as input.
//!
//! The usual trading agent concatenates account state onto the market features
//! and lets the network sort it out. This one keeps two circuits apart:
//!
//! - an exteroceptive [`MarketTrunk`] that only ever sees prices and volume,
//! - an interoceptive [`InteroceptiveCore`] that tracks four physiological
//!   signals ([`AccountBody`]) and acts on the trunk from outside: it gates the
//!   trunk's hidden units, biases the action values toward de-risking, and cools
//!   the exploration temperature as drawdown accelerates.
//!
//! The reward is drive reduction ([`homeostatic_reward`]): moving the body back
//! toward its setpoints pays, a little PnL pays, and liquidation costs a
//! terminal penalty.
//!
//! [`Trainer`] ties it together: a pool of [`MarketEnvironment`]s stepped in
//! parallel with rayon, a flat [`ReplayBuffer`], and DQN updates on
//! [`InteroceptiveAgent::train_step`].

pub mod agent;
pub mod buffer;
pub mod loss;
pub mod market_sim;
pub mod modulation;
pub mod shock;
pub mod state;
pub mod trainer;
pub mod trunk;

pub use agent::{AgentConfig, InteroceptiveAgent, QNetwork};
pub use buffer::{BatchData, ReplayBuffer};
pub use loss::{bellman_targets, homeostatic_reward, td_huber_loss};
pub use market_sim::{
    Action, MarketEnvironment, MarketTick, SimConfig, StepInfo, StepResult, synthetic_ticks,
    ticks_from_csv, ticks_from_flat,
};
pub use modulation::{InteroceptiveCore, temperature};
pub use shock::Shock;
pub use state::{AccountBody, SETPOINT};
pub use trainer::{TrainReport, Trainer, TrainerConfig};
pub use trunk::MarketTrunk;

/// The spec's name for a batch of rows. Every tensor here is a row-major
/// [`Matrix`](crate::matrix::Matrix): one row per sample.
pub type Tensor = crate::matrix::Matrix;
