# TASK SPECIFICATION: Interoceptive Trading Agent & Homeostatic RL Engine in RustingBrain

## Objective
Implement an end-to-end Interoceptive Deep Reinforcement Learning system for algorithmic trading and automated de-risking in pure Rust using `RustingBrain`. 

The system rejects the standard "concat all inputs" approach and instead implements a **Dual-Circuit Architecture**:
1. An **Exteroceptive Market Trunk** (processes raw market features).
2. An independent **Interoceptive Core** (tracks account physiology and dynamically modulates trunk representations, action logits, and exploration temperature).

Do NOT use PyTorch, Python, or external DL frameworks. Rely strictly on `RustingBrain` tensor/matrix abstractions and standard Rust primitives.

---

## 1. System Architecture & Module Boundaries

Organize the implementation across five modular components:

```
src/
├── interoception/
│   ├── state.rs          # 4D AccountBody struct, setpoints, and decay dynamics
│   └── modulation.rs     # Gain modulation, logit bias, and temperature scaling
├── network/
│   ├── trunk.rs          # Exteroceptive market encoder (MLP or 1D-Conv)
│   └── agent.rs          # Combined InteroceptiveAgent (Trunk + Core + Heads)
├── environment/
│   ├── market_sim.rs     # Tick/Candle replay engine with simulated fills & margin
│   └── shock.rs          # Black Swan & Liquidity shock injection augmentations
├── rl/
│   ├── buffer.rs         # Circular ReplayBuffer storing (s, b, a, r, s', b', done)
│   └── loss.rs           # Homeostatic Drive-Reduction loss & Bellman target update
└── training/
    └── trainer.rs        # High-throughput training loop (Rayon-parallelized envs)
```

---

## 2. Mathematical & Algorithmic Specifications

### 2.1. The Interoceptive State Vector ($b \in [0.0, 1.0]^4$)
Maintain an internal physiological state struct:
```rust
pub struct AccountBody {
    pub margin_stress: f32,      // Used Margin / Total Maintenance Margin threshold
    pub drawdown_velocity: f32,  // EMA of negative equity slope over last N steps
    pub slippage_pain: f32,      // Normalized gap between requested price and fill price
    pub exposure_fatigue: f32,   // Exponential accumulator of time spent in open leverage
}
```

* **Setpoints ($s^*$):** Target comfort baseline: `[0.10, 0.00, 0.05, 0.00]`.
* **State Evolution ($db/dt$):**
  At each environment tick, update $b_t$ with leaky-integrator dynamics toward baseline plus external perturbation:
  $$b_{t+1} = \text{clamp}\Big(b_t - \alpha \odot (b_t - s^*) + \Delta b_{\text{market\_event}}, \; 0.0, \; 1.0\Big)$$

### 2.2. Neuromodulatory Coupling
Do NOT append $b$ directly to market features. Use multiplicative gating and action-space steering:

1. **Gain Modulation on Hidden Activations ($h \in \mathbb{R}^{d}$):**
   $$h' = h \odot \Big(1.0 + \tanh\big(W_{\text{gain}} \cdot b + c_{\text{gain}}\big)\Big)$$
   *(Ensures baseline $h$ passes through cleanly when stress is near zero).*

2. **Action-Space Logit Biasing:**
   $$\text{logits}(a) = W_{\text{head}} h' + W_{\text{bias}} \cdot b$$
   *Initialize $W_{\text{bias}}$ such that high `margin_stress` strongly shifts logits toward `Action::DeRisk` / `Action::Flatten`.*

3. **Arousal-Driven Temperature Scaling:**
   $$\tau = \tau_0 \cdot \exp\big(-\beta \cdot b_{\text{drawdown\_velocity}}\big)$$
   *Under panic conditions, drop $\tau \to 0$ to force deterministic capital preservation.*

### 2.3. Homeostatic Drive-Reduction Reward
Replace raw PnL optimization with distance reduction in physiological drive space:

$$\mathcal{D}(b) = \sqrt{\sum_{i=1}^4 w_i \cdot (b_i - s_i^*)^2}$$
$$r_t = \Big(\mathcal{D}(b_t) - \mathcal{D}(b_{t+1})\Big) + \lambda \cdot \text{PnL}_t - \text{TerminalPenalty} \cdot \mathbb{I}(\text{liquidated})$$

---

## 3. Implementation Requirements

### A. Environment (`market_sim.rs`)
- **Input:** Replay historical price slices from flat binary buffers or CSV arrays: `[open, high, low, close, volume]`.
- **Order Execution:** Simulate realistic execution frictions:
  - Fixed commission fee (e.g., 0.04% taker).
  - Dynamic slippage model: $\text{slippage} = \kappa \cdot \frac{\text{order\_size}}{\text{tick\_volume}} \cdot \text{price}$.
- **Margin Account Math:**
  - Track `cash`, `position_size`, `entry_price`, `unrealized_pnl`, and `margin_ratio`.
  - Trigger **Biological Death (`liquidated = true`)** if `margin_ratio >= 1.0`.

### B. Stress Injector (`shock.rs`)
- Inject non-stationary market conditions into 15% of training episodes:
  - **Flash Crash:** Accelerate price drops by $3\times$ across a 20-step window.
  - **Liquidity Void:** Multiply slippage coefficient $\kappa$ by $10\times$ for 50 steps to simulate order book evaporation.

### C. Neural Architecture (`network/`)
- **Market Trunk:** 2-layer MLP (or 1D Temporal Convolution) mapping normalized lookback window $(N \times F)$ to hidden state $h \in \mathbb{R}^{128}$.
- **Interoceptive Core:**
  - Holds learned weight tensors: $W_{\text{gain}} \in \mathbb{R}^{128 \times 4}$, $W_{\text{bias}} \in \mathbb{R}^{A \times 4}$.
  - Computes modulated $h'$ and biased action logits.
- **Action Space ($A = 5$):**
  - `0: Hold / Do Nothing`
  - `1: Open / Add Long (Small)`
  - `2: Open / Add Short (Small)`
  - `3: De-risk (Close 50% Position)`
  - `4: Emergency Flatten (Close 100% to Cash)`

### D. RL Engine (`rl/`)
- Implement a memory-efficient circular `ReplayBuffer` with flat pre-allocated continuous memory.
- Provide a standard Deep Q-Network (DQN) or Policy-Gradient update step updating network weights via `RustingBrain` autograd/optimizer.

---

## 4. Required Interfaces & Rust Idioms

Ensure the public API aligns with the following structure:

```rust
pub enum Action {
    Hold = 0,
    BuySmall = 1,
    SellSmall = 2,
    DeRiskHalf = 3,
    EmergencyFlatten = 4,
}

pub struct StepResult {
    pub next_obs: Tensor,
    pub next_body: AccountBody,
    pub reward: f32,
    pub done: bool,
}

impl MarketEnvironment {
    pub fn new(price_data: Arc<Vec<MarketTick>>, config: SimConfig) -> Self;
    pub fn reset(&mut self) -> (Tensor, AccountBody);
    pub fn step(&mut self, action: Action) -> StepResult;
}

impl InteroceptiveAgent {
    pub fn forward(&self, obs: &Tensor, body: &AccountBody) -> (Tensor /* logits/Q */, Tensor /* h_mod */);
    pub fn select_action(&self, obs: &Tensor, body: &AccountBody, epsilon: f32) -> Action;
    pub fn train_step(&mut self, batch: &BatchData) -> f32 /* loss */;
}
```

---

## 5. Verification & Acceptance Criteria

1. **Compilation:** Clean compile with zero warnings using `cargo check` and `cargo test`.
2. **Deterministic Stress Reflex (Unit Test):**
   - Given an arbitrary market observation, run inference with `body.margin_stress = 0.05`. Record logits.
   - Run inference with identical observation but `body.margin_stress = 0.95`.
   - **Assertion:** Logit for `Action::EmergencyFlatten` must be significantly greater than in the calm state:
     $$\text{logit}_{\text{stress}}(\text{EmergencyFlatten}) - \text{logit}_{\text{calm}}(\text{EmergencyFlatten}) > \Delta_{\text{threshold}}$$
3. **Execution Speed:** Ensure step simulations run without heap allocation per step inside the inner replay loop.

