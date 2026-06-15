# Deep Hedging with Reinforcement Learning

This project implements deep reinforcement learning agents for hedging European call options under transaction costs and financing. It includes:

- **Chiarella‑Heston market simulator** (agent‑based model with momentum, fundamental, and volatility traders)
- **Simple 3‑state agent** (log price, normalized holdings, normalized time‑to‑maturity) – aligns with the paper *Deeper Hedging* (Gao et al., ICAIF 2023)
- **TD3 training** with double Q‑critics (mean + squared cost) and 2:1 critic‑to‑actor update ratio, enabling a risk‑sensitive objective (mean cost + λ·std)
- **Walk‑forward validation** on real ETF data (IWM, SPY) – scales each 21‑day window to start at $100
- **Init delta layer** – warm‑start idea adapted from the [deep‑hedging engine](https://github.com/alexander-dybdahl/deep-hedging/tree/main) by Alexander Dybdahl
- **Replay buffer & batch learning** – stores past transitions (state, action, reward, next state); each epoch adds 64 parallel trajectories (1,344 transitions) to a buffer of size 500,000, then samples mini‑batches of size 128 for stable, off‑policy updates

---

## 📁 Project Structure

```text
option_hedge/
├── train_simple.py
├── walk_forward_simple.py
├── config_simple.py
├── agent_simple.py
├── chiarella_heston_simple_env.py
├── replay_buffer.py
├── checkpoints/
│   └── simple/
│       ├── final_actor.pkl
│       └── best_actor.pkl
├── hedge_window_simple_*.png
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

1. **Clone the repository** (or copy the files to your local machine).

2. **Create a conda environment** (recommended):
```bash
conda create -n hedge-rl python=3.9
conda activate hedge-rl
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 🧠 Training the Simple Agent

The agent learns to hedge a short ATM call with 21 daily rebalancing steps, transaction costs (`kappa = 0.02`), and net financing cost.

Run:
```bash
python train_simple.py
```

Training progress is printed every 10 epochs:
```text
Epoch 0: cost=12.09, loss=14.61, buffer=1344
Epoch 10: cost=8.45, loss=10.22, buffer=14784
...
```

- **cost** = average total hedging cost (including final payoff, transaction costs, financing) excluding the upfront premium. Lower is better.
- **loss** = mean cost + λ * std (risk‑adjusted). Lower is better.

After training, the final model is saved in `checkpoints/simple/final_actor.pkl`.

---

## 📊 Walk‑Forward Validation on Real Data

Test the trained agent on historical IWM (or SPY) prices. The script:
- Downloads daily data from Yahoo Finance.
- Creates non‑overlapping 21‑day windows.
- Scales each window to start at $100 (preserves returns).
- Computes net profit = Black‑Scholes premium − total hedging cost.
- Compares against BSM delta hedge (with same transaction costs).

Run:
```bash
python walk_forward_simple.py --ticker IWM
```

Example output:
```text
Window 118: GBM Profit = $0.04
            CH  Profit = $1.08
            BSM Profit = $-1.36
Window 119: GBM Profit = $0.08
            CH  Profit = $0.11
            BSM Profit = $-1.67
...
Mean Net Profit ($): CH Agent = 0.45, BSM Delta = -0.50
Win Rate (%): CH Agent = 61.34%, BSM Delta = 52.10%
Sharpe Ratio: CH Agent = 3.262, BSM Delta = -1.670
```

---

## ⚙️ Configuration (`config_ch.py`)

Key parameters you may want to adjust:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `S0` | `100.0` | Initial stock price (scaling target) |
| `K` | `100.0` | Option strike price |
| `r` | `0.05` | Risk‑free rate (5% per year) |
| `sigma` | `0.20` | Constant volatility for Black‑Scholes |
| `T` | `1/12` | Option life (21 trading days) |
| `N_steps` | `21` | Number of rebalancing steps |
| `kappa` | `0.02` | Transaction cost rate (2% per trade) |
| `is_short_call` | `True` | We hedge a short call position |
| `lambda_risk` | `1.0` | Risk aversion (mean + λ·std) |
| `buffer_size` | `500000` | Replay buffer capacity |
| `num_epochs` | `500` | Training epochs (each = 64 parallel episodes) |
| `action_lbnd` / `ubnd` | `-2.0` / `2.0` | Hedge position limits (shares) |

### Chiarella‑Heston parameters (for training data generation)
These control the realism of the simulated market:
- `kappa_CH` – fundamental trader intensity (`0.5`)
- `beta` – momentum trader intensity (`0.5`)
- `alpha` – momentum decay (`1/6 ≈ 0.1667`)
- `phi`, `theta`, `sigma_vol`, `rho` – Heston volatility parameters

---

## 📈 Results (Walk-Forward Validation at .1% transaction cost)

### Profitability Metrics
| Metric | GBM Agent | CH Agent | BSM Delta |
| :--- | :--- | :--- | :--- |
| Mean Net Profit ($) | 0.33 | **0.45** | -0.50 |
| Std Net Profit ($) | **2.16** | 2.21 | 4.80 |
| Win Rate (%) | **61.34%** | **61.34%** | 52.10% |
| Total Profit ($) | 39.70 | **54.07** | -60.02 |

### Risk-Adjusted Metrics
| Metric | GBM Agent | CH Agent | BSM Delta |
| :--- | :--- | :--- | :--- |
| Sharpe Ratio | 2.449 | **3.262** | -1.670 |
| Sortino Ratio | 3.145 | **4.482** | -2.134 |
| Calmar Ratio | 125.253 | **152.803** | -19.823 |
| Max Drawdown ($) | **-17.02** | -22.55 | -89.62 |
| Max Drawdown (%) | **-31.69%** | -35.39% | -302.78% |
| VaR 95% ($) | -4.08 | **-3.60** | -9.95 |
| CVaR 95% ($) | -5.23 | **-4.88** | -12.31 |

### Example hedging on windows
![Hedge Window 1](hedge_window_1.png)
![Hedge Window 2](hedge_window_2.png)
![Hedge Window 3](hedge_window_3.png)

### Summary: Best Performer by Metric
* **Mean Net Profit ($):** CH Agent ($0.45)
* **Std Net Profit ($):** GBM Agent ($2.16) — *Lower is better*
* **Win Rate (%):** GBM/CH Agent (61.34%)
* **Total Profit ($):** CH Agent ($54.07)
* **Sharpe Ratio:** CH Agent (3.262)
* **Sortino Ratio:** CH Agent (4.482)
* **Calmar Ratio:** CH Agent (152.803)
* **Max Drawdown ($):** GBM Agent (-$17.02) — *Least negative is better*
* **VaR 95% ($):** CH Agent (-$3.60) — *Lowest risk is better*
* **CVaR 95% ($):** CH Agent (-$4.88) — *Lowest risk is better*

---

## 📚 References

1. Gao, K., Weston, S., Vytelingum, P., Stillman, N. R., Luk, W., & Guo, C. (2023). *Deeper Hedging: A New Agent‑based Model for Effective Deep Hedging*. ICAIF.
2. Bühler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). *Deep Hedging*. Quantitative Finance.
3. Fujimoto, S., van Hoof, H., & Meger, D. (2018). *Addressing Function Approximation Error in Actor‑Critic Methods (TD3)*. ICML.

---
