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

To test all windows (119 for IWM), use:
```bash
python walk_forward_simple.py --ticker IWM --windows 119
```

**Optional arguments:**
- `--checkpoint path.pkl` – load a different checkpoint (e.g., `best_actor.pkl`)
- `--no_plots` – skip generating hedge window plots

**Output includes:**
- **Profitability metrics:** mean net profit, win rate, total profit.
- **Risk metrics:** Sharpe, Sortino, Calmar, max drawdown, VaR, CVaR.
- **Plots:** hedge positions, cumulative P&L, profit bar chart (first 3 windows).
- **Hedge distribution histogram.**

Example output:
```text
Window 1: Simple Profit = $2.38 | BSM Profit = $0.68
...
Mean Net Profit ($): Simple Agent = 0.42, BSM Delta = -1.24
Win Rate (%): Simple Agent = 74.0%, BSM Delta = 54.0%
Sharpe Ratio: Simple Agent = 2.59, BSM Delta = -3.76
```

---

## ⚙️ Configuration (`config_simple.py`)

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

## 📈 Results Interpretation

**In training:**
- Costs start around $12 (random policy) and drop to $5‑7 (good policy).
- Even a perfect hedge cannot have zero cost because transaction costs and financing are unavoidable.
- If costs do not decrease, try increasing exploration noise or critic updates.

**In walk‑forward validation:**
- Positive net profit means the agent outperforms the premium received.
- Win rate > 50% means the agent makes money more often than it loses.
- Compare Sharpe ratio – higher is better.
- The simple agent should outperform BSM delta hedge when transaction costs are significant (here `kappa=0.02`).

---

## 🛠️ Troubleshooting

| Problem | Likely cause | Solution |
| :--- | :--- | :--- |
| `FileNotFoundError` when saving checkpoints | Directory missing | Run `mkdir -p checkpoints/simple` |
| `ValueError: All input arrays must have same shape` | State builder bug | Use the corrected `build_simple_state` from the final environment file |
| Yahoo Finance download fails | Network or rate limiting | Wait a few minutes, use VPN, or download CSV manually |
| Training cost does not decrease | Wrong reward sign or learning rate | Verify `c = -r` in `train_simple.py`; try `lr_actor = 5e-4` |
| BSM hedge has `nan` | Missing `is_short_call` attribute | Add `is_short_call = True` to `ConfigSimple` |

---

## 📚 References

1. Gao, K., Weston, S., Vytelingum, P., Stillman, N. R., Luk, W., & Guo, C. (2023). *Deeper Hedging: A New Agent‑based Model for Effective Deep Hedging*. ICAIF.
2. Bühler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). *Deep Hedging*. Quantitative Finance.
3. Fujimoto, S., van Hoof, H., & Meger, D. (2018). *Addressing Function Approximation Error in Actor‑Critic Methods (TD3)*. ICML.

---

## 📄 License

This project is provided for research and educational purposes. No warranty is implied.

---

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements.

**Happy hedging!** 📈