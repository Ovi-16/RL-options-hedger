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
🚀 Installation
Clone the repository (or copy the files to your local machine).

Create a conda environment (recommended):

Bash
conda create -n hedge-rl python=3.9
conda activate hedge-rl
Install dependencies:

Bash
pip install -r requirements.txt
🧠 Training the Simple Agent
The agent learns to hedge a short ATM call with 21 daily rebalancing steps, transaction costs (kappa = 0.02), and net financing cost.

Run:

Bash
python train_simple.py
Training progress is printed every 10 epochs:

Plaintext
Epoch 0: cost=12.09, loss=14.61, buffer=1344
Epoch 10: cost=8.45, loss=10.22, buffer=14784
...
cost = average total hedging cost (including final payoff, transaction costs, financing) excluding the upfront premium. Lower is better.

loss = mean cost + λ * std (risk‑adjusted). Lower is better.

After training, the final model is saved in checkpoints/simple/final_actor.pkl.

📊 Walk‑Forward Validation on Real Data
Test the trained agent on historical IWM (or SPY) prices. The script:

Downloads daily data from Yahoo Finance.

Creates non‑overlapping 21‑day windows.

Scales each window to start at $100 (preserves returns).

Computes net profit = Black‑Scholes premium − total hedging cost.

Compares against BSM delta hedge (with same transaction costs).

Run:

Bash
python walk_forward_simple.py --ticker IWM
To test all windows (119 for IWM), use:

Bash
python walk_forward_simple.py --ticker IWM --windows 119
Optional arguments:

--checkpoint path.pkl – load a different checkpoint (e.g., best_actor.pkl)

--no_plots – skip generating hedge window plots

Output includes:

Profitability metrics: mean net profit, win rate, total profit.

Risk metrics: Sharpe, Sortino, Calmar, max drawdown, VaR, CVaR.

Plots: hedge positions, cumulative P&L, profit bar chart (first 3 windows).

Hedge distribution histogram.