# walk_forward_all.py
# Compare Simple Agent (CH‑trained, 3‑state), GBM Agent (GBM‑trained, 4‑state with init delta),
# and BSM delta hedge on real IWM/SPY data.

import os
import pickle
import numpy as np
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
from config_simple import ConfigSimple
from config import Config as ConfigGBM
from env import black_scholes_delta, build_state
from agent_simple import SimpleActor
from agent import ActorWithInitDelta

# ================================================================
# Black-Scholes and helper functions (common)
# ================================================================

def black_scholes_price(S, K, T, r, sigma):
    if T <= 0:
        return max(0, S - K)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2)

def compute_max_drawdown(pnl_series):
    cumulative = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    drawdown_pct = drawdown / (np.abs(running_max) + 1e-8) * 100
    return np.min(drawdown), np.min(drawdown_pct)

def compute_sharpe_ratio(pnl_series, risk_free_rate=0.02):
    if len(pnl_series) == 0 or np.std(pnl_series) == 0:
        return 0
    excess = pnl_series - risk_free_rate / 252
    return np.sqrt(252) * np.mean(excess) / np.std(excess)

def compute_sortino_ratio(pnl_series, risk_free_rate=0.02):
    if len(pnl_series) == 0:
        return 0
    excess = pnl_series - risk_free_rate / 252
    downside = excess[excess < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return 0 if np.mean(excess) <= 0 else 100
    return np.sqrt(252) * np.mean(excess) / np.std(downside)

def compute_calmar_ratio(pnl_series):
    total = np.sum(pnl_series)
    _, max_dd_pct = compute_max_drawdown(pnl_series)
    return 0 if max_dd_pct == 0 else total / abs(max_dd_pct) * 100

def compute_var(pnl_series, conf=0.95):
    return np.percentile(pnl_series, (1-conf)*100)

def compute_cvar(pnl_series, conf=0.95):
    var = compute_var(pnl_series, conf)
    return np.mean(pnl_series[pnl_series <= var])

def download_stock_data(ticker='SPY', start='2015-01-01', end='2024-12-31'):
    print(f"Downloading {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)[['Close']].copy()
    df.columns = ['price']
    print(f"Downloaded {len(df)} days, range ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    return df

def create_rolling_windows(prices, window_size=21, step_size=21):
    windows, dates = [], []
    for i in range(0, len(prices) - window_size, step_size):
        windows.append(prices.iloc[i:i+window_size+1].values.flatten())
        dates.append({'start': prices.index[i], 'end': prices.index[i+window_size]})
    print(f"Created {len(windows)} non‑overlapping 21‑day windows")
    return windows, dates

# ================================================================
# Simple Agent (3‑state, CH‑trained)
# ================================================================

def build_simple_state(price, h_prev, tau, action_ubnd, T):
    return np.array([np.log(price), h_prev / action_ubnd, tau / T], dtype=np.float32)

def simulate_simple(actor, params, price_scaled, cfg, strike):
    n = len(price_scaled) - 1
    dt = 1/252
    h = 0.0
    total_cost = 0.0
    hedges = []
    for step in range(n):
        S = price_scaled[step]
        tau = cfg.T - step * dt
        tau = max(tau, 0.01)
        state = build_simple_state(S, h, tau, cfg.action_ubnd, cfg.T).reshape(1, -1)
        action = actor.apply(params, state)
        action = np.clip(action, cfg.action_lbnd, cfg.action_ubnd).item()
        S_next = price_scaled[step+1]
        V_t = black_scholes_price(S, strike, tau, cfg.r, cfg.sigma)
        V_next = black_scholes_price(S_next, strike, tau-dt, cfg.r, cfg.sigma)
        option_change = -(V_next - V_t)
        hedging_gain = h * (S_next - S)
        trans_cost = cfg.kappa * abs(action - h) * S_next
        financing_cost = cfg.r * (h * S - V_t) * dt
        reward = option_change + hedging_gain - trans_cost - financing_cost
        total_cost += -reward
        hedges.append(action)
        h = action
    payoff = max(price_scaled[-1] - strike, 0)
    total_cost += payoff
    return total_cost, -total_cost, hedges

# ================================================================
# GBM Agent (4‑state, with init delta)
# ================================================================

def simulate_gbm(actor, params, price_scaled, cfg, strike):
    n = len(price_scaled) - 1
    dt = 1/252
    h = 0.0
    total_cost = 0.0
    hedges = []
    for step in range(n):
        S = price_scaled[step]
        tau = cfg.T - step * dt
        tau = max(tau, 0.01)
        state = build_state(np.array([S]), tau, np.array([h]), strike, cfg.r, cfg.sigma, cfg.T)
        state = np.array(state).reshape(1, -1)
        action = actor.apply(params, state)
        action = np.clip(action, cfg.action_lbnd, cfg.action_ubnd).item()
        S_next = price_scaled[step+1]
        V_t = black_scholes_price(S, strike, tau, cfg.r, cfg.sigma)
        V_next = black_scholes_price(S_next, strike, tau-dt, cfg.r, cfg.sigma)
        option_change = -(V_next - V_t)
        hedging_gain = h * (S_next - S)
        trans_cost = cfg.kappa * abs(action - h) * S_next
        financing_cost = cfg.r * (h * S - V_t) * dt   # net financing
        reward = option_change + hedging_gain - trans_cost - financing_cost
        total_cost += -reward
        hedges.append(action)
        h = action
    payoff = max(price_scaled[-1] - strike, 0)
    total_cost += payoff
    return total_cost, -total_cost, hedges

# ================================================================
# BSM Delta Hedge (simplified, no financing)
# ================================================================

def simulate_bsm(price_path, cfg, strike):
    n = len(price_path) - 1
    dt = 1/252
    h = 0.0
    total_cost = 0.0
    hedges = []
    for step in range(n):
        S = price_path[step]
        tau = cfg.T - step * dt
        tau = max(tau, 0.01)
        delta = black_scholes_delta(S, strike, tau, cfg.r, cfg.sigma)
        if getattr(cfg, 'is_short_call', True):
            delta = -delta
        delta = np.clip(delta, cfg.action_lbnd, cfg.action_ubnd)
        S_next = price_path[step+1]
        hedging_error = (S_next - S) * h
        trans_cost = cfg.kappa * abs(delta - h) * S_next
        step_cost = -hedging_error + trans_cost
        total_cost += step_cost
        hedges.append(delta)
        h = delta
    payoff = max(price_path[-1] - strike, 0)
    total_cost += payoff
    return total_cost, -total_cost, hedges

# ================================================================
# Metrics and plotting (simplified, reuses earlier functions)
# ================================================================

def compute_all_metrics(results, agent_names):
    """results: list of dicts with keys 'simple_net', 'gbm_net', 'bsm_net'"""
    strategies = []
    for name in agent_names:
        profits = [r[f'{name.lower()}_net'] for r in results if not np.isnan(r[f'{name.lower()}_net'])]
        if profits:
            strategies.append((name, profits))
    if not strategies:
        return
    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS COMPARISON")
    print("="*80)
    all_metrics = {}
    for name, profits in strategies:
        profits = np.array(profits)
        mean_p = np.mean(profits)
        std_p = np.std(profits)
        win_rate = 100 * np.mean(profits > 0)
        total_p = np.sum(profits)
        sharpe = compute_sharpe_ratio(profits)
        sortino = compute_sortino_ratio(profits)
        calmar = compute_calmar_ratio(profits)
        max_dd_amt, max_dd_pct = compute_max_drawdown(profits)
        var95 = compute_var(profits, 0.95)
        cvar95 = compute_cvar(profits, 0.95)
        all_metrics[name] = {
            'mean': mean_p, 'std': std_p, 'win_rate': win_rate, 'total': total_p,
            'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar,
            'max_dd_amt': max_dd_amt, 'max_dd_pct': max_dd_pct,
            'var95': var95, 'cvar95': cvar95,
        }
    # Print tables
    print("\nPROFITABILITY METRICS")
    print(f"{'Metric':<20} ", end="")
    for name in all_metrics.keys():
        print(f"{name:>15}", end="")
    print("\n" + "-"*65)
    for label, key in [('Mean Net Profit ($)', 'mean'), ('Std Net Profit ($)', 'std'),
                       ('Win Rate (%)', 'win_rate'), ('Total Profit ($)', 'total')]:
        print(f"{label:<20} ", end="")
        for m in all_metrics.values():
            print(f"{m[key]:>15.2f}", end="")
        print()
    print("\nRISK-ADJUSTED METRICS")
    print(f"{'Metric':<20} ", end="")
    for name in all_metrics.keys():
        print(f"{name:>15}", end="")
    print("\n" + "-"*65)
    for label, key in [('Sharpe Ratio', 'sharpe'), ('Sortino Ratio', 'sortino'),
                       ('Calmar Ratio', 'calmar'), ('Max Drawdown ($)', 'max_dd_amt'),
                       ('Max Drawdown (%)', 'max_dd_pct'), ('VaR 95% ($)', 'var95'),
                       ('CVaR 95% ($)', 'cvar95')]:
        print(f"{label:<20} ", end="")
        for m in all_metrics.values():
            if key == 'max_dd_pct':
                print(f"{m[key]:>14.2f}%", end=" ")
            elif key in ['sharpe','sortino','calmar']:
                print(f"{m[key]:>15.3f}", end="")
            else:
                print(f"{m[key]:>15.2f}", end="")
        print()
    # Best performer summary
    print("\nSUMMARY: BEST PERFORMER BY METRIC")
    for label, key in [('Mean Net Profit ($)', 'mean'), ('Win Rate (%)', 'win_rate'),
                       ('Sharpe Ratio', 'sharpe'), ('Sortino Ratio', 'sortino'),
                       ('Calmar Ratio', 'calmar'), ('Max Drawdown (%)', 'max_dd_pct')]:
        best_name = None
        best_val = -np.inf if 'drawdown' not in key.lower() else np.inf
        for name, m in all_metrics.items():
            val = m[key]
            if 'drawdown' in key.lower():
                if val < best_val:
                    best_val = val
                    best_name = name
            else:
                if val > best_val:
                    best_val = val
                    best_name = name
        if key == 'max_dd_pct':
            print(f"{label:<20}: {best_name} ({best_val:.2f}%)")
        elif 'Ratio' in label:
            print(f"{label:<20}: {best_name} ({best_val:.3f})")
        else:
            print(f"{label:<20}: {best_name} (${best_val:.2f})")

# ================================================================
# Main
# ================================================================

def run_walk_forward_all(ticker='IWM', simple_ckpt='checkpoints/simple_0.25/final_actor.pkl',
                         gbm_ckpt='checkpoints/gbm_td3/epoch_250_actor.pkl', max_windows=None):
    print("="*80)
    print("WALK-FORWARD VALIDATION - ALL AGENTS")
    print(f"Ticker: {ticker} | 21‑day windows scaled to $100")
    print("Comparing: Simple Agent (CH) | GBM Agent (TD3) | BSM Delta")
    print("="*80)

    # Load data
    prices = download_stock_data(ticker)['price']
    windows, window_dates = create_rolling_windows(prices, 21, 21)
    if max_windows:
        windows = windows[:max_windows]
        window_dates = window_dates[:max_windows]
    print(f"\nTesting on {len(windows)} windows\n")

    # Load configs
    cfg_simple = ConfigSimple()
    cfg_gbm = ConfigGBM()
    # Ensure consistent kappa and action bounds (use simple config as reference)
    cfg_gbm.kappa = cfg_simple.kappa
    cfg_gbm.action_lbnd = cfg_simple.action_lbnd
    cfg_gbm.action_ubnd = cfg_simple.action_ubnd
    cfg_gbm.r = cfg_simple.r
    cfg_gbm.sigma = cfg_simple.sigma
    cfg_gbm.T = cfg_simple.T
    cfg_gbm.N_steps = cfg_simple.N_steps

    # Load simple agent
    try:
        with open(simple_ckpt, 'rb') as f:
            simple_params = pickle.load(f)
        simple_actor = SimpleActor()
        print(f"Loaded Simple Agent from {simple_ckpt}")
    except:
        print(f"❌ Simple agent not found at {simple_ckpt}")
        simple_actor = None

    # Load GBM agent
    try:
        with open(gbm_ckpt, 'rb') as f:
            gbm_params = pickle.load(f)
        gbm_actor = ActorWithInitDelta()
        print(f"Loaded GBM Agent from {gbm_ckpt}")
    except:
        print(f"❌ GBM agent not found at {gbm_ckpt}")
        gbm_actor = None

    results = []
    for idx, (window_prices, dates) in enumerate(zip(windows, window_dates)):
        scale = 100.0 / window_prices[0]
        scaled = window_prices * scale
        strike = 100.0
        premium = black_scholes_price(100.0, strike, 21/252, cfg_simple.r, cfg_simple.sigma)

        row = {'window': idx, 'start': dates['start'], 'end': dates['end'], 'premium': premium}

        # Simple agent
        if simple_actor:
            try:
                cost, pnl, hedges = simulate_simple(simple_actor, simple_params, scaled, cfg_simple, strike)
                row['simple_net'] = premium - cost
                row['simple_hedges'] = hedges
            except Exception as e:
                print(f"  Simple error: {e}")
                row['simple_net'] = np.nan

        # GBM agent
        if gbm_actor:
            try:
                cost, pnl, hedges = simulate_gbm(gbm_actor, gbm_params, scaled, cfg_gbm, strike)
                row['gbm_net'] = premium - cost
                row['gbm_hedges'] = hedges
            except Exception as e:
                print(f"  GBM error: {e}")
                row['gbm_net'] = np.nan

        # BSM
        try:
            cost, pnl, hedges = simulate_bsm(scaled, cfg_simple, strike)
            row['bsm_net'] = premium - cost
            row['bsm_hedges'] = hedges
        except Exception as e:
            print(f"  BSM error: {e}")
            row['bsm_net'] = np.nan

        results.append(row)
        print(f"Window {idx+1}: Simple={row.get('simple_net',0):.2f}  GBM={row.get('gbm_net',0):.2f}  BSM={row.get('bsm_net',0):.2f}")

    # Compute and print metrics for all agents present
    agent_names = []
    if simple_actor: agent_names.append('Simple')
    if gbm_actor: agent_names.append('GBM')
    agent_names.append('BSM')
    compute_all_metrics(results, agent_names)
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='IWM')
    parser.add_argument('--simple_ckpt', type=str, default='checkpoints/simple_0.25/final_actor.pkl')
    parser.add_argument('--gbm_ckpt', type=str, default='checkpoints/gbm_td3/epoch_250_actor.pkl')
    parser.add_argument('--windows', type=int, default=None)
    args = parser.parse_args()
    run_walk_forward_all(ticker=args.ticker, simple_ckpt=args.simple_ckpt,
                         gbm_ckpt=args.gbm_ckpt, max_windows=args.windows)