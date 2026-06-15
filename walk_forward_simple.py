# walk_forward_all.py
# Compare Simple Agent (CH‑trained, 3‑state), GBM Agent (GBM‑trained, 4‑state with init delta),
# and BSM delta hedge on real IWM/SPY data.
# Includes plots: hedge positions, cumulative P&L, net profit bars, hedge distributions.

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
    step_pnls = []
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
        step_cost = -reward
        total_cost += step_cost
        step_pnls.append(-step_cost)
        hedges.append(action)
        h = action
    payoff = max(price_scaled[-1] - strike, 0)
    total_cost += payoff
    step_pnls.append(-payoff)
    return total_cost, -total_cost, step_pnls, hedges

# ================================================================
# GBM Agent (4‑state, with init delta)
# ================================================================

def simulate_gbm(actor, params, price_scaled, cfg, strike):
    n = len(price_scaled) - 1
    dt = 1/252
    h = 0.0
    total_cost = 0.0
    step_pnls = []
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
        financing_cost = cfg.r * (h * S - V_t) * dt
        reward = option_change + hedging_gain - trans_cost - financing_cost
        step_cost = -reward
        total_cost += step_cost
        step_pnls.append(-step_cost)
        hedges.append(action)
        h = action
    payoff = max(price_scaled[-1] - strike, 0)
    total_cost += payoff
    step_pnls.append(-payoff)
    return total_cost, -total_cost, step_pnls, hedges

# ================================================================
# BSM Delta Hedge (simplified, no financing)
# ================================================================

def simulate_bsm(price_path, cfg, strike):
    n = len(price_path) - 1
    dt = 1/252
    h = 0.0
    total_cost = 0.0
    step_pnls = []
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
        step_pnls.append(-step_cost)
        hedges.append(delta)
        h = delta
    payoff = max(price_path[-1] - strike, 0)
    total_cost += payoff
    step_pnls.append(-payoff)
    return total_cost, -total_cost, step_pnls, hedges

# ================================================================
# Plotting functions (NEW)
# ================================================================

def plot_window_comparison(scaled_prices, strike, result, window_date, plot_num):
    """Plot price path, hedge positions, cumulative P&L, and net profit bar for one window."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    days = np.arange(len(scaled_prices))
    premium = result['premium']
    
    # Price and strike
    ax1 = axes[0,0]
    ax1.plot(days, scaled_prices, label='Stock price (scaled to $100)', color='black')
    ax1.axhline(y=strike, linestyle='--', color='gray', label=f'Strike = ${strike:.0f}')
    ax1.set_title(f'Window {plot_num}: {window_date["start"].date()} - {window_date["end"].date()}')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.text(0.02, 0.95, f'Option premium = ${premium:.2f}', transform=ax1.transAxes, fontsize=10, verticalalignment='top')
    
    # Hedge positions
    ax2 = axes[0,1]
    for name, key in [('Simple Agent', 'simple_hedges'), ('GBM Agent', 'gbm_hedges'), ('BSM', 'bsm_hedges')]:
        hedges = result.get(key)
        if hedges is not None and len(hedges) > 0:
            ax2.plot(days[:-1], hedges, label=name, linewidth=1.5)
    ax2.set_title('Hedge Positions (shares)')
    ax2.set_ylabel('Holding')
    ax2.legend()
    
    # Cumulative P&L
    ax3 = axes[1,0]
    for name, key in [('Simple Agent', 'simple_step_pnls'), ('GBM Agent', 'gbm_step_pnls'), ('BSM', 'bsm_step_pnls')]:
        pnls = result.get(key)
        if pnls is not None and len(pnls) > 0:
            cum = np.cumsum(pnls)
            ax3.plot(days[:len(cum)], cum, label=name, linewidth=1.5)
    ax3.set_title('Cumulative P&L from Hedging')
    ax3.set_ylabel('P&L ($)')
    ax3.legend()
    
    # Net profit bar chart
    ax4 = axes[1,1]
    net_profits = []
    names = []
    for name in ['Simple', 'GBM', 'BSM']:
        net = result.get(f'{name.lower()}_net')
        if net is not None and not np.isnan(net):
            net_profits.append(net)
            names.append(name)
    ax4.bar(names, net_profits, color=['blue', 'orange', 'green'])
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.set_title('Net Profit (Premium - Hedging Cost)')
    ax4.set_ylabel('Profit ($)')
    
    plt.tight_layout()
    plt.savefig(f'hedge_window_all_{plot_num}.png', dpi=150)
    plt.close()
    print(f"  Saved plot hedge_window_all_{plot_num}.png")

def plot_hedge_distributions(results, simple_exists, gbm_exists):
    """Plot histograms of hedge positions for all agents."""
    simple_hedges = []
    gbm_hedges = []
    bsm_hedges = []
    for r in results:
        if simple_exists and 'simple_hedges' in r:
            simple_hedges.extend(r['simple_hedges'])
        if gbm_exists and 'gbm_hedges' in r:
            gbm_hedges.extend(r['gbm_hedges'])
        if 'bsm_hedges' in r:
            bsm_hedges.extend(r['bsm_hedges'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = []
    if simple_exists:
        titles.append('Simple Agent')
    if gbm_exists:
        titles.append('GBM Agent')
    titles.append('BSM Delta')
    
    data_list = []
    if simple_exists:
        data_list.append(simple_hedges)
    if gbm_exists:
        data_list.append(gbm_hedges)
    data_list.append(bsm_hedges)
    
    for ax, data, title in zip(axes, data_list, titles):
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax.set_title(f'{title}\n(mean={np.mean(data):.2f}, std={np.std(data):.2f})')
            ax.set_xlabel('Hedge Position (shares)')
            ax.set_ylabel('Frequency')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig('hedge_distributions_all.png', dpi=150)
    plt.close()
    print("Saved hedge_distributions_all.png")

# ================================================================
# Metrics and printing (unchanged)
# ================================================================

def compute_all_metrics(results, agent_names):
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

def run_walk_forward_all(ticker='SPY', simple_ckpt='checkpoints/simple_0.25/final_actor.pkl',
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
        simple_exists = True
    except:
        print(f"❌ Simple agent not found at {simple_ckpt}")
        simple_actor = None
        simple_exists = False

    # Load GBM agent
    try:
        with open(gbm_ckpt, 'rb') as f:
            gbm_params = pickle.load(f)
        gbm_actor = ActorWithInitDelta()
        print(f"Loaded GBM Agent from {gbm_ckpt}")
        gbm_exists = True
    except:
        print(f"❌ GBM agent not found at {gbm_ckpt}")
        gbm_actor = None
        gbm_exists = False

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
                cost, pnl, step_pnls, hedges = simulate_simple(simple_actor, simple_params, scaled, cfg_simple, strike)
                row['simple_net'] = premium - cost
                row['simple_step_pnls'] = step_pnls
                row['simple_hedges'] = hedges
            except Exception as e:
                print(f"  Simple error: {e}")
                row['simple_net'] = np.nan

        # GBM agent
        if gbm_actor:
            try:
                cost, pnl, step_pnls, hedges = simulate_gbm(gbm_actor, gbm_params, scaled, cfg_gbm, strike)
                row['gbm_net'] = premium - cost
                row['gbm_step_pnls'] = step_pnls
                row['gbm_hedges'] = hedges
            except Exception as e:
                print(f"  GBM error: {e}")
                row['gbm_net'] = np.nan

        # BSM
        try:
            cost, pnl, step_pnls, hedges = simulate_bsm(scaled, cfg_simple, strike)
            row['bsm_net'] = premium - cost
            row['bsm_step_pnls'] = step_pnls
            row['bsm_hedges'] = hedges
        except Exception as e:
            print(f"  BSM error: {e}")
            row['bsm_net'] = np.nan

        results.append(row)
        print(f"Window {idx+1}: Simple={row.get('simple_net',0):.2f}  GBM={row.get('gbm_net',0):.2f}  BSM={row.get('bsm_net',0):.2f}")

        # Plot first 3 windows
        if idx < 3:
            plot_window_comparison(scaled, strike, row, dates, idx+1)

    # Plot hedge distributions after all windows
    plot_hedge_distributions(results, simple_exists, gbm_exists)

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
    parser.add_argument('--ticker', type=str, default='SPY')
    parser.add_argument('--simple_ckpt', type=str, default='checkpoints/simple_0.25/final_actor.pkl')
    parser.add_argument('--gbm_ckpt', type=str, default='checkpoints/gbm_td3/epoch_250_actor.pkl')
    parser.add_argument('--windows', type=int, default=None)
    args = parser.parse_args()
    run_walk_forward_all(ticker=args.ticker, simple_ckpt=args.simple_ckpt,
                         gbm_ckpt=args.gbm_ckpt, max_windows=args.windows)