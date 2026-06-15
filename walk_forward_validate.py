# walk_forward_validate.py - Simple CH estimation, scale to $100, with hedge distribution plot
# Modified: now processes ALL windows by default (--windows=None)

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import jax
import jax.numpy as jnp
import yfinance as yf
from scipy import stats
from config import cfg
from config_ch import cfg_ch
from env import build_state, black_scholes_delta
from agent import Actor
from agent_ch import ActorCHWithInitDelta

# ================================================================
# HELPER FUNCTIONS (unchanged)
# ================================================================

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        return S * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

def compute_max_drawdown(pnl_series):
    cumulative = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    drawdown_pct = drawdown / (np.abs(running_max) + 1e-8) * 100
    return np.min(drawdown), np.min(drawdown_pct)

def compute_sharpe_ratio(pnl_series, risk_free_rate=0.02):
    if len(pnl_series) == 0 or np.std(pnl_series) == 0:
        return 0
    excess_returns = pnl_series - risk_free_rate / 252
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

def compute_sortino_ratio(pnl_series, risk_free_rate=0.02):
    if len(pnl_series) == 0:
        return 0
    excess_returns = pnl_series - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0 if np.mean(excess_returns) <= 0 else 100
    return np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)

def compute_calmar_ratio(pnl_series):
    total_return = np.sum(pnl_series)
    _, max_dd_pct = compute_max_drawdown(pnl_series)
    if max_dd_pct == 0:
        return 0
    return total_return / abs(max_dd_pct) * 100

def compute_var(pnl_series, confidence=0.95):
    return np.percentile(pnl_series, (1 - confidence) * 100)

def compute_cvar(pnl_series, confidence=0.95):
    var = compute_var(pnl_series, confidence)
    return np.mean(pnl_series[pnl_series <= var])

def download_stock_data(ticker='IWM', start_date='2015-01-01', end_date='2024-12-31'):
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df = stock[['Close']].copy()
    df.columns = ['price']
    print(f"Downloaded {len(df)} days of data")
    print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    return df

def create_rolling_windows_atm(prices, window_size=21, step_size=21):
    windows = []
    dates = []
    strikes = []
    moneyness_list = []
    for start_idx in range(0, len(prices) - window_size, step_size):
        end_idx = start_idx + window_size
        S0 = prices.iloc[start_idx]
        strike = S0   # ATM
        window_prices = prices[start_idx:end_idx + 1].values.flatten()
        windows.append(window_prices)
        dates.append({
            'start': prices.index[start_idx],
            'end': prices.index[end_idx],
            'S0': S0,
            'strike': strike
        })
        strikes.append(strike)
        moneyness_list.append(S0 / strike)
    print(f"Created {len(windows)} rolling windows (ATM options, strike = S0)")
    return windows, dates, moneyness_list

def build_state_ch_simple(log_price, momentum, log_fundamental, variance, h_prev, tau, K, r, sigma, T, action_ubnd):
    """Build CH state from simple heuristics (no filter)."""
    log_price = jnp.asarray(log_price).reshape(-1)
    momentum = jnp.asarray(momentum).reshape(-1)
    log_fund = jnp.asarray(log_fundamental).reshape(-1)
    variance = jnp.asarray(variance).reshape(-1)
    h_norm = (jnp.asarray(h_prev).reshape(-1) / action_ubnd).clip(-1, 1)
    tau_norm = jnp.full_like(log_price, tau / T).clip(0, 1)
    return jnp.stack([log_price, momentum, log_fund, variance, h_norm, tau_norm], axis=-1)

def simulate_on_scaled_path_simple(actor, actor_params, price_path_scaled, cfg_used, strike_scaled, is_ch):
    """
    Simulate hedging on scaled price path using simple CH state estimation (no filter).
    """
    n_steps = len(price_path_scaled) - 1
    dt = 1/252
    S = price_path_scaled[0]
    h = 0.0
    total_cost = 0.0
    step_pnls = []
    daily_hedges = []
    
    # For CH simple state
    last_momentum = 0.0
    returns_history = []
    
    for step in range(n_steps):
        tau = cfg_used.T - step * dt
        tau = max(tau, 0.01)
        
        if is_ch:
            # Simple volatility (rolling 20-day)
            if step > 0:
                returns_history.append(np.log(S / price_path_scaled[max(0, step-1)]))
                lookback = min(20, len(returns_history))
                if lookback > 1:
                    realized_vol = np.std(returns_history[-lookback:]) * np.sqrt(252)
                else:
                    realized_vol = cfg_used.sigma
            else:
                realized_vol = cfg_used.sigma
            
            # Simple momentum (EMA)
            if step > 0:
                daily_return = (S - price_path_scaled[step-1]) / price_path_scaled[step-1]
                last_momentum = 0.1 * daily_return + 0.9 * last_momentum
            
            log_price = np.log(S)
            log_fundamental = log_price   # fundamental = current price
            variance = realized_vol ** 2
            
            state = build_state_ch_simple(log_price, last_momentum, log_fundamental, variance,
                                          h, tau, strike_scaled, cfg_used.r, cfg_used.sigma,
                                          cfg_used.T, cfg_used.action_ubnd)
        else:
            # GBM agent (simple state)
            state = build_state(jnp.array([S]), tau, jnp.array([h]),
                                strike_scaled, cfg_used.r, cfg_used.sigma, cfg_used.T)
        
        action = actor.apply(actor_params, state)
        action = jnp.clip(action, cfg_used.action_lbnd, cfg_used.action_ubnd)
        action = float(action.squeeze())
        
        S_next = price_path_scaled[step + 1]
        dS = S_next - S
        hedging_error = dS * h
        transaction_cost = cfg_used.kappa * abs(action - h) * S_next
        financing_cost = cfg_used.r * h * S * dt
        step_cost = -hedging_error + transaction_cost + financing_cost
        total_cost += step_cost
        step_pnls.append(-step_cost)
        daily_hedges.append(action)
        
        S = S_next
        h = action
    
    # Add option payoff
    payoff = max(price_path_scaled[-1] - strike_scaled, 0)
    total_cost += payoff
    step_pnls.append(-payoff)
    return total_cost, -total_cost, step_pnls, daily_hedges

def simulate_bsm_on_path(price_path, cfg_used, strike):
    """BSM delta hedge on any price path."""
    n_steps = len(price_path) - 1
    dt = 1/252
    h = 0.0
    total_cost = 0.0
    step_pnls = []
    daily_hedges = []
    for step in range(n_steps):
        S = price_path[step]
        S_next = price_path[step + 1]
        tau = cfg_used.T - step * dt
        tau = max(tau, 0.01)
        delta = black_scholes_delta(S, strike, tau, cfg_used.r, cfg_used.sigma)
        if cfg_used.is_short_call:
            delta = -delta
        delta = np.clip(delta, cfg_used.action_lbnd, cfg_used.action_ubnd)
        dS = S_next - S
        hedging_error = dS * h
        transaction_cost = cfg_used.kappa * abs(delta - h) * S_next
        financing_cost = cfg_used.r * h * S * dt
        step_cost = -hedging_error + transaction_cost + financing_cost
        total_cost += step_cost
        step_pnls.append(-step_cost)
        daily_hedges.append(delta)
        h = delta
    payoff = max(price_path[-1] - strike, 0)
    total_cost += payoff
    step_pnls.append(-payoff)
    return total_cost, -total_cost, step_pnls, daily_hedges

def load_agent(model_type):
    if model_type == 'gbm':
        checkpoint_path = "checkpoints/gbm/final_actor.pkl"
        cfg_used = cfg
        actor = Actor()
        is_ch = False
    elif model_type == 'ch':
        checkpoint_path = "checkpoints/ch/final_actor.pkl"
        cfg_used = cfg_ch
        actor = ActorCHWithInitDelta(K=cfg_used.K, r=cfg_used.r, T=cfg_used.T)
        is_ch = True
    else:
        return None, None, None, None
    try:
        with open(checkpoint_path, "rb") as f:
            actor_params = pickle.load(f)
        print(f"  Loaded {model_type.upper()} model")
        return actor, actor_params, cfg_used, is_ch
    except FileNotFoundError:
        print(f"  ❌ {model_type.upper()} model not found at {checkpoint_path}")
        return None, None, None, None

def plot_hedge_distributions(results):
    """Plot histograms of hedge positions (shares) for all agents."""
    gbm_hedges = []
    ch_hedges = []
    bsm_hedges = []
    
    for r in results:
        if 'gbm_hedges' in r and r['gbm_hedges'] is not None:
            gbm_hedges.extend(r['gbm_hedges'])
        if 'ch_hedges' in r and r['ch_hedges'] is not None:
            ch_hedges.extend(r['ch_hedges'])
        if 'bsm_hedges' in r and r['bsm_hedges'] is not None:
            bsm_hedges.extend(r['bsm_hedges'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, data, name in zip(axes, [gbm_hedges, ch_hedges, bsm_hedges],
                              ['GBM Agent', 'CH Agent', 'BSM Delta']):
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax.set_title(f'{name}\n(mean={np.mean(data):.2f}, std={np.std(data):.2f})')
            ax.set_xlabel('Hedge Position (shares)')
            ax.set_ylabel('Frequency')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(name)
    
    plt.tight_layout()
    plt.savefig('hedge_distributions.png', dpi=150)
    plt.close()
    print("Saved hedge_distributions.png")

def run_full_walk_forward(ticker='IWM', n_windows=None, plot_windows=True):
    """
    Walk-forward validation with 21‑day windows scaled to start at $100.
    Uses simple CH state estimation (no filter, no warm‑up).
    If n_windows is None, processes ALL available windows.
    """
    print("="*80)
    print(f"WALK-FORWARD VALIDATION - {ticker} (21‑day windows, scaled to $100, SIMPLE CH)")
    print("Comparing: GBM Agent | CH Agent (simple) | BSM Delta Hedge")
    print("="*80)

    stock_data = download_stock_data(ticker, '2015-01-01', '2024-12-31')
    prices = stock_data['price']

    # Create non‑overlapping 21‑day windows (ATM)
    windows, window_dates, _ = create_rolling_windows_atm(prices, window_size=21, step_size=21)

    if len(windows) == 0:
        print("No windows created.")
        return None

    # Apply window limit if specified
    if n_windows is not None:
        windows = windows[:n_windows]
        window_dates = window_dates[:n_windows]
    print(f"\nTesting on {len(windows)} windows (21 days each)\n")

    # Load models
    print("Loading models...")
    gbm_actor, gbm_params, gbm_cfg, _ = load_agent('gbm')
    ch_actor, ch_params, ch_cfg, ch_is_ch = load_agent('ch')
    if gbm_actor is None and ch_actor is None:
        print("No models found.")
        return None

    results = []
    plot_idx = 0

    for idx, (window_prices, dates) in enumerate(zip(windows, window_dates)):
        S0_orig = window_prices[0]
        scale = 100.0 / S0_orig
        scaled_window = window_prices * scale
        strike_scaled = 100.0   # ATM after scaling
        premium = black_scholes_price(100.0, strike_scaled, 21/252, cfg.r, cfg.sigma)

        window_result = {
            'window': idx,
            'start_date': dates['start'],
            'end_date': dates['end'],
            'S0_scaled': 100.0,
            'strike_scaled': strike_scaled,
            'premium': premium,
        }

        # GBM agent
        if gbm_actor is not None:
            try:
                cost, pnl, step_pnls, hedges = simulate_on_scaled_path_simple(
                    gbm_actor, gbm_params, scaled_window, gbm_cfg, strike_scaled, is_ch=False)
                net = premium - cost
                window_result['gbm_net'] = net
                window_result['gbm_step_pnls'] = step_pnls
                window_result['gbm_hedges'] = hedges
                print(f"Window {idx+1}: GBM Profit = ${net:.2f}")
            except Exception as e:
                print(f"  GBM error: {e}")
                window_result['gbm_net'] = np.nan

        # CH agent (simple estimation)
        if ch_actor is not None:
            try:
                cost, pnl, step_pnls, hedges = simulate_on_scaled_path_simple(
                    ch_actor, ch_params, scaled_window, ch_cfg, strike_scaled, is_ch=True)
                net = premium - cost
                window_result['ch_net'] = net
                window_result['ch_step_pnls'] = step_pnls
                window_result['ch_hedges'] = hedges
                print(f"        CH  Profit = ${net:.2f}")
            except Exception as e:
                print(f"  CH error: {e}")
                window_result['ch_net'] = np.nan

        # BSM delta hedge
        try:
            cost, pnl, step_pnls, hedges = simulate_bsm_on_path(scaled_window, cfg, strike_scaled)
            net = premium - cost
            window_result['bsm_net'] = net
            window_result['bsm_step_pnls'] = step_pnls
            window_result['bsm_hedges'] = hedges
            print(f"        BSM Profit = ${net:.2f}")
        except Exception as e:
            print(f"  BSM error: {e}")
            window_result['bsm_net'] = np.nan

        results.append(window_result)

        # Plot first 3 windows if requested
        if plot_windows and idx < 3:
            plot_idx += 1
            plot_window_comparison(scaled_window, strike_scaled, window_result, window_dates[idx], plot_idx)

    compute_all_metrics(results)
    plot_hedge_distributions(results)
    return results

def plot_window_comparison(scaled_prices, strike, result, window_date, plot_num):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()

    days = np.arange(len(scaled_prices))
    premium = result['premium']
    ax1.plot(days, scaled_prices, label='Stock price (scaled to $100)', color='black')
    ax1.axhline(y=strike, linestyle='--', color='gray', label=f'Strike = ${strike:.0f}')
    ax1.set_title(f'Window {plot_num}: {window_date["start"].date()} - {window_date["end"].date()}')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.text(0.02, 0.95, f'Option premium = ${premium:.2f}', transform=ax1.transAxes, fontsize=10, verticalalignment='top')

    for name, key in [('GBM', 'gbm_hedges'), ('CH', 'ch_hedges'), ('BSM', 'bsm_hedges')]:
        hedges = result.get(key)
        if hedges is not None:
            ax2.plot(days[:-1], hedges, label=name)
    ax2.set_title('Hedge Positions (shares)')
    ax2.set_ylabel('Holding')
    ax2.legend()

    for name, key in [('GBM', 'gbm_step_pnls'), ('CH', 'ch_step_pnls'), ('BSM', 'bsm_step_pnls')]:
        pnls = result.get(key)
        if pnls is not None:
            cum = np.cumsum(pnls)
            ax3.plot(days[:len(cum)], cum, label=name)
    ax3.set_title('Cumulative P&L from Hedging')
    ax3.set_ylabel('P&L ($)')
    ax3.legend()

    net_profits = []
    names = []
    for name in ['GBM', 'CH', 'BSM']:
        net = result.get(f'{name.lower()}_net')
        if net is not None and not np.isnan(net):
            net_profits.append(net)
            names.append(name)
    ax4.bar(names, net_profits, color=['blue', 'orange', 'green'])
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.set_title('Net Profit (Premium - Hedging Cost)')
    ax4.set_ylabel('Profit ($)')

    plt.tight_layout()
    plt.savefig(f'hedge_window_{plot_num}.png', dpi=150)
    plt.close()
    print(f"  Saved plot hedge_window_{plot_num}.png")

def compute_all_metrics(results):
    """Compute metrics for all strategies."""
    strategies = []
    if 'gbm_net' in results[0] and not np.isnan(results[0]['gbm_net']):
        strategies.append(('GBM Agent', [r['gbm_net'] for r in results]))
    if 'ch_net' in results[0] and not np.isnan(results[0]['ch_net']):
        strategies.append(('CH Agent', [r['ch_net'] for r in results]))
    strategies.append(('BSM Delta', [r['bsm_net'] for r in results]))

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

    print("\n" + "="*80)
    print("PROFITABILITY METRICS")
    print("="*80)
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

    print("\n" + "="*80)
    print("RISK-ADJUSTED METRICS")
    print("="*80)
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

    print("\n" + "="*80)
    print("SUMMARY: BEST PERFORMER BY METRIC")
    print("="*80)
    for label, key in [('Mean Net Profit ($)', 'mean'), ('Std Net Profit ($)', 'std'),
                       ('Win Rate (%)', 'win_rate'), ('Total Profit ($)', 'total'),
                       ('Sharpe Ratio', 'sharpe'), ('Sortino Ratio', 'sortino'),
                       ('Calmar Ratio', 'calmar'), ('Max Drawdown ($)', 'max_dd_amt'),
                       ('Max Drawdown (%)', 'max_dd_pct'), ('VaR 95% ($)', 'var95'),
                       ('CVaR 95% ($)', 'cvar95')]:
        best_name = None
        best_val = -np.inf if key not in ['max_dd_amt','max_dd_pct','var95','cvar95'] else np.inf
        for name, m in all_metrics.items():
            val = m[key]
            if key in ['max_dd_amt','max_dd_pct','var95','cvar95']:
                if val < best_val:
                    best_val = val
                    best_name = name
            else:
                if val > best_val:
                    best_val = val
                    best_name = name
        if key == 'max_dd_pct':
            print(f"{label:<20}: {best_name} ({best_val:.2f}%)")
        elif key in ['sharpe','sortino','calmar']:
            print(f"{label:<20}: {best_name} ({best_val:.3f})")
        else:
            print(f"{label:<20}: {best_name} (${best_val:.2f})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='SPY')
    parser.add_argument('--windows', type=int, default=None, help='Number of windows to test (default: all)')
    parser.add_argument('--no_plots', action='store_true')
    args = parser.parse_args()
    run_full_walk_forward(ticker=args.ticker, n_windows=args.windows, plot_windows=not args.no_plots)