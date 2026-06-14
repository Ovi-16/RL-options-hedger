# validate_jax.py - FAST VECTORIZED VALIDATION
# Uses same vectorized approach as training for speed

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle
from config import cfg
from env import build_state, step_vectorized, black_scholes_delta
from agent import Actor


def vectorized_simulate(rng_key, actor_params, n_paths, n_steps):
    """
    FULLY VECTORIZED simulation - processes all paths and steps in one go.
    Returns final costs for all paths.
    """
    # Initialize
    S = jnp.full(n_paths, cfg.S0)
    h = jnp.zeros(n_paths)
    total_cost = jnp.zeros(n_paths)
    
    # Pre-generate all random shocks
    eps_key, rng_key = random.split(rng_key)
    epsilons = random.normal(eps_key, shape=(n_steps, n_paths))
    
    # Create step indices (0, 1, 2, ..., n_steps-1)
    step_indices = jnp.arange(n_steps)
    
    def step_fn(carry, step_data):
        S, h, total_cost = carry
        eps, step = step_data  # Unpack epsilon and step index
        tau = cfg.T - step * cfg.dt
        tau = jnp.maximum(tau, 0.0)
        
        # Build state for all paths at once
        state = build_state(S, tau, h, cfg.K, cfg.r, cfg.sigma, cfg.T)
        
        # Get actions for all paths
        actions = actor.apply(actor_params, state)
        actions = jnp.clip(actions, cfg.action_lbnd, cfg.action_ubnd)
        
        # Vectorized step
        S_next = S * jnp.exp((cfg.r - 0.5 * cfg.sigma**2) * cfg.dt + cfg.sigma * eps * jnp.sqrt(cfg.dt))
        
        # Calculate costs vectorized
        dS = S_next - S
        hedging_error = dS * h
        transaction_cost = cfg.kappa * jnp.abs(actions - h) * S_next
        financing_cost = cfg.r * h * S * cfg.dt
        step_cost = -hedging_error + transaction_cost + financing_cost
        total_cost_next = total_cost + step_cost
        
        return (S_next, actions, total_cost_next), None
    
    # Zip epsilons and step indices together
    step_data = (epsilons, step_indices)
    
    # Scan through steps
    carry_init = (S, h, total_cost)
    final_carry, _ = jax.lax.scan(step_fn, carry_init, step_data)
    
    _, _, final_total_cost = final_carry
    return final_total_cost, rng_key


def compute_option_price_fast(actor_params, n_paths=10000, lambda_risk=cfg.lambda_risk):
    """
    Fast option price computation using vectorized simulation.
    """
    rng = random.PRNGKey(42)
    final_costs, _ = vectorized_simulate(rng, actor_params, n_paths, cfg.N_steps)
    final_costs = np.array(final_costs)
    
    expected_cost = np.mean(final_costs)
    std_cost = np.std(final_costs)
    ask_price = expected_cost + lambda_risk * std_cost
    
    # Bootstrap for confidence interval (faster with fewer iterations)
    n_bootstrap = 500
    bootstrap_means = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(final_costs), len(final_costs), replace=True)
        bootstrap_means.append(np.mean(final_costs[idx]) + lambda_risk * np.std(final_costs[idx]))
    
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    return {
        'ask_price': ask_price,
        'expected_cost': expected_cost,
        'std_cost': std_cost,
        'risk_premium': lambda_risk * std_cost,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_paths': n_paths
    }


def validate_fast(actor_params, n_trials=10000, seed=42):
    """
    Fast validation comparing RL policy against BSM delta hedge.
    """
    rng = random.PRNGKey(seed)
    
    # RL policy simulation
    rl_costs, rng = vectorized_simulate(rng, actor_params, n_trials, cfg.N_steps)
    
    # BSM delta policy - simulate using a direct function instead of a fake actor
    def bsm_action_func(state):
        price_norm = state[:, 0]
        time_norm = state[:, 2]
        S = price_norm * 2 * cfg.K
        tau = time_norm * cfg.T
        delta = black_scholes_delta(S, cfg.K, tau, cfg.r, cfg.sigma)
        return jnp.clip(delta, cfg.action_lbnd, cfg.action_ubnd)
    
    # Run BSM simulation directly with a function
    def simulate_bsm(rng_key, n_paths, n_steps):
        """BSM-specific simulation without actor"""
        S = jnp.full(n_paths, cfg.S0)
        h = jnp.zeros(n_paths)
        total_cost = jnp.zeros(n_paths)
        
        eps_key, rng_key = random.split(rng_key)
        epsilons = random.normal(eps_key, shape=(n_steps, n_paths))
        step_indices = jnp.arange(n_steps)
        
        def step_fn(carry, step_data):
            S, h, total_cost = carry
            eps, step = step_data
            tau = cfg.T - step * cfg.dt
            tau = jnp.maximum(tau, 0.0)
            
            # Build state
            state = build_state(S, tau, h, cfg.K, cfg.r, cfg.sigma, cfg.T)
            
            # BSM action
            actions = bsm_action_func(state)
            actions = jnp.clip(actions, cfg.action_lbnd, cfg.action_ubnd)
            
            # Step
            S_next = S * jnp.exp((cfg.r - 0.5 * cfg.sigma**2) * cfg.dt + cfg.sigma * eps * jnp.sqrt(cfg.dt))
            
            dS = S_next - S
            hedging_error = dS * h
            transaction_cost = cfg.kappa * jnp.abs(actions - h) * S_next
            financing_cost = cfg.r * h * S * cfg.dt
            step_cost = -hedging_error + transaction_cost + financing_cost
            total_cost_next = total_cost + step_cost
            
            return (S_next, actions, total_cost_next), None
        
        step_data = (epsilons, step_indices)
        carry_init = (S, h, total_cost)
        final_carry, _ = jax.lax.scan(step_fn, carry_init, step_data)
        _, _, final_total_cost = final_carry
        return final_total_cost, rng_key
    
    bsm_costs, _ = simulate_bsm(rng, n_trials, cfg.N_steps)
    
    # Convert to numpy
    rl_costs = np.array(rl_costs)
    bsm_costs = np.array(bsm_costs)
    
    # Calculate statistics
    def compute_stats(costs):
        return {
            'mean': np.mean(costs),
            'std': np.std(costs),
            'mean_pct': 100 * np.mean(costs) / cfg.S0,
            'std_pct': 100 * np.std(costs) / cfg.S0,
            'var_95': np.percentile(costs, 95),
            'cvar_95': np.mean(costs[costs >= np.percentile(costs, 95)])
        }
    
    rl_stats = compute_stats(rl_costs)
    bsm_stats = compute_stats(bsm_costs)
    
    # Compute option price
    rl_price = compute_option_price_fast(actor_params, n_paths=min(n_trials, 5000))
    bsm_price = black_scholes_price(cfg.S0, cfg.K, cfg.T, cfg.r, cfg.sigma)
    
    results = {
        'rl': rl_stats,
        'bsm': bsm_stats,
        'rl_price': rl_price,
        'bsm_price': bsm_price,
        'rl_costs': rl_costs,
        'bsm_costs': bsm_costs
    }
    
    return results


def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes price for European option"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


def plot_results(results):
    """Plot validation results"""
    rl_costs = results['rl_costs']
    bsm_costs = results['bsm_costs']
    
    model_name = "CH" if cfg.use_chiarella_heston else "GBM"
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram
    axes[0, 0].hist(bsm_costs, bins=50, alpha=0.5, label='BSM', density=True, color='blue')
    axes[0, 0].hist(rl_costs, bins=50, alpha=0.5, label='RL', density=True, color='red')
    axes[0, 0].set_xlabel('Hedging Cost ($)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title(f'Distribution of Hedging Costs ({model_name})')
    axes[0, 0].legend()
    
    # Boxplot
    axes[0, 1].boxplot([bsm_costs, rl_costs], tick_labels=['BSM', 'RL'])
    axes[0, 1].set_ylabel('Hedging Cost ($)')
    axes[0, 1].set_title(f'Cost Comparison ({model_name})')
    
    # QQ plot
    from scipy import stats
    stats.probplot(rl_costs, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f'Q-Q Plot (RL Costs vs Normal) - {model_name}')
    
    # Price comparison bar chart
    prices = [results['bsm_price'], results['rl_price']['ask_price']]
    labels = ['BSM', 'RL']
    axes[1, 1].bar(labels, prices, color=['blue', 'red'])
    axes[1, 1].set_ylabel('Option Price ($)')
    axes[1, 1].set_title(f'Option Price Comparison ({model_name})')
    
    plt.tight_layout()
    plt.savefig(f'validation_results_{model_name.lower()}.png', dpi=150)
    plt.show()
    print(f"\n✅ Saved validation plot to 'validation_results_{model_name.lower()}.png'")


def compute_action_statistics_fast(actor_params, n_samples=10000):
    """Fast action statistics using vectorized state generation"""
    model_name = "CH" if cfg.use_chiarella_heston else "GBM"
    
    print("\n" + "="*80)
    print(f"ACTION STATISTICS ACROSS RANDOM STATES ({model_name})")
    print("="*80)
    
    rng = random.PRNGKey(123)
    
    # Generate random states in batches for speed
    n_batches = 10
    batch_size = n_samples // n_batches
    
    all_actions = []
    all_deltas = []
    
    for batch in range(n_batches):
        S_batch = jax.random.uniform(rng, (batch_size,)) * 200
        tau_batch = jax.random.uniform(rng, (batch_size,)) * cfg.T
        h_batch = jax.random.uniform(rng, (batch_size,), minval=-2.0, maxval=2.0)
        
        # Build states in batch
        states = []
        for i in range(batch_size):
            state = build_state(
                jnp.array([S_batch[i]]), tau_batch[i], jnp.array([h_batch[i]]),
                cfg.K, cfg.r, cfg.sigma, cfg.T
            )
            states.append(state)
        states = jnp.vstack(states)
        
        # Get actions
        actions = actor.apply(actor_params, states)
        actions = np.array(actions)
        
        # Get deltas
        deltas = black_scholes_delta(S_batch, cfg.K, tau_batch, cfg.r, cfg.sigma)
        deltas = np.array(deltas)
        
        all_actions.extend(actions)
        all_deltas.extend(deltas)
        
        rng, _ = random.split(rng)
    
    actions = np.array(all_actions)
    deltas = np.array(all_deltas)
    
    print(f"\nAction statistics across {len(actions)} random states:")
    print(f"  Mean action: {np.mean(actions):.4f}")
    print(f"  Std action: {np.std(actions):.4f}")
    print(f"  Min action: {np.min(actions):.4f}")
    print(f"  Max action: {np.max(actions):.4f}")
    print(f"  Actions near upper bound (>1.5): {100*np.mean(actions > 1.5):.1f}%")
    print(f"  Actions near lower bound (<-1.5): {100*np.mean(actions < -1.5):.1f}%")
    print(f"\nCorrelation with BSM delta: {np.corrcoef(actions, deltas)[0,1]:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(actions, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero')
    axes[0].set_xlabel('Action (Shares Held)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Distribution of Agent Actions ({model_name})')
    axes[0].legend()
    
    axes[1].scatter(deltas, actions, alpha=0.3, s=1)
    axes[1].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x (BSM Delta)')
    axes[1].set_xlabel('BSM Delta')
    axes[1].set_ylabel('RL Action')
    axes[1].set_title(f'RL Action vs BSM Delta ({model_name})')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'action_distribution_{model_name.lower()}.png', dpi=150)
    plt.show()
    
    return actions, deltas


if __name__ == "__main__":
    model_name = "Chiarella-Heston" if cfg.use_chiarella_heston else "GBM"
    checkpoint_path = cfg.checkpoint_dir + "/final_actor.pkl"
    
    print("="*80)
    print(f"DEEP HEDGING VALIDATION SUITE - {model_name} MODEL (FAST VECTORIZED)")
    print("="*80)
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load trained actor
    actor = Actor()
    try:
        with open(checkpoint_path, "rb") as f:
            actor_params = pickle.load(f)
        print("✅ Loaded checkpoint successfully")
    except FileNotFoundError:
        print(f"❌ No checkpoint found at {checkpoint_path}")
        print("   Please train the model first by running: python train.py")
        exit(1)
    
    # Run fast validation
    print("\n" + "-"*80)
    print("PART 1: VALIDATION RESULTS (Vectorized)")
    print("-"*80)
    
    results = validate_fast(actor_params, n_trials=10000)
    plot_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print(f"VALIDATION RESULTS SUMMARY ({model_name})")
    print("="*60)
    print(f"BSM Theoretical Price: ${results['bsm_price']:.4f}")
    print(f"RL Ask Price: ${results['rl_price']['ask_price']:.4f}")
    print(f"  - Expected Cost: ${results['rl_price']['expected_cost']:.4f}")
    print(f"  - Risk Premium: ${results['rl_price']['risk_premium']:.4f}")
    print(f"  - 95% CI: [${results['rl_price']['ci_lower']:.4f}, ${results['rl_price']['ci_upper']:.4f}]")
    print(f"\nHedging Cost Comparison:")
    print(f"  BSM: Mean = ${results['bsm']['mean']:.4f}, Std = ${results['bsm']['std']:.4f}")
    print(f"  RL:  Mean = ${results['rl']['mean']:.4f}, Std = ${results['rl']['std']:.4f}")
    improvement = 100 * (results['bsm']['mean'] - results['rl']['mean']) / results['bsm']['mean']
    print(f"  Improvement: {improvement:.1f}% lower mean cost")
    
    # Action statistics (fast version)
    print("\n" + "-"*80)
    print("PART 2: ACTION STATISTICS")
    print("-"*80)
    actions, deltas = compute_action_statistics_fast(actor_params, n_samples=5000)
    
    # Diagnostic summary
    print("\n" + "="*80)
    print(f"DIAGNOSTIC SUMMARY ({model_name})")
    print("="*80)
    
    rl_mean = results['rl']['mean']
    rl_std = results['rl']['std']
    bsm_mean = results['bsm']['mean']
    bsm_std = results['bsm']['std']
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────┐
    │  HEDGING PERFORMANCE DIAGNOSTIC                                 │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  Average Cost:                                                  │
    │    BSM: ${bsm_mean:.4f}                                          │
    │    RL:  ${rl_mean:.4f}  ({100*(bsm_mean-rl_mean)/bsm_mean:+.1f}%) │
    │                                                                  │
    │  Cost Volatility (Risk):                                        │
    │    BSM: ${bsm_std:.4f}                                          │
    │    RL:  ${rl_std:.4f}  ({rl_std/bsm_std:.1f}x higher)           │
    │                                                                  │
    │  Ask Price (Mean + λ×Std, λ=1):                                 │
    │    BSM: ${bsm_mean + bsm_std:.4f}                               │
    │    RL:  ${rl_mean + rl_std:.4f}                                 │
    │                                                                  │
    ├─────────────────────────────────────────────────────────────────┤
    │  INTERPRETATION:                                                │
    │                                                                  │
    │  {'✓' if rl_mean < bsm_mean else '✗'} RL has {'lower' if rl_mean < bsm_mean else 'higher'} average cost                                          │
    │  {'✗' if rl_std > bsm_std else '✓'} RL has {'higher' if rl_std > bsm_std else 'lower'} risk (volatility)                     │
    │                                                                  │
    │  {'⚠️  Agent is GAMBLING' if rl_std > 2*bsm_std else '✓ Agent risk is reasonable'}                                         │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    # Additional diagnostics
    extreme_pct = 100 * np.mean(np.abs(actions) > 1.8)
    if extreme_pct > 10:
        print(f"⚠️  WARNING: {extreme_pct:.1f}% of actions are at extreme bounds (±{cfg.action_ubnd})")
        print("   The agent has learned to take maximum positions (gambling behavior)")
    else:
        print(f"✓ Only {extreme_pct:.1f}% of actions at extreme bounds - good")
    
    corr = np.corrcoef(actions, deltas)[0, 1]
    if corr > 0.7:
        print(f"✓ Agent actions correlate well with BSM delta (r={corr:.3f})")
    elif corr > 0.3:
        print(f"⚠️ Agent actions weakly correlate with BSM delta (r={corr:.3f})")
    else:
        print(f"❌ Agent actions show NO correlation with BSM delta (r={corr:.3f})")
    
    print("\n" + "="*80)
    print("Price Sensitivity Analysis")
    print("="*80)
    for test_S0 in [80, 90, 100, 110, 120]:
        original_S0 = cfg.S0
        cfg.S0 = test_S0
        price_info = compute_option_price_fast(actor_params, n_paths=5000)
        print(f"S0 = ${test_S0}: Ask Price = ${price_info['ask_price']:.4f}")
        cfg.S0 = original_S0
    
    print(f"\n✅ Validation complete for {model_name} model!")