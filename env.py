# env.py
# GBM environment with EXPANDED STATE (includes delta)
# Reward now raw P&L (no per-step penalty - penalty applied at terminal)

import jax
import jax.numpy as jnp
from config import cfg

def black_scholes_delta(S, K, tau, r, sigma):
    """
    Vectorized BSM delta for call option.
    Returns delta in [0, 1].
    Handles both scalar and array inputs.
    """
    epsilon = 1e-8
    # Ensure tau is positive and handle edge cases
    tau = jnp.maximum(tau, epsilon)
    sqrt_tau = jnp.sqrt(tau)
    d1 = (jnp.log(S/K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau + epsilon)
    # Normal CDF using error function
    norm_cdf = lambda x: 0.5 * (1 + jax.lax.erf(x / jnp.sqrt(2.0)))
    return norm_cdf(d1)


def gbm_step(S, dt, mu, sigma, eps):
    """Single GBM step (vectorised)."""
    return S * jnp.exp((mu - 0.5 * sigma**2) * dt + sigma * jnp.sqrt(dt) * eps)


def build_state(S, tau, h_prev, K, r, sigma, T):
    """
    Build normalized state vector with 4 features:
    1. price_norm: normalized price in [0, 1]
    2. delta: Black-Scholes delta (theoretical hedge ratio)
    3. time_norm: normalized time to maturity in [0, 1]
    4. h_norm: current holdings normalized to [-1, 1]
    
    All inputs can be arrays; outputs maintain the same shape.
    """
    # Ensure all inputs are at least 1D for consistent broadcasting
    S = jnp.asarray(S)
    tau = jnp.asarray(tau)
    h_prev = jnp.asarray(h_prev)
    
    # Get the broadcast shape
    target_shape = jnp.broadcast_shapes(S.shape, tau.shape, h_prev.shape)
    
    # Broadcast all to the same shape
    S = jnp.broadcast_to(S, target_shape)
    tau = jnp.broadcast_to(tau, target_shape)
    h_prev = jnp.broadcast_to(h_prev, target_shape)
    
    # Feature 1: Normalized price
    # Map S from [0, 2K] to [0, 1] (typical range for hedging)
    price_norm = S / (2 * K)
    price_norm = jnp.clip(price_norm, 0.0, 1.0)
    
    # Feature 2: Delta (the critical missing feature!)
    delta = black_scholes_delta(S, K, tau, r, sigma)
    # Delta is already in [0, 1], no normalization needed
    
    # Feature 3: Normalized time to maturity
    time_norm = tau / T
    time_norm = jnp.clip(time_norm, 0.0, 1.0)
    
    # Feature 4: Current holdings normalized to [-1, 1]
    # Since action bounds are [-2, 2], normalize to [-1, 1] for better NN input
    h_norm = h_prev / cfg.action_ubnd
    h_norm = jnp.clip(h_norm, -1.0, 1.0)
    
    # Stack along the last axis (adds feature dimension)
    # All arrays now have same shape, so this works
    state = jnp.stack([price_norm, delta, time_norm, h_norm], axis=-1)
    
    return state


def reward_components(S_prev, S_next, h_prev, h_next, t_step):
    """
    Calculate raw P&L components (NO risk penalty here).
    Returns:
    - hedging_error: P&L from price movement
    - transaction_cost: cost of rebalancing
    - financing_cost: interest on borrowed money
    - total_cost: sum of all costs (what we want to minimize)
    - total_pnl: profit (negative of total_cost)
    """
    dt = cfg.dt
    r = cfg.r
    
    # Hedging error (P&L from price change)
    dS = S_next - S_prev
    hedging_error = dS * h_prev
    
    # Transaction cost (paid when changing position)
    transaction_cost = cfg.kappa * jnp.abs(h_next - h_prev) * S_next
    
    # Financing cost: interest on borrowed money to hold shares
    financing_cost = r * h_prev * S_prev * dt
    
    # Total cost (positive = bad)
    total_cost = -hedging_error + transaction_cost + financing_cost
    
    # P&L (positive = good)
    total_pnl = -total_cost
    
    return {
        'hedging_error': hedging_error,
        'transaction_cost': transaction_cost,
        'financing_cost': financing_cost,
        'total_cost': total_cost,
        'total_pnl': total_pnl
    }


def step_vectorized(carry, eps, actions):
    """
    Single environment step.
    Returns new carry and step info (NO variance penalty here).
    """
    S, h_prev, total_cost_so_far, t = carry
    dt = cfg.dt
    mu = cfg.r
    sigma = cfg.sigma
    K = cfg.K
    r = cfg.r
    T = cfg.T
    
    # Ensure all tensors have consistent shapes
    S = jnp.asarray(S)
    h_prev = jnp.asarray(h_prev)
    eps = jnp.asarray(eps)
    actions = jnp.asarray(actions)
    t = jnp.asarray(t)
    
    # Next price
    S_next = gbm_step(S, dt, mu, sigma, eps)
    h_next = actions
    
    # Calculate step P&L components
    components = reward_components(S, S_next, h_prev, h_next, t)
    
    # Update cumulative cost
    total_cost_next = total_cost_so_far + components['total_cost']
    
    # Time to maturity for next state
    tau_next = T - (t + 1) * dt
    tau_next = jnp.maximum(tau_next, 0.0)  # Ensure non-negative
    
    # Build next state (uses h_next for holdings)
    next_state = build_state(S_next, tau_next, h_next, K, r, sigma, T)
    
    # Check if done (at final step)
    done_array = (t + 1 >= cfg.N_steps).astype(jnp.float32)
    
    new_carry = (S_next, h_next, total_cost_next, t + 1)
    
    step_info = {
        'reward': components['total_pnl'],  # Raw P&L (positive = good)
        'total_pnl': -total_cost_next,       # Cumulative P&L so far
        'next_state': next_state,
        'done': done_array,
        'components': components
    }
    
    return new_carry, step_info
# env.py - Add this function for CH state building

def build_state_ch(state_dict, tau, h_prev, K, r, T):
    """
    Build state for Chiarella-Heston agent.
    State = [log_price, momentum, log_fundamental, variance, h_prev, tau]
    """
    log_price = state_dict['log_price']
    momentum = state_dict['momentum']
    log_fundamental = state_dict['log_fundamental']
    variance = state_dict['variance']
    
    # Normalize holdings to [-1, 1]
    h_norm = jnp.clip(h_prev / cfg.action_ubnd, -1.0, 1.0)
    
    # Normalize tau to [0, 1]
    tau_norm = tau / T
    
    return jnp.stack([log_price, momentum, log_fundamental, variance, h_norm, tau_norm], axis=-1)