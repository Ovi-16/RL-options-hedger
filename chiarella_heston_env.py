# chiarella_heston_env.py
# Chiarella-Heston environment with 6-dimensional state including tau

import jax
import jax.numpy as jnp
import numpy as np


def black_scholes_delta_ch(S, K, tau, r, sigma):
    """BSM delta for call option - used for warm start"""
    epsilon = 1e-8
    tau = jnp.maximum(tau, epsilon)
    sqrt_tau = jnp.sqrt(tau)
    d1 = (jnp.log(S/K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau + epsilon)
    norm_cdf = lambda x: 0.5 * (1 + jax.lax.erf(x / jnp.sqrt(2.0)))
    return norm_cdf(d1)


def build_state_ch(state_dict, tau, h_prev, K, r, sigma, T, action_ubnd):
    """
    CH state: [log_price, momentum, log_fundamental, variance, h_norm, tau_norm]
    Total 6 features - includes tau explicitly
    """
    log_price = jnp.asarray(state_dict['log_price']).reshape(-1)
    momentum = jnp.asarray(state_dict['momentum']).reshape(-1)
    log_fundamental = jnp.asarray(state_dict['log_fundamental']).reshape(-1)
    variance = jnp.asarray(state_dict['variance']).reshape(-1)
    h_prev = jnp.asarray(h_prev).reshape(-1)
    
    # Normalize holdings to [-1, 1]
    h_norm = h_prev / action_ubnd
    h_norm = jnp.clip(h_norm, -1.0, 1.0)
    
    # Normalize tau to [0, 1]
    tau_norm = jnp.full_like(log_price, tau / T)
    tau_norm = jnp.clip(tau_norm, 0.0, 1.0)
    
    # Stack: [log_price, momentum, log_fundamental, variance, h_norm, tau_norm]
    return jnp.stack([log_price, momentum, log_fundamental, variance, h_norm, tau_norm], axis=-1)


def ch_step(state_dict, h_prev, total_cost, t, actions, rng_key, cfg):
    """Chiarella-Heston step - vectorized across paths"""
    log_price = state_dict['log_price']
    momentum = state_dict['momentum']
    log_fundamental = state_dict['log_fundamental']
    variance = state_dict['variance']
    price = state_dict['price']
    
    dt = cfg.dt
    K = cfg.K
    r = cfg.r
    T = cfg.T
    
    # CH parameters from config
    kappa_ch = cfg.kappa_CH
    sigma_F = cfg.sigma_F
    beta = cfg.beta
    gamma = cfg.gamma
    alpha = cfg.alpha
    omega = cfg.omega
    phi = cfg.phi
    theta = cfg.theta
    sigma_vol = cfg.sigma_vol
    rho = cfg.rho
    g = r - 0.5 * sigma_F**2
    
    # Split RNG
    keys = jax.random.split(rng_key, 5)
    key_F = keys[0]
    key_S = keys[1]
    key_V = keys[2]
    rng_key = keys[3]
    
    n_paths = price.shape[0]
    
    # 1. Fundamental value evolution (GBM)
    dW_F = jax.random.normal(key_F, shape=(n_paths,)) * jnp.sqrt(dt)
    log_fundamental_next = log_fundamental + g * dt + sigma_F * dW_F
    fundamental_price = jnp.exp(log_fundamental_next)
    
    # 2. Volatility evolution (Heston)
    dW_V = jax.random.normal(key_V, shape=(n_paths,)) * jnp.sqrt(dt)
    variance_next = variance + phi * (theta - variance) * dt
    variance_next += sigma_vol * jnp.sqrt(jnp.maximum(variance, 0)) * dW_V
    variance_next = jnp.maximum(variance_next, 1e-8)
    
    # 3. Correlated noise for price (leverage effect)
    eps_V = dW_V / jnp.sqrt(dt)
    eps_S = jax.random.normal(key_S, shape=(n_paths,))
    eps_S_corr = rho * eps_V + jnp.sqrt(1 - rho**2) * eps_S
    
    # 4. Price change from three trader types
    fundamental_demand = kappa_ch * (fundamental_price - price) * dt
    momentum_demand = beta * jnp.tanh(gamma * momentum) * dt
    vol_demand = omega * jnp.sqrt(jnp.maximum(variance, 0)) * eps_S_corr * jnp.sqrt(dt)
    
    dS = fundamental_demand + momentum_demand + vol_demand
    price_next = jnp.maximum(price + dS, 0.01)
    
    # Update momentum signal (EMA of returns)
    log_price_next = jnp.log(price_next)
    dP = log_price_next - log_price
    momentum_next = (1 - alpha) * momentum + alpha * (dP / dt)
    
    # Calculate P&L components
    h_next = actions
    dS_actual = price_next - price
    hedging_error = dS_actual * h_prev
    transaction_cost = cfg.kappa * jnp.abs(h_next - h_prev) * price_next
    financing_cost = r * h_prev * price * dt
    total_cost_next = total_cost + (-hedging_error + transaction_cost + financing_cost)
    
    # New state dict
    new_state_dict = {
        'log_price': log_price_next,
        'momentum': momentum_next,
        'log_fundamental': log_fundamental_next,
        'variance': variance_next,
        'price': price_next
    }
    
    # Build next state
    tau_next = T - (t + 1) * dt
    next_state = build_state_ch(new_state_dict, tau_next, h_next, K, r, cfg.sigma, T, cfg.action_ubnd)
    
    new_carry = (new_state_dict, h_next, total_cost_next, t + 1)
    step_info = {
        'reward': -(-hedging_error + transaction_cost + financing_cost),
        'next_state': next_state,
        'done': (t + 1 >= cfg.N_steps).astype(jnp.float32)
    }
    
    return new_carry, step_info, rng_key


def get_initial_carry_ch(n_paths, S0):
    """Initialize CH carry for parallel paths"""
    state_dict = {
        'log_price': jnp.full(n_paths, jnp.log(S0)),
        'momentum': jnp.zeros(n_paths),
        'log_fundamental': jnp.full(n_paths, jnp.log(S0)),
        'variance': jnp.full(n_paths, 0.04),
        'price': jnp.full(n_paths, S0)
    }
    h = jnp.zeros(n_paths)
    total_cost = jnp.zeros(n_paths)
    t = jnp.zeros(n_paths, dtype=int)
    return (state_dict, h, total_cost, t)


def get_actions(actor, actor_params, state, add_noise=False, rng_key=None, noise_scale=0.1, cfg=None):
    """Get actions from actor"""
    if cfg is None:
        from config_ch import cfg_ch
        cfg = cfg_ch
    
    actions = actor.apply(actor_params, state)
    if add_noise and rng_key is not None:
        noise = jax.random.normal(rng_key, shape=actions.shape) * noise_scale
        actions = actions + noise
        actions = jnp.clip(actions, cfg.action_lbnd, cfg.action_ubnd)
    return actions




def simulate_vectorized_batch_ch(rng_key, actor, actor_params, replay_buffer, cfg, 
                                  training=True, exploration_noise_scale=0.2):
    """Vectorized batch simulation with configurable exploration noise"""
    N_parallel = cfg.N_parallel
    N_steps = cfg.N_steps
    
    carry = get_initial_carry_ch(N_parallel, cfg.S0)
    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []
    
    rng = rng_key
    
    for step in range(N_steps):
        state_dict, h, total_cost, t = carry
        tau = cfg.T - step * cfg.dt
        tau = jnp.maximum(tau, 0.0)
        
        state = build_state_ch(state_dict, tau, h, cfg.K, cfg.r, cfg.sigma, cfg.T, cfg.action_ubnd)
        
        noise_key, rng = jax.random.split(rng)
        if training:
            # Use the passed exploration_noise_scale
             actions = get_actions(actor, actor_params, state, 
                                  add_noise=True, rng_key=noise_key, 
                                  noise_scale=exploration_noise_scale,
                                  cfg=cfg)  # ← ADD THIS
        else:
            actions = get_actions(actor, actor_params, state, 
                                  add_noise=False, 
                                  cfg=cfg)  # ← ADD THIS
        
        new_carry, step_info, rng = ch_step(state_dict, h, total_cost, t, actions, rng, cfg)
        carry = new_carry
        
        all_states.append(np.array(state))
        all_actions.append(np.array(actions))
        all_rewards.append(np.array(step_info['reward']))
        all_next_states.append(np.array(step_info['next_state']))
        all_dones.append(np.array(step_info['done']))
    
    if training:
        for step in range(N_steps):
            for i in range(N_parallel):
                replay_buffer.push(
                    all_states[step][i],
                    all_actions[step][i],
                    all_rewards[step][i],
                    all_next_states[step][i],
                    all_dones[step][i]
                )
    
    state_dict, h, total_cost, t = carry
    return total_cost, rng