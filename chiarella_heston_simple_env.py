# chiarella_heston_simple_env.py
import jax
import jax.numpy as jnp
import numpy as np
from config_simple import ConfigSimple
from scipy.stats import norm   # for scalar fallback, but we'll use jnp

def black_scholes_price(S, K, tau, r, sigma):
    """
    Vectorized Black-Scholes price for arrays.
    tau can be an array; we handle tau <= 0 with jnp.where.
    """
    # Ensure tau is at least epsilon for numerical stability
    tau_safe = jnp.maximum(tau, 1e-8)
    d1 = (jnp.log(S/K) + (r + 0.5*sigma**2)*tau_safe) / (sigma*jnp.sqrt(tau_safe))
    d2 = d1 - sigma*jnp.sqrt(tau_safe)
    # Standard normal CDF using jax.scipy.special.ndtr (more reliable)
    from jax.scipy.stats import norm as jax_norm
    call = S * jax_norm.cdf(d1) - K * jnp.exp(-r*tau_safe) * jax_norm.cdf(d2)
    # For tau <= 0, payoff is max(S-K, 0)
    call = jnp.where(tau <= 0, jnp.maximum(S - K, 0.0), call)
    return call

def build_simple_state(price, h_prev, tau, action_ubnd, T):
    log_price = jnp.log(price)
    h_norm = h_prev / action_ubnd
    tau_norm = jnp.full_like(log_price, tau / T)
    return jnp.stack([log_price, h_norm, tau_norm], axis=-1)

def simple_step(state_dict, h_prev, total_cost, t, actions, rng_key, cfg):
    """
    Evolve CH dynamics (hidden states). Agent only sees simple state.
    Returns next simple state and reward.
    """
    # Unpack CH state
    log_price = state_dict['log_price']
    momentum = state_dict['momentum']
    log_fundamental = state_dict['log_fundamental']
    variance = state_dict['variance']
    price = state_dict['price']
    dt = cfg.dt
    r = cfg.r
    sigma = cfg.sigma
    T = cfg.T

    # CH parameters
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
    g = r - 0.5*sigma_F**2

    # Random numbers
    keys = jax.random.split(rng_key, 5)
    key_F, key_S, key_V = keys[0], keys[1], keys[2]
    n_paths = price.shape[0]

    # Fundamental
    dW_F = jax.random.normal(key_F, (n_paths,)) * jnp.sqrt(dt)
    log_fundamental_next = log_fundamental + g*dt + sigma_F*dW_F
    fundamental_price = jnp.exp(log_fundamental_next)

    # Variance
    dW_V = jax.random.normal(key_V, (n_paths,)) * jnp.sqrt(dt)
    variance_next = variance + phi*(theta - variance)*dt + sigma_vol*jnp.sqrt(jnp.maximum(variance,0))*dW_V
    variance_next = jnp.maximum(variance_next, 1e-8)

    # Correlated price noise
    eps_V = dW_V / jnp.sqrt(dt)
    eps_S = jax.random.normal(key_S, (n_paths,))
    eps_S_corr = rho*eps_V + jnp.sqrt(1 - rho**2)*eps_S

    # Price change
    fundamental_demand = kappa_ch * (fundamental_price - price) * dt
    momentum_demand = beta * jnp.tanh(gamma * momentum) * dt
    vol_demand = omega * jnp.sqrt(jnp.maximum(variance,0)) * eps_S_corr * jnp.sqrt(dt)
    dS = fundamental_demand + momentum_demand + vol_demand
    price_next = jnp.maximum(price + dS, 0.01)

    # Momentum update
    log_price_next = jnp.log(price_next)
    dP = log_price_next - log_price
    momentum_next = (1 - alpha)*momentum + alpha*(dP/dt)

    # Hedging P&L components
    h_next = actions
    dS_actual = price_next - price
    hedging_gain = h_prev * dS_actual
    trans_cost = cfg.kappa * jnp.abs(h_next - h_prev) * price_next

    # Option prices (for reward)
    tau = T - t * dt          # t is an array of shape (n_paths,)
    tau_next = tau - dt
    V_t = black_scholes_price(price, cfg.K, tau, r, sigma)
    V_next = black_scholes_price(price_next, cfg.K, tau_next, r, sigma)
    option_change = -(V_next - V_t)
    financing_cost = r * (h_prev * price - V_t) * dt

    # Reward = profit
    reward = option_change + hedging_gain - trans_cost - financing_cost
    step_cost = -reward
    total_cost_next = total_cost + step_cost

    # New CH state dict
    new_state_dict = {
        'log_price': log_price_next,
        'momentum': momentum_next,
        'log_fundamental': log_fundamental_next,
        'variance': variance_next,
        'price': price_next
    }

    # Build simple next state for agent
    tau_next_val = T - (t+1)*dt
    next_state = build_simple_state(price_next, h_next, tau_next_val, cfg.action_ubnd, T)

    new_carry = (new_state_dict, h_next, total_cost_next, t+1)
    step_info = {
        'reward': reward,
        'next_state': next_state,
        'done': (t+1 >= cfg.N_steps).astype(jnp.float32)
    }
    return new_carry, step_info, keys[3]  # return leftover rng

def get_initial_carry_simple(n_paths, S0):
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

def simulate_vectorized_batch_simple(rng_key, actor, actor_params, replay_buffer, cfg,
                                     training=True, exploration_noise_scale=0.2):
    N_parallel = cfg.N_parallel
    N_steps = cfg.N_steps
    carry = get_initial_carry_simple(N_parallel, cfg.S0)
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
        # Build agent's observation (simple state)
        state = build_simple_state(state_dict['price'], h, tau, cfg.action_ubnd, cfg.T)

        # Get action from actor
        noise_key, rng = jax.random.split(rng)
        if training:
            actions = actor.apply(actor_params, state)
            noise = jax.random.normal(noise_key, actions.shape) * exploration_noise_scale
            actions = actions + noise
            actions = jnp.clip(actions, cfg.action_lbnd, cfg.action_ubnd)
        else:
            actions = actor.apply(actor_params, state)
            actions = jnp.clip(actions, cfg.action_lbnd, cfg.action_ubnd)

        # Step the environment
        new_carry, step_info, rng = simple_step(state_dict, h, total_cost, t, actions, rng, cfg)
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
    _, _, total_cost, _ = carry
    return total_cost, rng