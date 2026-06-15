# agent_ch.py - Chiarella-Heston Actor with Double Critic

import jax
import jax.numpy as jnp
import flax.linen as nn


class ActorCHWithInitDelta(nn.Module):
    """
    Actor for Chiarella-Heston market.
    State includes: [log_price, momentum, log_fundamental, variance, h_norm, tau_norm]
    Computes BSM delta from price, volatility, and tau for warm start.
    """
    hidden_dims: tuple = (128, 128, 64)
    action_lbnd: float = -2.0
    action_ubnd: float = 2.0
    K: float = 100.0
    r: float = 0.05
    T: float = 1.0
    
    @nn.compact
    def __call__(self, state):
        # Ensure state is 2D
        if state.ndim == 1:
            state = state[None, :]
        
        # CH state: [log_price, momentum, log_fundamental, variance, h_norm, tau_norm]
        log_price = state[:, 0]
        variance = state[:, 3]
        tau_norm = state[:, 5]
        
        # Compute actual price, volatility, and tau
        S = jnp.exp(log_price)
        sigma = jnp.sqrt(jnp.maximum(variance, 1e-8))
        tau = tau_norm * self.T
        tau_safe = jnp.maximum(tau, 1e-8)
        
        # Compute BSM delta (warm start)
        d1 = (jnp.log(S/self.K) + (self.r + 0.5 * sigma**2) * tau_safe) / (sigma * jnp.sqrt(tau_safe))
        norm_cdf = lambda x: 0.5 * (1 + jax.lax.erf(x / jnp.sqrt(2.0)))
        bsm_delta = norm_cdf(d1)
        
        # For SHORT call, use negative delta
        bsm_delta = -bsm_delta
        
        # Learn adjustment using full CH state
        x = state
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        
        adjustment = nn.Dense(1)(x).squeeze(-1)
        
        action = bsm_delta + adjustment
        action = jnp.clip(action, self.action_lbnd, self.action_ubnd)
        
        return action


def create_ch_networks(rng, state_dim, cfg):
    """
    Create CH actor and Double Critic.
    Uses the same DoubleCritic class from agent.py
    """
    from agent import DoubleCritic
    
    actor = ActorCHWithInitDelta(
        K=cfg.K, 
        r=cfg.r, 
        T=cfg.T,
        action_lbnd=cfg.action_lbnd,
        action_ubnd=cfg.action_ubnd
    )
    critic = DoubleCritic()  # ← Double Critic!
    
    dummy_state = jnp.ones((1, state_dim))
    dummy_action = jnp.ones((1,))
    
    rng, init_rng = jax.random.split(rng)
    actor_params = actor.init(init_rng, dummy_state)
    critic_params = critic.init(init_rng, dummy_state, dummy_action)
    
    return actor, critic, actor_params, critic_params