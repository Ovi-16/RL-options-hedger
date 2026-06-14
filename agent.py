# agent.py - Updated with Init Delta Layer and Entropy

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from config import cfg


class InitDeltaLayer(nn.Module):
    """
    Pre-trained layer that outputs BSM delta.
    This gives the agent a warm start - it already knows the theoretical hedge.
    """
    @nn.compact
    def __call__(self, state):
        # Delta is the 2nd feature in state (index 1)
        # This layer simply extracts and passes it through
        delta = state[:, 1]  # Shape: (batch,)
        return delta


class ActorWithInitDelta(nn.Module):
    """
    Actor that starts from BSM delta and learns adjustments.
    NO internal dropout noise - exploration handled externally in get_actions().
    """
    hidden_dims: tuple = (128, 128, 64)
    
    @nn.compact
    def __call__(self, state):
        # Extract BSM delta from state (warm start) - feature index 1
        bsm_delta = state[:, 1]
        
        # Learn adjustment to delta
        x = state
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        
        # Output adjustment (can be positive or negative)
        adjustment = nn.Dense(1)(x).squeeze(-1)
        
        # Final action = delta + adjustment
        action = bsm_delta + adjustment
        action = jnp.clip(action, cfg.action_lbnd, cfg.action_ubnd)
        
        return action

# Keep original Actor for comparison
class Actor(nn.Module):
    """Original actor without init delta"""
    hidden_dims: tuple = (128, 128, 64)
    
    @nn.compact
    def __call__(self, state):
        x = state
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        action = nn.Dense(1)(x).squeeze(-1)
        return jnp.clip(action, cfg.action_lbnd, cfg.action_ubnd)


class DoubleCritic(nn.Module):
    """Two independent critics for Q-learning"""
    hidden_dims: tuple = (256, 256)
    
    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action[:, None]], axis=-1)
        
        # Q1 network
        x1 = x
        for dim in self.hidden_dims:
            x1 = nn.Dense(dim)(x1)
            x1 = nn.relu(x1)
        q1 = nn.Dense(1)(x1).squeeze(-1)
        
        # Q2 network
        x2 = x
        for dim in self.hidden_dims:
            x2 = nn.Dense(dim)(x2)
            x2 = nn.relu(x2)
        q2 = nn.Dense(1)(x2).squeeze(-1)
        
        return q1, q2


def create_networks(rng, state_dim, use_init_delta=True):
    """Create networks with optional init delta layer"""
    if use_init_delta:
        actor = ActorWithInitDelta()
    else:
        actor = Actor()
    critic = DoubleCritic()
    
    dummy_state = jnp.ones((1, state_dim))
    dummy_action = jnp.ones((1,))
    
    # Need separate RNG for dropout if using entropy
    rng, dropout_rng = jax.random.split(rng)
    actor_params = actor.init({'params': rng, 'dropout': dropout_rng}, dummy_state)
    critic_params = critic.init(rng, dummy_state, dummy_action)
    
    return actor, critic, actor_params, critic_params


def create_optimizers():
    """Create optimizers for actor and critic"""
    actor_opt = optax.adam(learning_rate=cfg.lr_actor)
    critic_opt = optax.adam(learning_rate=cfg.lr_critic)
    return actor_opt, critic_opt
# agent.py - Add this new class for CH agent

class ActorCHWithInitDelta(nn.Module):
    """
    Actor for Chiarella-Heston market.
    State includes: [log_price, momentum, log_fundamental, variance, holdings, tau]
    Computes BSM delta from price, volatility, and tau for warm start.
    """
    hidden_dims: tuple = (128, 128, 64)
    
    @nn.compact
    def __call__(self, state):
        # CH state: [log_price, momentum, log_fundamental, variance, holdings, tau]
        log_price = state[:, 0]
        variance = state[:, 3]
        tau = state[:, 5]
        
        # Compute BSM delta from CH state variables
        S = jnp.exp(log_price)
        sigma = jnp.sqrt(jnp.maximum(variance, 1e-8))
        tau_safe = jnp.maximum(tau, 1e-8)
        
        d1 = (jnp.log(S/self.K) + (self.r + 0.5 * sigma**2) * tau_safe) / (sigma * jnp.sqrt(tau_safe))
        norm_cdf = lambda x: 0.5 * (1 + jax.lax.erf(x / jnp.sqrt(2.0)))
        bsm_delta = norm_cdf(d1)
        
        # For SHORT call (what you're hedging), use negative delta
        bsm_delta = -bsm_delta
        
        # Learn adjustment using full CH state
        x = state
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        
        adjustment = nn.Dense(1)(x).squeeze(-1)
        
        action = bsm_delta + adjustment
        action = jnp.clip(action, cfg.action_lbnd, cfg.action_ubnd)
        
        return action
def create_optimizers(cfg=None):
    """Create optimizers for actor and critic"""
    if cfg is None:
        from config import cfg as default_cfg
        cfg = default_cfg
    actor_opt = optax.adam(learning_rate=cfg.lr_actor)
    critic_opt = optax.adam(learning_rate=cfg.lr_critic)
    return actor_opt, critic_opt