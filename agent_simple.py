# agent_simple.py
import jax
import jax.numpy as jnp
import flax.linen as nn
from config_simple import ConfigSimple

class SimpleActor(nn.Module):
    hidden_dims: tuple = (128, 128, 64)
    @nn.compact
    def __call__(self, state):
        x = state
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        action = nn.Dense(1)(x).squeeze(-1)
        return jnp.clip(action, ConfigSimple.action_lbnd, ConfigSimple.action_ubnd)

class SimpleDoubleCritic(nn.Module):
    hidden_dims: tuple = (256, 256)
    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action[:, None]], axis=-1)
        # Q1
        x1 = x
        for dim in self.hidden_dims:
            x1 = nn.Dense(dim)(x1)
            x1 = nn.relu(x1)
        q1 = nn.Dense(1)(x1).squeeze(-1)
        # Q2
        x2 = x
        for dim in self.hidden_dims:
            x2 = nn.Dense(dim)(x2)
            x2 = nn.relu(x2)
        q2 = nn.Dense(1)(x2).squeeze(-1)
        return q1, q2

def create_simple_networks(rng, state_dim):
    actor = SimpleActor()
    critic = SimpleDoubleCritic()
    dummy_state = jnp.ones((1, state_dim))
    dummy_action = jnp.ones((1,))
    actor_params = actor.init(rng, dummy_state)
    critic_params = critic.init(rng, dummy_state, dummy_action)
    return actor, critic, actor_params, critic_params