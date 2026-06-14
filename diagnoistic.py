# diagnostic.py
import jax.numpy as jnp
from agent import Actor
from validate_jax import load_actor

actor = Actor()
params = load_actor("checkpoints/final_actor.pkl")

# Test different moneyness values
moneyness_values = jnp.array([0.8, 1.0, 1.2, 1.5, 2.0])
tau = 0.5  # half way to expiry
h_prev = 0.0

for m in moneyness_values:
    state = jnp.array([[m, tau, h_prev]])
    action = actor.apply(params, state)
    print(f"Moneyness {m:.2f} -> Hedge Ratio {action[0]:.3f}")