# train.py - Fully updated with Init Delta Layer, Entropy, and Risk Penalty

import os
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import optax
from config import cfg
from env import build_state, step_vectorized
from agent import create_networks, create_optimizers
from replay_buffer import ReplayBuffer


def get_actions(actor, actor_params, state, add_noise=False, rng_key=None, noise_scale=0.1):
    """Get actions from actor, optionally adding exploration noise"""
    # Get actions (no internal RNGs needed)
    actions = actor.apply(actor_params, state)
    
    if add_noise and rng_key is not None:
        # Add Gaussian noise for exploration
        noise = jax.random.normal(rng_key, shape=actions.shape) * noise_scale
        actions = actions + noise
        actions = jnp.clip(actions, cfg.action_lbnd, cfg.action_ubnd)
    
    return actions


def simulate_vectorized_batch(rng_key, actor, actor_params, replay_buffer, training=True):
    """
    Vectorized batch simulation (FAST!).
    Returns final costs and updates buffer.
    """
    N_parallel = cfg.N_parallel
    N_steps = cfg.N_steps
    
    # Initialize paths (all parallel)
    S = jnp.full(N_parallel, cfg.S0)
    h = jnp.zeros(N_parallel)
    total_cost = jnp.zeros(N_parallel)
    t = jnp.zeros(N_parallel, dtype=int)
    
    # Pre-generate all random shocks (vectorized)
    eps_key, rng_key = jax.random.split(rng_key)
    epsilons = jax.random.normal(eps_key, shape=(N_steps, N_parallel))
    
    # Storage for transitions
    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []
    
    for step in range(N_steps):
        eps = epsilons[step]
        tau = cfg.T - step * cfg.dt
        tau = jnp.maximum(tau, 0.0)
        
        state = build_state(S, tau, h, cfg.K, cfg.r, cfg.sigma, cfg.T)
        
        if training:
            noise_key, rng_key = jax.random.split(rng_key)
            actions = get_actions(actor, actor_params, state, 
                                  add_noise=True, rng_key=noise_key, noise_scale=0.2)
        else:
            actions = get_actions(actor, actor_params, state, add_noise=False)
        
        carry = (S, h, total_cost, t)
        new_carry, step_info = step_vectorized(carry, eps, actions)
        S, h, total_cost, t = new_carry
        
        # Store for replay buffer
        all_states.append(np.array(state))
        all_actions.append(np.array(actions))
        all_rewards.append(np.array(step_info['reward']))
        all_next_states.append(np.array(step_info['next_state']))
        all_dones.append(np.array(step_info['done']))
    
    # Add all transitions to replay buffer
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
    
    return total_cost, rng_key


def compute_risk_adjusted_loss(final_costs, lambda_risk=cfg.lambda_risk):
    """Compute risk-adjusted loss: E[Cost] + λ * Std(Cost)"""
    mean_cost = jnp.mean(final_costs)
    std_cost = jnp.std(final_costs)
    return mean_cost + lambda_risk * std_cost


def save_checkpoint(actor_params, critic_params, filename):
    """Save model checkpoints"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(f"{filename}_actor.pkl", "wb") as f:
        pickle.dump(actor_params, f)
    with open(f"{filename}_critic.pkl", "wb") as f:
        pickle.dump(critic_params, f)
    print(f"Saved checkpoint to {filename}")


def compute_actor_loss(actor, critic, actor_params, critic_params, s, a_prev=None, use_entropy=True):
    """
    Compute actor loss with entropy bonus and action penalties.
    This prevents the gambling behavior (always -2.0).
    
    Parameters:
    - actor: the actor network object
    - critic: the critic network object  
    - actor_params: parameters for the actor
    - critic_params: parameters for the critic
    - s: current states (batch)
    - a_prev: previous actions (for change penalty)
    - use_entropy: whether to use entropy bonus
    """
    # Get actions from actor
    a_pred = actor.apply(actor_params, s)
    
    # Get Q-values from critic
    q1_pred, _ = critic.apply(critic_params, s, a_pred)
    
    # Standard loss: maximize Q
    standard_loss = -jnp.mean(q1_pred)
    
    if use_entropy:
        # 1. Entropy bonus: reward diverse actions (higher std = more exploration)
        action_std = jnp.std(a_pred)
        entropy_bonus = cfg.entropy_coeff * action_std
        
        # 2. Magnitude penalty: punish extreme actions (gambling prevention)
        magnitude_penalty = cfg.action_penalty * jnp.mean(a_pred**2)
        
        # 3. Bound penalty: encourage actions away from saturation
        bound_penalty = 0.005 * jnp.mean(jnp.tanh(jnp.abs(a_pred)))
        
        # 4. Change penalty: encourage smooth hedging (if previous action available)
        if a_prev is not None:
            change_penalty = 0.01 * jnp.mean((a_pred - a_prev)**2)
        else:
            change_penalty = 0.0
        
        total_loss = standard_loss - entropy_bonus + magnitude_penalty + bound_penalty + change_penalty
    else:
        total_loss = standard_loss
    
    return total_loss
def main():
    print("="*60)
    print("TRAINING DEEP HEDGING AGENT (with Init Delta + Entropy)")
    print("="*60)
    
    rng = jax.random.PRNGKey(42)
    state_dim = cfg.state_dim
    
    print(f"State dimension: {state_dim}")
    print(f"Action bounds: [{cfg.action_lbnd}, {cfg.action_ubnd}]")
    print(f"Parallel paths per batch: {cfg.N_parallel}")
    print(f"Total epochs: {cfg.num_epochs}")
    print(f"Init delta layer: {cfg.use_init_delta}")
    print(f"Entropy coefficient: {cfg.entropy_coeff}")
    print(f"Action penalty: {cfg.action_penalty}")
    print(f"Risk aversion λ: {cfg.lambda_risk}")
    print("="*60)
    
    # Create networks with init delta layer
    rng, init_rng = jax.random.split(rng)
    actor, critic, actor_params, critic_params = create_networks(
        init_rng, state_dim, use_init_delta=cfg.use_init_delta
    )
    
    # Initialize target networks (separate copies - these are the actual network objects)
    target_actor = actor  # Same architecture, separate params
    target_critic = critic  # Same architecture, separate params
    target_actor_params = actor_params
    target_critic_params = critic_params
    
    # Create optimizers
    actor_opt, critic_opt = create_optimizers()
    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)
    
    # Replay buffer
    replay_buffer = ReplayBuffer(cfg.buffer_size, state_dim)
    
    best_loss = float('inf')
    
    for epoch in range(cfg.num_epochs):
        # Collect experience (vectorized)
        rng, subkey = jax.random.split(rng)
        final_costs, rng = simulate_vectorized_batch(
            subkey, actor, actor_params, replay_buffer, training=True
        )
        
        # Current loss for monitoring
        current_loss = compute_risk_adjusted_loss(final_costs)
        mean_cost = jnp.mean(final_costs)
        std_cost = jnp.std(final_costs)
        
        # Update agent if enough samples
        if len(replay_buffer) >= cfg.batch_size:
            for update_step in range(cfg.update_freq):
                # Sample batch
                batch = replay_buffer.sample(cfg.batch_size)
                s, a, r, s_next, done = map(jnp.array, batch)
                
                # --- Critic Update (Double Q-learning) ---
                # Get next actions from target actor
                a_next = target_actor.apply(target_actor_params, s_next)
                
                # Get Q-values from target critic
                q1_next, q2_next = target_critic.apply(target_critic_params, s_next, a_next)
                target_q = r + cfg.gamma * (1.0 - done) * jnp.minimum(q1_next, q2_next)
                
                # Current Q-values
                q1, q2 = critic.apply(critic_params, s, a)
                critic_loss_val = jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
                
                # Update critic
                critic_grads = jax.grad(lambda p: critic_loss_val)(critic_params)
                critic_grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), critic_grads)
                critic_updates, critic_opt_state = critic_opt.update(critic_grads, critic_opt_state)
                critic_params = optax.apply_updates(critic_params, critic_updates)
                
                # --- Actor Update (with entropy and penalties) ---
                actor_loss_val = compute_actor_loss(
                actor,           # Pass the actor network object
                critic,          # Pass the critic network object
                actor_params, 
                critic_params, 
                s, 
                a_prev=a, 
                use_entropy=cfg.use_entropy
            )

                actor_grads = jax.grad(lambda p: actor_loss_val)(actor_params)
                actor_grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), actor_grads)
                actor_updates, actor_opt_state = actor_opt.update(actor_grads, actor_opt_state)
                actor_params = optax.apply_updates(actor_params, actor_updates)
                
                # Soft update target networks
                target_actor_params = jax.tree_util.tree_map(
                    lambda p, tp: cfg.tau * p + (1 - cfg.tau) * tp,
                    actor_params, target_actor_params)
                target_critic_params = jax.tree_util.tree_map(
                    lambda p, tp: cfg.tau * p + (1 - cfg.tau) * tp,
                    critic_params, target_critic_params)
        
        # Logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {current_loss:.2f}, Mean Cost = {mean_cost:.2f}, Std = {std_cost:.2f}, Buffer = {len(replay_buffer)}")
        
        # Save checkpoint
        if epoch % cfg.save_every == 0 and epoch > 0:
            save_checkpoint(actor_params, critic_params, f"{cfg.checkpoint_dir}/epoch_{epoch}")
        
        if float(current_loss) < best_loss:
            best_loss = float(current_loss)
            save_checkpoint(actor_params, critic_params, f"{cfg.checkpoint_dir}/best")
    
    # Final checkpoint
    save_checkpoint(actor_params, critic_params, f"{cfg.checkpoint_dir}/final")
    print("\n" + "="*60)
    print(f"TRAINING COMPLETE. Best loss: {best_loss:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()