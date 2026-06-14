# train_ch.py - TD3-style training with multiple updates per epoch

import os
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import optax
from config_ch import cfg_ch
from replay_buffer import ReplayBuffer
from chiarella_heston_env import simulate_vectorized_batch_ch
from agent_ch import create_ch_networks


def get_actions(actor, actor_params, state, add_noise=False, rng_key=None, noise_scale=0.1):
    """Get actions from actor"""
    actions = actor.apply(actor_params, state)
    if add_noise and rng_key is not None:
        noise = jax.random.normal(rng_key, shape=actions.shape) * noise_scale
        actions = actions + noise
        actions = jnp.clip(actions, cfg_ch.action_lbnd, cfg_ch.action_ubnd)
    return actions


def compute_risk_adjusted_loss(final_costs, lambda_risk=cfg_ch.lambda_risk):
    mean_cost = jnp.mean(final_costs)
    std_cost = jnp.std(final_costs)
    return mean_cost + lambda_risk * std_cost


def save_checkpoint(actor_params, critic_params, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(f"{filename}_actor.pkl", "wb") as f:
        pickle.dump(actor_params, f)
    with open(f"{filename}_critic.pkl", "wb") as f:
        pickle.dump(critic_params, f)
    print(f"Saved checkpoint to {filename}")


def create_optimizers(cfg):
    """Create optimizers for actor and critic"""
    actor_opt = optax.adam(learning_rate=cfg.lr_actor)
    critic_opt = optax.adam(learning_rate=cfg.lr_critic)
    return actor_opt, critic_opt


def apply_update(params, grads, opt, opt_state):
    """Apply gradient update to parameters"""
    updates, new_opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


def soft_update(target_params, source_params, tau):
    """Soft update target networks: target = tau * source + (1 - tau) * target"""
    return jax.tree_util.tree_map(
        lambda tp, sp: tau * sp + (1 - tau) * tp,
        target_params, source_params
    )


def main():
    print("="*60)
    print("TRAINING CHIARELLA-HESTON AGENT (TD3-style)")
    print("="*60)
    
    cfg = cfg_ch
    state_dim = cfg.state_dim
    checkpoint_dir = "checkpoints/ch"
    
    print(f"State dimension: {state_dim}")
    print(f"Action bounds: [{cfg.action_lbnd}, {cfg.action_ubnd}]")
    print(f"Parallel paths per batch: {cfg.N_parallel}")
    print(f"Steps per path: {cfg.N_steps}")
    print(f"Total steps per epoch: {cfg.N_parallel * cfg.N_steps}")
    print(f"Buffer size: {cfg.buffer_size}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning starts: {cfg.learning_starts}")
    print(f"Gradient steps: {cfg.gradient_steps} ({'as many as steps collected / batch_size' if cfg.gradient_steps == -1 else cfg.gradient_steps})")
    print(f"Policy delay: {cfg.policy_delay}")
    print(f"Total epochs: {cfg.num_epochs}")
    print("="*60)
    
    rng = jax.random.PRNGKey(42)
    
    # Create CH networks with Double Critic
    rng, init_rng = jax.random.split(rng)
    actor, critic, actor_params, critic_params = create_ch_networks(init_rng, state_dim, cfg)
    
    # Target networks (same architecture, separate params)
    target_actor_params = actor_params
    target_critic_params = critic_params
    
    # Create optimizers
    actor_opt, critic_opt = create_optimizers(cfg)
    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)
    
    # Replay buffer
    replay_buffer = ReplayBuffer(cfg.buffer_size, state_dim)
    best_loss = float('inf')
    
    for epoch in range(cfg.num_epochs):
        rng, subkey = jax.random.split(rng)
        
        # 1. COLLECT EXPERIENCE (ONE EPOCH OF DATA)
        final_costs, rng = simulate_vectorized_batch_ch(
            subkey, actor, actor_params, replay_buffer, cfg, training=True
        )
        
        current_loss = compute_risk_adjusted_loss(final_costs, cfg.lambda_risk)
        mean_cost = jnp.mean(final_costs)
        std_cost = jnp.std(final_costs)
        
        # 2. DETERMINE NUMBER OF GRADIENT UPDATES
        n_updates = 0
        if len(replay_buffer) >= cfg.learning_starts:
            if cfg.gradient_steps == -1:
                # Use as many updates as steps collected ÷ batch_size
                steps_this_epoch = cfg.N_parallel * cfg.N_steps
                n_updates = max(1, steps_this_epoch // cfg.batch_size)
            else:
                n_updates = cfg.gradient_steps
            
            # 3. PERFORM ALL UPDATES FOR THIS EPOCH
            for update_step in range(n_updates):
                # Sample batch
                batch = replay_buffer.sample(cfg.batch_size)
                s, a, r, s_next, done = map(jnp.array, batch)
                
                # --- Critic Update (always update critics) ---
                # Get next actions from target actor
                a_next = actor.apply(target_actor_params, s_next)
                
                # Add target policy noise (TD3 trick for smoother targets)
                rng, noise_key = jax.random.split(rng)
                noise = jax.random.normal(noise_key, shape=a_next.shape) * 0.2
                a_next = a_next + noise
                a_next = jnp.clip(a_next, cfg.action_lbnd, cfg.action_ubnd)
                
                # Compute target Q values
                q1_next, q2_next = critic.apply(target_critic_params, s_next, a_next)
                target_q = r + cfg.gamma * (1.0 - done) * jnp.minimum(q1_next, q2_next)
                
                # Current Q values
                q1, q2 = critic.apply(critic_params, s, a)
                critic_loss_val = jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
                
                # Update critic
                critic_grads = jax.grad(lambda p: critic_loss_val)(critic_params)
                critic_grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), critic_grads)
                critic_params, critic_opt_state = apply_update(critic_params, critic_grads, critic_opt, critic_opt_state)
                
                # --- Actor Update (only every policy_delay updates) ---
                if update_step % cfg.policy_delay == 0:
                    def actor_loss(p):
                        a_pred = actor.apply(p, s)
                        q1_pred, _ = critic.apply(critic_params, s, a_pred)
                        action_penalty = cfg.action_penalty * jnp.mean(a_pred**2)
                        return -jnp.mean(q1_pred) + action_penalty
                    
                    actor_grads = jax.grad(actor_loss)(actor_params)
                    actor_grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), actor_grads)
                    actor_params, actor_opt_state = apply_update(actor_params, actor_grads, actor_opt, actor_opt_state)
                    
                    # Soft update target networks
                    target_actor_params = soft_update(target_actor_params, actor_params, cfg.tau_soft)
                    target_critic_params = soft_update(target_critic_params, critic_params, cfg.tau_soft)
        
        # Logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {current_loss:.2f}, Mean Cost = {mean_cost:.2f}, Std = {std_cost:.2f}, Buffer = {len(replay_buffer)}")
            print(f"  Updates this epoch: {n_updates}")
        
        # Save checkpoint
        if epoch % cfg.save_every == 0 and epoch > 0:
            save_checkpoint(actor_params, critic_params, f"{checkpoint_dir}/epoch_{epoch}")
        
        if float(current_loss) < best_loss:
            best_loss = float(current_loss)
            save_checkpoint(actor_params, critic_params, f"{checkpoint_dir}/best")
    
    save_checkpoint(actor_params, critic_params, f"{checkpoint_dir}/final")
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()