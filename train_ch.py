
import os
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt   # NEW: for plotting
from config_ch import cfg_ch
from replay_buffer import ReplayBuffer
from chiarella_heston_env import simulate_vectorized_batch_ch
from agent_ch import create_ch_networks
from chiarella_heston_env import get_initial_carry_ch, build_state_ch, ch_step, get_actions

# NEW: Helper function to simulate a few paths without storing to buffer
def simulate_sample_paths(actor, actor_params, cfg, rng_key, n_paths=5):
    """
    Simulate a small number of paths (no exploration noise, no buffer).
    Returns: list of price paths, hedge positions, and actions for plotting.
    """
    carry = get_initial_carry_ch(n_paths, cfg.S0)
    all_prices = []
    all_actions = []
    all_hedges = []
    state_dict, h, total_cost, t = carry
    rng = rng_key

    for step in range(cfg.N_steps):
        tau = cfg.T - step * cfg.dt
        tau = jnp.maximum(tau, 0.0)
        state = build_state_ch(state_dict, tau, h, cfg.K, cfg.r, cfg.sigma, cfg.T, cfg.action_ubnd)
        actions = get_actions(actor, actor_params, state, add_noise=False )
        # Store
        all_prices.append(np.array(state_dict['price']))
        all_hedges.append(np.array(h))
        all_actions.append(np.array(actions))
        # Step
        new_carry, step_info, rng = ch_step(state_dict, h, total_cost, t, actions, rng, cfg)
        state_dict, h, total_cost, t = new_carry
    # Add final price
    all_prices.append(np.array(state_dict['price']))
    return all_prices, all_actions, all_hedges

# NEW: Plotting function
def plot_epoch_summary(epoch, actor, actor_params, cfg, rng_key, save_dir="training_plots"):
    """
    Generate and save:
    - Hedge position histogram (across all steps and paths)
    - Sample price paths
    - Hedge positions over time for a few paths
    """
    os.makedirs(save_dir, exist_ok=True)
    n_paths = 10   # simulate 10 paths for statistics
    prices, actions, hedges = simulate_sample_paths(actor, actor_params, cfg, rng_key, n_paths)
    # Convert to numpy arrays: shapes (steps+1, n_paths) and (steps, n_paths)
    prices_arr = np.array(prices)          # (steps+1, n_paths)
    hedges_arr = np.array(hedges)          # (steps, n_paths)
    actions_arr = np.array(actions)        # (steps, n_paths)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # 1. Hedge position histogram (all steps, all paths)
    all_hedges_flat = hedges_arr.flatten()
    axes[0,0].hist(all_hedges_flat, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0,0].axvline(x=0, color='red', linestyle='--')
    axes[0,0].set_title(f'Epoch {epoch}: Hedge Position Distribution\nmean={np.mean(all_hedges_flat):.2f}, std={np.std(all_hedges_flat):.2f}')
    axes[0,0].set_xlabel('Hedge Position (shares)')
    axes[0,0].set_ylabel('Frequency')
    
    # 2. Sample price paths (first 3 paths)
    for i in range(min(3, n_paths)):
        axes[0,1].plot(prices_arr[:, i], label=f'Path {i+1}')
    axes[0,1].set_title('Sample Price Paths')
    axes[0,1].set_xlabel('Time step')
    axes[0,1].set_ylabel('Price ($)')
    axes[0,1].legend()
    
    # 3. Hedge positions over time for first 3 paths
    for i in range(min(3, n_paths)):
        axes[1,0].plot(hedges_arr[:, i], label=f'Path {i+1}')
    axes[1,0].set_title('Hedge Positions Over Time')
    axes[1,0].set_xlabel('Time step')
    axes[1,0].set_ylabel('Shares')
    axes[1,0].legend()
    
    # 4. Actions (same as hedges in this environment, but we plot separately)
    for i in range(min(3, n_paths)):
        axes[1,1].plot(actions_arr[:, i], label=f'Path {i+1}')
    axes[1,1].set_title('Actions (Hedge Changes)')
    axes[1,1].set_xlabel('Time step')
    axes[1,1].set_ylabel('Action')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch:04d}.png", dpi=150)
    plt.close()
    print(f"Saved training plot to {save_dir}/epoch_{epoch:04d}.png")
def compute_theoretical_min_cost(cfg):
    """
    Compute theoretical minimum possible cost for benchmarking.
    This helps determine if your model is learning well.
    """
    # Unhedged strategy (delta=0 always)
    # Expected cost = |payoff| at maturity (on average)
    # For ATM call, expected payoff ≈ 0.4 * S0 = 40
    unhedged_cost_estimate = 40.0  # Rough estimate for ATM call
    
    # Perfect hedging (no hedging error, only transaction costs)
    # Each step: you might need to adjust by ~0.5 delta on average
    # Transaction cost per adjustment = kappa * |delta_change| * price
    avg_delta_change = 0.5  # Rough estimate
    avg_price = cfg.S0  # 100
    cost_per_step = cfg.kappa * avg_delta_change * avg_price  # 0.02 * 0.5 * 100 = 1.0
    perfect_hedge_min = cost_per_step * cfg.N_steps  # 1.0 * 50 = 50
    
    # Absolute theoretical lower bound (no transaction costs, perfect hedge)
    # This is essentially zero, but with costs it's >0
    absolute_lower_bound = 0.0
    
    print("="*60)
    print("THEORETICAL COST BENCHMARKS:")
    print("="*60)
    print(f"Unhedged strategy (do nothing):      ~${unhedged_cost_estimate:.2f}")
    print(f"Perfect hedging (transaction only):   ~${perfect_hedge_min:.2f}")
    print(f"Absolute lower bound (no costs):      ${absolute_lower_bound:.2f}")
    print("="*60)
    
    # FIX: Check if current_mean_cost exists and is not None
    if hasattr(cfg, 'current_mean_cost') and cfg.current_mean_cost is not None:
        print(f"Your current cost:                    ~${cfg.current_mean_cost:.2f}")
    else:
        print(f"Your current cost:                    ~Not yet available")
    
    print("\nInterpretation:")
    print(f"- If cost > {unhedged_cost_estimate:.0f}: WORSE than doing nothing")
    print(f"- If cost < {perfect_hedge_min:.0f}: BETTER than perfect hedge (impossible!)")
    print(f"- Good range: ${perfect_hedge_min:.0f} to ${unhedged_cost_estimate:.0f}")
    print("="*60)
    
    return perfect_hedge_min, unhedged_cost_estimate


def evaluate_random_policy(cfg, rng_key, num_episodes=10):
    """
    Evaluate a random policy to establish baseline.
    This shows if the environment itself is reasonable.
    """
    print("\nEvaluating RANDOM policy baseline...")
    
    # Create random network (or just use random actions)
    total_costs = []
    
    for episode in range(num_episodes):
        rng_key, subkey = jax.random.split(rng_key)
        
        # Generate one batch with random actions
        # (You'll need to modify simulate_vectorized_batch_ch to accept random actions)
        # For now, just print estimate
        pass
    
    print(f"Random policy expected cost: ~${50:.2f} (estimate)")
    return 50.0

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


def soft_update(target_params, source_params, tau):
    """Soft update target networks: target = tau * source + (1 - tau) * target"""
    return jax.tree_util.tree_map(
        lambda t, s: tau * s + (1 - tau) * t,
        target_params, source_params
    )


def main():
    print("="*60)
    print("TRAINING CHIARELLA-HESTON AGENT (TRUE TD3 with Delayed Updates)")
    print("="*60)
    
    cfg = cfg_ch
    state_dim = cfg.state_dim
    checkpoint_dir = "checkpoints/ch_td3_2"

    cfg.current_mean_cost = None  # Will be updated in training loop
    
    # Run diagnostics
    print("\n" + "="*60)
    print("PRE-TRAINING DIAGNOSTICS")
    print("="*60)
    
    # Calculate theoretical bounds
    perfect_min, unhedged_cost = compute_theoretical_min_cost(cfg)
    
    # Evaluate random policy (optional - requires modification)
    # random_cost = evaluate_random_policy(cfg, rng)
    
    print(f"\nTarget: Get Mean Cost below ${unhedged_cost:.0f}")
    print(f"Goal: ${perfect_min:.0f} - ${unhedged_cost:.0f} is realistic range")


    print(f"State dimension: {state_dim}")
    print(f"Action bounds: [{cfg.action_lbnd}, {cfg.action_ubnd}]")
    print(f"Parallel paths per batch: {cfg.N_parallel}")
    print(f"Total epochs: {cfg.num_epochs}")
    print(f"TD3 Delayed Update: Actor updates every {cfg.actor_update_freq} critic updates")
    print(f"Target update frequency: Every {cfg.target_update_freq} actor updates")
    print(f"Critic updates per epoch: {cfg.critic_updates_per_epoch}")
    print(f"Target noise scale: {cfg.target_noise_scale}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print("="*60)
    
    rng = jax.random.PRNGKey(42)
    
    # ============================================================
    # STEP 1: Create BOTH current and target network OBJECTS
    # ============================================================
    # Create current networks
    rng, init_rng = jax.random.split(rng)
    actor, critic, actor_params, critic_params = create_ch_networks(init_rng, state_dim, cfg)
    
    # Create TARGET networks (separate network objects with same architecture)
    # We need to re-initialize with different RNG to get separate objects
    rng, target_init_rng = jax.random.split(rng)
    target_actor, target_critic, target_actor_params, target_critic_params = create_ch_networks(
        target_init_rng, state_dim, cfg
    )
    
    # Initialize target parameters to match current parameters (essential for TD3)
    target_actor_params = actor_params  # Start identical
    target_critic_params = critic_params  # Start identical
    
    # Create optimizers
    actor_opt, critic_opt = create_optimizers(cfg)
    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)
    
    # Replay buffer
    replay_buffer = ReplayBuffer(cfg.buffer_size, state_dim)
    best_loss = float('inf')
    
    # Tracking metrics
    critic_update_counter = 0
    actor_update_counter = 0
    exploration_noise_scale = cfg.exploration_noise_scale
    for epoch in range(cfg.num_epochs):
        # ============================================================
        # Data collection with DECAYING exploration noise
        # ============================================================
        rng, subkey = jax.random.split(rng)
        
        # Pass exploration noise scale to simulation
        final_costs, rng = simulate_vectorized_batch_ch(
            subkey, actor, actor_params, replay_buffer, cfg, 
            training=True,
            exploration_noise_scale=exploration_noise_scale  # Add this
        )
        
        # Decay exploration noise
        exploration_noise_scale *= cfg.exploration_noise_decay
        exploration_noise_scale = max(exploration_noise_scale, 0.05) 
        
        # Log current performance
        current_loss = compute_risk_adjusted_loss(final_costs, cfg.lambda_risk)
        mean_cost = jnp.mean(final_costs)
        std_cost = jnp.std(final_costs)
        
        # ============================================================
        # STEP 3: Perform MULTIPLE CRITIC UPDATES (TD3 core)
        # ============================================================
        if len(replay_buffer) >= cfg.batch_size:
            for update_step in range(cfg.critic_updates_per_epoch):
                # Sample random batch from replay buffer
                batch = replay_buffer.sample(cfg.batch_size)
                s, a, r, s_next, done = map(jnp.array, batch)
                
                # Split RNG for this update
                rng, noise_key = jax.random.split(rng)
                
                # --- TD3 STEP 1: ADD TARGET POLICY NOISE ---
                # Smoothing regularization (TD3 trick #2)
                # NOW using target_actor (the network object) and target_actor_params
                a_next = target_actor.apply(target_actor_params, s_next)
                noise = jax.random.normal(noise_key, shape=a_next.shape) * cfg.target_noise_scale
                noise = jnp.clip(noise, -cfg.target_noise_clip, cfg.target_noise_clip)
                a_next = a_next + noise
                a_next = jnp.clip(a_next, cfg.action_lbnd, cfg.action_ubnd)
                
                # --- CRITIC UPDATE (Always do this) ---
                # Get Q-values from target critic (using target_critic network object)
                q1_next, q2_next = target_critic.apply(target_critic_params, s_next, a_next)

                #TODO:  removed values below 

                c=-r

                target1 = c + cfg.gamma * (1 - done) * q1_next

                target2 = c**2 + 2 * cfg.gamma * c * q1_next + (cfg.gamma**2) * (1 - done) * q2_next
                
                q1, q2 = critic.apply(critic_params, s, a)
                critic_loss_val = jnp.mean((q1 - target1)**2 + (q2 - target2)**2)
                
                # Current Q-values
                # target_q = r + cfg.gamma * (1.0 - done) * jnp.minimum(q1_next, q2_next)
                # q1, q2 = critic.apply(critic_params, s, a)
                # critic_loss_val = jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
                
                # Update critic
                critic_grads = jax.grad(lambda p: critic_loss_val)(critic_params)
                critic_grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), critic_grads)
                critic_updates, critic_opt_state = critic_opt.update(critic_grads, critic_opt_state)
                critic_params = optax.apply_updates(critic_params, critic_updates)
                
                critic_update_counter += 1
                
                # ============================================================
                # STEP 4: DELAYED ACTOR UPDATE (TD3 trick #1)
                # Update actor ONLY every actor_update_freq critic updates
                # ============================================================
                #TODO:changes made in the actor_loss
                if critic_update_counter % cfg.actor_update_freq == 0:
                    def actor_loss(p):
                        a_pred = actor.apply(p, s)
                        q1_pred, q2_pred = critic.apply(critic_params, s, a_pred)
                        mean_cost = q1_pred
                        var_cost = jnp.maximum(q2_pred - mean_cost**2, 1e-8)
                        std_cost = jnp.sqrt(var_cost)
                        risk_adjusted = mean_cost + cfg.lambda_risk * std_cost
                        return jnp.mean(risk_adjusted)   # <-- scalar
                    # --- ACTOR UPDATE ---
                    # def actor_loss(p):
                    #     a_pred = actor.apply(p, s)
                    #     q1_pred, _ = critic.apply(critic_params, s, a_pred)
                        
                    #     # Action penalty to prevent extreme actions
                    #     action_penalty = cfg.action_penalty * jnp.mean(a_pred**2)
                        
                    #     # TD3: Maximize Q (negative loss) with penalty
                    #     return -jnp.mean(q1_pred) + action_penalty
                    
                    actor_grads = jax.grad(actor_loss)(actor_params)
                    actor_grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), actor_grads)
                    actor_updates, actor_opt_state = actor_opt.update(actor_grads, actor_opt_state)
                    actor_params = optax.apply_updates(actor_params, actor_updates)
                    
                    actor_update_counter += 1
                    
                    # ============================================================
                    # STEP 5: SOFT UPDATE TARGET NETWORKS
                    # Update BOTH target_actor_params AND target_critic_params
                    # ============================================================
                    if actor_update_counter % cfg.target_update_freq == 0:
                        target_actor_params = soft_update(target_actor_params, actor_params, cfg.tau_soft)
                        target_critic_params = soft_update(target_critic_params, critic_params, cfg.tau_soft)
        
        # ============================================================
        # LOGGING AND CHECKPOINTING
        # ============================================================
        
          # FIX: Only calculate progress if perfect_min < unhedged_cost
        if epoch % 10 == 0:
            plot_rng = jax.random.PRNGKey(epoch * 12345)
            plot_epoch_summary(epoch, actor, actor_params, cfg, plot_rng)
            if perfect_min < unhedged_cost:
                progress = (unhedged_cost - float(mean_cost)) / (unhedged_cost - perfect_min) * 100
                progress = max(0, min(100, progress))  # Clamp between 0-100
                print(f"Epoch {epoch:3d}: Loss = {current_loss:.2f}, "
                    f"Mean Cost = {mean_cost:.2f}, Std = {std_cost:.2f}, "
                    f"Buffer = {len(replay_buffer)}, "
                    f"Progress = {progress:.1f}% to target")
            else:
                # If bounds are inverted, just print basic info
                print(f"Epoch {epoch:3d}: Loss = {current_loss:.2f}, "
                    f"Mean Cost = {mean_cost:.2f}, Std = {std_cost:.2f}, "
                    f"Buffer = {len(replay_buffer)}")
            
                # Your existing detailed print
            print(f"Epoch {epoch:3d}: Loss = {current_loss:.2f}, "
                f"Mean Cost = {mean_cost:.2f}, Std = {std_cost:.2f}, "
                f"Buffer = {len(replay_buffer)}, "
                f"Critic Updates = {critic_update_counter}, "
                f"Actor Updates = {actor_update_counter}")
        # Save checkpoint
        if epoch % cfg.save_every == 0 and epoch > 0:
            save_checkpoint(actor_params, critic_params, f"{checkpoint_dir}/epoch_{epoch}")
        
        if float(current_loss) < best_loss:
            best_loss = float(current_loss)
            save_checkpoint(actor_params, critic_params, f"{checkpoint_dir}/best")
    
    # Final checkpoint
    save_checkpoint(actor_params, critic_params, f"{checkpoint_dir}/final")
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Total critic updates: {critic_update_counter}")
    print(f"Total actor updates: {actor_update_counter}")
    print(f"Actor/Critic update ratio: {actor_update_counter/critic_update_counter:.3f}")


if __name__ == "__main__":
    main()