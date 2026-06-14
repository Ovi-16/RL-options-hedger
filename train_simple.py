# train_simple.py
import os
import pickle
import jax
import jax.numpy as jnp
import optax
from config_simple import ConfigSimple
from replay_buffer import ReplayBuffer
from chiarella_heston_simple_env import simulate_vectorized_batch_simple
from agent_simple import create_simple_networks, SimpleActor, SimpleDoubleCritic

cfg = ConfigSimple()
state_dim = 3
checkpoint_dir = "checkpoints/simple_0.25"

# Create checkpoint directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)

def create_optimizers():
    return optax.adam(cfg.lr_actor), optax.adam(cfg.lr_critic)

def soft_update(target_params, source_params, tau):
    return jax.tree_util.tree_map(lambda t, s: tau * s + (1 - tau) * t, target_params, source_params)

def main():
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    actor, critic, actor_params, critic_params = create_simple_networks(init_rng, state_dim)

    # Create target networks (separate instances)
    target_actor = SimpleActor()
    target_critic = SimpleDoubleCritic()
    # Initialize them with the same parameters as current networks
    target_actor_params = actor_params
    target_critic_params = critic_params

    actor_opt, critic_opt = create_optimizers()
    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)

    replay_buffer = ReplayBuffer(cfg.buffer_size, state_dim)
    best_loss = float('inf')
    critic_update_counter = 0
    actor_update_counter = 0
    exploration_noise_scale = cfg.exploration_noise_scale

    for epoch in range(cfg.num_epochs):
        rng, subkey = jax.random.split(rng)
        final_costs, rng = simulate_vectorized_batch_simple(
            subkey, actor, actor_params, replay_buffer, cfg,
            training=True, exploration_noise_scale=exploration_noise_scale
        )
        exploration_noise_scale *= cfg.exploration_noise_decay
        exploration_noise_scale = max(exploration_noise_scale, 0.05)

        current_loss = jnp.mean(final_costs) + cfg.lambda_risk * jnp.std(final_costs)
        mean_cost = jnp.mean(final_costs)

        if len(replay_buffer) >= cfg.batch_size:
            for _ in range(cfg.critic_updates_per_epoch):
                batch = replay_buffer.sample(cfg.batch_size)
                s, a, r, s_next, done = map(jnp.array, batch)
                rng, noise_key = jax.random.split(rng)

                # Target action with smoothing
                a_next = target_actor.apply(target_actor_params, s_next)
                noise = jax.random.normal(noise_key, a_next.shape) * cfg.target_noise_scale
                noise = jnp.clip(noise, -cfg.target_noise_clip, cfg.target_noise_clip)
                a_next = a_next + noise
                a_next = jnp.clip(a_next, cfg.action_lbnd, cfg.action_ubnd)

                q1_next, q2_next = target_critic.apply(target_critic_params, s_next, a_next)

                c = -r
                target1 = c + cfg.gamma_discount * (1 - done) * q1_next
                target2 = c**2 + 2 * cfg.gamma_discount * c * q1_next + (cfg.gamma_discount**2) * (1 - done) * q2_next

                q1, q2 = critic.apply(critic_params, s, a)
                critic_loss = jnp.mean((q1 - target1)**2 + (q2 - target2)**2)

                critic_grads = jax.grad(lambda p: critic_loss)(critic_params)
                critic_grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), critic_grads)
                critic_updates, critic_opt_state = critic_opt.update(critic_grads, critic_opt_state)
                critic_params = optax.apply_updates(critic_params, critic_updates)
                critic_update_counter += 1

                if critic_update_counter % cfg.actor_update_freq == 0:
                    def actor_loss(p):
                        a_pred = actor.apply(p, s)
                        q1_pred, q2_pred = critic.apply(critic_params, s, a_pred)
                        mean_c = q1_pred
                        var_c = jnp.maximum(q2_pred - mean_c**2, 1e-8)
                        std_c = jnp.sqrt(var_c)
                        return jnp.mean(mean_c + cfg.lambda_risk * std_c)

                    actor_grads = jax.grad(actor_loss)(actor_params)
                    actor_grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), actor_grads)
                    actor_updates, actor_opt_state = actor_opt.update(actor_grads, actor_opt_state)
                    actor_params = optax.apply_updates(actor_params, actor_updates)
                    actor_update_counter += 1

                    if actor_update_counter % cfg.target_update_freq == 0:
                        target_actor_params = soft_update(target_actor_params, actor_params, cfg.tau_soft)
                        target_critic_params = soft_update(target_critic_params, critic_params, cfg.tau_soft)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: cost={mean_cost:.2f}, loss={current_loss:.2f}, buffer={len(replay_buffer)}")
        if epoch % cfg.save_every == 0 and epoch > 0:
            with open(f"{checkpoint_dir}/epoch_{epoch}_actor.pkl", "wb") as f:
                pickle.dump(actor_params, f)

        if current_loss < best_loss:
            best_loss = current_loss
            with open(f"{checkpoint_dir}/best_actor.pkl", "wb") as f:
                pickle.dump(actor_params, f)

    with open(f"{checkpoint_dir}/final_actor.pkl", "wb") as f:
        pickle.dump(actor_params, f)
    print("Training complete.")

if __name__ == "__main__":
    main()