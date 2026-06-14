# train.py - TD3 for GBM environment with risk-adjusted cost objective

import os
import pickle
import jax
import jax.numpy as jnp
import optax
from config import cfg
from env import build_state, step_vectorized
from agent import create_networks
from replay_buffer import ReplayBuffer

def get_actions(actor, actor_params, state, add_noise=False, rng_key=None, noise_scale=0.1):
    actions = actor.apply(actor_params, state)
    if add_noise and rng_key is not None:
        noise = jax.random.normal(rng_key, shape=actions.shape) * noise_scale
        actions = actions + noise
        actions = jnp.clip(actions, cfg.action_lbnd, cfg.action_ubnd)
    return actions

def simulate_vectorized_batch(rng_key, actor, actor_params, replay_buffer, training=True):
    """Simulate N_parallel paths, collect transitions, return final costs."""
    N_parallel = cfg.N_parallel
    N_steps = cfg.N_steps
    S = jnp.full(N_parallel, cfg.S0)
    h = jnp.zeros(N_parallel)
    total_cost = jnp.zeros(N_parallel)
    t = jnp.zeros(N_parallel, dtype=int)

    # Pre‑generate shocks
    eps_key, rng_key = jax.random.split(rng_key)
    epsilons = jax.random.normal(eps_key, shape=(N_steps, N_parallel))

    all_states, all_actions, all_rewards, all_next_states, all_dones = [], [], [], [], []

    for step in range(N_steps):
        eps = epsilons[step]
        tau = cfg.T - step * cfg.dt
        tau = jnp.maximum(tau, 0.0)
        state = build_state(S, tau, h, cfg.K, cfg.r, cfg.sigma, cfg.T)

        if training:
            noise_key, rng_key = jax.random.split(rng_key)
            actions = get_actions(actor, actor_params, state,
                                  add_noise=True, rng_key=noise_key,
                                  noise_scale=cfg.exploration_noise_scale)
        else:
            actions = get_actions(actor, actor_params, state, add_noise=False)

        carry = (S, h, total_cost, t)
        new_carry, step_info = step_vectorized(carry, eps, actions)
        S, h, total_cost, t = new_carry

        all_states.append(jnp.array(state))
        all_actions.append(jnp.array(actions))
        all_rewards.append(jnp.array(step_info['reward']))   # raw P&L
        all_next_states.append(jnp.array(step_info['next_state']))
        all_dones.append(jnp.array(step_info['done']))

    if training:
        for step in range(N_steps):
            for i in range(N_parallel):
                replay_buffer.push(
                    jnp.array(all_states[step][i]),
                    jnp.array(all_actions[step][i]),
                    jnp.array(all_rewards[step][i]),
                    jnp.array(all_next_states[step][i]),
                    jnp.array(all_dones[step][i])
                )
    return total_cost, rng_key

def soft_update(target_params, source_params, tau):
    return jax.tree_util.tree_map(lambda t, s: tau * s + (1 - tau) * t, target_params, source_params)

def main():
    print("="*60)
    print("TD3 TRAINING FOR GBM HEDGING (risk‑adjusted cost objective)")
    print("="*60)
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    actor, critic, actor_params, critic_params = create_networks(init_rng, cfg.state_dim, use_init_delta=cfg.use_init_delta)

    # Target networks (separate instances)
    target_actor = type(actor)()
    target_critic = type(critic)()
    target_actor_params = actor_params
    target_critic_params = critic_params

    actor_opt = optax.adam(cfg.lr_actor)
    critic_opt = optax.adam(cfg.lr_critic)
    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)

    replay_buffer = ReplayBuffer(cfg.buffer_size, cfg.state_dim)
    best_loss = float('inf')
    critic_update_counter = 0
    actor_update_counter = 0
    exploration_noise_scale = cfg.exploration_noise_scale

    for epoch in range(cfg.num_epochs):
        rng, subkey = jax.random.split(rng)
        final_costs, rng = simulate_vectorized_batch(
            subkey, actor, actor_params, replay_buffer, training=True
        )
        # Decay exploration noise
        exploration_noise_scale *= cfg.exploration_noise_decay
        exploration_noise_scale = max(exploration_noise_scale, 0.05)

        current_loss = jnp.mean(final_costs) + cfg.lambda_risk * jnp.std(final_costs)
        mean_cost = jnp.mean(final_costs)

        if len(replay_buffer) >= cfg.batch_size:
            for _ in range(cfg.critic_updates_per_epoch):
                batch = replay_buffer.sample(cfg.batch_size)
                s, a, r, s_next, done = map(jnp.array, batch)
                rng, noise_key = jax.random.split(rng)

                # ----- target action with smoothing (TD3 trick) -----
                a_next = target_actor.apply(target_actor_params, s_next)
                noise = jax.random.normal(noise_key, shape=a_next.shape) * cfg.target_noise_scale
                noise = jnp.clip(noise, -cfg.target_noise_clip, cfg.target_noise_clip)
                a_next = a_next + noise
                a_next = jnp.clip(a_next, cfg.action_lbnd, cfg.action_ubnd)

                # ----- target Q values -----
                q1_next, q2_next = target_critic.apply(target_critic_params, s_next, a_next)
                # convert reward to cost: c = -r
                c = -r
                target_q = c + cfg.gamma * (1.0 - done) * jnp.minimum(q1_next, q2_next)

                # ----- critic loss (MSE) -----
                q1, q2 = critic.apply(critic_params, s, a)
                critic_loss = jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)

                critic_grads = jax.grad(lambda p: critic_loss)(critic_params)
                critic_grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), critic_grads)
                critic_updates, critic_opt_state = critic_opt.update(critic_grads, critic_opt_state)
                critic_params = optax.apply_updates(critic_params, critic_updates)
                critic_update_counter += 1

                # ----- delayed actor update -----
                if critic_update_counter % cfg.actor_update_freq == 0:
                    def actor_loss(p):
                        a_pred = actor.apply(p, s)
                        q1_pred, _ = critic.apply(critic_params, s, a_pred)
                        # We want to minimise cost = -reward, so we maximise Q (which is cost)
                        # But our critic now learns expected total cost (positive = bad).
                        # Actually critic learns target_q = c + gamma * ... where c = -reward.
                        # So higher Q means higher cost → we want to minimise it.
                        # Thus actor loss = +jnp.mean(q1_pred)  (minimise cost)
                        return jnp.mean(q1_pred)

                    actor_grads = jax.grad(actor_loss)(actor_params)
                    actor_grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), actor_grads)
                    actor_updates, actor_opt_state = actor_opt.update(actor_grads, actor_opt_state)
                    actor_params = optax.apply_updates(actor_params, actor_updates)
                    actor_update_counter += 1

                    # soft update target networks
                    target_actor_params = soft_update(target_actor_params, actor_params, cfg.tau)
                    target_critic_params = soft_update(target_critic_params, critic_params, cfg.tau)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: MeanCost = {mean_cost:.2f}, Loss = {current_loss:.2f}, Buffer = {len(replay_buffer)}")
        if epoch % cfg.save_every == 0 and epoch > 0:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            with open(f"{cfg.checkpoint_dir}/epoch_{epoch}_actor.pkl", "wb") as f:
                pickle.dump(actor_params, f)

        if current_loss < best_loss:
            best_loss = current_loss
            with open(f"{cfg.checkpoint_dir}/best_actor.pkl", "wb") as f:
                pickle.dump(actor_params, f)

    with open(f"{cfg.checkpoint_dir}/final_actor.pkl", "wb") as f:
        pickle.dump(actor_params, f)
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()