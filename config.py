# config.py
class Config:
    # Market parameters
    is_short_call = True
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1/12
    N_steps = 21
    dt = T / N_steps

    # Transaction cost (2% per trade)
    kappa = 0.0025   # changed from 0.0025 to match paper / realistic

    # Risk aversion for final objective
    lambda_risk = 1.0

    # RL hyperparameters (TD3)
    gamma = 0.99
    tau = 0.005               # soft update rate
    lr_actor = 3e-4
    lr_critic = 3e-4
    buffer_size = 500000      # larger buffer
    batch_size = 128
    N_parallel = 64
    num_epochs = 500          # train longer
    steps_per_epoch = 100     # not directly used in vectorised env

    # Action bounds
    action_lbnd = -2.0
    action_ubnd = 2.0

    # TD3 specific
    actor_update_freq = 2        # update actor every 2 critic updates
    target_noise_scale = 0.2     # noise for target smoothing
    target_noise_clip = 0.5      # clip noise
    critic_updates_per_epoch = 200   # number of critic updates per epoch

    # Exploration noise (decaying)
    exploration_noise_scale = 0.3
    exploration_noise_decay = 0.995

    # Init delta layer & entropy (optional – you can keep or remove)
    use_init_delta = True
    use_entropy = False        # entropy not standard in TD3, but can keep
    entropy_coeff = 0.01
    action_penalty = 0.02

    # State dimension (now 4)
    state_dim = 4

    # Checkpointing
    checkpoint_dir = "checkpoints/gbm_td3"
    save_every = 50

cfg = Config()