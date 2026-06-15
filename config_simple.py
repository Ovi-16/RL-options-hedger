class ConfigSimple:
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1/12
    N_steps = 21
    dt = T / N_steps
    kappa = 0.0025

    # CH parameters (for simulation only)
    kappa_CH = 0.5
    sigma_F = 0.2
    beta = 0.5
    gamma = 10.0
    alpha = 1/6
    omega = 0.3
    phi = 2.0
    theta = 0.0484
    sigma_vol = 0.5
    rho = -0.7

    lambda_risk = 1.0
    gamma_discount = 0.99
    lr_actor = 1e-3
    lr_critic = 1e-3
    buffer_size = 500000
    batch_size = 128
    N_parallel = 64
    num_epochs = 1000
    action_lbnd = -2.0
    action_ubnd = 2.0
    actor_update_freq = 2
    target_update_freq = 1
    target_noise_scale = 0.2
    target_noise_clip = 0.5
    critic_updates_per_epoch = 200
    exploration_noise_scale = 0.3
    exploration_noise_decay = 0.995
    state_dim = 3
    save_every = 50
    tau_soft = 0.005
    is_short_call = True 