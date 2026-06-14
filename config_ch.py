# config_ch.py
# Chiarella-Heston specific configuration

class ConfigCH:
    # ================================================================
    # MARKET PARAMETERS
    # ================================================================
    S0 = 100.0          # initial stock price
    K = 100.0           # strike price
    r = 0.05            # risk‑free rate
    sigma = 0.2         # volatility (used for BSM delta warm start)
    T = 1/12             # time to maturity (years)
    N_steps = 21       # number of rebalancing steps
    dt = T / N_steps    # time step

    # Transaction costs
    kappa = 0.02        # proportional transaction cost (2%)
    
    # ================================================================
    # CHIARELLA-HESTON MODEL PARAMETERS (Project A)
    # Calibrated to S&P 500 data (1999-2023)
    # ================================================================
    
    # Fundamental trader parameters
    kappa_CH = 0.5      # κ - demand intensity from fundamental traders
    sigma_F = 0.2       # σ_F - volatility of fundamental value
    g = r - 0.5 * sigma_F**2  # volatility-adjusted drift (auto-computed)
    
    # Momentum trader parameters
    beta = 0.5          # β - momentum demand intensity
    gamma = 10.0        # γ - saturation level
    alpha = 1/6         # α - decay rate (5-day horizon, from paper)
    
    # Volatility trader parameters (Heston-style)
    omega = 0.3         # ω - volatility trader intensity
    phi = 2.0           # φ - mean reversion rate
    theta = 0.0484      # θ - long-term variance
    sigma_vol = 0.5     # σ - volatility of volatility
    rho = -0.7          # ρ - correlation (leverage effect)
    
    # ================================================================
    # RISK & PRICING
    # ================================================================
    lambda_risk = 1.0   # Risk aversion for pricing
    is_short_call = True
    
    # ================================================================
    # RL HYPERPARAMETERS
    # ================================================================
    gamma = 0.99        # discount factor
    tau_soft = 0.005    # soft update for target networks (renamed to avoid confusion)
    lr_actor = 1e-3
    lr_critic = 1e-3
    buffer_size = 500000    # INCREASED to 1M (standard TD3)
    batch_size = 128       # INCREASED for better gradient estimates
    N_parallel = 64
    # learning_starts = 10000  # NEW: fill buffer this much before learning
    
    # TD3-specific parameters
    # gradient_steps = -1      # NEW: -1 = as many updates as steps collected
    # policy_delay = 2         # NEW: update actor every 2 critic updates
    # update_freq = 20         # Number of updates per epoch (used if gradient_steps = -1)
    
    # Training control
    num_epochs = 500
    steps_per_epoch = 100
    
    # Action bounds
    action_lbnd = -2.0
    action_ubnd = 2.0
    
    # Init delta layer and penalties
    use_init_delta = True
    use_entropy = True
    entropy_coeff = 0.01
    action_penalty = 0.02
    
    # Checkpointing
    save_every = 50
    
    # State dimension (CH uses 6)
    state_dim = 6

    actor_update_freq = 2          # TD3 trick #1: Update actor every 2 critic updates
    target_update_freq = 1         # Update target networks every actor update (or less)
    target_noise_scale = 0.2       # TD3 trick #2: Target policy smoothing noise
    target_noise_clip = 0.5        # Clip noise to avoid extreme values
    critic_updates_per_epoch = 200
     # Improved exploration
    exploration_noise_scale = 0.3  # Add this - more exploration early
    exploration_noise_decay = 0.995  # Decay over time
cfg_ch = ConfigCH()