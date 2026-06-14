# config.py

# config.py
# All hyperparameters for the RL hedging project

class Config:
    # Market parameters
    S0 = 100.0          # initial stock price
    K = 100.0           # strike price
    r = 0.05            # risk‑free rate (also used as drift)
    sigma = 0.2         # volatility
    T = 1.0             # time to maturity (years)
    N_steps = 50       # number of rebalancing steps
    dt = T / N_steps    # time step

    # Transaction costs
    kappa = 0.02        # proportional transaction cost (2%)
    
    # Risk aversion for pricing (used in final option price calculation)
    lambda_risk = 1.0   # Risk aversion for variance penalty in pricing
    
    # RL hyperparameters
    gamma = 0.99        # discount factor
    tau = 0.005         # soft update for target networks
    lr_actor = 3e-4     # Increased from 1e-5 for faster learning
    lr_critic = 3e-4    # Same for both critics
    buffer_size = 100000
    batch_size = 64
    N_parallel =128    # number of parallel environments per batch
    update_freq = 2     # Update actor every 2 critic updates (TD3 style)
    
    # Training control
    num_epochs = 200    # Increased for convergence
    steps_per_epoch = 100
    
    # Action bounds (allowing shorting and leverage)
    action_lbnd = -2.0  # Can short up to 2 shares
    action_ubnd = 2.0   # Can hold up to 2 shares
    
    # State dimension (expanded from 3 to 4)
    state_dim = 4       # [price_norm, delta, time_norm, h_prev]
    
    # Checkpointing
    checkpoint_dir = "checkpoints"
    save_every = 50
    # In config.py - Add these to Config class
    use_init_delta = True      # Enable init delta layer (Project B feature)
    use_entropy = True         # Enable entropy bonus
    entropy_coeff = 0.01       # Strength of entropy bonus
    action_penalty = 0.02      # Penalty for extreme actions


    save_every = 50
    
    # ================================================================
    # CHECKPOINT DIRECTORY (AUTO-SELECTS BASED ON MODEL TYPE)
    # ================================================================
    # @property
    # def checkpoint_dir(self):
    #     if self.use_chiarella_heston:
    #         return "checkpoints/ch"
    #     else:
    #         return "checkpoints/gbm"
    
    # State dimension (will be set dynamically by train.py)
    state_dim = 4


cfg = Config()