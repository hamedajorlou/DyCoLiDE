"""
Optimal Hyperparameter Configurations for CoLiDE Variants
Tuned for different graph sizes.

Usage:
    from hyperparameter_configs import get_config

    config = get_config(n_nodes=100, method='batch')
    model.fit(X, **config)
"""

import numpy as np


# =============================================================================
# OPTIMAL CONFIGURATIONS FOR 100 NODES
# =============================================================================

# Optimal for 100 nodes, ~400 edges, 2000 samples
# Based on tuning with lambda1=[0.02, 0.05, 0.1], lr=[0.0002, 0.0003]
CONFIG_100_NODES_BATCH = {
    'lambda1': 0.05,           # L1 regularization (controls sparsity)
    'T': 5,                    # Number of outer iterations
    'mu_init': 1.0,            # Initial augmented Lagrangian penalty
    'mu_factor': 0.1,          # Penalty reduction factor
    's': [1.0, 0.9, 0.8, 0.7, 0.6],  # Acyclicity parameters

    # Batch-specific parameters
    'batch_size': 100,         # Samples per batch
    'n_batches_warm': 15000,   # = 3000 * (100/20) - warmup batches
    'n_batches_final': 40000,  # = 8000 * (100/20) - final iteration batches
    'lr': 0.0002,              # Learning rate (Adam)
    'checkpoint': 500,         # Check convergence every N batches
    'beta_1': 0.99,            # Adam momentum parameter
    'beta_2': 0.999,           # Adam second moment parameter
}

CONFIG_100_NODES_ONLINE = {
    'lambda1': 0.05,
    'T': 5,
    'mu_init': 1.0,
    'mu_factor': 0.1,
    's': [1.0, 0.9, 0.8, 0.7, 0.6],

    # Online-specific parameters
    'n_samples_warm': 25000,   # = 5000 * (100/20) - warmup samples
    'n_samples_final': 60000,  # = 12000 * (100/20) - final iteration samples
    'lr': 0.0002,
    'checkpoint': 500,
    'beta_1': 0.99,
    'beta_2': 0.999,
    'update_freq': 1,          # Update W every N samples (1=true online)
}

CONFIG_100_NODES_STANDARD = {
    'lambda1': 0.05,
    'T': 5,
    'mu_init': 1.0,
    'mu_factor': 0.1,
    's': [1.0, 0.9, 0.8, 0.7, 0.6],

    # Standard full-batch parameters
    'warm_iter': 30000,        # Warmup iterations per outer loop
    'max_iter': 60000,         # Final iteration count
    'lr': 0.0002,
    'checkpoint': 1000,
    'beta_1': 0.99,
    'beta_2': 0.999,
}


# =============================================================================
# OPTIMAL CONFIGURATIONS FOR 200 NODES
# =============================================================================

# Optimal for 200 nodes, ~800 edges, 4000 samples
# TUNED: Grid search completed 2026-01-07
# Best results from testing 7 different configurations
# Key findings:
# - T=6 gives best SHD for Batch (151 vs 180-257 for T=5)
# - lambda1=0.07 gives best SHD for Online (101 vs 140-249)
# - Batch multiplier 4.0/10.0 is optimal (vs 3.0/8.0 or 5.0/12.0)

CONFIG_200_NODES_BATCH = {
    # BEST CONFIG: T=6, SHD=151, TPR=0.9489, FDR=0.1326, Time=34.4min
    'lambda1': 0.05,           # Tuned: tested [0.03, 0.05, 0.07]
    'T': 6,                    # Tuned: T=6 significantly better than T=5
    'mu_init': 1.0,
    'mu_factor': 0.1,
    's': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],  # Extended for T=6

    # Batch-specific parameters (tuned)
    'batch_size': 100,         # Tested: 100 vs 150, no significant difference
    'n_batches_warm': 40000,   # = 4.0 * 1000 * (200/20) - optimal multiplier
    'n_batches_final': 100000, # = 10.0 * 1000 * (200/20) - optimal multiplier
    'lr': 0.0002,              # Tuned: 0.0002 better than 0.00015
    'checkpoint': 500,
    'beta_1': 0.99,
    'beta_2': 0.999,
}

CONFIG_200_NODES_ONLINE = {
    # BEST CONFIG: lambda1=0.07, SHD=101, TPR=0.9524, FDR=0.0772, Time=42.7min
    'lambda1': 0.07,           # Tuned: 0.07 much better than 0.03 or 0.05
    'T': 5,                    # T=6 gives marginal improvement (SHD 140 vs 101)
    'mu_init': 1.0,
    'mu_factor': 0.1,
    's': [1.0, 0.9, 0.8, 0.7, 0.6],

    # Online-specific parameters (tuned)
    'n_samples_warm': 100000,  # = 2.5 * n_batches_warm (4.0 multiplier)
    'n_samples_final': 250000, # = 2.5 * n_batches_final (10.0 multiplier)
    'lr': 0.0002,              # Tuned: 0.0002 works well
    'checkpoint': 500,
    'beta_1': 0.99,
    'beta_2': 0.999,
    'update_freq': 1,
}

CONFIG_200_NODES_STANDARD = {
    # Standard method NOT recommended for 200 nodes (too slow)
    # Use batch or online instead
    'lambda1': 0.05,
    'T': 5,
    'mu_init': 1.0,
    'mu_factor': 0.1,
    's': [1.0, 0.9, 0.8, 0.7, 0.6],
    'warm_iter': 60000,        # Would take hours
    'max_iter': 120000,
    'lr': 0.00015,
    'checkpoint': 1000,
    'beta_1': 0.99,
    'beta_2': 0.999,
}


# =============================================================================
# ADAPTIVE CONFIGURATION (scales with graph size)
# =============================================================================

def get_adaptive_config(n_nodes, n_samples, method='batch', base_nodes=20):
    """
    Generate adaptive configuration that scales with graph size.

    Parameters:
    -----------
    n_nodes : int
        Number of nodes in the graph
    n_samples : int
        Number of samples available
    method : str
        One of: 'batch', 'online', 'standard'
    base_nodes : int
        Base size for scaling (default 20)

    Returns:
    --------
    config : dict
        Configuration dictionary
    """
    scale_factor = n_nodes / base_nodes

    # Base configuration
    config = {
        'lambda1': 0.05,
        'T': 5 if n_nodes <= 150 else 6,
        'mu_init': 1.0,
        'mu_factor': 0.1,
        's': [1.0, 0.9, 0.8, 0.7, 0.6] if n_nodes <= 150 else [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        'beta_1': 0.99,
        'beta_2': 0.999,
    }

    # Method-specific scaling
    if method == 'batch':
        # Superlinear scaling for iterations
        iter_scale = scale_factor ** 1.3  # n^1.3 scaling

        config.update({
            'batch_size': min(150, max(100, n_samples // 20)),
            'n_batches_warm': int(3000 * iter_scale),
            'n_batches_final': int(8000 * iter_scale),
            'lr': 0.0002 if n_nodes <= 100 else 0.00015,
            'checkpoint': 500,
        })

    elif method == 'online':
        # Online needs more samples (2-2.5x batch iterations)
        iter_scale = scale_factor ** 1.3

        config.update({
            'n_samples_warm': int(5000 * iter_scale),
            'n_samples_final': int(12000 * iter_scale),
            'lr': 0.0002 if n_nodes <= 100 else 0.00015,
            'checkpoint': 500,
            'update_freq': 1,
        })

    elif method == 'standard':
        # Linear scaling (not recommended for large graphs)
        config.update({
            'warm_iter': int(30000 * scale_factor),
            'max_iter': int(60000 * scale_factor),
            'lr': 0.0002 if n_nodes <= 100 else 0.00015,
            'checkpoint': 1000,
        })

    return config


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def get_config(n_nodes, method='batch', adaptive=False, **kwargs):
    """
    Get optimal configuration for a given graph size.

    Parameters:
    -----------
    n_nodes : int
        Number of nodes in the graph
    method : str
        One of: 'batch', 'online', 'standard'
    adaptive : bool
        If True, use adaptive scaling instead of fixed configs
    **kwargs : dict
        Override specific parameters

    Returns:
    --------
    config : dict
        Configuration dictionary

    Examples:
    ---------
    >>> config = get_config(100, method='batch')
    >>> config = get_config(200, method='online')
    >>> config = get_config(150, method='batch', adaptive=True)
    >>> config = get_config(100, method='batch', lambda1=0.1)  # Override lambda1
    """
    if adaptive:
        # Determine n_samples from kwargs or use default
        n_samples = kwargs.pop('n_samples', n_nodes * 20)
        config = get_adaptive_config(n_nodes, n_samples, method)
    else:
        # Use predefined configs
        if n_nodes <= 100:
            if method == 'batch':
                config = CONFIG_100_NODES_BATCH.copy()
            elif method == 'online':
                config = CONFIG_100_NODES_ONLINE.copy()
            elif method == 'standard':
                config = CONFIG_100_NODES_STANDARD.copy()
            else:
                raise ValueError(f"Unknown method: {method}")

        elif n_nodes <= 200:
            if method == 'batch':
                config = CONFIG_200_NODES_BATCH.copy()
            elif method == 'online':
                config = CONFIG_200_NODES_ONLINE.copy()
            elif method == 'standard':
                config = CONFIG_200_NODES_STANDARD.copy()
            else:
                raise ValueError(f"Unknown method: {method}")
        else:
            # For >200 nodes, always use adaptive
            n_samples = kwargs.pop('n_samples', n_nodes * 20)
            config = get_adaptive_config(n_nodes, n_samples, method)

    # Override with any provided kwargs
    config.update(kwargs)

    return config


# =============================================================================
# HELPER FUNCTION: Print configuration
# =============================================================================

def print_config(config, method='batch'):
    """Print configuration in a readable format"""
    print(f"Configuration for {method.upper()} method:")
    print("=" * 60)

    # Common parameters
    print(f"  lambda1:     {config['lambda1']}")
    print(f"  T:           {config['T']}")
    print(f"  lr:          {config['lr']}")
    print(f"  mu_init:     {config['mu_init']}")
    print(f"  mu_factor:   {config['mu_factor']}")

    # Method-specific
    if method == 'batch':
        print(f"\n  Batch-specific:")
        print(f"    batch_size:        {config['batch_size']}")
        print(f"    n_batches_warm:    {config['n_batches_warm']}")
        print(f"    n_batches_final:   {config['n_batches_final']}")
        total = (config['T'] - 1) * config['n_batches_warm'] + config['n_batches_final']
        print(f"    Total batches:     {total}")

    elif method == 'online':
        print(f"\n  Online-specific:")
        print(f"    n_samples_warm:    {config['n_samples_warm']}")
        print(f"    n_samples_final:   {config['n_samples_final']}")
        print(f"    update_freq:       {config['update_freq']}")
        total = (config['T'] - 1) * config['n_samples_warm'] + config['n_samples_final']
        print(f"    Total samples:     {total}")

    elif method == 'standard':
        print(f"\n  Standard-specific:")
        print(f"    warm_iter:         {config['warm_iter']}")
        print(f"    max_iter:          {config['max_iter']}")
        total = (config['T'] - 1) * config['warm_iter'] + config['max_iter']
        print(f"    Total iterations:  {total}")

    print("=" * 60)


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("Optimal Hyperparameter Configurations\n")

    print("\n" + "="*70)
    print("100 NODES - BATCH METHOD")
    print("="*70)
    config = get_config(100, method='batch')
    print_config(config, 'batch')

    print("\n" + "="*70)
    print("100 NODES - ONLINE METHOD")
    print("="*70)
    config = get_config(100, method='online')
    print_config(config, 'online')

    print("\n" + "="*70)
    print("200 NODES - BATCH METHOD (TO BE TUNED)")
    print("="*70)
    config = get_config(200, method='batch')
    print_config(config, 'batch')

    print("\n" + "="*70)
    print("200 NODES - ONLINE METHOD (TO BE TUNED)")
    print("="*70)
    config = get_config(200, method='online')
    print_config(config, 'online')

    print("\n" + "="*70)
    print("150 NODES - ADAPTIVE CONFIG")
    print("="*70)
    config = get_config(150, method='batch', adaptive=True, n_samples=3000)
    print_config(config, 'batch')
