import numpy as np
n_s = 6
n_r = 32

channel_matrix = np.random.randn(n_r, n_s) # Change This However You like :))

CONFIG = {
    "P": 120,
    "var": 1,
    "beta": 0.1,
    "n_t": 128,
    "n_r": n_r,
    "n_s": n_s,
    "n_t_rf": 6,
    "n_r_rf": 6,
    "n_cl": 8,
    "n_ray": 10,
    "v_rf_a": 100,
    "v_rf_iteration": 1000,
    "channel_matrix": channel_matrix
}
