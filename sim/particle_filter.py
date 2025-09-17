import numpy as np
from scipy.stats import norm

from monte import sim_hz
from system_model import *


# Particle Filter Constants
NUM_PARTICLES = 20
PERCENT_EFFECTIVE = 0.4
NUM_EFFECTIVE_THRESHOLD = int( NUM_PARTICLES * PERCENT_EFFECTIVE )

x_0_std = 15

def pf_init():

    return ( np.random.normal(0, x_0_std, size=(NUM_STATES, NUM_PARTICLES)), 
             np.ones(shape=(NUM_PARTICLES)) * ( 1 / NUM_PARTICLES ) )


# takes in particles & accelerametor measurement & gives back the new state of particles
def prediction_step(x_k, u_k):
    
    return A @ x_k + (B @ u_k)[:, None] + np.random.normal(0, np.sqrt(process_noise_variance))

def update_step(sensor_measurement, x_k, w_k):

    # extract positions (3 x NUM_PARTICLES)
    positions = x_k[0:3, :]   # rows [x, y, z]
    diff = positions - CENTER[:, None]
    estimated_distances = np.linalg.norm(diff, axis=0)   # shape (NUM_PARTICLES,)

    likelihoods = norm.pdf(sensor_measurement,   # the actual measurement z(k)
                       loc=estimated_distances,  # each particle's predicted measurement
                       scale=np.sqrt(measurement_noise_variance))
    
    w_k = w_k @ likelihoods
    w_k /= w_k.sum()

    eff_particles = 1.0 / np.sum(w_k**2)
    if eff_particles <= NUM_EFFECTIVE_THRESHOLD:
        
        # build CDF
        cdf = np.cumsum(w_k)
        cdf[-1] = 1.0  # guard
        # evenly spaced thresholds with one random offset
        u0 = np.random.rand() / NUM_PARTICLES
        u = u0 + (np.arange(NUM_PARTICLES) / NUM_PARTICLES)
        idx = np.searchsorted(cdf, u, side='left')       # (N,)

        x_k = x_k[:, idx]                                # copy all state rows
        w_k = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES)                        # reset to uniform

    return x_k, w_k

def run_pf_for_all_runs(monte_data):
    """
    Expects:
      monte_data['state_sum'] : (R, T_sim, 6)
      monte_data['acc_sum']   : (R, T_sim, 3)

    Uses globals: NUM_STATES, NUM_PARTICLES, sim_hz, imu_hz,
                  CENTER, measurement_noise_variance,
                  pf_init(), prediction_step(), update_step()
    Adds to monte_data:
      'x_k'        : (R, T_pf, NUM_STATES, NUM_PARTICLES)
      'w_k'        : (R, T_pf, NUM_PARTICLES)
      'x_estimate' : (R, T_pf, NUM_STATES)
    """
    S_all = monte_data['state_sum']   # (R, T_sim, 6)
    A_all = monte_data['acc_sum']     # (R, T_sim, 3)
    Runs, T_sim, _ = S_all.shape

    # PF update every 'step_div' sim ticks
    step_div = int(sim_hz // imu_hz)
    if step_div < 1:
        raise ValueError("imu_hz must be <= sim_hz and yield an integer ratio for this path.")
    T_pf = 1 + (T_sim - 1) // step_div   # include t=0

    # Allocate
    x_k_all = np.zeros((Runs, T_pf, NUM_STATES, NUM_PARTICLES))
    w_k_all = np.full((Runs, T_pf, NUM_PARTICLES), 1.0 / NUM_PARTICLES)
    x_est_all = np.zeros((Runs, T_pf, NUM_STATES))

    for r in range(Runs):
        traj = S_all[r]     # (T_sim, 6)
        acc  = A_all[r]     # (T_sim, 3)

        # init (t = 0)
        x_k_all[r, 0], w_k_all[r, 0] = pf_init()
        x_est_all[r, 0] = x_k_all[r, 0] @ w_k_all[r, 0]

        pf_idx = 1
        for s in range(1, T_sim):
            # only update PF on IMU ticks
            if (s % step_div) != 0:
                continue

            # Prediction with current acceleration sample
            x_k_all[r, pf_idx] = prediction_step(x_k_all[r, pf_idx - 1], acc[s])

            # Range sensor measurement (to CENTER) with noise
            z = np.linalg.norm(traj[s, :3] - CENTER) + np.random.normal(
                0.0, np.sqrt(measurement_noise_variance)
            )

            # Measurement update (your update_step returns (x_k_new, w_k_new))
            x_k_all[r, pf_idx], w_k_all[r, pf_idx] = update_step(
                z, x_k_all[r, pf_idx], w_k_all[r, pf_idx - 1]
            )

            # State estimate as weighted mean
            x_est_all[r, pf_idx] = x_k_all[r, pf_idx] @ w_k_all[r, pf_idx]
            pf_idx += 1

    # Stash results back
    monte_data['x_k'] = x_k_all
    monte_data['w_k'] = w_k_all
    monte_data['x_estimate'] = x_est_all

    return monte_data
