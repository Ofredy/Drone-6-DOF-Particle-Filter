import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from monte import sim_hz
from system_model import *


# Particle Filter Constants
NUM_PARTICLES = 10000
PERCENT_EFFECTIVE = 0.4
NUM_EFFECTIVE_THRESHOLD = int( NUM_PARTICLES * PERCENT_EFFECTIVE )

x_0_std = 20

def pf_init():

    x_0_all = np.zeros((NUM_STATES, NUM_PARTICLES))
    x_0_all[0:3, :] = np.random.normal(0.0, x_0_std, size=(3, NUM_PARTICLES))
    return x_0_all, np.full(NUM_PARTICLES, 1.0/NUM_PARTICLES)


# takes in particles & accelerametor measurement & gives back the new state of particles
def prediction_step(x_k, u_k):
    
    return A @ x_k + (B @ u_k)[:, None] + np.random.normal(0, np.sqrt(process_noise_variance))

def update_step(sensor_measurement, x_k, w_k):
    """
    sensor_measurement : (M,)  vector of ranges (one per beacon)
    x_k                : (6, N) particles after prediction
    w_k                : (N,)   prior weights
    Uses globals: BEACONS, measurement_noise_variance, NUM_PARTICLES, NUM_EFFECTIVE_THRESHOLD
    """
    # positions and predicted ranges to each beacon
    positions = x_k[0:3, :]                              # (3, N)
    diff = positions.T[None, :, :] - BEACONS[:, None, :] # (M, N, 3)
    d_hat = np.linalg.norm(diff, axis=2)                 # (M, N)  <-- axis=2 is the xyz norm

    # measurement noise: scalar or per-beacon
    sigma = np.sqrt(np.asarray(measurement_noise_variance, float))
    if sigma.ndim == 0:
        sigma = np.full(BEACONS.shape[0], float(sigma))
    sigma = sigma[:, None]                               # (M, 1) for broadcasting

    # per-beacon likelihoods -> (M, N)
    lk_per = norm.pdf(sensor_measurement[:, None], loc=d_hat, scale=sigma)

    # combine beacons: product across M -> (N,)
    lk = np.prod(np.maximum(lk_per, 1e-300), axis=0)

    # weight update (elementwise) + robust normalize
    w_k = w_k * lk
    s = w_k.sum()
    if not np.isfinite(s) or s <= 0.0:
        w_k = np.full_like(w_k, 1.0 / w_k.size)
    else:
        w_k /= s

    # resample if needed (systematic)
    eff_particles = 1.0 / np.sum(w_k**2)
    if eff_particles <= NUM_EFFECTIVE_THRESHOLD:
        cdf = np.cumsum(w_k); cdf[-1] = 1.0
        u0 = np.random.rand() / NUM_PARTICLES
        u = u0 + (np.arange(NUM_PARTICLES) / NUM_PARTICLES)
        idx = np.searchsorted(cdf, u, side='left')
        x_k = x_k[:, idx]
        w_k = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES)

    return x_k, w_k

def plot_pred_update_step(truth_pos, x_pred, w_pred, x_post, w_post, step_idx=None):
    """
    Debug one PF step: compare prediction vs update and show beacons (XY).
      truth_pos : (3,) true [x,y,z] at this time
      x_pred    : (NUM_STATES, N) particles AFTER prediction (prior)
      w_pred    : (N,)           weights BEFORE update   (prior)
      x_post    : (NUM_STATES, N) particles AFTER update (posterior)
      w_post    : (N,)           weights AFTER update    (posterior)
    Uses global BEACONS = (M,3).
    """
    def _norm(w):
        s = np.sum(w)
        return (np.ones_like(w)/w.size) if (not np.isfinite(s) or s <= 0) else (w/s)

    w_pred = _norm(np.asarray(w_pred))
    w_post = _norm(np.asarray(w_post))

    pos_pred = x_pred[0:3, :]   # (3, N)
    pos_post = x_post[0:3, :]   # (3, N)

    mu_pred = pos_pred @ w_pred    # (3,)
    mu_post = pos_post @ w_post    # (3,)

    e_pred = np.linalg.norm(mu_pred - truth_pos)
    e_post = np.linalg.norm(mu_post - truth_pos)

    def _sizes(w):
        w = w / (w.max() + 1e-12)
        return 10.0 + 120.0 * w

    # beacons (XY)
    beacons_xy = None
    if 'BEACONS' in globals():
        B = np.asarray(BEACONS)
        if B.ndim == 2 and B.shape[1] >= 2:
            beacons_xy = B[:, :2]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # PRIOR
    axs[0].scatter(pos_pred[0], pos_pred[1], s=_sizes(w_pred), alpha=0.35, label='particles')
    axs[0].scatter([truth_pos[0]], [truth_pos[1]], marker='*', s=140, label='truth', zorder=5)
    axs[0].scatter([mu_pred[0]], [mu_pred[1]], marker='o', s=70, label='mean', zorder=5)
    if beacons_xy is not None:
        axs[0].scatter(beacons_xy[:,0], beacons_xy[:,1], marker='X', s=90, label='beacons', zorder=6)
        # annotate beacon indices
        for i, (bx, by) in enumerate(beacons_xy):
            axs[0].text(bx, by, f'B{i}', ha='left', va='bottom', fontsize=8)
    axs[0].set_title(f'Prediction (prior){"" if step_idx is None else f" | step {step_idx}"}\n‖mean error‖={e_pred:.3g}')
    axs[0].set_xlabel('x'); axs[0].set_ylabel('y'); axs[0].grid(True, alpha=0.3); axs[0].legend()
    axs[0].set_aspect('equal', 'box')

    # POSTERIOR
    axs[1].scatter(pos_post[0], pos_post[1], s=_sizes(w_post), alpha=0.35, label='particles')
    axs[1].scatter([truth_pos[0]], [truth_pos[1]], marker='*', s=140, label='truth', zorder=5)
    axs[1].scatter([mu_post[0]], [mu_post[1]], marker='o', s=70, label='mean', zorder=5)
    if beacons_xy is not None:
        axs[1].scatter(beacons_xy[:,0], beacons_xy[:,1], marker='X', s=90, label='beacons', zorder=6)
        for i, (bx, by) in enumerate(beacons_xy):
            axs[1].text(bx, by, f'B{i}', ha='left', va='bottom', fontsize=8)
    axs[1].set_title(f'Update (posterior){"" if step_idx is None else f" | step {step_idx}"}\n‖mean error‖={e_post:.3g}')
    axs[1].set_xlabel('x'); axs[1].set_ylabel('y'); axs[1].grid(True, alpha=0.3); axs[1].legend()
    axs[1].set_aspect('equal', 'box')

    plt.show()

    return {
        'mean_prior':  mu_pred,
        'mean_post':   mu_post,
        'err_prior':   e_pred,
        'err_post':    e_post,
    }

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
            x_pred_dbg = x_k_all[r, pf_idx].copy()
            w_pred_dbg = w_k_all[r, pf_idx - 1].copy()

            # Range sensor measurement (to CENTER) with noise
            z = np.linalg.norm(traj[s, :3] - BEACONS, axis=1) + np.random.normal(
                0.0, np.sqrt(measurement_noise_variance)
            )

            # Measurement update (your update_step returns (x_k_new, w_k_new))
            x_k_all[r, pf_idx], w_k_all[r, pf_idx] = update_step(
                z, x_k_all[r, pf_idx], w_k_all[r, pf_idx - 1]
            )

            #_ = plot_pred_update_step(
            #                              truth_pos=traj[s, :3],
            #                              x_pred=x_pred_dbg, w_pred=w_pred_dbg,
            #                              x_post=x_k_all[r, pf_idx], w_post=w_k_all[r, pf_idx],
            #                              step_idx=pf_idx
            #                          )

            # State estimate as weighted mean
            x_est_all[r, pf_idx] = x_k_all[r, pf_idx] @ w_k_all[r, pf_idx]
            pf_idx += 1

    # Stash results back
    monte_data['x_k'] = x_k_all
    monte_data['w_k'] = w_k_all
    monte_data['x_estimate'] = x_est_all

    return monte_data
