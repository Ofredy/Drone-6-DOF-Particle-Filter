import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from monte import sim_hz
from system_model import *


# Particle Filter Constants
NUM_PARTICLES = 500
PERCENT_EFFECTIVE = 0.4
NUM_EFFECTIVE_THRESHOLD = int( NUM_PARTICLES * PERCENT_EFFECTIVE )

pos_0_std = 0.25
vel_0_std = 0.5


def pf_init(sensor_measurement):
    """
    Initialize PF particles given a single UWB range measurement.

    Args:
        sensor_measurement : float, measured range to anchor

    Returns:
        x_0_all : (NUM_STATES, NUM_PARTICLES)
        w_0_all : (NUM_PARTICLES,)
    """
    x_0_all = np.zeros((NUM_STATES, NUM_PARTICLES))

    # --- sample radii around the measured distance ---
    radii = np.random.normal(sensor_measurement, pos_0_std, size=NUM_PARTICLES)
    radii = np.clip(radii, 0.1, None)  # avoid negative/zero

    # --- sample directions in spherical cap (facing -y into the room) ---
    theta = np.arccos(1 - np.random.rand(NUM_PARTICLES) * (1 - np.cos(np.deg2rad(60))))  # half-angle ~60°
    phi = np.random.uniform(0.0, 2.0*np.pi, size=NUM_PARTICLES)  # full azimuth range

    # unit vectors
    ux = np.sin(theta) * np.cos(phi)
    uy = -np.abs(np.cos(theta))   # bias directions into -y
    uz = np.sin(theta) * np.sin(phi)
    dirs = np.vstack((ux, uy, uz))

    # --- particle positions ---
    pos = BEACONS[0][:, None] + radii * dirs

    # enforce indoor bounds (resample if out of bounds)
    for i in range(NUM_PARTICLES):
        while not (X_LIM[0] <= pos[0, i] <= X_LIM[1] and
                   Y_LIM[0] <= pos[1, i] <= Y_LIM[1] and
                   Z_LIM[0] <= pos[2, i] <= Z_LIM[1]):
            r = np.random.normal(sensor_measurement, pos_0_std)
            t = np.arccos(1 - np.random.rand() * (1 - np.cos(np.deg2rad(60))))
            p = np.random.uniform(0.0, 2.0*np.pi)
            u = np.array([
                np.sin(t) * np.cos(p),
                -np.abs(np.cos(t)),
                np.sin(t) * np.sin(p)
            ])
            pos[:, i] = BEACONS[0] + r * u

    # --- set state array ---
    x_0_all[0:3, :] = pos
    x_0_all[3:, :] = np.random.normal(0.0, vel_0_std, size=(3, NUM_PARTICLES))

    # --- uniform weights ---
    return x_0_all, np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES)

# takes in particles & accelerametor measurement & gives back the new state of particles
def prediction_step(x_k, u_k):
    
    return A @ x_k + (B @ u_k)[:, None] + B @ np.random.normal(0.0, np.sqrt(process_noise_variance), size=(3, x_k.shape[1]))

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
    Debug one PF step in 3D: compare prediction vs update and show beacons.
      truth_pos : (3,) true [x,y,z] at this time
      x_pred    : (NUM_STATES, N) particles AFTER prediction (prior)
      w_pred    : (N,)           weights BEFORE update   (prior)
      x_post    : (NUM_STATES, N) particles AFTER update (posterior)
      w_post    : (N,)           weights AFTER update    (posterior)
    Uses global BEACONS = (M,3). Uses X_LIM/Y_LIM/Z_LIM if present.
    """
    def _norm(w):
        w = np.asarray(w).astype(float)
        s = np.sum(w)
        return (np.ones_like(w)/w.size) if (not np.isfinite(s) or s <= 0) else (w/s)

    def _sizes(w):
        w = w / (w.max() + 1e-12)
        return 6.0 + 120.0 * w

    def _set_bounds(ax):
        try:
            ax.set_xlim(X_LIM[0], X_LIM[1])
            ax.set_ylim(Y_LIM[0], Y_LIM[1])
            ax.set_zlim(Z_LIM[0], Z_LIM[1])
        except Exception:
            # fallback: auto data bounds with padding
            all_pts = np.column_stack([pos_pred, pos_post, truth_pos.reshape(3,1)])
            mn = all_pts.min(axis=1); mx = all_pts.max(axis=1)
            pad = 0.05 * (mx - mn + 1e-6)
            ax.set_xlim(mn[0]-pad[0], mx[0]+pad[0])
            ax.set_ylim(mn[1]-pad[1], mx[1]+pad[1])
            ax.set_zlim(mn[2]-pad[2], mx[2]+pad[2])

    w_pred = _norm(w_pred)
    w_post = _norm(w_post)

    pos_pred = x_pred[0:3, :]   # (3, N)
    pos_post = x_post[0:3, :]   # (3, N)

    mu_pred = pos_pred @ w_pred    # (3,)
    mu_post = pos_post @ w_post    # (3,)

    e_pred = np.linalg.norm(mu_pred - truth_pos)
    e_post = np.linalg.norm(mu_post - truth_pos)

    B = None
    if 'BEACONS' in globals():
        B = np.asarray(BEACONS)
        if B.ndim != 2 or B.shape[1] < 3:
            B = None

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # PRIOR (3D)
    ax1.scatter(pos_pred[0], pos_pred[1], pos_pred[2], s=_sizes(w_pred), alpha=0.28, label='particles')
    ax1.scatter([truth_pos[0]], [truth_pos[1]], [truth_pos[2]], marker='*', s=160, label='truth', zorder=5)
    ax1.scatter([mu_pred[0]], [mu_pred[1]], [mu_pred[2]], marker='o', s=80, label='mean', zorder=6)
    if B is not None:
        ax1.scatter(B[:,0], B[:,1], B[:,2], marker='X', s=100, label='beacons', zorder=7)
        for i, (bx, by, bz) in enumerate(B):
            ax1.text(bx, by, bz, f'B{i}', fontsize=8, ha='left', va='bottom')

    ax1.set_title(f'Prediction (prior){"" if step_idx is None else f" | step {step_idx}"}\n‖mean error‖={e_pred:.3g}')
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z'); ax1.grid(True, alpha=0.3); ax1.legend(loc='upper left')
    _set_bounds(ax1)

    # POSTERIOR (3D)
    ax2.scatter(pos_post[0], pos_post[1], pos_post[2], s=_sizes(w_post), alpha=0.28, label='particles')
    ax2.scatter([truth_pos[0]], [truth_pos[1]], [truth_pos[2]], marker='*', s=160, label='truth', zorder=5)
    ax2.scatter([mu_post[0]], [mu_post[1]], [mu_post[2]], marker='o', s=80, label='mean', zorder=6)
    if B is not None:
        ax2.scatter(B[:,0], B[:,1], B[:,2], marker='X', s=100, label='beacons', zorder=7)
        for i, (bx, by, bz) in enumerate(B):
            ax2.text(bx, by, bz, f'B{i}', fontsize=8, ha='left', va='bottom')

    ax2.set_title(f'Update (posterior){"" if step_idx is None else f" | step {step_idx}"}\n‖mean error‖={e_post:.3g}')
    ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z'); ax2.grid(True, alpha=0.3); ax2.legend(loc='upper left')
    _set_bounds(ax2)

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
        x_k_all[r, 0], w_k_all[r, 0] = pf_init(np.linalg.norm( traj[0, :3] - BEACONS[0] ))
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

            #if pf_idx > 20:
            #    import pdb; pdb.set_trace()
#
            _ = plot_pred_update_step(
                                          truth_pos=traj[s, :3],
                                          x_pred=x_pred_dbg, w_pred=w_pred_dbg,
                                          x_post=x_k_all[r, pf_idx], w_post=w_k_all[r, pf_idx],
                                          step_idx=pf_idx
                                      )

            # State estimate as weighted mean
            x_est_all[r, pf_idx] = x_k_all[r, pf_idx] @ w_k_all[r, pf_idx]
            pf_idx += 1

    # Stash results back
    monte_data['x_k'] = x_k_all
    monte_data['w_k'] = w_k_all
    monte_data['x_estimate'] = x_est_all

    return monte_data
