import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from monte import sim_hz
from system_model import *

# Particle Filter Constants
NUM_PARTICLES = 500
PERCENT_EFFECTIVE = 0.2
NUM_EFFECTIVE_THRESHOLD = int( NUM_PARTICLES * PERCENT_EFFECTIVE )

_rng = np.random.default_rng()
BETA = 0.6
REJUVENATION_SCALE = 0.25

pos_0_std = 0.25
vel_0_std = 0.5


def pf_init():
    """
    Initialize PF particles uniformly within the room bounds.
    Ignores any measurement; use this when you want a pure room-uniform prior.

    Returns:
        x_0_all : (NUM_STATES, NUM_PARTICLES)
        w_0_all : (NUM_PARTICLES,)
    """
    # Positions: uniform across the room
    x = np.random.uniform(X_LIM[0], X_LIM[1], size=NUM_PARTICLES)
    y = np.random.uniform(Y_LIM[0], Y_LIM[1], size=NUM_PARTICLES)
    z = np.random.uniform(Z_LIM[0], Z_LIM[1], size=NUM_PARTICLES)
    pos = np.vstack([x, y, z])  # (3, N)

    # Velocities: small Gaussian around 0
    vx = np.random.normal(0.0, vel_0_std, size=NUM_PARTICLES)
    vy = np.random.normal(0.0, vel_0_std, size=NUM_PARTICLES)
    vz = np.random.normal(0.0, vel_0_std, size=NUM_PARTICLES)

    # State stack
    x_0_all = np.zeros((NUM_STATES, NUM_PARTICLES))
    x_0_all[0:3, :] = pos
    x_0_all[3,   :] = vx
    x_0_all[4,   :] = vy
    x_0_all[5,   :] = vz

    # Uniform weights
    w_0_all = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES)
    return x_0_all, w_0_all

# takes in particles & accelerametor measurement & gives back the new state of particles
def prediction_step(x_k, u_k):
    # deterministic motion update
    x_pred = A @ x_k + (B @ u_k)[:, None]

    # add process noise directly to [x,y,z,vx,vy,vz]
    # tune these stds, not all equal!
    pos_drift_std = 0.05   # meters per step, try 2cm per IMU tick
    vel_drift_std = 0.00  # m/s per step, try 5cm/s per IMU tick

    noise = np.zeros_like(x_pred)
    noise[0:3, :] = _rng.normal(0.0, pos_drift_std, size=(3, x_k.shape[1]))
    noise[3:6, :] = _rng.normal(0.0, vel_drift_std, size=(3, x_k.shape[1]))

    return x_pred + noise

def jitter_particles_diag(X, process_noise_variance, kappa=0.3, rng=_rng):
    """
    Rejuvenation for diagonal Q:
      Q = diag(process_noise_variance), where process_noise_variance can be:
        - scalar (same variance for all states), or
        - (d,) array for per-state variance.
    Adds Gaussian noise: N(0, kappa * Q) to each particle.
    """
    X = np.asarray(X, dtype=np.float64)
    d, N = X.shape

    # Normalize variance to per-state vector
    if np.isscalar(process_noise_variance):
        var_vec = float(process_noise_variance) * np.ones(d, dtype=np.float64)
    else:
        var_vec = np.asarray(process_noise_variance, dtype=np.float64)
        if var_vec.shape != (d,):
            raise ValueError(f"process_noise_variance must be scalar or shape ({d},), got {var_vec.shape}")

    std_vec = np.sqrt(np.maximum(0.0, kappa) * np.maximum(var_vec, 0.0))  # ensure non-neg
    noise = rng.standard_normal(size=(d, N)) * std_vec[:, None]           # broadcast per-state std
    return X + noise

def residual_resample(weights, rng=_rng):
    """
    Residual resampling (low variance):
    - Deterministically allocate floor(N * w_i) copies.
    - Multinomial draw the remaining R copies from residual probs.
    Returns: indices (N,)
    """
    w = np.asarray(weights, dtype=np.float64)
    w_sum = w.sum()
    if w_sum <= 0 or not np.isfinite(w_sum):
        # fallback: uniform
        N = len(w)
        return rng.integers(low=0, high=N, size=N, endpoint=False)

    w = w / w_sum
    N = w.size

    Ns = np.floor(N * w).astype(int)          # integer copy counts
    R = N - Ns.sum()                           # how many left to draw
    idx = np.repeat(np.arange(N), Ns)          # deterministic part

    if R > 0:
        residual = N * w - Ns                  # fractional leftovers
        res_sum = residual.sum()
        if res_sum > 0 and np.isfinite(res_sum):
            p = residual / res_sum
        else:
            p = np.full(N, 1.0 / N)
        idx_res = rng.choice(N, size=R, replace=True, p=p)
        idx = np.concatenate([idx, idx_res])

    rng.shuffle(idx)                           # avoid ordering bias
    return idx

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
    w_k = w_k * (lk ** BETA)
    s = w_k.sum()
    if not np.isfinite(s) or s <= 0.0:
        w_k = np.full_like(w_k, 1.0 / w_k.size)
    else:
        w_k /= s

    # resample if needed (systematic)
    eff_particles = 1.0 / np.sum(w_k**2)
    if eff_particles <= NUM_EFFECTIVE_THRESHOLD:
        # --- residual resampling ---
        idx = residual_resample(w_k)
        x_k = x_k[:, idx]
        w_k = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES, dtype=np.float64)
        x_k = jitter_particles_diag(x_k, process_noise_variance, kappa=REJUVENATION_SCALE)

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
    step_div_imu = int(sim_hz // imu_hz)
    if step_div_imu < 1:
        raise ValueError("imu_hz must be <= sim_hz and yield an integer ratio for this path.")
    T_pf = 1 + (T_sim - 1) // step_div_imu   # include t=0

    step_div_rng = int(sim_hz // ranging_hz)
    if step_div_rng < 1 or not np.isclose(sim_hz, step_div_rng * ranging_hz):
        raise ValueError("ranging_hz must be <= sim_hz and divide it evenly.")

    # Allocate
    x_k_all = np.zeros((Runs, T_pf, NUM_STATES, NUM_PARTICLES))
    w_k_all = np.full((Runs, T_pf, NUM_PARTICLES), 1.0 / NUM_PARTICLES)
    x_est_all = np.zeros((Runs, T_pf, NUM_STATES))

    for r in range(Runs):
        print("run: %d/%d" % (r, Runs))
        traj = S_all[r]     # (T_sim, 6)
        acc  = A_all[r]     # (T_sim, 3)

        # init (t = 0)
        x_k_all[r, 0], w_k_all[r, 0] = pf_init()
        x_est_all[r, 0] = x_k_all[r, 0] @ w_k_all[r, 0]

        pf_idx = 1
        for s in range(1, T_sim):
            # only update PF on IMU ticks
            if (s % step_div_imu) != 0:
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
            if ( s % step_div_rng ) == 0:
                x_k_all[r, pf_idx], w_k_all[r, pf_idx] = update_step(
                    z, x_k_all[r, pf_idx], w_k_all[r, pf_idx - 1]
                )

            else:
                # no update this tick: carry weights forward
                w_k_all[r, pf_idx] = w_k_all[r, pf_idx - 1]

            #_ = plot_pred_update_step(
            #                                  truth_pos=traj[s, :3],
            #                                  x_pred=x_pred_dbg, w_pred=w_pred_dbg,
            #                                  x_post=x_k_all[r, pf_idx], w_post=w_k_all[r, pf_idx],
            #                                  step_idx=pf_idx
            #                              )

            #if pf_idx > 20:
            #    import pdb; pdb.set_trace()

            # State estimate as weighted mean
            x_est_all[r, pf_idx] = x_k_all[r, pf_idx] @ w_k_all[r, pf_idx]
            pf_idx += 1

    # Stash results back
    monte_data['x_k'] = x_k_all
    monte_data['w_k'] = w_k_all
    monte_data['x_estimate'] = x_est_all

    return monte_data
