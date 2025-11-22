import numpy as np

from system_model import *

# =============================
# Firefly-based Particle Filter
# =============================

# Particle / swarm constants
NUM_PARTICLES = 200
DEBUG_FIREFLY_PLOTS = False

_rng = np.random.default_rng()
vel_0_std = 0.1

# Prediction parameters
# Extra PF process noise (on top of IMU noise already in acc_sum)
PF_POS_NOISE_STD = 0.00   # [m]   small extra position noise per step
PF_VEL_NOISE_STD = 0.00   # [m/s] small extra velocity noise per step

# Velocity damping (fraction per second): e.g., 0.5 => ~50% decay in 1 s
VEL_DAMPING_1_PER_S = 0.9

# Firefly Algorithm hyperparameters
FIREFLY_BETA0  = 1.0    # can even bump a bit
FIREFLY_GAMMA  = 0.1    # attraction decays faster with distance
FIREFLY_ALPHA  = 0.1   # smaller random jiggle
FIREFLY_ITERS  = 2      # fewer internal iterations

# Soft-weighted state estimate temperature (bigger = smoother, smaller = peakier)
SOFT_TEMP = 1.0


def firefly_pf_init():
    """
    Initialize firefly/particle positions uniformly within the room bounds.

    Returns:
        x_0_all : (NUM_STATES, NUM_PARTICLES)
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

    return x_0_all

def prediction_step(x_k, u_k):
    """
    x_k : (6, N) particle states [x, y, z, vx, vy, vz]
    u_k : (3,)   control acceleration input (already includes IMU noise)
    """
    # 1) Deterministic motion update from system model
    x_pred = A @ x_k + (B @ u_k)[:, None]

    dt = pf_dt

    # 2) Velocity damping so they don't run away forever
    #    v <- (1 - lambda*dt) * v  with lambda = VEL_DAMPING_1_PER_S
    vel_decay = max(0.0, 1.0 - VEL_DAMPING_1_PER_S * dt)
    x_pred[3:6, :] *= vel_decay

    # 3) Small extra process noise (on top of IMU noise already in u_k)
    if PF_POS_NOISE_STD > 0 or PF_VEL_NOISE_STD > 0:
        pos_noise = np.random.normal(
            0.0, PF_POS_NOISE_STD, size=x_pred[0:3, :].shape
        )
        vel_noise = np.random.normal(
            0.0, PF_VEL_NOISE_STD, size=x_pred[3:6, :].shape
        )
        x_pred[0:3, :] += pos_noise
        x_pred[3:6, :] += vel_noise

    # 4) Respawn any particle that leaves the valid room volume
    pos = x_pred[0:3, :]
    out_of_bounds = (
        (pos[0, :] < X_LIM[0]) | (pos[0, :] > X_LIM[1]) |
        (pos[1, :] < Y_LIM[0]) | (pos[1, :] > Y_LIM[1]) |
        (pos[2, :] < Z_LIM[0]) | (pos[2, :] > Z_LIM[1])
    )
    idx = np.where(out_of_bounds)[0]
    if idx.size > 0:
        # respawn near uniform room for now (you could make this local later)
        pos[0, idx] = np.random.uniform(X_LIM[0], X_LIM[1], idx.size)
        pos[1, idx] = np.random.uniform(Y_LIM[0], Y_LIM[1], idx.size)
        pos[2, idx] = np.random.uniform(Z_LIM[0], Z_LIM[1], idx.size)
        x_pred[3, idx] = np.random.normal(0.0, vel_0_std, idx.size)
        x_pred[4, idx] = np.random.normal(0.0, vel_0_std, idx.size)
        x_pred[5, idx] = np.random.normal(0.0, vel_0_std, idx.size)

    return x_pred

def firefly_update_step(sensor_measurement, x_k):
    """
    Firefly-algorithm-style update step.

    sensor_measurement : (M,)  vector of ranges (one per beacon)
    x_k                : (6, N) particles AFTER prediction

    Returns:
        x_k_new : (6, N) updated particles
    """
    x_k = np.asarray(x_k, dtype=np.float64)
    sensor_measurement = np.asarray(sensor_measurement, dtype=np.float64)

    pos = x_k[0:3, :]                # (3, N)
    N = pos.shape[1]

    # ----- 1) Compute cost J_i = sum_m (d_hat - z_m)^2 -----
    diff = pos.T[None, :, :] - BEACONS[:, None, :]   # (M, N, 3)
    d_hat = np.linalg.norm(diff, axis=2)             # (M, N)
    resid = d_hat - sensor_measurement[:, None]      # (M, N)
    J = np.sum(resid**2, axis=0)                     # (N,)

    room_extent = np.array(
        [X_LIM[1] - X_LIM[0],
         Y_LIM[1] - Y_LIM[0],
         Z_LIM[1] - Z_LIM[0]],
        dtype=np.float64
    )
    alpha_vec = FIREFLY_ALPHA * room_extent  # (3,)

    # ----- 2) Firefly movement -----
    for _ in range(FIREFLY_ITERS):
        for i in range(N):
            for j in range(N):
                if J[j] < J[i]:
                    rij = np.linalg.norm(pos[:, i] - pos[:, j])
                    beta_ij = FIREFLY_BETA0 * np.exp(-FIREFLY_GAMMA * (rij**2))
                    eps_vec = (np.random.rand(3) - 0.5) * alpha_vec
                    pos[:, i] = pos[:, i] + beta_ij * (pos[:, j] - pos[:, i]) + eps_vec

        # recompute cost after move
        diff = pos.T[None, :, :] - BEACONS[:, None, :]
        d_hat = np.linalg.norm(diff, axis=2)
        resid = d_hat - sensor_measurement[:, None]
        J = np.sum(resid**2, axis=0)

    # --- Respawn any out-of-bounds particles after firefly movement ---
    out_of_bounds = (
        (pos[0, :] < X_LIM[0]) | (pos[0, :] > X_LIM[1]) |
        (pos[1, :] < Y_LIM[0]) | (pos[1, :] > Y_LIM[1]) |
        (pos[2, :] < Z_LIM[0]) | (pos[2, :] > Z_LIM[1])
    )
    idx = np.where(out_of_bounds)[0]
    if idx.size > 0:
        pos[0, idx] = np.random.uniform(X_LIM[0], X_LIM[1], idx.size)
        pos[1, idx] = np.random.uniform(Y_LIM[0], Y_LIM[1], idx.size)
        pos[2, idx] = np.random.uniform(Z_LIM[0], Z_LIM[1], idx.size)
        x_k[3, idx] = np.random.normal(0.0, vel_0_std, idx.size)
        x_k[4, idx] = np.random.normal(0.0, vel_0_std, idx.size)
        x_k[5, idx] = np.random.normal(0.0, vel_0_std, idx.size)

    x_k_new = x_k.copy()
    x_k_new[0:3, :] = pos

    return x_k_new

def soft_weighted_estimate(x_k, sensor_measurement, tau=SOFT_TEMP):
    """
    For single-beacon cases, returns the mode (most likely particle) instead of mean.
    """
    x_k = np.asarray(x_k, dtype=np.float64)
    sensor_measurement = np.asarray(sensor_measurement, dtype=np.float64)

    pos = x_k[0:3, :]
    N = pos.shape[1]

    diff = pos.T[None, :, :] - BEACONS[:, None, :]
    d_hat = np.linalg.norm(diff, axis=2)
    resid = d_hat - sensor_measurement[:, None]
    J = np.sum(resid**2, axis=0)

    # Numerically stable weights
    J_min = np.min(J)
    J_shift = J - J_min
    if tau <= 0:
        tau = 1e-6

    w_unnorm = np.exp(-J_shift / tau)
    w_sum = np.sum(w_unnorm)
    if w_sum == 0.0:
        w = np.full(N, 1.0 / N)
    else:
        w = w_unnorm / w_sum

    # --- MODE instead of mean ---
    idx_mode = np.argmax(w)
    x_est = x_k[:, idx_mode]

    return x_est, w

def brightest_particle_estimate(x_k, sensor_measurement):
    """
    State estimate = brightest particle = particle with minimum cost J.

    x_k               : (6, N)
    sensor_measurement: (M,)
    Returns:
        x_est : (6,)
    """
    x_k = np.asarray(x_k, dtype=np.float64)
    sensor_measurement = np.asarray(sensor_measurement, dtype=np.float64)

    pos = x_k[0:3, :]                # (3, N)
    N = pos.shape[1]

    diff = pos.T[None, :, :] - BEACONS[:, None, :]   # (M, N, 3)
    d_hat = np.linalg.norm(diff, axis=2)             # (M, N)
    resid = d_hat - sensor_measurement[:, None]      # (M, N)
    J = np.sum(resid**2, axis=0)                     # (N,)

    idx_best = np.argmin(J)
    return x_k[:, idx_best]

def run_firefly_for_all_runs(monte_data, sim_hz):
    """
    Firefly "PF" loop.

    Expects:
      monte_data['state_sum'] : (R, T_sim, 6)
      monte_data['acc_sum']   : (R, T_sim, 3)

    Adds to monte_data:
      'x_k'        : (R, T_pf, NUM_STATES, NUM_PARTICLES)
      'x_estimate' : (R, T_pf, NUM_STATES)
      'w_k'        : weights for each PF step
    """
    S_all = monte_data['state_sum']   # (R, T_sim, 6)
    A_all = monte_data['acc_sum']     # (R, T_sim, 3)
    Runs, T_sim, _ = S_all.shape

    # PF update every 'step_div_imu' sim ticks
    step_div_imu = int(sim_hz // imu_hz)
    if step_div_imu < 1:
        raise ValueError("imu_hz must be <= sim_hz and yield an integer ratio.")

    T_pf = 1 + (T_sim - 1) // step_div_imu   # include t=0

    if ranging_hz <= 0:
        raise ValueError("ranging_hz must be positive.")

    step_div_rng = int(round(sim_hz / ranging_hz))
    if not np.isclose(sim_hz / ranging_hz, step_div_rng, atol=1e-6):
        raise ValueError("ranging_hz must divide sim_hz evenly (can be fractional).")

    # Allocate
    x_k_all = np.zeros((Runs, T_pf, NUM_STATES, NUM_PARTICLES))
    w_k_all = np.full((Runs, T_pf, NUM_PARTICLES), 1.0 / NUM_PARTICLES)
    x_est_all = np.zeros((Runs, T_pf, NUM_STATES))

    for r in range(Runs):
        print(f"firefly run: {r+1}/{Runs}")
        traj = S_all[r]     # (T_sim, 6)
        acc  = A_all[r]     # (T_sim, 3)

        # init (t = 0)
        x_k_all[r, 0] = firefly_pf_init()
        x_est_all[r, 0] = np.mean(x_k_all[r, 0], axis=1)
        pf_idx = 1

        has_first_measurement = False  # <-- start plotting only after this flips

        for s in range(1, T_sim):
            # only update on IMU ticks
            if (s % step_div_imu) != 0:
                continue

            # ----- BEFORE PREDICTION (for plotting) -----
            x_before = x_k_all[r, pf_idx - 1].copy()
            w_before = w_k_all[r, pf_idx - 1].copy()

            # Prediction with current acceleration sample
            x_k_all[r, pf_idx] = prediction_step(
                x_k_all[r, pf_idx - 1], acc[s]
            )
            x_after_pred = x_k_all[r, pf_idx].copy()

            # Range sensor measurement (to beacons) with noise
            if (s % step_div_rng) == 0:
                from particle_filter import plot_pred_update_step

                z = np.linalg.norm(traj[s, :3] - BEACONS, axis=1) + np.random.normal(
                    0.0, np.sqrt(measurement_noise_variance), size=len(BEACONS)
                )

                # ----- SAVE PRIOR (after prediction, before Firefly update) -----
                x_prior = x_after_pred
                w_prior = w_before

                # ----- Firefly update (posterior) -----
                x_k_all[r, pf_idx] = firefly_update_step(
                    z, x_k_all[r, pf_idx]
                )
                x_post = x_k_all[r, pf_idx].copy()

                # 🔥 Soft-weighted state estimate using Firefly costs
                x_est, w_post = soft_weighted_estimate(x_post, z, tau=SOFT_TEMP)
                x_est_all[r, pf_idx] = x_est
                w_k_all[r, pf_idx]   = w_post

                # ----- PLOT PRIOR vs POSTERIOR (measurement update) -----
                if DEBUG_FIREFLY_PLOTS and has_first_measurement:
                    _ = plot_pred_update_step(
                        truth_pos=traj[s, :3],
                        x_pred=x_prior,  w_pred=w_prior,
                        x_post=x_post,   w_post=w_post,
                        step_idx=pf_idx
                    )

                # mark that we've now had at least one measurement
                has_first_measurement = True

            else:
                # weighted prediction-only estimate (no new z)
                w_prev = w_before
                w_sum = np.sum(w_prev)
                if w_sum <= 0:
                    w_prev = np.full_like(w_prev, 1.0 / w_prev.size)
                else:
                    w_prev = w_prev / w_sum

                x_est_all[r, pf_idx] = np.sum(x_after_pred * w_prev[None, :], axis=1)
                w_k_all[r, pf_idx]   = w_prev

                # ----- PLOT PREVIOUS vs PREDICTED (prediction-only step) -----
                if DEBUG_FIREFLY_PLOTS and has_first_measurement:
                    from particle_filter import plot_pred_update_step
                    _ = plot_pred_update_step(
                        truth_pos=traj[s, :3],
                        x_pred=x_before,      w_pred=w_before,
                        x_post=x_after_pred,  w_post=w_prev,
                        step_idx=pf_idx
                    )

            pf_idx += 1
            if pf_idx >= T_pf:
                break

    # Stash results back
    monte_data['x_k'] = x_k_all
    monte_data['w_k'] = w_k_all
    monte_data['x_estimate'] = x_est_all
    return monte_data
    