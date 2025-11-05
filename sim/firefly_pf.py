import numpy as np

from system_model import *

# =============================
# Firefly-based Particle Filter
# =============================

# Particle / swarm constants
NUM_PARTICLES = 10
PERCENT_EFFECTIVE = 0.3
NUM_EFFECTIVE_THRESHOLD = int(NUM_PARTICLES * PERCENT_EFFECTIVE)

_rng = np.random.default_rng()

# Initial velocity spread
pos_0_std = 0.25
vel_0_std = 0.5

# Firefly Algorithm hyperparameters
FIREFLY_BETA0  = 1.0    # max attractiveness
FIREFLY_GAMMA  = 1.0    # light absorption coefficient
FIREFLY_ALPHA  = 0.02   # random step factor (fraction of room extent)
FIREFLY_ITERS  = 1      # internal FA iterations per measurement update


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
    u_k : (3,)   control acceleration input

    Uses globals:
        A, B,
        pf_dt,
        process_noise_std,
        X_LIM, Y_LIM, Z_LIM
    """
    # Deterministic motion update
    x_pred = A @ x_k + (B @ u_k)[:, None]

    # Process noise: random accel model
    sigma_a = float(process_noise_std)   # accel noise std [m/s^2]
    dt = pf_dt
    pos_std = 0.5 * (dt**2) * sigma_a
    vel_std = dt * sigma_a

    diag_std = np.array([
        pos_std, pos_std, pos_std,
        vel_std, vel_std, vel_std
    ], dtype=float)

    noise = np.random.normal(
        loc=0.0,
        scale=diag_std[:, None],
        size=x_pred.shape
    )
    x_pred = x_pred + noise

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

    # ----- 3) Clip positions -----
    pos[0, :] = np.clip(pos[0, :], X_LIM[0], X_LIM[1])
    pos[1, :] = np.clip(pos[1, :], Y_LIM[0], Y_LIM[1])
    pos[2, :] = np.clip(pos[2, :], Z_LIM[0], Z_LIM[1])

    x_k_new = x_k.copy()
    x_k_new[0:3, :] = pos

    return x_k_new

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
      'w_k'        : dummy (uniform) weights for shape compatibility
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
    w_k_all = np.full((Runs, T_pf, NUM_PARTICLES), 1.0 / NUM_PARTICLES)  # dummy
    x_est_all = np.zeros((Runs, T_pf, NUM_STATES))

    for r in range(Runs):
        print(f"firefly run: {r+1}/{Runs}")
        traj = S_all[r]     # (T_sim, 6)
        acc  = A_all[r]     # (T_sim, 3)

        # init (t = 0)
        x_k_all[r, 0] = firefly_pf_init()
        # no measurement yet → just take mean of particles as initial estimate
        x_est_all[r, 0] = np.mean(x_k_all[r, 0], axis=1)

        pf_idx = 1
        for s in range(1, T_sim):
            # only update on IMU ticks
            if (s % step_div_imu) != 0:
                continue

            # Prediction with current acceleration sample
            x_k_all[r, pf_idx] = prediction_step(
                x_k_all[r, pf_idx - 1], acc[s]
            )

            # Range sensor measurement (to beacons) with noise
            if (s % step_div_rng) == 0:
                z = np.linalg.norm(traj[s, :3] - BEACONS, axis=1) + np.random.normal(
                    0.0, np.sqrt(measurement_noise_variance), size=len(BEACONS)
                )

                # Firefly update
                x_k_all[r, pf_idx] = firefly_update_step(
                    z, x_k_all[r, pf_idx]
                )

                # 🔥 State estimate = brightest particle
                x_est_all[r, pf_idx] = brightest_particle_estimate(
                    x_k_all[r, pf_idx], z
                )
            else:
                # no range update this tick: hold previous estimate
                x_est_all[r, pf_idx] = x_est_all[r, pf_idx - 1]

            pf_idx += 1
            if pf_idx >= T_pf:
                break

    # Stash results back
    monte_data['x_k'] = x_k_all
    monte_data['w_k'] = w_k_all   # mostly unused, kept for compatibility
    monte_data['x_estimate'] = x_est_all

    return monte_data
