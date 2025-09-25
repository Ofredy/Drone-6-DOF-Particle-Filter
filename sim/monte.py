from collections import deque  # (kept if you later want to animate)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  # (unused in this script, but handy)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

from system_model import *
import particle_filter


# =========================
# Global Config / Constants
# =========================
np.random.seed(69)

# Monte Carlo
NUM_MONTE_RUNS = 1

sim_time = 50  # seconds
sim_hz = 200   # integrator rate (dt = 1/simulation_hz)
sim_dt = 1 / sim_hz


# =========================
# Dynamics & Integrator
# =========================
def drone_x_dot(x_n, acceleration_n):
    """
    6-state continuous-time dynamics:
      x = [x, y, z, vx, vy, vz]
    dx/dt = [vx, vy, vz, ax, ay, az]
    """
    return np.array([
        x_n[3],             # dx/dt = vx
        x_n[4],             # dy/dt = vy
        x_n[5],             # dz/dt = vz
        acceleration_n[0],  # dvx/dt = ax
        acceleration_n[1],  # dvy/dt = ay
        acceleration_n[2],  # dvz/dt = az
    ])

def runge_kutta(get_x_dot, x_0, t_0, t_f, dt, accel_fn):
    """
    4th-order Runge-Kutta integrator with an acceleration provider accel_fn(t).
    Returns a dict with:
      'state_sum' : (N, 6) array of states over time
      'acc_sum'   : (N, 3) array of accelerations used at each step
    """
    steps = int((t_f - t_0) / dt)
    state_summary = np.zeros((steps, NUM_STATES))
    acceleration_summary = np.zeros((steps, 3))

    t = t_0
    x_n = x_0.copy()

    for k in range(steps):
        a = accel_fn(t)  # possibly deterministic + noise
        state_summary[k] = x_n
        acceleration_summary[k] = a

        k1 = dt * get_x_dot(x_n, a)
        k2 = dt * get_x_dot(x_n + 0.5 * k1, a)
        k3 = dt * get_x_dot(x_n + 0.5 * k2, a)
        k4 = dt * get_x_dot(x_n + k3, a)
        x_n = x_n + (k1 + 2*k2 + 2*k3 + k4) / 6.0

        t += dt

    return {'state_sum': state_summary, 'acc_sum': acceleration_summary}

# ====================================
# Ellipsoid Reference Motion (Center)
# ====================================
def ellipsoid_pos_vel_acc(t, rx, ry, rz, w_th, w_ph, th0=0.0, ph0=0.0):
    """
    Ellipsoid param:
      x = rx cosθ cosφ
      y = ry cosθ sinφ
      z = rz sinθ
    θ(t) = th0 + w_th t,  φ(t) = ph0 + w_ph t
    """
    th = th0 + w_th * t
    ph = ph0 + w_ph * t
    cth, sth = np.cos(th), np.sin(th)
    cph, sph = np.cos(ph), np.sin(ph)

    # position (relative to center)
    x = rx * cth * cph
    y = ry * cth * sph
    z = rz * sth

    # velocities (θ', φ' constants)
    vx = -rx * sth * cph * w_th - rx * cth * sph * w_ph
    vy = -ry * sth * sph * w_th + ry * cth * cph * w_ph
    vz =  rz * cth * w_th

    # accelerations (θ'', φ'' = 0)
    ax = -rx * cth * cph * (w_th**2 + w_ph**2) + 2 * rx * sth * sph * w_th * w_ph
    ay = -ry * cth * sph * (w_th**2 + w_ph**2) - 2 * ry * sth * cph * w_th * w_ph
    az = -rz * sth * (w_th**2)

    pos = np.array([x, y, z])
    vel = np.array([vx, vy, vz])
    acc = np.array([ax, ay, az])
    return pos, vel, acc

def make_ellipsoid_accel_provider(rx, ry, rz, w_th, w_ph, th0=0.0, ph0=0.0, noise_std=0.0):
    pos0_rel, vel0, _ = ellipsoid_pos_vel_acc(0.0, rx, ry, rz, w_th, w_ph, th0, ph0)

    x0 = np.zeros(6)
    x0[0:3] = pos0_rel + CENTER
    x0[3:6] = vel0

    def accel_fn(t):
        _, _, acc = ellipsoid_pos_vel_acc(t, rx, ry, rz, w_th, w_ph, th0, ph0)
        if noise_std > 0:
            acc = acc + np.random.normal(0.0, noise_std, size=3)
        return acc
    return accel_fn, x0

# ====================================
# Monte Carlo Trajectory Generation
# ====================================
def generate_trajectories():
    monte_state = []
    monte_acc   = []

    for _ in range(NUM_MONTE_RUNS):
        rx, ry, rz = 6.0, 6.0, 2.0
        w_th = np.random.uniform(0.05, 0.2)
        w_ph = np.random.uniform(0.05, 0.2)
        th0  = np.random.uniform(-np.pi/4, np.pi/4)
        ph0  = np.random.uniform(0, 2*np.pi)

        accel_fn, x0 = make_ellipsoid_accel_provider(
            rx, ry, rz, w_th, w_ph, th0, ph0
        )

        res = runge_kutta(
            drone_x_dot, x0,
            t_0=0.0, t_f=sim_time,
            dt=1.0/sim_hz,
            accel_fn=accel_fn
        )

        # pull from dict returned by integrator
        monte_state.append(res['state_sum'])  # (T, 6)
        monte_acc.append(res['acc_sum'])      # (T, 3)

    monte_data = {
        'state_sum': np.stack(monte_state, axis=0),  # (R, T, 6)
        'acc_sum':   np.stack(monte_acc,   axis=0),  # (R, T, 3)
    }
    return monte_data

# =========================
# Plotting
# =========================
def plot_trajectories(trajectories, fig_num=1, save_as_png=False, dpi=300):
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(111, projection='3d')

    # plot the beacon at the center
    ax.scatter([CENTER[0]], [CENTER[1]], [CENTER[2]], s=80, marker='X', label='Beacon', zorder=5)

    # plot each Monte trajectory
    for run_hash in trajectories:
        S = run_hash['state_sum']  # (N, 6)
        ax.plot(S[:, 0], S[:, 1], S[:, 2], linewidth=1.0)

    ax.set_title('Spherical Trajectories around Beacon (0, 0, 1)')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(loc='upper left')

    if save_as_png:
        plt.savefig('rover_trajectories.png', format='png', dpi=dpi)

    plt.show()


def plot_pf_xyz_est_vs_truth(monte_data, run_idx=0, sim_hz=None, imu_hz=None):
    """
    Plots x,y,z PF estimate (solid) vs truth (dashed) over time for one run.

    Inputs:
      monte_data['x_estimate'] : (R, T_pf, NS) or (T_pf, NS)
      monte_data['state_sum']  : (R, T_sim, 6) or (T_sim, 6)
      Optional: monte_data['t_pf'] (T_pf,), or imu_hz / dt_pf for time axis.
                If sim_hz & imu_hz given (or in monte_data), truth is decimated;
                else it is interpolated to PF length.
    """
    X = np.asarray(monte_data['x_estimate'])
    S = np.asarray(monte_data['state_sum'])
    if X.ndim == 2: X = X[None, ...]
    if S.ndim == 2: S = S[None, ...]

    est = X[run_idx, :, :3]   # (T_pf, 3)
    tru = S[run_idx, :, :3]   # (T_sim, 3)
    T_pf, T_sim = est.shape[0], tru.shape[0]

    # time axis for PF
    if 't_pf' in monte_data:
        t = np.asarray(monte_data['t_pf'])[:T_pf]
        xlabel = 'Time (s)'
    else:
        if imu_hz is None: imu_hz = monte_data.get('imu_hz', None)
        if sim_hz is None: sim_hz = monte_data.get('sim_hz', None)
        if imu_hz:
            t = np.arange(T_pf, dtype=float) / float(imu_hz)
            xlabel = 'Time (s)'
        elif 'dt_pf' in monte_data and monte_data['dt_pf']:
            t = np.arange(T_pf, dtype=float) * float(monte_data['dt_pf'])
            xlabel = 'Time (s)'
        else:
            t = np.arange(T_pf, dtype=float)
            xlabel = 'Sample'

    # align truth to PF length
    if T_sim == T_pf:
        tru_pf = tru
    elif (sim_hz is not None) and (imu_hz is not None) and (sim_hz % imu_hz == 0):
        step = int(sim_hz // imu_hz)
        tru_pf = tru[::step][:T_pf]
    else:
        # interpolate truth by index onto PF grid
        idx_sim = np.arange(T_sim, dtype=float)
        idx_pf  = np.linspace(0, T_sim - 1, T_pf)
        tru_pf = np.column_stack([np.interp(idx_pf, idx_sim, tru[:, i]) for i in range(3)])

    # plot
    labels = ['x', 'y', 'z']
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, est[:, i], label=f'{labels[i]} estimate')
        ax.plot(t, tru_pf[:, i], linestyle='--', label=f'{labels[i]} truth')
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(f'PF Position Estimate vs Truth (run {run_idx})')
    fig.tight_layout()
    plt.show()

    return {'t': t, 'est_xyz': est, 'truth_xyz': tru_pf}


# =========================
# Main
# =========================
if __name__ == "__main__":

    monte_data = generate_trajectories()
    #plot_trajectories(monte_data, fig_num=1, save_as_png=False, dpi=300)

    monte_data = particle_filter.run_pf_for_all_runs(monte_data)
    plot_pf_xyz_est_vs_truth(monte_data)
