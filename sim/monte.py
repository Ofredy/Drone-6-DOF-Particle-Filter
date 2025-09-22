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
NUM_MONTE_RUNS = 5

sim_time = 50  # seconds
sim_hz = 200   # integrator rate (dt = 1/simulation_hz)
sim_dt = 1 / sim_hz


# =========================
# Dynamics & Integrator
# =========================
def rover_x_dot(x_n, acceleration_n):
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
        rx, ry, rz = 40.0, 40.0, 10.0
        w_th = np.random.uniform(0.05, 0.2)
        w_ph = np.random.uniform(0.05, 0.2)
        th0  = np.random.uniform(-np.pi/4, np.pi/4)
        ph0  = np.random.uniform(0, 2*np.pi)

        accel_fn, x0 = make_ellipsoid_accel_provider(
            rx, ry, rz, w_th, w_ph, th0, ph0, noise_std=0.05
        )

        res = runge_kutta(
            rover_x_dot, x0,
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


def plot_pf_enorm_all_runs(monte_data, sim_hz=None, imu_hz=None):
    """
    One plot: ||position error|| vs time for ALL runs.
    Handles runs of different lengths.

    Inputs (either stacked arrays or lists):
      monte_data['x_estimate'] : (R,T,NS) or (T,NS) or list of (T,NS)
      monte_data['state_sum']  : (R,T,6)  or (T,6)  or list of (T,6)
      Optional:
        monte_data['w_k']      : (R,T,N) or (T,N) or list, used to trim unfilled PF rows
        sim_hz / imu_hz        : rates to enable clean decimation; else uses index interpolation
    Returns: (enorm_list, ts_list) — lists of per-run arrays
    """
    # helpers to normalize to list-of-runs
    def _to_runs(arr):
        if isinstance(arr, list):
            return [np.asarray(a) for a in arr]
        A = np.asarray(arr)
        if A.ndim == 3:  # (R,T,feat)
            return [A[r] for r in range(A.shape[0])]
        if A.ndim == 2:  # (T,feat)
            return [A]
        raise ValueError("unexpected array shape for runs")

    X_runs = _to_runs(monte_data['x_estimate'])
    S_runs = _to_runs(monte_data['state_sum'])
    if len(X_runs) != len(S_runs):
        raise ValueError(f"Runs mismatch: x_estimate has {len(X_runs)}, state_sum has {len(S_runs)}")

    W_runs = None
    if 'w_k' in monte_data:
        try:
            W_runs = _to_runs(monte_data['w_k'])
        except Exception:
            W_runs = None  # optional

    if sim_hz is None: sim_hz = monte_data.get('sim_hz', None)
    if imu_hz is None: imu_hz = monte_data.get('imu_hz', None)

    ts_list, enorm_list = [], []

    for r in range(len(X_runs)):
        est = np.asarray(X_runs[r])[:, :3]   # (T_pf,3)
        tru = np.asarray(S_runs[r])[:, :3]   # (T_sim,3)

        # trim PF rows to what actually got filled
        if W_runs is not None:
            w = np.asarray(W_runs[r])
            ws = np.sum(w, axis=-1) if w.ndim == 2 else w
            valid = np.where(np.isfinite(ws) & (ws > 0))[0]
            if valid.size > 0:
                est = est[:valid[-1] + 1]

        # also drop trailing rows that are all-zeros/NaN (in case no w_k)
        mask = np.any(np.isfinite(est), axis=1) & (np.linalg.norm(est, axis=1) > 0)
        if mask.any():
            last = np.nonzero(mask)[0][-1]
            est = est[:last + 1]

        Tp, Ts = est.shape[0], tru.shape[0]

        # align truth to PF
        if Tp == Ts:
            tru_al = tru
        elif sim_hz and imu_hz and (sim_hz % imu_hz == 0):
            step = int(sim_hz // imu_hz)
            if Ts >= step*Tp:
                tru_al = tru[::step][:Tp]
            else:
                # fallback interpolate if decimation would underflow
                idx_s = np.arange(Ts, dtype=float)
                idx_p = np.linspace(0, Ts - 1, Tp)
                tru_al = np.column_stack([np.interp(idx_p, idx_s, tru[:, i]) for i in range(3)])
        elif Ts % Tp == 0:
            tru_al = tru[::(Ts // Tp)][:Tp]
        else:
            idx_s = np.arange(Ts, dtype=float)
            idx_p = np.linspace(0, Ts - 1, Tp)
            tru_al = np.column_stack([np.interp(idx_p, idx_s, tru[:, i]) for i in range(3)])

        # error norm and time
        err = est - tru_al
        e_norm = np.linalg.norm(err, axis=1)               # (Tp,)
        t = (np.arange(Tp, dtype=float) / float(imu_hz)) if imu_hz else np.arange(Tp, dtype=float)

        ts_list.append(t)
        enorm_list.append(e_norm)

    # single plot: all runs, different lengths okay
    plt.figure()
    for r, (t, e) in enumerate(zip(ts_list, enorm_list)):
        plt.plot(t, e, label=f'run {r}')
    plt.xlabel('Time (s)' if imu_hz else 'Sample')
    plt.ylabel('‖position error‖')
    plt.title('PF Position Error Norm vs Time (all runs)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return enorm_list, ts_list


# =========================
# Main
# =========================
if __name__ == "__main__":

    monte_data = generate_trajectories()
    #plot_trajectories(monte_data, fig_num=1, save_as_png=False, dpi=300)

    monte_data = particle_filter.run_pf_for_all_runs(monte_data)
    plot_pf_enorm_all_runs(monte_data)
