from collections import deque  # (kept if you later want to animate)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  # (unused in this script, but handy)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

# =========================
# Global Config / Constants
# =========================
np.random.seed(69)

# Monte Carlo
NUM_MONTE_RUNS = 50

simulation_time = 50  # seconds
simulation_hz = 200   # integrator rate (dt = 1/simulation_hz)
sim_dt = 1 / simulation_hz


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

    trajectories = []
    for _ in range(NUM_MONTE_RUNS):
        rx = 40.0  # x semi-axis
        ry = 40.0  # y semi-axis
        rz = 10.0  # z semi-axis

        w_th = np.random.uniform(0.05, 0.2)
        w_ph = np.random.uniform(0.05, 0.2)
        th0  = np.random.uniform(-np.pi/4, np.pi/4)
        ph0  = np.random.uniform(0, 2*np.pi)

        accel_fn, x0 = make_ellipsoid_accel_provider(rx, ry, rz, w_th, w_ph, th0, ph0, noise_std=0.05)

        tmp = runge_kutta(rover_x_dot, x0,
                          t_0=0.0, t_f=simulation_time,
                          dt=1.0/simulation_hz,
                          accel_fn=accel_fn)
        trajectories.append(tmp)
    return trajectories

def add_process_noise_to_trajectories(trajectories):
    """
    Optional: injects additional zero-mean Gaussian process noise into stored histories.
    """
    for i, run_hash in enumerate(trajectories):
        run_hash['state_sum'] += np.random.normal(0, np.sqrt(process_noise_variance), size=run_hash['state_sum'].shape)
        run_hash['acc_sum']   += np.random.normal(0, np.sqrt(process_noise_variance), size=run_hash['acc_sum'].shape)
        trajectories[i] = run_hash
    return trajectories

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

# =========================
# Main
# =========================
if __name__ == "__main__":

    monte_data = generate_trajectories()

    # Optional: add extra process noise (comment out if you want clean paths)
    # rover_trajectories = add_process_noise_to_rover_trajectories(rover_trajectories)

    plot_trajectories(monte_data, fig_num=1, save_as_png=False, dpi=300)
