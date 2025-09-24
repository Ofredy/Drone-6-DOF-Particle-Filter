import numpy as np  


# System Constants
NUM_STATES = 6  # [x, y, z, vx, vy, vz]

# Noise / process config
process_noise_variance = 0.135  # optional extra noise if you use add_process_noise()
measurement_noise_variance = 1.0

# Rates
imu_hz = 100         # IMU update/logging rate (can be different from integrator)
pf_dt = 1 / imu_hz
ranging_hz = 100

# Beacons
CENTER = np.array([0.0,  0.0,  1.0]) # central of drone movement(0,0,1)
BEACONS = np.array([
    [0.0,  25.0,  1.0],   # was your old CENTER
    [-25.0, -25.0,  5.0],
    [25.0, -25.0,  2.0],
], dtype=float)

CENTER = BEACONS[0, :]  # central beacon at (0,0,1)

A = np.array([
                [1, 0, 0, pf_dt,      0,      0],
                [0, 1, 0,     0,  pf_dt,      0],
                [0, 0, 1,     0,      0,  pf_dt],
                [0, 0, 0,     1,      0,      0],
                [0, 0, 0,     0,      1,      0],
                [0, 0, 0,     0,      0,      1]
            ])

B = np.array([
                 [0.5*pf_dt**2,            0,            0],
                 [           0, 0.5*pf_dt**2,            0],
                 [           0,            0, 0.5*pf_dt**2],
                 [       pf_dt,            0,            0],
                 [           0,        pf_dt,            0],
                 [           0,            0,        pf_dt]
             ])
