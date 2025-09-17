import numpy as np  


# System Constants
NUM_STATES = 6  # [x, y, z, vx, vy, vz]

# Noise / process config
process_noise_variance = 0.125  # optional extra noise if you use add_process_noise()
measurement_noise_variance = 0.125

# Rates
imu_hz = 100         # IMU update/logging rate (can be different from integrator)
pf_dt = 1 / imu_hz
ranging_hz = 20

# Beacons
CENTER = np.array([0.0, 0.0, 1.0])  # central beacon at (0,0,1)

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
