import numpy as np  


# System Constants
NUM_STATES = 6  # [x, y, z, vx, vy, vz]

# Noise / process config
process_noise_variance = 0.0025 # optional extra noise if you use add_process_noise()
measurement_noise_variance = 0.0225

# Rates
imu_hz = 100         # IMU update/logging rate (can be different from integrator)
pf_dt = 1 / imu_hz
ranging_hz = 100

# Beacons
CENTER = np.array([0.0,  0.0,  1.0]) # central of drone movement(0,0,1)
BEACONS = np.array([
    [0.0,  25.0,  2.5],  
    #[-25.0, -25.0,  0.0],
    #[25.0, -25.0,  5.0],
], dtype=float)

# Indoor Space Definition
X_LIM = [ -25.0, 25.0 ]
Y_LIM = [ -25.0, 25.0 ]
Z_LIM = [ -10.0, 10.0 ]

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
