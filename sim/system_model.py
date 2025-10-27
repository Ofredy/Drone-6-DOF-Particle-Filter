import numpy as np  


# System Constants
NUM_STATES = 6  # [x, y, z, vx, vy, vz]

# Noise / process config
process_noise_std = 0.0025
process_noise_variance = np.sqrt(process_noise_std)
measurement_noise_std = 0.01
measurement_noise_variance = np.sqrt(measurement_noise_std)

# Rates
imu_hz = 20         # IMU update/logging rate (can be different from integrator)
pf_dt = 1 / imu_hz
ranging_hz = 10

# Beacons
CENTER = np.array([0.0,  0.0,  1.0]) # central of drone movement(0,0,1)
BEACONS = np.array([
    [-12.0,    12.0,  0.0],  
    [-12.0,   -12.0,  12.0],
    [12.0,  -12.0,  7.0],
    #[  0.0,  -8.3, 25.0],  # new high anchor
], dtype=float)

# Indoor Space Definition
X_LIM = [ -12.0, 12.0 ]
Y_LIM = [ -12.0, 12.0 ]
Z_LIM = [   0.0, 12.0 ]

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
