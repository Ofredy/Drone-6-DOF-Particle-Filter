import numpy as np

from system_model import *


# Particle Filter Constants
NUM_PARTICLES = 20
PERCENT_EFFECTIVE = 0.4
NUM_EFFECTIVE_THRESHOLD = int( NUM_PARTICLES * PERCENT_EFFECTIVE )

x_0_std = 15

def pf_init():

    return np.random.normal(0, x_0_std), np.ones(shape=(NUM_PARTICLES)) * ( 1 / NUM_PARTICLES )


# takes in particles & accelerametor measurement & gives back the new state of particles
def prediction_step(x_k, u_k):
    
    return A @ x_k + B @ u_k + np.random.normal(0, np.sqrt(process_noise_variance))

def update_step(x_k, w_k):

    pass

def particle_filter(monte_data, sim_time, sim_dt):
    
    sim_time = monte_data.shape[2]
    num_pf_time_steps = sim_time // pf_dt

    trajectory_history = monte_data['state_sum']
    acc_history = monte_data['acc_sum']

    # particles history
    x_k = np.zeros(shape=(num_pf_time_steps, NUM_STATES, NUM_PARTICLES))
    w_k = np.ones(shape=(num_pf_time_steps, NUM_PARTICLES)) * ( 1 / NUM_PARTICLES )

    # state estimate history
    x_estimate = np.zeros(shape=(num_pf_time_steps, NUM_STATES))

    # state estimate initialization
    x_estimate[0], w_k[0] = pf_init()

    for acc_meas_time in range(0, sim_time//sim_dt, 2):

        x_k = prediction_step(x_k, acc_history[acc_meas_time, :])

        if acc_meas_time % ( 1/ranging_hz ):

            update_step()

        num_effective_particles = 10
        if num_effective_particles < NUM_EFFECTIVE_THRESHOLD:

            # resampling logic
            pass

        x_estimate[acc_meas_time] = x_k @ w_k
