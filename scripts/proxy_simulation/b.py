"""
Proxy simulation to investigate performance upper bound.

Uses the [exafmm library](https://github.com/exafmm/exafmm-t)
"""


import numpy as np
from exafmm import laplace

# Constants
NUM_PARTICLES = 70000
DIM = 3
BOX_SIZE = 10.0
TIMESTEPS = 10
DELTA_T = 0.001
TEMPERATURE = 300
BOLTZMANN_CONST = 1.380649e-23
EPSILON_0 = 8.854187817e-12

# Initialization
np.random.seed(42)

# Initialize positions (random within a cubic box)
positions = np.random.uniform(0, BOX_SIZE, size=(NUM_PARTICLES, DIM))

# Initialize velocities (Maxwell-Boltzmann distribution)
std_dev = np.sqrt(BOLTZMANN_CONST * TEMPERATURE)
velocities = np.random.normal(0, std_dev, size=(NUM_PARTICLES, DIM))

# Initialize charges (random values between -1 and 1)
charges = np.random.uniform(-1.0, 1.0, size=NUM_PARTICLES)

# Masses of particles (all equal for simplicity)
masses = np.ones(NUM_PARTICLES)

# Initialize FMM-related objects
laplace_fmm = laplace.LaplaceFMM(p=5, ncrit=64) # Order p=5 and ncrit=64
sources = laplace.init_sources(positions, charges)
targets = laplace.init_targets(positions)
tree = laplace.setup(sources, targets, laplace_fmm)

def compute_forces_laplace(tree, fmm):
    """
    Compute forces on all particles using the Laplace method.
    """
    # evaluate potentials and gradients
    trg_values = laplace.evaluate(tree, fmm)

    # Extract gradients (forces) from the trg_values array
    forces = -trg_values[:, 1:] # trg_values[:, 1:4] contains gradients
    return forces

def velocity_verlet(positions, velocities, forces, masses, delta_t):
    """
    Velocity Verlet integration for one time step.
    """
    # Update positions
    positions += velocities * delta_t + 0.5 * forces / masses[:, None] * delta_t**2

    # Update forces
    laplace.clear_values(tree) # Clean previous values
    laplace.update_charges(tree, charges) # Update source charges
    forces_new = compute_forces_laplace(tree, laplace_fmm)
    
    # Update velocities
    velocities += 0.5 * (forces + forces_new) / masses[:, None] * delta_t
    
    return forces_new

import time

# Simulation loop
print(f"Starting simulation with {NUM_PARTICLES} particles for {TIMESTEPS} time steps.")
forces = compute_forces_laplace(tree, laplace_fmm) # Initial forces
total_time = time.time()
for step in range(TIMESTEPS):
    dt = time.time()
    # Integrate using the Velocity Verlet algorithm
    forces = velocity_verlet(positions, velocities, forces, masses, DELTA_T)

    # Periodic boundary conditions
    positions %= BOX_SIZE
    
    dt = time.time() - dt

    # Print step info
    print(f"Time step {step + 1}/{TIMESTEPS} completed in {dt:.1f} seconds")
total_time = time.time() - total_time
print(f"{10**7} time steps would take {total_time / TIMESTEPS * 10**7 / (60*60):.1f} hours")
print("Simulation completed.")