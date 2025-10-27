"""
Proxy simulation to investigate performance upper bound.

Uses the [exafmm library](https://github.com/exafmm/exafmm-t)
"""

import numpy as np
import exafmm.laplace as laplace
import exafmm.helmholtz as helmholtz
import time

# Constants
NUM_PARTICLES = 70000
TIMESTEP = 0.001       # Example timestep
NUM_STEPS = 10
BOX_SIZE = 10.0        # Example box size for periodic boundary conditions
TEMPERATURE = 1.0      # Reduced Temperature
MASS = 1.0             # Unit mass

# Function to initialize particle positions and velocities
def initialize_particles(num_particles, box_size, temperature, mass):
    """
    Initializes particles randomly within the box
    and velocities from a Maxwell-Boltzmann distribution.
    """
    positions = np.random.rand(num_particles, 3) * box_size

    # Generate velocities from a Maxwell-Boltzmann distribution
    sigma = np.sqrt(temperature / mass)  # Standard deviation
    velocities = np.random.normal(0, sigma, (num_particles, 3))

    return positions, velocities

# Function to compute forces using exafmm
def compute_forces(positions, charges, fmm_lib, tree, fmm):
    """
    Computes forces on particles using the exafmm library.

    Args:
        positions (np.ndarray): Particle positions (n_particles x 3).
        charges (np.ndarray): Particle charges (n_particles).
        fmm_lib (module): exafmm.laplace or exafmm.helmholtz.
        tree: The FMM tree object.
        fmm: The FMM object.

    Returns:
        np.ndarray: Forces on each particle (n_particles x 3).
    """
    # update charges in the FMM tree
    fmm_lib.update_charges(tree, charges)

    # clear target values (potentials and gradients)
    fmm_lib.clear_values(tree)

    # Evaluate the FMM to get potentials and gradients
    result = fmm_lib.evaluate(tree, fmm)

    # Extract forces from the gradient (result[:, 1:])
    # Negative gradient is force
    forces = -result[:, 1:]

    return forces

# Function to apply periodic boundary conditions
def apply_periodic_boundary_conditions(positions, box_size):
    """
    Applies periodic boundary conditions to particle positions.
    """
    positions = np.mod(positions, box_size)
    return positions

# Velocity Verlet integration
def velocity_verlet(positions, velocities, forces, mass, timestep, box_size):
    """
    Performs one step of Velocity Verlet integration.
    """
    # Update positions
    positions += velocities * timestep + 0.5 * forces / mass * timestep**2
    positions = apply_periodic_boundary_conditions(positions, box_size)

    # Update velocities (first half)
    velocities += 0.5 * forces / mass * timestep

    return positions, velocities

def velocity_verlet_forces(positions, velocities, forces, mass, timestep, box_size, fmm_lib, tree, fmm):
    """
    Performs one step of Velocity Verlet integration, including force calculation.
    """
    # Update positions
    positions += velocities * timestep + 0.5 * forces / mass * timestep**2
    positions = apply_periodic_boundary_conditions(positions, box_size)

    # Update velocities (first half)
    velocities += 0.5 * forces / mass * timestep

    # Compute new forces
    new_forces = compute_forces(positions, charges, fmm_lib, tree, fmm)

    # Update velocities (second half)
    velocities += 0.5 * new_forces / mass * timestep

    return positions, velocities, new_forces

# Main simulation
if __name__ == "__main__":
    # Initialize particle positions, velocities, and charges
    positions, velocities = initialize_particles(NUM_PARTICLES, BOX_SIZE, TEMPERATURE, MASS)
    charges = np.ones(NUM_PARTICLES)  # Example: all particles have charge +1

    # Choose FMM method (Laplace or Helmholtz)
    use_helmholtz = False  # Change to True to use Helmholtz

    if use_helmholtz:
        positions = positions.astype("complex128")
        velocities = velocities.astype("complex128")
        charges = charges.astype("complex128")
        fmm_lib = helmholtz
        # Initialize Helmholtz FMM
        p = 8 # Expansion order
        ncrit = 100 # Max particles per leaf node
        wavek = 0.8 # Example wavenumber
        fmm = fmm_lib.HelmholtzFmm(p=p, ncrit=ncrit, wavek=wavek)

    else:
        fmm_lib = laplace
        # Initialize Laplace FMM
        p = 5 # Expansion order
        ncrit = 100 # Max particles per leaf node
        fmm = fmm_lib.LaplaceFmm(p=p, ncrit=ncrit)

    # Initialize sources and targets using the same positions
    sources = fmm_lib.init_sources(positions, charges)
    targets = fmm_lib.init_targets(positions)

    # Setup the FMM tree
    start_time = time.time()
    tree = fmm_lib.setup(sources, targets, fmm)
    setup_time = time.time() - start_time
    print(f"FMM setup time: {setup_time:.4f} seconds")

    # Compute initial forces
    forces = compute_forces(positions, charges, fmm_lib, tree, fmm)

    # Run the simulation
    print("Starting simulation...")
    total_time = time.time()
    for step in range(NUM_STEPS):
        start_time = time.time()
        
        positions, velocities, forces = velocity_verlet_forces(
            positions, velocities, forces,
            MASS, TIMESTEP, BOX_SIZE, fmm_lib, tree, fmm
        )

        step_time = time.time() - start_time
        print(f"Step {step + 1}/{NUM_STEPS}, Time: {step_time:.4f} seconds")

    total_time = time.time() - total_time
    print(f"{10**7} time steps would take {(total_time / NUM_STEPS * 10**7) / (60*60):.1f} hours")