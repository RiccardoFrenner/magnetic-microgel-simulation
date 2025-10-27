"""This file contains the set of parameter variations (configs) to be tested when investigating their impact on gel properties (e.g., stiffness, structure)."""

gel_configs = {
    "charge_per_bead": [1 / 20, 3 / 20, 5 / 20],
    "volume_fraction": [0.3, 0.45, 0.6],
    "crosslink_percentage": [0.2, 0.7],
    "agent_force_field": ["none", "harmonic", "exp"],
}
