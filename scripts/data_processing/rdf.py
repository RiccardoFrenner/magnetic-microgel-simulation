"""
This script loads 3D particle positions from a NumPy file to calculate the Radial Distribution Function (RDF) using espressomd. The goal is to analyze the structural characteristics of the input data. The resulting RDF plot is saved as a uniquely named PNG file, and a log entry records the link between the output filename and the source data file.
"""

import random
import string
import argparse
from pathlib import Path

import espressomd
import espressomd.observables

import numpy as np
import matplotlib.pyplot as plt


# Function to generate a random string of a given length
# def generate_random_string(length=12):
#     letters = string.ascii_lowercase
#     return ''.join(random.choice(letters) for i in range(length))


# Parse arguments
parser = argparse.ArgumentParser(description="Load points from a numpy file and calculate RDF.")
parser.add_argument("points_file", type=Path, help="Path to the numpy file containing points.")
args = parser.parse_args()


# Load points from the numpy file
points = np.load(args.points_file)
box_l = points.max() - points.min()

# Initialize the system and add particles
system = espressomd.System(box_l=[box_l]*3)
system.part.add(pos=points)

# RDF calculation
rdf_obs = espressomd.observables.RDF(ids1=system.part.all().id, min_r=0, max_r=6, n_r_bins=100)
rs = rdf_obs.bin_centers()
rdf = rdf_obs.calculate()

# Generate random filename
# random_string = generate_random_string()
random_string = hex(np.abs(hash(str(args.points_file))))
output_filename = f"{random_string}.png"

# Plot and save the RDF plot
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(rs, rdf, label='simulated')
plt.legend()
plt.xlabel('r')
plt.ylabel('RDF')
plt.savefig(output_filename)
plt.close()

# Append the random string and points file path to "points_to_string.txt"
with open("points_to_string.txt", "a") as f:
    f.write(f"{random_string},{str(args.points_file)}\n")

print(f"Plot saved as {output_filename}. Random string and file path appended to points_to_string.txt.")
