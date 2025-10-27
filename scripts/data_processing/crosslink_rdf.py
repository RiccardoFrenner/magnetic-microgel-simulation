"""
Write out the radial distribution function of 'crosslink positions' (see file `crosslink_positions.py`)
"""

import argparse
from pathlib import Path


import espressomd
import espressomd.observables

import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("bead_pos_path", type=Path)
args = parser.parse_args()


bead_pos = np.load(args.bead_pos_path)

gel_dir_path = args.bead_pos_path.parent
while "current_checkpoint.txt" not in [p.name for p in gel_dir_path.iterdir()]:
    gel_dir_path = gel_dir_path.parent

current_checkpoint_name = (gel_dir_path / "current_checkpoint.txt").read_text()
current_checkpoint_path = gel_dir_path / "checkpoints" / current_checkpoint_name

crosslink_ids = np.load(current_checkpoint_path / "crosslinks.npy")

points = np.vstack([bead_pos[crosslink_ids[:, 0]], bead_pos[crosslink_ids[:, 1]]])






# RDF


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
random_string = hex(np.abs(hash(str(args.bead_pos_path))))
out_dir = Path("data/processed")
out_dir.mkdir(exist_ok=True, parents=True)
output_filename = f"{random_string}.png"

# Plot and save the RDF plot
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 7))
ax1.plot(rs, rdf)
ax2.semilogy(rs, rdf)
plt.xlabel('r')
plt.ylabel('RDF')
plt.savefig(out_dir / output_filename)
plt.close()

# Append the random string and points file path to "points_to_string.txt"
with open(out_dir / "points_to_string.txt", "a") as f:
    f.write(f"{random_string},{str(args.bead_pos_path)}\n")

print(f"Plot saved as {out_dir / output_filename}. Random string and file path appended to points_to_string.txt.")
