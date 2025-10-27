"""
Write out 'positions' of crosslinks. A crosslinks position is the center between the two beads it connects.
"""

import argparse
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("bead_pos_path", type=Path)
parser.add_argument("out_path", type=Path)
args = parser.parse_args()


bead_pos = np.load(args.bead_pos_path)

gel_dir_path = args.bead_pos_path.parent
while "current_checkpoint.txt" not in [p.name for p in gel_dir_path.iterdir()]:
    gel_dir_path = gel_dir_path.parent

current_checkpoint_name = (gel_dir_path / "current_checkpoint.txt").read_text()
current_checkpoint_path = gel_dir_path / "checkpoints" / current_checkpoint_name

crosslink_ids = np.load(current_checkpoint_path / "crosslinks.npy")

crosslink_pos = np.vstack([bead_pos[crosslink_ids[:, 0]], bead_pos[crosslink_ids[:, 1]]])

random_string = hex(np.abs(hash(str(args.points_file))))
output_filename = f"{random_string}.png"

np.save(args.out_path / output_filename, crosslink_pos)