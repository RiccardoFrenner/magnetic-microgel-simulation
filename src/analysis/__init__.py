from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from common import CheckpointDir, GelDir


@dataclass(frozen=True)
class GelProxy:
    n_chains: int
    bead_points: np.ndarray
    crosslinks: np.ndarray

    @classmethod
    def from_gel_dir(cls, gel_dir_path: Path):
        gel_dir = GelDir(gel_dir_path)
        latest_checkpoint = CheckpointDir(gel_dir.current_checkpoint_dir)
        return cls(
            n_chains=gel_dir.config.n_chains,
            bead_points=latest_checkpoint.get_bead_points(),
            crosslinks=latest_checkpoint.crosslinks_numpy(),
        )

    @cached_property
    def n_crosslinks(self) -> int:
        return self.crosslinks.shape[0]

    @cached_property
    def n_beads(self) -> int:
        return self.bead_points.shape[0]

    @cached_property
    def n_beads_per_chain(self) -> int:
        n = self.n_beads / self.n_chains
        assert np.ceil(n) == np.floor(n)
        return int(n)

    @cached_property
    def chain_pairs(self) -> np.ndarray:
        """All bonds without crosslinks.
        [(i1, j1), (i2, j2), ...]

        shape = (n_chains * (n_beads_per_chain - 1))
        shape = (n_beads - n_chains, 2)
        """
        n = self.n_beads_per_chain
        m = self.n_chains

        base_seq = np.arange(n - 1)
        offsets = np.arange(m)[:, np.newaxis] * (n + 0)
        result = base_seq + offsets
        a = result.ravel()
        return np.stack([a, a + 1], axis=1)

    @cached_property
    def chain_pos(self) -> np.ndarray:
        """
        [chain1, chain2, ...] | chain=[b1, b2, ...] | b=[x,y,z]

        shape = (n_chains, n_beads_per_chain, 3)
        """
        return np.array(np.split(self.bead_points, self.n_chains))

    @cached_property
    def crosslink_pos(self) -> np.ndarray:
        """
        [(p11, p12), (p21, p22), ...] | p=[x,y,z]

        shape = (n_crosslinks, 2, 3)
        """
        return np.array(
            [[self.bead_points[i], self.bead_points[j]] for i, j in self.crosslinks]
        )

    @cached_property
    def crosslink_centers(self):
        """
        [c1, c2, ...] | c=[x,y,z]

        shape = (n_crosslinks, 3)
        """
        return self.crosslink_pos.mean(axis=1)

    @cached_property
    def n_beads_between_crosslinks(self):
        "Number of beads between two crosslinks"

        n_beads_between = []
        chain_length = self.bead_points.shape[0] // self.n_chains

        n_chains_so_far = 0
        crosslink_bead_indices = sorted(self.crosslinks.flatten())
        for i in range(len(crosslink_bead_indices) - 1):
            b0 = crosslink_bead_indices[i+0]
            b1 = crosslink_bead_indices[i+1]

            chain_idx_0 = b0 // chain_length
            chain_idx_1 = b1 // chain_length

            if chain_idx_0 != chain_idx_1:
                n_chains_so_far += 1
                print(n_chains_so_far, chain_idx_0, chain_idx_1)
                # we don't count dangling ends as beads between crosslinks since we want a pore size estimate here
                continue

            if b1-b0 == 0:
                print("NOT GOOD"*50)
                print(b1, b0)
                for cl in self.crosslinks:
                    print(cl)
                print("="*200)
                print()
                raise ValueError("ASDSAD")

            n_beads_between.append(b1 - b0 - 1)

        return n_beads_between