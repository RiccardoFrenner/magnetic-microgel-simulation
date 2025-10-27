from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from random import randint
from typing import Any

import constants
from utils.mymath import sphere_volume


@dataclass
class ProgramOptions:
    continue_past_crosslinking: bool = True
    visualize_ogl: bool = False
    visualize_mpl: bool = False
    exit_on_checkpoint: bool = False
    no_checkpoints: bool = False
    debug_forcefield_scale: float = 1.0

    @classmethod
    def from_dict(cls, d: dict[str, Any], consume: bool = True) -> "ProgramOptions":
        options = ProgramOptions()
        for k, v in copy.deepcopy(list(d.items())):
            if k not in options.__dict__:
                continue
            options.__dict__[k] = v
            if consume:
                d.pop(k)
        return options


@dataclass(eq=True)
class Config:
    new_rng_state: bool = False  # debugging flag
    seed: int = randint(1000, 2**31 - 1)

    n_chains: int = 400
    chain_length: int = 80

    # every 20th monomer has a charge of +1 eV
    # a bead has a diameter of 1 nm
    # a PNIPAM monomer has a diameter of about 0.3 - 0.4 nm
    # thus we say that a bead is 3 monomers
    # thus a bead has a charge of 3/20
    charge_per_bead: float = 3 / 20

    steps_melt_eq_max: int = 10**5
    steps_per_cl_iteration: int = 1  # performance tradeoff. Ideally this is 1

    # TODO: Make this dynamic such that not all different gel configs run for the same amount of steps
    # even though they require less. 10^7 is a good upper bound such that all configs are equilibrated
    steps_gel_eq_max: int = 10**7
    steps_mmgel_eq_max: int = 10**7

    # This is a tradeoff between cross-linking and equilibration time
    volume_fraction: float = 0.2  # during confinement

    crosslink_percentage: float = 0.2  # 0 -> no crosslinks, 1 -> maximum number of crosslinks
    agent_force_field: str = "harmonic"  # choose from {none, harmonic, exp}
    n_agents_per_crosslink: float = 1.0
    initial_diff_steps: int = 10**4  # minimum is 10**4 but 10**5 to be very safe

    # choose from {harmonic, fene}
    bond_type: str = "fene"

    # MNP config
    mnp_volume_fraction: float = 0.075
    # This is actually the negative mnp charge to simplify file naming
    mnp_charge: float = -constants.EXPERIMENTAL_NP_CHARGE
    mnp_diameter: float = constants.NANOPARTICLE.diameter
    # TODO: This is just a testing/debugging parameter. Should be a program option?
    strict_mnp_count: bool = False

    @property
    def constraint_radius(self) -> float:
        total_bead_volume = self.n_beads * constants.BEAD.volume
        constraint_volume = total_bead_volume / self.volume_fraction
        r = (3 / 4 * constraint_volume / math.pi) ** (1 / 3)
        return r

    @property
    def initial_box_l(self) -> float:
        # Too large: P3M throws complex value not zero error
        # return 20*self.constraint_radius
        # Still too large: P3M throws complex value not zero error
        # return 14*self.constraint_radius
        # Too large because simulation takes too long and RAM cost too high
        # return 11*self.constraint_radius
        # Too small because gels get much larger than 170 sigma
        # return 6*self.constraint_radius

        # Let's see how this goes.
        # return 280.0  # approx = 10*self.constraint_radius
        x = 280 / (400 * 80) ** 0.5
        return round(x * self.n_beads**0.5, 1)  # = 280 for 400*80 beads

    @property
    def final_box_l(self) -> float:
        # TODO: We could make the box expand dynamically as the gel increases its size considerably during the whole simulation
        return self.initial_box_l

    @property
    def n_beads(self) -> int:
        return self.n_chains * self.chain_length

    @property
    def bead_density(self) -> float:
        return self.n_beads / sphere_volume(self.constraint_radius)

    @property
    def max_crosslinks(self) -> int:
        return self.n_beads // 2

    @property
    def n_mnps(self) -> int:
        return int(self.mnp_volume_fraction * self.final_box_l**3 / constants.NANOPARTICLE.volume)

    @property
    def n_crosslinks(self) -> int:
        # desired number of crosslinks
        return int(self.crosslink_percentage * self.max_crosslinks)

    def to_file(self, file_path: Path) -> None:
        file_path.write_text(json.dumps(asdict(self), indent=2))
        # with open(file_path, "w", encoding="utf-8") as f:
        #     for k, v in asdict(self).items():
        #         f.write(f"{k} = {v}\n")

    @classmethod
    def from_file(cls, file_path: Path) -> "Config":
        # if file_path.suffix == ".json":
        return cls(**json.loads(file_path.read_text()))
        # elif file_path.suffix == ".ini":
        # return cls(**read_ini_file(file_path))

    @classmethod
    def compare_configs(cls, conf1: "Config", conf2: "Config") -> bool:
        # print each member where they are different
        conf1_dict = asdict(conf1)
        conf2_dict = asdict(conf2)
        for k, v in conf1_dict.items():
            if k == "seed":
                continue

            if v != conf2_dict[k]:
                print(f"{k}: {v} != {conf2_dict[k]}")
                return False
        return True

    def __post_init__(self):
        # how many crosslinks are epiricial needed to ensure full connectivity
        if self.n_crosslinks < 5 * self.n_chains:
            raise ValueError(
                f"Too few crosslinks to ensure full connectivity: {self.n_crosslinks=} < {5*self.n_chains=}"
            )
        if self.crosslink_percentage > 1.0:
            raise ValueError(f"{self.crosslink_percentage=} > 1.0")

        constants.NANOPARTICLE = constants.ParticleTypeData(
            constants.NANOPARTICLE.ptype, constants.NANOPARTICLE.charge, self.mnp_diameter
        )


if __name__ == "__main__":
    # conf = Config(n_chains=70, chain_length=60, volume_fraction=0.6, crosslink_percentage=0.4)
    conf = Config(n_chains=400, chain_length=80, volume_fraction=0.2, crosslink_percentage=0.4)
    print(f"Num. crosslinks per chain: {conf.n_crosslinks / conf.n_chains:g}")
    print(f"{conf.n_chains=}")
    print(f"{conf.chain_length=}")
    print(f"{conf.initial_box_l=}")
    print(f"{conf.final_box_l=}")
    print(f"{conf.volume_fraction=}")
    print(f"{conf.constraint_radius=}")
    print(f"{conf.bead_density=}")
    print(f"{conf.n_crosslinks=}")
    print(f"{conf.n_mnps=}")
