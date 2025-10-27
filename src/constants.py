from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True, eq=True)
class ParticleTypeData:
    used_ptypes: ClassVar[set] = set()

    ptype: int
    charge: float = 0.0
    diameter: float = 1.0

    # def __post_init__(self):
    #     if self.ptype in ParticleTypeData.used_ptypes:
    #         raise ValueError(f"Particle of type '{self.ptype}' already exists")
    #     self.used_ptypes.add(self.ptype)

    @property
    def radius(self) -> float:
        return self.diameter / 2

    @property
    def volume(self) -> float:
        return 4 / 3 * math.pi * self.radius**3


BEAD = ParticleTypeData(
    ptype=0,
)
# CHARGED_BEAD = ParticleTypeData(
#     ptype=1,
#     charge=1.0,
# )
NANOPARTICLE = ParticleTypeData(
    ptype=2,
    diameter=10.0,
)
GEL_BOUNDARY = ParticleTypeData(
    ptype=3,
    diameter=0.0,
)
CROSSLINK_AGENT = ParticleTypeData(
    ptype=4,
    diameter=0.5,
)
# CROSSLINK_AGENT_BOUNDARY = ParticleTypeData(
#     ptype=5,
# )


ION_DIAMETER = 1.0
# SALT_P = ParticleTypeData(
#     ptype=6,
#     diameter=ION_DIAMETER,
#     charge=1.0,
# )
# SALT_N = ParticleTypeData(
#     ptype=7,
#     diameter=ION_DIAMETER,
#     charge=-1.0,
# )
BEAD_COUNTER_ION = ParticleTypeData(
    ptype=8,
    diameter=ION_DIAMETER,
    charge=-1.0,
)
NP_COUNTER_ION = ParticleTypeData(
    ptype=9,
    diameter=ION_DIAMETER,
    charge=1.0,
)

COUNTERION_TO_ION = {
    BEAD_COUNTER_ION.ptype: BEAD.ptype,
    NP_COUNTER_ION.ptype: NANOPARTICLE.ptype,
}

ION_TO_COUNTERION = {
    BEAD.ptype: BEAD_COUNTER_ION.ptype,
    NANOPARTICLE.ptype: NP_COUNTER_ION.ptype,
}

ION_SPECIES = list(p for p in COUNTERION_TO_ION.keys())

NM = 1.0  # 1 nm
KT = 1.0
EPS = 1.0
ELEMENTARY_CHARGE = 1.0
assert ELEMENTARY_CHARGE == 1.0, "Some of the above assumes this"

BJERRUM_LENGTH = 2.0  # for water

EXPERIMENTAL_NP_CHARGE = -14.0  # From https://doi.org/10.1021/acs.jpcb.5b03778

LANGEVIN_GAMMA = 5.0
MU_0 = 1.0

TIME_STEP = 0.01

DIPOLAR_LAMBDA = 0.58  # From Regine paper
# Prefactor `γ~ = γ μ^2  = μ0 μ^2 / (4π)` contains dipole moment since
# it is the same for all particles.
# This can be simplified by the dipolar interaction parameter
# `λ = γ μ^2 / (kBT σ^3)` => `γ~ = λ kBT σ^3`
PREFACTOR = DIPOLAR_LAMBDA * NANOPARTICLE.diameter**3 * KT
DP3M_PARAMS: dict[str, Any] = {
    "prefactor": PREFACTOR,
    "accuracy": 1e-4,
    "r_cut": 6.0,  # Needed for tuning to work
    "tune": True,
}
P3M_PARAMS: dict[str, Any] = {
    "prefactor": KT * BJERRUM_LENGTH * ELEMENTARY_CHARGE**2,
    "accuracy": 1e-3,
    "check_neutrality": True,
    "charge_neutrality_tolerance": 1e-10,
    "tune": True,
}

FENE_PARAMS = dict(
    k=30.0 * EPS / BEAD.diameter,
    d_r_max=1.5 * BEAD.diameter,
    r_0=0.0 * BEAD.diameter,
)

HARMONIC_PARAMS = dict(
    k=10.0 * EPS / BEAD.diameter,
    r_0=1.0 * BEAD.diameter,
)
