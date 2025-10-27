from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import Any, Literal, Optional

import espressomd
import espressomd.constraints
import espressomd.magnetostatics
import espressomd.polymer
import espressomd.shapes
import matplotlib.pyplot as plt
import numpy as np
from espressomd import System
from espressomd.interactions import FeneBond, HarmonicBond

import constants
from analysis import magnetic
from common import log
from constants import ParticleTypeData
from utils import myrandom
from utils.mymath import PointCloud, plot_ellipsoid
from visualization import plotting


def add_counterions(
    *,
    system: System,
    rng: np.random.Generator,
    pdata: constants.ParticleTypeData,
    radius: Optional[float] = None,
) -> None:
    total_counterion_charge = -system.part.select(type=pdata.ptype).q.sum()
    n_cions = int(round(abs(total_counterion_charge)))
    cion_charge = float(total_counterion_charge) / n_cions

    center = np.copy(system.box_l) / 2
    if radius is None:
        pos = myrandom.uniform_box(rng=rng, n=n_cions, box_length=system.box_l, center=center)
    else:
        pos = myrandom.random_uniform_ball(rng=rng, n=n_cions, R=radius, center=center)

    cion_ptype = constants.ION_TO_COUNTERION[pdata.ptype]
    cion_type = np.full(n_cions, cion_ptype) if n_cions > 1 else cion_ptype

    pdata_kwargs = dict(
        pos=pos,
        type=cion_type,
        q=np.full(n_cions, cion_charge),
    )
    pdata_kwargs = _fix_pdata_shapes(pdata_kwargs)
    system.part.add(**pdata_kwargs)


def add_polymer(*, system: System, chain_length: int, charge_per_bead: float, seed: int) -> np.ndarray:
    initial_bead_positions = espressomd.polymer.linear_polymer_positions(
        n_polymers=1,
        beads_per_chain=chain_length,
        bond_length=constants.BEAD.diameter,
        seed=seed,
        min_distance=0.6 * constants.BEAD.diameter,
        respect_constraints=True,
    )[0]
    chain = system.part.add(
        pos=initial_bead_positions,
        type=np.full(len(initial_bead_positions), constants.BEAD.ptype),
        q=np.full(len(initial_bead_positions), charge_per_bead),
    )
    # intentionally not done here
    # for b1, b2 in itertools.pairwise(chain):
    #     b1.add_bond((bond, b2))

    return chain.id


def add_bond_type(*, system: System, bond_type: str):
    if bond_type == "fene":
        log.info("Creating fene bond")
        bead_bond = FeneBond(**constants.FENE_PARAMS)
    elif bond_type == "harmonic":
        log.info("Creating harmonic bond")
        bead_bond = HarmonicBond(**constants.HARMONIC_PARAMS)
    else:
        raise ValueError(f"Unknown bond type '{bond_type}'")
    system.bonded_inter.add(bead_bond)

    return bead_bond


def add_spherical_constraint(*, system: System, radius: float):
    confinement_shape = espressomd.shapes.Sphere(center=system.box_l / 2, radius=radius, direction=-1)  # type: ignore
    confinement = espressomd.constraints.ShapeBasedConstraint(
        shape=confinement_shape,
        penetrable=False,
        only_positive=True,
        particle_type=constants.GEL_BOUNDARY.ptype,
    )
    system.constraints.add(confinement)


def add_wca(system: System, p1: ParticleTypeData, p2: ParticleTypeData) -> None:
    system.non_bonded_inter[p1.ptype, p2.ptype].wca.set_params(
        epsilon=constants.EPS, sigma=(p1.diameter + p2.diameter) / 2
    )


def _fix_pdata_shapes(pdata: dict[str, Any]) -> dict[str, Any]:
    for k, v in pdata.items():
        pdata[k] = np.squeeze(v)

    positions = pdata["pos"]
    if np.ndim(positions) != 2:
        n_particles = 1
    else:
        n_particles = np.shape(positions)[0]

    if n_particles == 1:
        for k, v in pdata.items():
            if np.size(v) == 1:
                pdata[k] = v.item()

    return pdata


def add_nanoparticles(
    *,
    system,
    volume_fraction: float,
    rng: np.random.Generator,
    mnp_charge: float,
    max_bead_overlap: float = 0.2 * constants.BEAD.diameter,
    strict_mnp_count=True,
):
    box_volume = float(np.prod(system.box_l.copy()))
    mnp_volume = constants.NANOPARTICLE.volume
    total_mnp_volume = volume_fraction * box_volume
    n_nanoparticles = int(round(total_mnp_volume / mnp_volume))

    log.info(f"Trying to create {n_nanoparticles} MNPs")
    pos_candidates = myrandom.uniform_box(
        rng=rng,
        box_length=system.box_l[0],
        center=system.box_l / 2,
        n=n_nanoparticles,
    )

    beads = system.part.select(type=constants.BEAD.ptype)
    min_dist = constants.NANOPARTICLE.radius + constants.BEAD.radius - max_bead_overlap

    from scipy.spatial import distance

    # shape (n_beads, n_nanoparticles)
    distances = distance.cdist(beads.pos, pos_candidates)
    has_mnp_ok_overlap = np.prod(distances > min_dist, axis=0).astype(bool)
    assert len(has_mnp_ok_overlap) == pos_candidates.shape[0]
    # remove mnp positions with too much overlap
    mnp_positions = pos_candidates[has_mnp_ok_overlap]
    n_nanoparticles = len(mnp_positions)
    if strict_mnp_count:
        assert n_nanoparticles >= 10, f"Too few ({n_nanoparticles} < 10) MNPs"

    if n_nanoparticles < 1:
        return

    log.info(f"Adding {n_nanoparticles} MNPs")
    pdata = {
        "pos": mnp_positions,
        # NOTE: strength of dipole moment is set by the dipolar
        # interaction prefactor
        # assume all have a dipm of 1.0
        "dip": myrandom.random_uniform_ball(
            rng=rng,
            n=n_nanoparticles,
            on_surface=True,
            keepdim=True,
        ),
        "type": np.full(n_nanoparticles, constants.NANOPARTICLE.ptype),
        # TODO: Why should only MNPs have a rotation?
        "rotation": np.full((n_nanoparticles, 3), True, dtype=bool),
        "q": np.full(n_nanoparticles, mnp_charge),
    }
    pdata = _fix_pdata_shapes(pdata)
    pdata["rotation"] = pdata["rotation"].tolist()
    nanoparticles = system.part.add(**pdata)

    return nanoparticles


def init_magnetostatics(system: System, dp3m_params: dict[str, Any]) -> None:
    log.info("Initializing magnetostatics")
    pdict_before_dp3m = system.part.all().to_dict()

    dipolar_interaction = espressomd.magnetostatics.DipolarP3M(**dp3m_params)
    try:
        system.actors.add(dipolar_interaction)
    except AttributeError:  # espresso 4.3 does not have actors member anymore
        system.magnetostatics.solver = dipolar_interaction
    dp3m_params.update(dipolar_interaction.get_params())
    dp3m_params["tune"] = False

    # undo particle change due to tuning
    system.part.all().remove()
    system.part.add(pdict_before_dp3m)


def run_minimize(
    *,
    system: System,
    steps_per_iteration=10,
    min_iterations=0,
    max_iterations=10**9,
    min_rel_energy_change=0.05,
    min_force=None,
    **steepest_descent_params,
):
    if steepest_descent_params == dict():
        system.integrator.run(0)
        max_force = np.max(np.linalg.norm(system.part.all().f, axis=1))
        # max_disp = 0.005 * constants.BEAD.diameter # previously
        max_disp = 0.01 * constants.BEAD.diameter
        steepest_descent_params = {
            "f_max": 0,
            "gamma": max_disp / max_force,
            "max_displacement": max_disp,
        }
    else:
        system.integrator.run(0)
        max_force = np.max(np.linalg.norm(system.part.all().f, axis=1))

    try:
        system.thermostat.suspend()
    except AttributeError:  # espresso 4.3 does not have this attribute anymore
        system.thermostat.turn_off()
    system.integrator.set_steepest_descent(**steepest_descent_params)

    max_force_below_counter = 0

    energy = float(system.analysis.energy()["total"])
    log.info(f"Energy/Force at minimization start: {energy:.2e}/{max_force:.2e}")
    for i in range(max_iterations):
        t0 = time.time()
        system.integrator.run(steps_per_iteration)
        gc.collect()
        energy_new = float(system.analysis.energy()["total"])
        # Prevent division by zero errors:
        if abs(energy) < sys.float_info.epsilon:
            log.info(f"WARNING: Energy too small to continue minimization: {energy}")
            break
        relative_energy_change = (energy - energy_new) / energy
        energy = energy_new
        log.info(f"Minimization force: {max_force:>8.2e}")

        # adapively choose gamma such that the displacement of the particles
        # scales with the force acting on it, but where the particle with the
        # largest force is displaces by a maximum of
        # `steepest_descent_params["max_displacement"]`
        max_force = np.max(np.linalg.norm(system.part.all().f, axis=1))
        steepest_descent_params["gamma"] = min(
            steepest_descent_params["max_displacement"] / max_force, 10
        )  # cap gamma at 10
        system.integrator.set_steepest_descent(**steepest_descent_params)

        dt = time.time() - t0

        if min_force is None:
            if i > min_iterations and abs(relative_energy_change) < min_rel_energy_change:
                break
        else:
            # force has to be below the limit for a few time steps
            if max_force < min_force:
                max_force_below_counter += 1
                if i > min_iterations and max_force_below_counter >= 10:
                    break
            else:
                max_force_below_counter = 0

    try:
        system.thermostat.recover()
    except AttributeError:  # espresso 4.3 does not have this attribute anymore
        system.thermostat.set_langevin(
            kT=constants.KT,
            gamma=constants.LANGEVIN_GAMMA,
            seed=time.time_ns() % (2**31 - 1),  # TODO: ideally I would like to get the previous state somehow?
        )
    system.integrator.set_vv()
    energy = system.analysis.energy()["total"]
    log.info("")
    log.info(f"Energy/Force at minimization end: {energy:.2e}/{max_force:.2e}")


class Visualizer:
    def __init__(
        self,
        system: System,
        img_dir: Path,
        operation_mode: Literal["off", "2d", "full"],
    ) -> None:
        self._system = system
        self._img_dir = img_dir
        self._operation_mode = operation_mode

        if self._operation_mode != "full":
            self._visualizer = None
            return

        window_size = 2048
        import espressomd.visualization

        self._visualizer = espressomd.visualization.openGLLive(
            system,
            particle_coloring="type",
            particle_sizes="auto",
            background_color=[1, 1, 1],
            window_size=[window_size, window_size],
            draw_box=True,
            draw_axis=False,
            # camera_position=(system.box_l[0]/2,
            #                 system.box_l[1]/2,
            #                 system.box_l[2]),
        )

    def screenshot(self, name: str) -> None:
        if self._operation_mode == "off":
            return

        self._img_dir.mkdir(exist_ok=True, parents=True)
        ogl_img_path = self._img_dir / f"OGL_{name}.png"
        mpl_img_path = self._img_dir / f"MPL_{name}.png"

        # if any of them exists, the state (and thus screenshot) would
        # probably no longer match the image name
        if ogl_img_path.exists() or mpl_img_path.exists():
            return

        self._ogl_screenshot(ogl_img_path)
        self._mpl_screenshot(mpl_img_path)

    def _ogl_screenshot(self, img_path: Path) -> None:
        if self._visualizer is None:
            return
        self._visualizer.screenshot(str(img_path))

    def _mpl_screenshot(self, img_path: Path) -> None:
        particles = self._system.part.all()
        pos = particles.pos_folded
        species = particles.type
        bead_pos = pos[species == constants.BEAD.ptype]

        fig = plt.figure(figsize=(12, 24))

        # 3D scatterplot with ellipsoid
        ax3d = fig.add_subplot(211, projection="3d")
        plotting.myscatter(ax3d, pos, species, dim=3, only_beads=False)
        ellipsoid = PointCloud(bead_pos).ellipsoid_of_inertia
        plot_ellipsoid(
            ax3d,
            *ellipsoid.unnormalized_axes,
            center=bead_pos.mean(axis=0),
            color="black",
        )
        # plotting.plot_box(ax3d, self._system.box_l[0], dim=3)

        # 2D scatterplot
        ax = fig.add_subplot(212)
        ax.set_aspect("equal")
        plotting.myscatter(ax, pos, species)
        # plotting.plot_box(ax, self._system.box_l[0], dim=2)

        # fig.savefig(str(img_dir / f"MPL_{img_path.name}"))
        fig.savefig(str(img_path))


def remove_excess_mnps(system: System, rng: np.random.Generator, strict_mnp_count=True) -> None:
    nanoparticles = system.part.select(type=constants.NANOPARTICLE.ptype)
    are_inside = magnetic.are_nanoparticles_in_gel(
        bead_positions=system.part.select(type=constants.BEAD.ptype).pos,
        np_positions=nanoparticles.pos,
    )
    log.info("are_inside")
    log.info(are_inside)
    log.info(np.count_nonzero(are_inside))
    log.info("")

    mnps_ids_to_be_deleted = [
        p.id
        for is_inside, p in zip(are_inside, nanoparticles, strict=True)  # type: ignore
        if not is_inside
    ]

    log.info(f"Removing {len(mnps_ids_to_be_deleted)}/{len(nanoparticles)} MNPs")

    # sanity check
    if len(mnps_ids_to_be_deleted) == 0:
        points = system.part.all().pos
        types = system.part.all().type
        point_file_path = Path(f"./tmp/{id(system)}.npz")
        point_file_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(point_file_path, pos=points, types=types)
        raise RuntimeError(
            "It is unlikely that all MNPs are inside the gel. "
            "Seems rather like a bug."
            f"Wrote data file to: {point_file_path}"
        )

    n_inside = len(nanoparticles) - len(mnps_ids_to_be_deleted)
    if n_inside / len(nanoparticles) < 0.01:
        raise RuntimeError(f"Too few ({n_inside}) MNPs are inside gel")

    system.part.by_ids(mnps_ids_to_be_deleted).remove()
    nanoparticles = system.part.select(type=constants.NANOPARTICLE.ptype)

    # remove MNP counter ions to ensure charge neutrality
    mnp_counterions = system.part.select(type=constants.NP_COUNTER_ION.ptype)
    assert mnp_counterions is not None
    n_conterions_to_remove = int(abs(nanoparticles.q.sum() + mnp_counterions.q.sum()))
    log.info(f"Removing {n_conterions_to_remove}/{len(mnp_counterions)} MNP counterions")
    ids_to_remove = rng.choice(mnp_counterions.id, n_conterions_to_remove, replace=False)
    system.part.by_ids(ids_to_remove).remove()
    mnp_counterions = system.part.select(type=constants.NP_COUNTER_ION.ptype)

    # I have to modify counter ion charge as well
    # in case it does not divide
    if abs(mnp_counterions.q.sum() + nanoparticles.q.sum()) > 1e-12:
        new_mnp_charge = -nanoparticles.q.sum() / len(mnp_counterions)
        log.info(
            "Removing MNPs made the system charged. "
            "MNP counterion charge has to be modified from "
            f"{mnp_counterions.q[0]:.2e} to {new_mnp_charge:.2e}"
        )
        mnp_counterions.q = new_mnp_charge

    n_mnps_left = len(nanoparticles.id)
    if strict_mnp_count:
        assert n_mnps_left > 1, f"Too few ({n_mnps_left} < 2) MNPs left after removal"


def tune_skin(*, system: System, min_skin=0.2, max_skin=2.0, tol=0.2, int_steps=10) -> None:
    time_before_tune = system.time  # skin tuning changes time (d3pm not)
    pdict_before = system.part.all().to_dict()

    log.info(f"Tuning skin ... (was {system.cell_system.skin})")
    system.cell_system.tune_skin(
        min_skin=min_skin,
        max_skin=max_skin,
        tol=tol,
        int_steps=int_steps,
    )
    log.info(f"Tuned skin to {system.cell_system.skin}")

    # undo particle change due to tuning
    system.part.all().remove()
    system.part.add(pdict_before)
    system.time = time_before_tune
