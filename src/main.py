from __future__ import annotations

import argparse
import datetime
import gc
import itertools
import json
import time
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Optional

import espressomd
import espressomd.electrostatics
import networkx as nx
import numpy as np

import constants
import core
import crosslinking
from analysis.network import graph_from_beads
from common import CheckpointDir, GelDir, init_file_logging, log
from config import Config, ProgramOptions
from utils import myjson

INITIAL_SKIN = 0.4


def read_ini_file(path: Path) -> dict[str, Any]:
    config = ConfigParser()
    config.read(path)
    return {k: v for k, v in config["DEFAULT"].items()}


def parse_commandline():
    parser = argparse.ArgumentParser(
        "Create a magnetic microgel",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("gel_dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n_chains", type=int)
    parser.add_argument("--chain_length", type=int)
    parser.add_argument("--debug_forcefield_scale", type=float, default=1.0)
    parser.add_argument("--charge_per_bead", type=float)
    parser.add_argument("--volume_fraction", type=float)
    parser.add_argument("--crosslink_percentage", type=float)
    parser.add_argument("--agent_force_field", type=str)
    parser.add_argument("--mnp_charge", type=float)
    parser.add_argument("--strict_mnp_count", type=bool)
    parser.add_argument("--continue_past_crosslinking", type=lambda s: bool(int(s)))
    parser.add_argument("--bond_type", type=str)
    parser.add_argument("--mnp_volume_fraction", type=float)
    parser.add_argument("--mnp_diameter", type=float)
    parser.add_argument(
        "--steps_melt_eq_max",
        type=int,
        help="Number of time steps to equilibrate the polymer melt (not crosslinked) for.",
    )
    parser.add_argument(
        "--steps_gel_eq_max",
        type=int,
        help="Number of time steps to equilibrate the crosslinked gel for.",
    )
    parser.add_argument(
        "--steps_mmgel_eq_max",
        type=int,
        help="Number of time steps to equilibrate the magnetic microgel (mmgel) for.",
    )
    parser.add_argument("--n_agents_per_crosslink", type=float)
    parser.add_argument(
        "--initial_diff_steps",
        type=int,
        help="Number of time steps to let the agents diffuse before crosslinking starts",
    )

    parser.add_argument("--visualize_ogl", action="store_true", default=False)
    parser.add_argument("--visualize_mpl", action="store_true", default=False)
    parser.add_argument("--no_checkpoints", action="store_true", default=False)
    clargs = vars(parser.parse_args())

    print(clargs)
    gel_dir = GelDir(path=clargs.pop("gel_dir"))
    program_options = ProgramOptions.from_dict(clargs)
    print(program_options)
    print()
    if gel_dir.config_path.exists():
        config = Config.from_file(gel_dir.config_path)
    else:
        config = Config(**clargs)

    return (gel_dir, program_options, config)


def get_p3m_params(charge_per_bead: float, box_l: float) -> dict[str, Any]:
    def make_even(n: int) -> int:
        if n % 2 == 0:
            return n
        return n + 1

    p3m_params = constants.P3M_PARAMS.copy()

    if abs(charge_per_bead - 0.05) < 0.01:
        p3m_params.update(
            dict(
                # mesh=300,
                mesh=make_even(int(0.6 * box_l)),
                r_cut=3.0,
                tune=True,
            )
        )
    elif abs(charge_per_bead - 0.15) < 0.01:
        p3m_params.update(
            dict(
                # mesh=360,
                mesh=make_even(int(0.7 * box_l)),
                r_cut=3.0,
                tune=True,
            )
        )
    elif abs(charge_per_bead - 0.25) < 0.01:
        p3m_params.update(
            dict(
                # mesh=420,
                mesh=make_even(int(0.8 * box_l)),
                r_cut=3.0,
                tune=True,
            )
        )
    else:
        raise ValueError(f"Unexpected charge per bead: {charge_per_bead}")

    return p3m_params


class IntegrationCallback:
    def __init__(self, *, fun, interval: int) -> None:
        self.fun = fun
        self.interval = max(1, interval)

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)


def integrate_system(
    *,
    system: espressomd.System,
    start_steps: int,
    end_steps: int,
    control_file_path: Path,
    loop_name: str,
    callbacks: Optional[list[IntegrationCallback]] = None,
):
    def is_stop_requested() -> bool:
        try:
            data = read_ini_file(control_file_path)
            return data.get(f"{loop_name}_stop", False)
        except FileNotFoundError:
            return False

    if start_steps >= end_steps:
        return

    # if is_stop_requested():
    #     return

    log.info(f"Integrating {loop_name} from {start_steps} to {end_steps}")

    STEPS_PER_IT = 100
    CONTROL_FILE_CHECK_INTERVAL_IN_SECONDS = 4 * 60.0
    time_since_last_check = 2 * CONTROL_FILE_CHECK_INTERVAL_IN_SECONDS

    dt_callbacks = 0
    dt = 0
    while start_steps < end_steps:
        dt = time.time()
        if time_since_last_check > CONTROL_FILE_CHECK_INTERVAL_IN_SECONDS:
            # if is_stop_requested():
            #     return
            time_since_last_check = 0

        dt_int = time.time()
        system.integrator.run(STEPS_PER_IT)
        dt_int = time.time() - dt_int
        start_steps += STEPS_PER_IT

        yield start_steps

        log.info(
            f"{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | "
            f"{system.time:>14.3f} | "
            f"{start_steps:>9}/{end_steps} | "
            f"{loop_name} integration took {dt_int:.2e} seconds for {STEPS_PER_IT}"
            f" steps (={STEPS_PER_IT / dt_int:.2f} steps per second)"
            f" (dt_cb={dt_callbacks:.2e}s)"
        )

        # this has to be after the yield otherwise state and
        # checkpoint name are not synced.
        dt_callbacks = time.time()
        if callbacks is not None:
            for callback in callbacks:
                if start_steps % callback.interval == 0:
                    callback(start_steps)
        dt_callbacks = time.time() - dt_callbacks
        dt = time.time() - dt
        time_since_last_check += dt


class Microgel:
    def __init__(self, *, gel_dir: GelDir, program_options: ProgramOptions, config: Config):
        self.gel_dir = gel_dir
        self.config = config
        self.program_options = program_options
        log.info(self.config)
        self.config.to_file(self.gel_dir.config_path)

        # ====================== checkpoint loading ======================
        loaded_ckpt_dir = CheckpointDir(path=self.gel_dir.current_checkpoint_dir)
        log.info(f"Loading checkpoint data from {loaded_ckpt_dir.path}")
        log.info(
            f"{(loaded_ckpt_dir.path / 'main_ckpt.json').resolve()} exists={(loaded_ckpt_dir.path / 'main_ckpt.json').resolve().exists()}"
        )
        ckpt_data = loaded_ckpt_dir.sdata()
        log.info(f"Data: {ckpt_data}")

        self.is_confinement_gone = ckpt_data.get("is_confinement_gone", False)
        self.is_melt_minimized = ckpt_data.get("is_melt_minimized", False)
        self.is_skin_tuned = ckpt_data.get("is_skin_tuned", False)
        self.steps_melt_eq = ckpt_data.get("steps_melt_eq", 0)
        self.steps_gel_eq = ckpt_data.get("steps_gel_eq", 0)
        self.is_ferrofluid_minimized = ckpt_data.get("is_ferrofluid_minimized", False)
        self.steps_mmgel_eq = ckpt_data.get("steps_mmgel_eq", 0)

        self.rng = np.random.default_rng(config.seed)
        if not config.new_rng_state:  # This is for debugging purposes
            self.rng.__setstate__(ckpt_data.get("rng_state", self.rng.__getstate__()))

        if "crosslinking_state" in ckpt_data:
            log.info("Loading crosslinking state")
            cl_ckpt = ckpt_data["crosslinking_state"]
            self.crosslinking_state = crosslinking.CrosslinkingState(
                active_agent_ids=loaded_ckpt_dir.active_agents(),
                crosslinks=loaded_ckpt_dir.crosslinks(),
                is_initialized=cl_ckpt["is_initialized"],
                has_finished=cl_ckpt["has_finished"],
                iterations=cl_ckpt["iterations"],
            )
        else:
            self.crosslinking_state = crosslinking.CrosslinkingState()

        # ====================== setup system ======================
        log.info("Setting up system")
        self.system = espressomd.System(
            box_l=[ckpt_data.get("box_l", config.initial_box_l)] * 3,
            periodicity=[True] * 3,
        )
        self.system.time_step = constants.TIME_STEP
        self.system.cell_system.skin = ckpt_data.get("skin", INITIAL_SKIN)
        self.system.cell_system.set_regular_decomposition(use_verlet_lists=True)
        self.system.time = ckpt_data.get("sim_time", 0.0)
        log.info(f"Time at start: {self.system.time}")
        grid = self.system.cell_system.node_grid
        n_cores = np.prod(grid)
        log.info(f"MPI Config grid: {list(grid)}")
        log.info(f"MPI Config cores: {n_cores}")
        log.info(f"Box: {self.system.box_l}")

        from packaging.version import parse

        if parse(np.__version__) >= parse("2.0"):
            rng_state = self.rng.bit_generator.state["state"]["state"]
        else:
            rng_state = self.rng.__getstate__()["state"]["state"]

        self.system.thermostat.set_langevin(
            kT=constants.KT,
            gamma=constants.LANGEVIN_GAMMA,
            seed=rng_state % (2**31 - 1),
        )
        self.visualizer = core.Visualizer(
            self.system,
            img_dir=self.gel_dir.img_dir,
            operation_mode="full"
            if program_options.visualize_ogl
            else ("2d" if program_options.visualize_mpl else "off"),
        )

        # ====================== setup interactions ======================
        for p1, p2 in itertools.combinations_with_replacement(
            [
                constants.BEAD,
                constants.GEL_BOUNDARY,
                constants.BEAD_COUNTER_ION,
            ],
            2,
        ):
            if p1 == constants.GEL_BOUNDARY and p1 == p2:
                continue
            core.add_wca(self.system, p1, p2)

        self.bead_bond = core.add_bond_type(system=self.system, bond_type=self.config.bond_type)
        self.system.bonded_inter.add(self.bead_bond)

        if not self.is_confinement_gone:
            core.add_spherical_constraint(system=self.system, radius=config.constraint_radius)

        # ====================== setup melt ======================
        if len(loaded_ckpt_dir.pdict()) > 0:
            fixed_dict = loaded_ckpt_dir.pdict()
            print(list(fixed_dict.keys()))
            # this is only if accidentally checkpoints before mnp where done with a "#define Dipoles" on and now it is not on anymore (also "#define ROTATION" but I always use them together so ...)
            if not espressomd.has_features("DIPOLES") and "dipm" in fixed_dict:
                del fixed_dict["quat"]
                del fixed_dict["dipm"]
                del fixed_dict["ext_torque"]
                del fixed_dict["omega_body"]
                del fixed_dict["omega_lab"]
                del fixed_dict["torque_lab"]
                del fixed_dict["rotation"]
            self.system.part.add(**fixed_dict)

        if len(self.system.part.select(type=constants.BEAD.ptype)) == 0:
            log.info(f"Creating {config.n_chains} polymers")
            for i in range(config.n_chains):
                core.add_polymer(
                    system=self.system,
                    chain_length=config.chain_length,
                    charge_per_bead=config.charge_per_bead,
                    seed=config.seed,
                )
            core.add_counterions(
                system=self.system,
                rng=self.rng,
                pdata=constants.BEAD,
                radius=config.constraint_radius,
            )
            self.visualizer.screenshot("MeltRandom")

        assert abs(self.beads.q.sum()) > 0
        assert abs(self.bead_counterions.q.sum()) > 0

        # create polymer bonds
        for i, (b1, b2) in enumerate(itertools.pairwise(self.beads)):
            # do not connect ends of different chains
            if (i + 1) % config.chain_length == 0:
                continue
            b1.add_bond((self.bead_bond, b2))

    def run(self):
        # ====================== remove overlap ======================
        if not self.is_melt_minimized:
            log.info("Minimizing polymer melt")
            core.run_minimize(system=self.system, min_force=300.0)
            self.is_melt_minimized = True
            self.visualizer.screenshot("MeltMinimized")
            self.write_checkpoint("MeltMinimized")

        # ====================== setup electrostatics ======================
        log.info("Setting up electrostatics")

        if self.gel_dir.p3m_params_path.exists():
            p3m_params = json.loads(self.gel_dir.p3m_params_path.read_text())
        else:
            p3m_params = get_p3m_params(self.config.charge_per_bead, self.system.box_l[0])

        pdict_before_p3m = self.system.part.all().to_dict()
        p3m = espressomd.electrostatics.P3M(**p3m_params)
        try:
            self.system.actors.add(p3m)  # p3m only starts tuning here
        except AttributeError:  # espresso 4.3 has no actors anymore
            self.system.electrostatics.solver = p3m

        # tune skin if the parameters have just been set
        if not self.is_skin_tuned:
            core.tune_skin(system=self.system)
            self.is_skin_tuned = True

        log.info(f"Energy post p3m 1: {self.system.analysis.energy()['total']}")

        # undo particle changes due to tuning
        try:
            system.actors  # trigger exception to check if we use espresso 4.3
            if p3m_params["tune"]:
                log.info("Undo tuning")
                self.system.part.all().remove()
                self.system.part.add(pdict_before_p3m)
        except UnboundLocalError:  # espresso 4.3 complains about existing particles when I do the above
            pass

        log.info(f"Energy post p3m 2: {self.system.analysis.energy()['total']}")

        p3m_params.update(p3m.get_params())
        p3m_params["tune"] = False
        self.gel_dir.p3m_params_path.write_text(json.dumps(p3m_params, cls=myjson.NumpyJSONEncoder, indent=4))

        log.info(f"Energy post p3m 3: {self.system.analysis.energy()['total']}")

        system = self.system
        config = self.config
        rng = self.rng

        # ====================== equilibrate melt ======================
        for self.steps_melt_eq in integrate_system(
            system=system,
            start_steps=self.steps_melt_eq,
            end_steps=config.steps_melt_eq_max,
            control_file_path=self.gel_dir.control_file_path,
            loop_name="MeltEq",
            callbacks=[
                IntegrationCallback(
                    fun=lambda step: self.write_checkpoint(f"MeltEq_{step}"),
                    interval=config.steps_melt_eq_max // 5,
                ),
                IntegrationCallback(
                    fun=lambda step: self.visualizer.screenshot(f"MeltEq_{step}"),
                    interval=config.steps_melt_eq_max // 5,
                ),
            ],
        ):
            pass

        self.visualizer.screenshot("MeltEqDone")
        self.write_checkpoint("MeltEqDone")
        log.info(f"Simulation Time: {system.time:g}")
        log.info(f"Energy post melt eq: {system.analysis.energy()['total']}")

        # ====================== cross-linking ======================
        def _are_bonded(p_id_0: int, p_id_1: int) -> bool:
            p0 = system.part.by_id(p_id_0)
            p1 = system.part.by_id(p_id_1)
            return (self.bead_bond, p1.id) in p0.bonds

        def _create_bond(p_id_0: int, p_id_1: int) -> None:
            p0 = system.part.by_id(p_id_0)
            p1 = system.part.by_id(p_id_1)
            p0.add_bond((self.bead_bond, p1))

        # init cross-linking
        log.info(f"Init cl with {len(self.crosslinking_state.crosslinks)}")
        self.crosslinker = crosslinking.AgentCrosslinker(
            system=system,
            rng=rng,
            program_options=self.program_options,
            are_bonded=_are_bonded,
            create_bond=_create_bond,
            confinement_radius=config.constraint_radius,
            desired_n_crosslinks=config.n_crosslinks,
            cl_config=crosslinking.AgentCrosslinkConfig(
                max_dist=1.2 * constants.BEAD.diameter,
                max_bonds_per_bead=3,  # this way only end beads can have more than one crosslink
                n_agents_per_crosslink=config.n_agents_per_crosslink,
                force_field_type=config.agent_force_field,
                initial_diff_steps=config.initial_diff_steps,
                chain_length=config.chain_length,
                output_path=self.gel_dir.path,
            ),
            state=self.crosslinking_state,
        )
        log.info(f"Energy post cl 1: {system.analysis.energy()['total']}")
        self.crosslinker.init()
        log.info(f"Energy post cl 2: {system.analysis.energy()['total']}")

        self.run_crosslinking()
        log.info(f"Energy post cl 3: {system.analysis.energy()['total']}")
        if not self.program_options.continue_past_crosslinking:
            return

        # ====================== equilibrate gel ======================
        for self.steps_gel_eq in integrate_system(
            system=system,
            start_steps=self.steps_gel_eq,
            end_steps=config.steps_gel_eq_max,
            control_file_path=self.gel_dir.control_file_path,
            loop_name="GelEq",
            callbacks=[
                IntegrationCallback(
                    fun=lambda step: self.write_checkpoint(f"GelEq_{step}"),
                    interval=config.steps_gel_eq_max // 20,
                ),
                IntegrationCallback(
                    fun=lambda step: self.visualizer.screenshot(f"GelEq_{step}"),
                    interval=config.steps_gel_eq_max // 5,
                ),
                IntegrationCallback(
                    fun=lambda step: self.write_bead_positions(folder_name="gel_eq", file_name=f"BeadPos_{step}.npy"),
                    interval=10000,  # was 5000
                ),
            ],
        ):
            pass

        self.visualizer.screenshot("GelEqDone")
        self.write_checkpoint("GelEqDone")
        log.info(f"Simulation Time: {system.time:g}")

        # ====================== ferrofluid interactions ======================
        log.info("Setting up ferrofluid interactions")
        assert all(system.periodicity)
        espressomd.assert_features(["ROTATION"])
        espressomd.assert_features(["DIPOLES"])

        # Is about 2.5x slower than the regular decomposition on 8 MPI cores with 600 MNPs
        # on the bw uni cluster
        # system.cell_system.set_hybrid_decomposition(
        #     n_square_types={constants.NANOPARTICLE.ptype},
        #     cutoff_regular=constants.BEAD.diameter * 2**(1/6),
        # )

        for p1, p2 in itertools.combinations_with_replacement(
            [
                constants.BEAD,
                constants.NANOPARTICLE,
                constants.NP_COUNTER_ION,
            ],
            2,
        ):
            if p1.ptype == constants.BEAD.ptype and p2.ptype == p1.ptype:
                continue
            core.add_wca(system, p1, p2)

        for pdata in [constants.NANOPARTICLE, constants.NP_COUNTER_ION]:
            core.add_wca(system, constants.BEAD_COUNTER_ION, pdata)

        # ====================== add ferrofluid ======================
        if len(system.part.select(type=constants.NANOPARTICLE.ptype)) < 1:
            log.info(f"Adding MNPs of diameter {constants.NANOPARTICLE.diameter}")
            core.add_nanoparticles(
                system=system,
                rng=rng,
                mnp_charge=-config.mnp_charge,
                volume_fraction=config.mnp_volume_fraction,
                strict_mnp_count=config.strict_mnp_count,
            )
            core.add_counterions(
                system=system,
                rng=rng,
                pdata=constants.NANOPARTICLE,
            )
            self.visualizer.screenshot("FerrofluidRandom")

        # ====================== remove overlap ======================
        if not self.is_ferrofluid_minimized:
            log.info("Removing MNP overlap")

            # NOTE: Minimization does not converge if other particles are fixed, therefore it is commented out
            # system.part.all().fix = [True]*3
            # nanoparticles.fix = [False]*3
            # mnp_counterions.fix = [False]*3
            core.run_minimize(system=system, min_force=300.0)
            # system.part.all().fix = [False]*3

            self.is_ferrofluid_minimized = True
            self.is_skin_tuned = False
            self.visualizer.screenshot("FerrofluidMinimized")
            self.write_checkpoint("FerrofluidMinimized")

        log.info("Setting up magnetostatics")
        DP3M_PARAMS_PATH = self.gel_dir.path / "dp3m_params.json"
        if DP3M_PARAMS_PATH.exists():
            dp3m_params = json.loads(DP3M_PARAMS_PATH.read_text())
        else:
            dp3m_params = constants.DP3M_PARAMS.copy()
        core.init_magnetostatics(system, dp3m_params)
        DP3M_PARAMS_PATH.write_text(json.dumps(dp3m_params, cls=myjson.NumpyJSONEncoder))

        # tune skin if the parameters have just been set
        if not self.is_skin_tuned:
            core.tune_skin(system=self.system)
            self.is_skin_tuned = True

        # ====================== equilibrate mmgel ======================
        for self.steps_mmgel_eq in integrate_system(
            system=system,
            start_steps=self.steps_mmgel_eq,
            end_steps=config.steps_mmgel_eq_max,
            control_file_path=self.gel_dir.control_file_path,
            loop_name="MMGelEq",
            callbacks=[
                IntegrationCallback(
                    fun=lambda step: self.write_checkpoint(f"MMGelEq_{step}"),
                    interval=config.steps_mmgel_eq_max // 200,
                ),
                IntegrationCallback(
                    fun=lambda step: self.visualizer.screenshot(f"MMGelEq_{step}"),
                    interval=config.steps_mmgel_eq_max // 5,
                ),
                IntegrationCallback(
                    fun=lambda step: self.write_bead_positions(folder_name="mmgel_eq", file_name=f"BeadPos_{step}.npy"),
                    interval=1500,
                ),
                IntegrationCallback(
                    fun=lambda step: self.write_mnp_positions(folder_name="mmgel_eq", file_name=f"MNPPos_{step}.npy"),
                    interval=1500,
                ),
            ],
        ):
            pass

        self.visualizer.screenshot("MMGelEqDone")
        self.write_checkpoint("MMGelEqDone")
        log.info(f"Simulation Time: {system.time:g}")

    def run_crosslinking(self):
        system = self.system
        crosslinker = self.crosslinker
        bead_bond = self.bead_bond
        CONFIG = self.config
        rng = self.rng

        log.info(f"Energy run cl 1: {system.analysis.energy()['total']}")
        crosslinking_already_done = crosslinker.state.has_finished
        log.info(f"Starting crosslinking with {len(crosslinker.state.crosslinks)} crosslinks")
        all_bead_ids = set(system.part.select(type=constants.BEAD.ptype).id)
        for cl_ids in crosslinker.state.crosslinks.copy():
            # in case there are crosslinks for beads which have been removed
            # remove them from the crosslink list
            bid0, bid1 = cl_ids
            if not (bid0 in all_bead_ids and bid1 in all_bead_ids):
                log.info(f"Removing crosslink {cl_ids}")
                crosslinker.state.crosslinks.remove(cl_ids)
                continue

            b1, b2 = system.part.by_ids(cl_ids)
            b1.add_bond((bead_bond, b2))
        log.info(f"Cross-links left after removal: {len(crosslinker.state.crosslinks)}")
        log.info(f"Energy ckpt cls: {system.analysis.energy()['total']}")

        if crosslinking_already_done:
            return

        if crosslinker.state.iterations == 0:
            self.write_checkpoint(f"GelCrosslinked_{crosslinker.state.iterations}")
        log.info("Cross-linking gel")
        while not crosslinker.state.has_finished:
            crosslinker.update()
            int_dt = time.time()
            system.integrator.run(CONFIG.steps_per_cl_iteration)
            int_dt = time.time() - int_dt
            log.info(f"Integration dt={int_dt:.1e}s")
            if crosslinker.state.iterations % 20 == 0:
                self.write_checkpoint(f"GelCrosslinked_{crosslinker.state.iterations}")
            gc.collect()

        log.info("Cross-linking done: Removing confinement, saving file")

        # Cross-linking may remove particles thus they must be cleared
        system.auto_update_accumulators.clear()

        # Remove free chains and counter ions and cross-links
        # between free chains
        bead_ids = system.part.select(type=constants.BEAD.ptype).id
        G = graph_from_beads(bead_ids, np.array(list(crosslinker.crosslinks)), CONFIG.chain_length)
        components = list(nx.connected_components(G))
        largest_component = max(components, key=len)
        free_bead_ids = np.array(list(set(bead_ids) - set(largest_component)))
        log.info(f"Removing {len(free_bead_ids)} beads ({len(free_bead_ids) // CONFIG.chain_length} chains)")

        # remove cross-links
        for b1, b2 in self.crosslinking_state.crosslinks.copy():
            if b1 in free_bead_ids or b2 in free_bead_ids:
                self.crosslinking_state.crosslinks.remove((b1, b2))

        cions_to_remove_ids = rng.choice(
            system.part.select(type=constants.BEAD_COUNTER_ION.ptype).id,
            int(round(abs(system.part.by_ids(free_bead_ids).q.sum()))),
            replace=False,
        )

        particle_ids_to_remove = cions_to_remove_ids.tolist() + free_bead_ids.tolist()
        assert len(set(particle_ids_to_remove)) == len(particle_ids_to_remove)
        assert set(particle_ids_to_remove).issubset(set(system.part.all().id))
        particles_to_remove = system.part.by_ids(particle_ids_to_remove)
        particles_to_remove.remove()

        # make charge neutral
        cions = system.part.select(type=constants.BEAD_COUNTER_ION.ptype)
        cions.q = -system.part.select(type=constants.BEAD.ptype).q.sum() / len(cions)
        log.info(f"Total charge after chain removal: {system.part.all().q.sum()}")

        system.constraints.clear()
        self.is_confinement_gone = True
        self.visualizer.screenshot("GelCrosslinked")
        self.write_checkpoint("GelCrosslinkedDone")

        if len(system.part.select(type=constants.BEAD.ptype)) < CONFIG.n_beads // 2:
            log.error(f"Not enough {len(system.part.select(type=constants.BEAD.ptype))} beads left")
            exit()

    def write_bead_positions(self, folder_name: str, file_name: str) -> None:
        log.info(f"Writing bead positions into {folder_name}, {file_name}")
        folder = self.gel_dir.path / f"raw/{folder_name}"
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / f"{file_name}", self.beads.pos)

    def write_mnp_positions(self, folder_name: str, file_name: str) -> None:
        log.info(f"Writing MNP positions into {folder_name}, {file_name}")
        folder = self.gel_dir.path / f"raw/{folder_name}"
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / f"{file_name}", self.mnps.pos)

    @property
    def beads(self):
        return self.system.part.select(type=constants.BEAD.ptype)

    @property
    def mnps(self):
        return self.system.part.select(type=constants.NANOPARTICLE.ptype)

    @property
    def bead_counterions(self):
        return self.system.part.select(type=constants.BEAD_COUNTER_ION.ptype)

    # @property
    # def ckpt_dir(self):
    #     return CheckpointDir(path=self.gel_dir.current_checkpoint_dir)

    def write_checkpoint(self, name: str) -> None:
        if self.program_options.no_checkpoints:
            return

        path = self.gel_dir.checkpoints_dir / name
        ckpt_dir = CheckpointDir(path=path)
        if path.exists():
            return

        log.info(f"Writing checkpoint {path}")

        # save small data
        main_dict = {
            "rng_state": self.rng.__getstate__(),
            "skin": self.system.cell_system.skin,
            "box_l": self.system.box_l[0],
            "sim_time": self.system.time,
            "is_confinement_gone": self.is_confinement_gone,
            "is_melt_minimized": self.is_melt_minimized,
            "is_skin_tuned": self.is_skin_tuned,
            "steps_melt_eq": self.steps_melt_eq,
            "steps_gel_eq": self.steps_gel_eq,
            "is_ferrofluid_minimized": self.is_ferrofluid_minimized,
            "steps_mmgel_eq": self.steps_mmgel_eq,
        }
        if self.crosslinking_state.is_initialized:
            crosslinking_state = {
                "is_initialized": self.crosslinking_state.is_initialized,
                "has_finished": self.crosslinking_state.has_finished,
                "iterations": self.crosslinking_state.iterations,
            }
            main_dict["crosslinking_state"] = crosslinking_state

        main_ckpt_path = path / "main_ckpt.json"
        main_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(main_ckpt_path, "w") as f:
            json.dump(main_dict, f, cls=myjson.NumpyJSONEncoder, indent=4)

        # save particle data
        pdict = self.system.part.all().to_dict()
        if len(pdict) != 0:
            pdict.pop("bonds")
            with open(path / "pdict.json", "w") as f:
                json.dump(pdict, f, cls=myjson.NumpyJSONEncoder)

        # save big crosslinking data
        if self.crosslinking_state.is_initialized:
            ckpt_dir.crosslinks_path.parent.mkdir(parents=True, exist_ok=True)
            log.info(f"Saving crosslinks to {ckpt_dir.crosslinks_path}")
            np.save(
                ckpt_dir.crosslinks_path,
                np.array(list(self.crosslinking_state.crosslinks)),
            )
            np.save(
                ckpt_dir.active_agents_path,
                np.array(self.crosslinking_state.active_agent_ids),
            )

        current_ckpt_name = path.relative_to(self.gel_dir.checkpoints_dir)
        self.gel_dir.current_checkpoint_tracker.write_text(str(current_ckpt_name))
        log.info(f"Checkpoint {path} written")


def main():
    gel_dir, program_options, config = parse_commandline()
    gel_dir.path.mkdir(parents=True, exist_ok=True)

    init_file_logging(gel_dir.path)

    mmgel = Microgel(
        gel_dir=gel_dir,
        program_options=program_options,
        config=config,
    )
    mmgel.run()


if __name__ == "__main__":
    main()
