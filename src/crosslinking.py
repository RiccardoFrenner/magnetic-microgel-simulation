from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, TypeAlias

import espressomd
import espressomd.accumulators
import espressomd.constraints
import espressomd.interactions
import espressomd.observables
import networkx as nx
import numpy as np
from scipy import spatial

import common
import constants
import core
import forcefields
import physics
from analysis.network import graph_from_beads
from common import log
from config import ProgramOptions
from utils import myjson, myrandom

ParticleHandle: TypeAlias = Any
ParticleSlice: TypeAlias = Any


def distance_based_crosslink_probability(distance):
    a = physics.fene_potential(distance, **constants.FENE_PARAMS)
    b = physics.wca_potential(distance, epsilon=constants.EPS, sigma=constants.BEAD.diameter)
    return np.exp(-(a + b)) * 1e30


@dataclass
class ConstraintCrosslinkingProbabilities:
    constraint_radius: float
    string_to_fun: dict = field(default_factory=dict)

    @property
    def R(self) -> float:
        return self.constraint_radius

    def crosslink_probability_constant(self, r):
        assert r >= 0
        assert r <= self.R
        vol = self.R
        p = 1.0
        return p / vol

    def crosslink_probability_linear_center_high(self, r):
        assert r >= 0
        assert r <= self.R
        vol = 0.5 * self.R
        p = -r / self.R + 1
        return p / vol

    def crosslink_probability_linear_edge_high(self, r):
        vol = 0.5 * self.R
        p = r / self.R
        return p / vol

    def crosslink_probability_nonlinear(self, r):
        # a (r + b)**2
        # a b**2 = 1
        # a (R + b)**2 = 0
        # a = 1 / R**2
        # b = -R
        a = 1 / self.R**2
        b = -self.R
        p = a * (r + b) ** 2
        vol = self.R / 3
        return p / vol

    def prob_fun_from_string(self, s: str):
        fun = self.string_to_fun[s]
        return lambda bond_length, distance_from_center: distance_based_crosslink_probability(bond_length) * fun(
            distance_from_center
        )

    def __post_init__(self):
        self.string_to_fun = {
            "constant": self.crosslink_probability_constant,
            "linear_high_edge": self.crosslink_probability_linear_edge_high,
            "linear_low_edge": self.crosslink_probability_linear_center_high,
            "nonlinear_low_edge": self.crosslink_probability_nonlinear,
        }


@dataclass(kw_only=True, eq=True)
class CrosslinkingState:
    active_agent_ids: list[int] = field(default_factory=list)
    crosslinks: set[tuple[int, int]] = field(default_factory=set)
    is_initialized: bool = False
    has_finished: bool = False
    iterations: int = 0


@dataclass
class AgentCrosslinkConfig:
    max_dist: float
    max_bonds_per_bead: int
    n_agents_per_crosslink: float
    force_field_type: str
    initial_diff_steps: int
    chain_length: int
    output_path: Path
    ensure_all_chains_are_crosslinked: bool = True
    enable_debug_output: bool = False
    with_agent_interactions: bool = False

    def __post_init__(self):
        assert self.n_agents_per_crosslink >= 1


class AgentCrosslinker:
    """
    What does it need from the **environment/state**?
    - Knowledge:
        - confinement radius
        - neighbor search
    - Tools (aka. state changing stuff):
        - bond creation
        - particle creation
        - particle deletion
        - force field setup
        - integration
        - minimization
    """

    def __init__(
        self,
        *,
        system: espressomd.System,
        rng: np.random.Generator,
        are_bonded: Callable[[int, int], bool],
        create_bond: Callable[[int, int], None],
        confinement_radius: float,
        desired_n_crosslinks: int,
        program_options: ProgramOptions,
        cl_config: AgentCrosslinkConfig,
        state: CrosslinkingState,
    ):
        self.are_bonded = are_bonded
        self.create_bond = create_bond
        self.system = system
        self.rng = rng
        self.desired_n_crosslinks = desired_n_crosslinks
        self.program_options = program_options
        self.cl_config = cl_config
        self.confinement_radius = confinement_radius
        self.ff_index = -1

        self.state = state

        core.add_wca(self.system, constants.CROSSLINK_AGENT, constants.GEL_BOUNDARY)

        if self.cl_config.with_agent_interactions:
            # Does this improve distribution of exp agents?
            # -> A little bit but not enough, therefore `False` by default
            core.add_wca(self.system, constants.CROSSLINK_AGENT, constants.CROSSLINK_AGENT)

            # Is this needed to 1) not form big clumps of beads
            # and 2) to ensure all polymers get cross-linked?
            # -> 1) TODO
            # -> 2) No. It's actually worse and requires more crosslinks
            # on average to create a fully connected cluster
            core.add_wca(self.system, constants.CROSSLINK_AGENT, constants.BEAD)

    def init(self) -> None:
        if not self.state.is_initialized:
            log.info("Initializing agent crosslinker")
            self._spawn_agents(self.confinement_radius)
            # needs to happen after spawning agents
            self.ff_index = self._init_force_field(self.confinement_radius)

            agent_ids: list[int] = [a.id for a in self._agents()]
            self.state.active_agent_ids = agent_ids.copy()

            self._diffuse_agents()
            self.state.is_initialized = True
            self._dt = self.system.time
        else:
            # needs to also happen if already initialized but not finished
            # since it was loaded from a checkpoint
            if not self.state.has_finished:
                self.ff_index = self._init_force_field(self.confinement_radius)

    @property
    def crosslinks(self):
        return self.state.crosslinks

    @property
    def active_agent_ids(self):
        return self.state.active_agent_ids

    def update(self) -> None:
        """A single cross-link iteration.
        Return whether it is finished or not"""

        if self.state.has_finished:
            return

        self._not_finished_update()
        self.state.has_finished = self._has_finished()

        if self.state.has_finished:
            log.info("")
            self._cleanup()
            return

    def _is_bead_pair_valid(self, id1: int, id2: int) -> bool:
        assert id1 < id2, (id1, id2)

        system = self.system
        crosslinks = self.crosslinks

        id_pair = (id1, id2)
        p0 = system.part.by_id(id_pair[0])
        p1 = system.part.by_id(id_pair[1])

        if system.distance(p0, p1) > self.cl_config.max_dist:
            return False

        # filter invalid crosslinks
        if id_pair in crosslinks:
            return False

        # filter beads that are neighbors on the same polymer chain
        if self.are_bonded(*id_pair):
            return False
        # for safety the other way around as well
        if self.are_bonded(id_pair[1], id_pair[0]):
            return False

        def count_crosslinks(i: int) -> int:
            c = 0
            for cl in self.crosslinks:
                if i in cl:
                    c += 1
            return c

        n_crosslinks_per_bead = np.array([count_crosslinks(i) for i in [id1, id2]])
        # filter beads that have too many crosslinks already
        # if any(len(p.bonds) >= self.cl_config.max_bonds_per_bead for p in [p0, p1]):  # does not work because p.bonds are asymmetric in espresso
        if np.any(
            n_crosslinks_per_bead >= 1
        ):  # TODO: chain end can also only have 1 crosslink this way, which makes them special as all other beads can have 3 bonds, which may not be desired.
            return False

        return True

    def _not_finished_update(self) -> None:
        assert len(self.active_agent_ids) > 0, "AgentCrosslinker has no agents"
        if self.desired_n_crosslinks == 0:
            self.state.iterations += 1
            return

        agent_pos_dir = self.cl_config.output_path / "raw/agents"
        agent_pos_dir.mkdir(exist_ok=True, parents=True)
        agent_path = agent_pos_dir / f"{common.simtime_to_int(self.system.time)}_agent_pos.npy"
        np.save(agent_path, self._agents().pos)

        valid_agent_bead_bead_pairs, search_dt = self._find_bond_actors()

        agents_ids_to_consider = list(valid_agent_bead_bead_pairs.keys())
        self.rng.shuffle(agents_ids_to_consider)
        all_beads = self.system.part.select(type=constants.BEAD.ptype)
        all_agents = self.system.part.select(type=constants.CROSSLINK_AGENT.ptype)
        debug_info = {
            "sim_time": self.system.time,
            "all_bead_pos": all_beads.pos,
            "all_bead_ids": all_beads.id,
            "all_agent_pos": all_agents.pos,
            "all_agent_ids": all_agents.id,
            "active_agent_ids": self.active_agent_ids.copy(),
            "crosslinks": list(self.crosslinks),
            "all_agent_candidates": agents_ids_to_consider,
            "visited_agent_candidates": [],
            "visited_bead_pair_lists": [],
        }
        n_lost_by_chance = 0
        for agent_id in agents_ids_to_consider:
            assert agent_id in self.active_agent_ids, agent_id

            bead_pair_list = valid_agent_bead_bead_pairs[agent_id]
            agent = self.system.part.by_id(agent_id)
            for b1, b2 in bead_pair_list.copy():
                # remove invalid bead pairs from list because they can
                # become invalid again after crosslinks have been made
                if not self._is_bead_pair_valid(b1.id, b2.id):
                    bead_pair_list.remove((b1, b2))

            if len(bead_pair_list) == 0:
                continue

            b1, b2 = self.rng.choice(bead_pair_list)

            debug_info["visited_agent_candidates"].append(agent.id)
            debug_info["visited_bead_pair_lists"].append([[b1.id, b2.id] for b1, b2 in bead_pair_list])

            # favor bonds with distance close to equilibrium distance
            # (approx a bead diameter (but not exactly!))
            pair_distance = self.system.distance(b1, b2)
            rel_deviation = abs(pair_distance - constants.BEAD.diameter) / constants.BEAD.diameter
            # G = graph_from_beads(
            #     all_beads.id,
            #     np.array(list(self.crosslinks)),
            #     self.cl_config.chain_length,
            # )
            # # this may be a little unphysical but otherwise I cannot ensure full connectivity easily
            # rel_deviation *= 100 if nx.has_path(G, b1.id, b2.id) else 1
            # not needed since I use all pairs anyway now
            if self.rng.uniform(0, 1) > min(1, np.exp(-rel_deviation)):
                n_lost_by_chance += 1
                continue

            self._create_crosslink(agent, b1, b2)

            if self._has_finished():
                break

        if self.cl_config.enable_debug_output:
            debug_info_dir = self.cl_config.output_path / "raw/debug/agents"
            debug_info_dir.mkdir(exist_ok=True, parents=True)
            (debug_info_dir / f"debug_agents_{common.simtime_to_int(self.system.time)}.json").write_text(
                json.dumps(debug_info, indent=2, cls=myjson.NumpyJSONEncoder)
            )

        log.info(
            f"it={self.state.iterations:<10} | "
            f"n_possible_cls={len(valid_agent_bead_bead_pairs):>6} | "
            f"{n_lost_by_chance=:>6} | "
            f"search_time={search_dt:>.1e} | "
            f"real_time={datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | "
            f"sim_time={self.system.time:>8.3f} | #Crosslinks so far "
            f"{len(self.state.crosslinks):>6}/{self.desired_n_crosslinks:<6} "
            f"|"
        )

        self.state.iterations += 1

    def _has_finished(self) -> bool:
        if not len(self.crosslinks) >= self.desired_n_crosslinks:
            return False

        bead_ids = self.system.part.select(type=constants.BEAD.ptype).id

        if self.cl_config.ensure_all_chains_are_crosslinked and not self._are_all_chains_crosslinked():
            n_beads = len(bead_ids)
            n_chains = n_beads // self.cl_config.chain_length
            log.info(
                "INFO: Cross-linking not yet done since not all chains are "
                "cross-linked "
                f"{self._n_chains_crosslinked()}/{n_chains}"
            )
            return False

        # check that all are connected to a big cluster
        G = graph_from_beads(bead_ids, np.array(list(self.crosslinks)), self.cl_config.chain_length)
        components = list(nx.connected_components(G))
        largest_component = max(components, key=len)
        if len(largest_component) < len(bead_ids):
            log.info(
                f"{id(self)} | All chains are crosslinked but they do not form a single "
                f"cluster but {len(components)} clusters with"
                f"{list(map(len, components))} beads each instead"
            )
            return False

        if len(self.crosslinks) > self.desired_n_crosslinks:
            log.warning(
                "Cross-linking has required"
                f" {len(self.crosslinks) - self.desired_n_crosslinks}"
                " more cross-links than specified to cross-link all chains"
            )

        return True

    def _create_crosslink(self, agent: ParticleHandle, bead0: ParticleHandle, bead1: ParticleHandle) -> None:
        assert bead0.id < bead1.id, (bead0.id, bead1.id)
        self.active_agent_ids.remove(agent.id)
        self.create_bond(bead0.id, bead1.id)
        self.crosslinks.add((bead0.id, bead1.id))

    def _spawn_agents(self, confinement_radius: float) -> None:
        system = self.system
        rng = self.rng

        n_agents = int(np.ceil(self.cl_config.n_agents_per_crosslink * self.desired_n_crosslinks))
        log.info(f"Creating {n_agents} agents to form {self.desired_n_crosslinks} cross-links")

        _ = system.part.add(
            pos=myrandom.random_uniform_ball(
                rng=rng,
                n=n_agents,
                R=confinement_radius - constants.BEAD.diameter,
                center=system.box_l / 2,
            ),
            type=np.full(n_agents, constants.CROSSLINK_AGENT.ptype),
        )

        if self.cl_config.with_agent_interactions:
            # NOTE: Minimize does not converge if we fix the other ones, therefore it is commented out
            # self.system.part.all().fix = [True]*3
            # self._agents().fix = [False]*3
            core.run_minimize(system=self.system, min_force=300.0)
            # self.system.part.all().fix = [False]*3

    def _init_force_field(self, confinement_radius: float) -> int:
        system = self.system

        force_field_type = self.cl_config.force_field_type
        log.info(f"Setting up {force_field_type} force field")

        ff_index = -1
        force_fun = None
        if force_field_type == "harmonic":
            _force_fun = partial(
                forcefields.harm_force_fun,
                center=system.box_l / 2,
                equil_distance=0.9 * confinement_radius,
            )
            force_fun = lambda x: _force_fun(x) * self.program_options.debug_forcefield_scale
        elif force_field_type == "exp":
            _force_fun = partial(
                forcefields.exp_force_fun,
                center=system.box_l / 2,
                equil_distance=confinement_radius,
            )
            force_fun = lambda x: _force_fun(x) * self.program_options.debug_forcefield_scale
        elif force_field_type == "none":
            ...
        else:
            raise ValueError("No valid force field specified")

        if force_field_type != "none":
            assert force_fun is not None
            # one cell should fit at least 4 times into
            # the gel radius and divide box_l
            box_l = system.box_l[0]
            grid_spacing = np.full(3, box_l / (box_l // (confinement_radius / 4) + 1))

            FIELD_STRENGTH = 1000.0
            field_data = FIELD_STRENGTH * espressomd.constraints.ForceField.field_from_fn(
                system.box_l, grid_spacing, force_fun
            )
            log.info("Maximum field force: %f" % np.max(np.ravel(np.linalg.norm(field_data, axis=-1))))

            agents_ids = [int(a.id) for a in self._agents()]
            assert len(agents_ids) > 0, "No agents to add force field to"
            F = espressomd.constraints.ForceField(
                field=field_data,
                grid_spacing=grid_spacing,
                particle_scales={a_id: 1.0 for a_id in agents_ids},
                default_scale=0.0,
            )
            log.info("Using %s force field" % force_field_type)
            ff_index = len(system.constraints)
            system.constraints.add(F)

        return ff_index

    def _diffuse_agents(self) -> None:
        system = self.system

        log.info("Initial agent diffusion without cross-linking")
        system.integrator.set_vv()

        # run until 2 % of agents are within 2 bead diameters of the constraint radius
        agents = self._agents()

        def enough_agents_at_boundary() -> bool:
            distance_from_center = np.linalg.norm(agents.pos - system.box_l / 2, axis=-1)
            distance_from_constraint = np.abs(self.confinement_radius - distance_from_center)
            n_boundary_agents = np.count_nonzero(distance_from_constraint < 2 * constants.BEAD.diameter)
            return n_boundary_agents > len(agents) * 0.02

        STEPS_PER_IT = max(10, self.cl_config.initial_diff_steps // 20)
        # NOTE: Once we checkpoint during diffusion this has to be adapted
        diffusion_steps = 0
        diffusion_out_path = self.cl_config.output_path / "raw/agent_diffusion"
        diffusion_out_path.mkdir(exist_ok=True, parents=True)
        np.save(diffusion_out_path / "0_agent_pos_diffusing.npy", agents.pos)
        while diffusion_steps < self.cl_config.initial_diff_steps or not enough_agents_at_boundary():
            log.info(f"    Agent diffusion time step: {diffusion_steps}")
            system.integrator.run(STEPS_PER_IT)
            diffusion_steps += STEPS_PER_IT
            np.save(
                diffusion_out_path / f"{diffusion_steps}_agent_pos_diffusing.npy",
                agents.pos,
            )
        log.info("Finished agent diffusion")

    def _cleanup(self) -> None:
        self._agents().remove()

        if self.ff_index >= 0:
            self.system.constraints.remove(self.system.constraints[self.ff_index])

        log.info("Cleaned up cross-linker")

    def _agents(self) -> ParticleSlice:
        return self.system.part.select(type=constants.CROSSLINK_AGENT.ptype)

    def _get_crosslinked_chain_indices(self) -> np.ndarray:
        """Return all bead ids of all chains which are not crosslinked"""
        n_beads = len(self.system.part.select(type=constants.BEAD.ptype))

        def pid_to_chain_index(i: int) -> int:
            assert i < n_beads, f"Beads don't seem to start at 0 index {i} >= {n_beads}"
            return i // self.cl_config.chain_length

        crosslinked_chains: set[int] = set()
        for b0, b1 in self.crosslinks:
            crosslinked_chains.add(pid_to_chain_index(b0))
            crosslinked_chains.add(pid_to_chain_index(b1))

        return np.array(list(crosslinked_chains))

    def _n_chains_crosslinked(self) -> int:
        return len(self._get_crosslinked_chain_indices())

    def _are_all_chains_crosslinked(self) -> bool:
        n_beads = len(self.system.part.select(type=constants.BEAD.ptype))
        n_chains = n_beads // self.cl_config.chain_length
        return self._n_chains_crosslinked() >= n_chains

    def _get_n_crosslinks_per_chain(self) -> np.ndarray:
        n_beads = len(self.system.part.select(type=constants.BEAD.ptype))

        def pid_to_chain_index(i: int) -> int:
            assert i < n_beads, f"Beads don't seem to start at 0 index {i} >= {n_beads}"
            return self._pid_to_chain_index(i)

        n_crosslinks_per_chain: list[int] = [0 for _ in range(n_beads // self.cl_config.chain_length)]
        for b0, b1 in self.crosslinks:
            n_crosslinks_per_chain[pid_to_chain_index(b0)] += 1
            n_crosslinks_per_chain[pid_to_chain_index(b1)] += 1

        return np.array(n_crosslinks_per_chain)

    def _pid_to_chain_index(self, pid: int) -> int:
        return pid // self.cl_config.chain_length

    def _find_bond_actors(
        self,
    ) -> tuple[dict[int, list[tuple[ParticleHandle, ParticleHandle]]], float]:
        # Candidate search
        # 1. get (bead, bead), (agent, agent), (bead, agent) pairs
        # 2. remove all except (bead, agent) pairs
        # 3. collect all pairs with the same agent id
        # 4. for each of these collections check if two beads are close
        # 5. if so, create a crosslink
        search_t0 = time.time()

        beads = self.system.part.select(type=constants.BEAD.ptype)
        bead_points = beads.pos
        beads = list(beads)
        bead_kdtree = spatial.KDTree(bead_points, balanced_tree=False)  # balanced_tree=False is faster
        bead_agent_pairs: list[tuple[ParticleHandle, ParticleHandle]] = []
        for active_agent_id in self.active_agent_ids:
            active_agent = self.system.part.by_id(active_agent_id)
            close_bead_indices = bead_kdtree.query_ball_point(
                active_agent.pos.squeeze(), self.cl_config.max_dist, p=2, workers=1
            )
            for bi in close_bead_indices:
                bead_agent_pairs.append((beads[bi], active_agent))

        agent_collections: dict[int, list[ParticleHandle]] = {}
        for bead, agent in bead_agent_pairs:
            agent_collections.setdefault(agent.id, []).append(bead)
        valid_agent_bead_bead_pairs: dict[int, list[tuple[ParticleHandle, ParticleHandle]]] = {}
        for agent_id in agent_collections:
            for b1, b2 in itertools.combinations(agent_collections[agent_id], 2):
                if b1.id > b2.id:
                    b1, b2 = b2, b1

                if not self._is_bead_pair_valid(b1.id, b2.id):
                    continue

                valid_agent_bead_bead_pairs.setdefault(agent_id, []).append((b1, b2))

        search_t1 = time.time()
        search_dt = search_t1 - search_t0

        return valid_agent_bead_bead_pairs, search_dt
