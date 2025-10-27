import json
import logging
import logging.handlers
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import Config
from utils import mymath

LOG_BUFFER = 10

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
log.addHandler(ch)


def init_file_logging(out_dir: Path):
    fh = logging.FileHandler(out_dir / "main.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    memory_handler = logging.handlers.MemoryHandler(
        LOG_BUFFER, logging.ERROR, target=fh
    )
    log.addHandler(memory_handler)


def simtime_to_int(t: float) -> int:
    return int(round(100 * t))


def folder_to_file_ts(
    folder_path: Path, time_position: int = -1, pattern="*.npy"
) -> pd.DataFrame:
    return (
        pd.DataFrame.from_records(
            [
                {"path": p, "timestep": int(p.stem.split("_")[time_position])}
                for p in folder_path.glob(pattern)
            ],
            columns=["path", "timestep"],
        )
        .sort_values("timestep")
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class GelDir:
    path: Path

    @classmethod
    def is_gel_dir(cls, path) -> bool:
        path = Path(path)
        if not path.is_dir():
            return False

        return (path / "current_checkpoint.txt").exists()

    @classmethod
    def from_inner(cls, path) -> "GelDir":
        path = Path(path)
        if cls.is_gel_dir(path):
            return cls(path)

        for parent in path.parents:
            if cls.is_gel_dir(parent):
                return cls(parent)

        raise ValueError(f"File '{path}' is not inside a gel directory with a config")

    @property
    def config_path(self) -> Path:
        return self.path / "config.json"

    @property
    def config(self) -> Config:
        return Config.from_file(self.config_path)

    @property
    def raw_data_path(self) -> Path:
        return self.path / "raw/"

    @property
    def gel_eq_path(self) -> Path:
        return self.raw_data_path / "gel_eq/"

    def beads_gel_eq(self) -> pd.DataFrame:
        return folder_to_file_ts(self.gel_eq_path)

    @property
    def mmgel_eq_path(self) -> Path:
        return self.raw_data_path / "mmgel_eq/"

    def beads_mmgel_eq(self) -> pd.DataFrame:
        return folder_to_file_ts(self.mmgel_eq_path)

    @property
    def agents_eq_path(self) -> Path:
        return self.raw_data_path / "agents/"

    def agents_eq(self) -> pd.DataFrame:
        return folder_to_file_ts(self.agents_eq_path, time_position=0)

    @property
    def agents_diffusing_path(self) -> Path:
        return self.raw_data_path / "agent_diffusion/"

    def agents_diffusing(self) -> pd.DataFrame:
        return folder_to_file_ts(self.agents_diffusing_path, time_position=0)

    @property
    def n_agents(self) -> int:
        return len(np.load(next(self.agents_diffusing_path.iterdir())))

    @property
    def crosslinks(self):
        return CheckpointDir(self.checkpoints_dir / "GelCrosslinkedDone").crosslinks()
    
    @property
    def crosslinks_numpy(self):
        return CheckpointDir(self.checkpoints_dir / "GelCrosslinkedDone").crosslinks_numpy()
    
    @property
    def n_crosslinks(self) -> int:
        # return len(CheckpointDir(self.current_checkpoint_dir).crosslinks())
        return len(self.crosslinks)

    @property
    def p3m_params_path(self) -> Path:
        return self.path / "p3m_params.json"

    def p3m_params(self) -> dict[str, Any]:
        return json.loads(self.p3m_params_path.read_text())

    @property
    def control_file_path(self) -> Path:
        # this is an ini file so that I can modify it more easily
        # programmatically
        return self.path / "control.ini"

    @property
    def dp3m_params_path(self) -> Path:
        return self.path / "dp3m_params.json"

    def dp3m_params(self) -> dict[str, Any]:
        return json.loads(self.dp3m_params_path.read_text())

    @property
    def current_checkpoint_tracker(self) -> Path:
        """Only contains the name of the current checkpoint. Not the full path!"""
        return self.path / "current_checkpoint.txt"

    @property
    def checkpoints_dir(self) -> Path:
        return self.path / "checkpoints"

    @property
    def current_checkpoint_dir(self) -> Path:
        if not self.current_checkpoint_tracker.exists():
            return self.checkpoints_dir / "first"
        current_ckpt_name = self.current_checkpoint_tracker.read_text().rstrip()
        return self.checkpoints_dir / current_ckpt_name

    @property
    def img_dir(self) -> Path:
        return self.path / "images"


@dataclass(frozen=True)
class CheckpointDir:
    path: Path

    def sdata(self) -> dict[str, Any]:
        """Small value data"""
        path = self.path / "main_ckpt.json"
        if not path.exists():
            return dict()
        return json.loads(path.read_text())

    def pdict(self) -> dict[str, Any]:
        """Particle data"""
        path = self.path / "pdict.json"
        if not path.exists():
            return dict()
        return json.loads(path.read_text())

    def get_bead_points(self):
        import constants

        pdict = self.pdict()
        bead_indices = np.array(pdict["type"]) == constants.BEAD.ptype
        bead_points = np.array(pdict["pos"])[bead_indices]
        return bead_points

    @property
    def crosslinks_path(self) -> Path:
        return self.path / "crosslinks.npy"

    def crosslinks_numpy(self) -> np.ndarray:
        path = self.crosslinks_path
        if not path.exists():
            return np.array([])
        return np.load(path, allow_pickle=False)

    def crosslinks(self) -> set[tuple[int, int]]:
        return set(tuple([a, b]) for a, b in self.crosslinks_numpy())

    @property
    def active_agents_path(self) -> Path:
        return self.path / "active_agents.npy"

    def active_agents(self) -> list[int]:
        path = self.active_agents_path
        if not path.exists():
            return list()
        return np.load(path, allow_pickle=False).tolist()


def online_average_npy_files(file_paths, data_post_processor, with_ci=False):
    if with_ci:
        return mymath.online_mean(
            (data_post_processor(np.load(path)) for path in file_paths)
        )
