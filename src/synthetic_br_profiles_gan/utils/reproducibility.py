"""Utilitários de reprodutibilidade."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeedState:
    """Informações sobre a configuração de seed de uma execução."""

    seed: int
    pythonhashseed: str | None
    tensorflow_seeded: bool
    torch_seeded: bool
    deterministic_ops_requested: bool
    notes: list[str]


def set_global_seed(
    seed: int,
    deterministic_ops: bool = True,
    seed_tensorflow: bool = True,
    seed_torch: bool = True,
) -> SeedState:
    """Fixa seeds para Python, NumPy, TensorFlow e PyTorch quando disponíveis."""
    notes: list[str] = []
    previous_hash_seed = os.environ.get("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = str(seed)
    if previous_hash_seed not in {None, str(seed)}:
        notes.append("PYTHONHASHSEED was changed after interpreter startup; hash determinism may be limited.")

    if deterministic_ops:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    tensorflow_seeded = False
    if seed_tensorflow:
        try:
            import tensorflow as tf

            tf.random.set_seed(seed)
            tensorflow_seeded = True
        except ImportError:
            notes.append("TensorFlow is not installed; TensorFlow seed was not set.")
    else:
        notes.append("TensorFlow seed skipped for this model.")

    torch_seeded = False
    if seed_torch:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if deterministic_ops:
                torch.use_deterministic_algorithms(True, warn_only=True)
            torch_seeded = True
        except ImportError:
            notes.append("PyTorch is not installed; PyTorch/CTGAN seed was not set.")
    else:
        notes.append("PyTorch seed skipped for this model.")

    state = SeedState(
        seed=int(seed),
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        tensorflow_seeded=tensorflow_seeded,
        torch_seeded=torch_seeded,
        deterministic_ops_requested=deterministic_ops,
        notes=notes,
    )
    LOGGER.info("seed_configured", extra={"seed_state": state.__dict__})
    return state


def seed_state_to_dict(state: SeedState) -> dict[str, Any]:
    """Serializa o estado de seed para manifestos."""
    return {
        "seed": state.seed,
        "pythonhashseed": state.pythonhashseed,
        "tensorflow_seeded": state.tensorflow_seeded,
        "torch_seeded": state.torch_seeded,
        "deterministic_ops_requested": state.deterministic_ops_requested,
        "notes": state.notes,
    }
