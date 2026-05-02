"""Utilitarios de reprodutibilidade."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Fixa seeds de Python, NumPy e TensorFlow quando disponivel."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
    except ImportError:
        return

    tf.random.set_seed(seed)

