"""Containers for optional outputs produced during training."""

from dataclasses import dataclass
from typing import Any

import nest
import numpy as np


@dataclass
class TrainingOutputs:
    """Optional outputs returned by a training run."""

    loss: np.ndarray
    recurrent_events: dict[str, Any]
    output_events: dict[str, Any]
    recurrent_neurons: nest.NodeCollection