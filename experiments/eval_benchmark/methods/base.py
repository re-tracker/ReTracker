from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MethodOutput:
    pred_tracks_xy: np.ndarray  # (N,T,2) float32
    pred_occluded: np.ndarray  # (N,T) bool
    runtime_sec: float
    meta: dict[str, Any] = field(default_factory=dict)


class BaseMethod(ABC):
    name: str

    @abstractmethod
    def load(self, device: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, frames_uint8: np.ndarray, query_points_tyx: np.ndarray) -> MethodOutput:
        raise NotImplementedError
