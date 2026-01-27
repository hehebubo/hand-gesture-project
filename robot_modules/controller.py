from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


Bounds = Dict[str, Tuple[float, float]]


class TeleopController:
    def __init__(
        self,
        target_center: Optional[Tuple[float, float, float]] = None,
        target_gain: Optional[Tuple[float, float, float]] = None,
        workspace_bounds: Optional[Bounds] = None,
        hand_pos_is_0_1: bool = False,
        alpha: float = 0.15,
    ):
        self.target_center = np.asarray(
            target_center if target_center is not None else (0.0, 0.0, 0.0),
            dtype=np.float64,
        )
        self.target_gain = np.asarray(
            target_gain if target_gain is not None else (1.0, 1.0, 1.0),
            dtype=np.float64,
        )
        self.workspace_bounds: Bounds = workspace_bounds or {
            "x": (0.10, 0.80),
            "y": (-0.30, 0.30),
            "z": (0.15, 0.80),
        }
        self.hand_pos_is_0_1 = hand_pos_is_0_1
        self.alpha = float(alpha)
        self._pos_filt: Optional[np.ndarray] = None

    def reset_filter(self):
        self._pos_filt = None

    def _clip(self, pos: np.ndarray) -> np.ndarray:
        for idx, axis in enumerate(("x", "y", "z")):
            bounds = self.workspace_bounds.get(axis)
            if bounds is None:
                continue
            pos[idx] = np.clip(pos[idx], bounds[0], bounds[1])
        return pos

    def process(self, raw_pos: np.ndarray) -> np.ndarray:
        """
        raw_pos: (x, y, z) from hand tracker.
        Returns filtered target position in robot coordinates.
        """
        raw = np.asarray(raw_pos, dtype=np.float64).reshape(3)
        if self.hand_pos_is_0_1:
            raw = raw - 0.5
        target = (raw * self.target_gain) + self.target_center

        target = self._clip(target)
        if self._pos_filt is None:
            self._pos_filt = target
        else:
            self._pos_filt = (1.0 - self.alpha) * self._pos_filt + self.alpha * target

        return self._clip(self._pos_filt.copy())
