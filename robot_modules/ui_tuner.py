from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    import omni.ui as ui
except Exception:  # pragma: no cover - UI is optional in headless runs
    ui = None


class TeleopTunerWindow:
    def __init__(self, teleop_ctrl, *, title: str = "Teleop Tuner") -> None:
        self._teleop = teleop_ctrl
        self._window = None
        self._center_models: Dict[int, ui.SimpleFloatModel] = {}
        self._gain_models: Dict[int, ui.SimpleFloatModel] = {}
        self._bounds_models: Dict[str, Tuple[ui.SimpleFloatModel, ui.SimpleFloatModel]] = {}
        self._alpha_model: ui.SimpleFloatModel | None = None
        if ui is None:
            print("[ui] omni.ui unavailable, teleop tuner disabled.")
            return

        self._default_center = np.asarray(self._teleop.target_center, dtype=np.float64).copy()
        self._default_gain = np.asarray(self._teleop.target_gain, dtype=np.float64).copy()
        bounds = dict(self._teleop.workspace_bounds) if self._teleop.workspace_bounds else {}
        self._default_bounds = {
            "x": tuple(bounds.get("x", (0.0, 0.0))),
            "y": tuple(bounds.get("y", (0.0, 0.0))),
            "z": tuple(bounds.get("z", (0.0, 0.0))),
        }
        self._default_alpha = float(getattr(self._teleop, "alpha", 0.15))

        self._window = ui.Window(title, width=360, height=520)
        with self._window.frame:
            scroll_policy = getattr(
                ui.ScrollBarPolicy,
                "SCROLLBAR_ALWAYS_ON",
                getattr(
                    ui.ScrollBarPolicy,
                    "ALWAYS_ON",
                    getattr(
                        ui.ScrollBarPolicy,
                        "ALWAYS",
                        getattr(
                            ui.ScrollBarPolicy,
                            "SCROLLBAR_AS_NEEDED",
                            None,
                        ),
                    ),
                ),
            )
            if scroll_policy is None:
                scroll_policy = ui.ScrollBarPolicy.__members__.get(
                    "SCROLLBAR_AS_NEEDED",
                    next(iter(ui.ScrollBarPolicy.__members__.values())),
                )
            with ui.ScrollingFrame(vertical_scrollbar_policy=scroll_policy):
                with ui.VStack(spacing=6, height=0):
                    ui.Label("Target Center (m)")
                    self._hint("Shift the end-effector target center (global offset).")
                    self._vec3_fields(
                        "Center",
                        np.asarray(self._teleop.target_center, dtype=np.float64),
                        self._set_center_axis,
                        step=0.01,
                        default_values=self._default_center,
                        model_store=self._center_models,
                        help_texts=(
                            f"X: shift forward/back (default {self._default_center[0]:.3f})",
                            f"Y: shift left/right (default {self._default_center[1]:.3f})",
                            f"Z: shift up/down (default {self._default_center[2]:.3f})",
                        ),
                    )
                    ui.Separator()

                    ui.Label("Target Gain")
                    self._hint("Scale the hand movement (larger = bigger motion).")
                    self._vec3_fields(
                        "Gain",
                        np.asarray(self._teleop.target_gain, dtype=np.float64),
                        self._set_gain_axis,
                        step=0.05,
                        default_values=self._default_gain,
                        model_store=self._gain_models,
                        help_texts=(
                            f"X: scale forward/back (default {self._default_gain[0]:.3f})",
                            f"Y: scale left/right (default {self._default_gain[1]:.3f})",
                            f"Z: scale up/down (default {self._default_gain[2]:.3f})",
                        ),
                    )
                    ui.Separator()

                    ui.Label("Workspace Bounds (m)")
                    self._hint("Clamp reachable workspace; out-of-range is clipped.")
                    self._bounds_fields(
                        dict(self._teleop.workspace_bounds) if self._teleop.workspace_bounds else {},
                        default_bounds=self._default_bounds,
                        model_store=self._bounds_models,
                    )
                    ui.Separator()

                    ui.Label("Smoothing Alpha")
                    self._hint("Smoothing: higher = faster/more jitter, lower = smoother/slower.")
                    self._float_field(
                        "alpha",
                        float(getattr(self._teleop, "alpha", 0.15)),
                        self._set_alpha,
                        step=0.01,
                        default_value=self._default_alpha,
                        model_store="alpha",
                        help_text=f"Smoothing (default {self._default_alpha:.3f})",
                    )
                    ui.Separator()
                    ui.Button("Reset All", clicked_fn=self._reset_all)

    def _hint(self, text: str):
        ui.Label(text, style={"color": 0xFF999999, "font_size": 12})

    def _float_field(
        self,
        label: str,
        value: float,
        on_change,
        *,
        step: float = 0.01,
        default_value: float | None = None,
        model_store: str | None = None,
        help_text: str | None = None,
    ):
        with ui.HStack(height=0):
            ui.Label(label, width=90)
            model = ui.SimpleFloatModel(float(value))
            ui.FloatDrag(model, step=step, width=140)
            model.add_value_changed_fn(lambda m: on_change(float(m.get_value_as_float())))
            if model_store == "alpha":
                self._alpha_model = model
            if default_value is not None:
                ui.Button(
                    "Reset",
                    width=60,
                    clicked_fn=lambda dv=default_value, m=model: self._reset_float(m, dv, on_change),
                )
            if help_text:
                ui.Label(help_text, style={"color": 0xFF888888, "font_size": 12})

    def _vec3_fields(
        self,
        prefix: str,
        values: np.ndarray,
        setter,
        *,
        step: float = 0.01,
        default_values: np.ndarray | None = None,
        model_store: Dict[int, ui.SimpleFloatModel] | None = None,
        help_texts: Tuple[str, str, str] | None = None,
    ):
        axes = ("X", "Y", "Z")
        for idx, axis in enumerate(axes):
            with ui.HStack(height=0):
                ui.Label(f"{prefix} {axis}", width=90)
                model = ui.SimpleFloatModel(float(values[idx]))
                ui.FloatDrag(model, step=step, width=140)
                model.add_value_changed_fn(
                    lambda m, i=idx: setter(i, float(m.get_value_as_float()))
                )
                if model_store is not None:
                    model_store[idx] = model
                if default_values is not None:
                    default_val = float(default_values[idx])
                    ui.Button(
                        "Reset",
                        width=60,
                        clicked_fn=lambda dv=default_val, m=model, i=idx: self._reset_axis(m, dv, setter, i),
                    )
                if help_texts:
                    ui.Label(help_texts[idx], style={"color": 0xFF888888, "font_size": 12})

    def _bounds_fields(
        self,
        bounds: Dict[str, Tuple[float, float]],
        *,
        default_bounds: Dict[str, Tuple[float, float]] | None = None,
        model_store: Dict[str, Tuple[ui.SimpleFloatModel, ui.SimpleFloatModel]] | None = None,
    ):
        axis_hints = {
            "x": "Forward/Back limits",
            "y": "Left/Right limits",
            "z": "Up/Down limits",
        }
        for axis in ("x", "y", "z"):
            lo, hi = bounds.get(axis, (0.0, 0.0))
            d_lo, d_hi = (0.0, 0.0)
            if default_bounds and axis in default_bounds:
                d_lo, d_hi = default_bounds[axis]
            with ui.HStack(height=0):
                ui.Label(f"{axis.upper()} min", width=90)
                model_min = ui.SimpleFloatModel(float(lo))
                ui.FloatDrag(model_min, step=0.01, width=120)
                ui.Label("max", width=40)
                model_max = ui.SimpleFloatModel(float(hi))
                ui.FloatDrag(model_max, step=0.01, width=120)
                if model_store is not None:
                    model_store[axis] = (model_min, model_max)
                if default_bounds is not None:
                    ui.Button(
                        "Reset",
                        width=60,
                        clicked_fn=lambda a=axis, mn=model_min, mx=model_max, dl=d_lo, dh=d_hi: self._reset_bounds(
                            a, mn, mx, dl, dh
                        ),
                    )
                ui.Label(
                    f"{axis_hints[axis]} (default {d_lo:.3f}~{d_hi:.3f})",
                    style={"color": 0xFF888888, "font_size": 12},
                )

                def set_min(m, a=axis, max_model=model_max):
                    val = float(m.get_value_as_float())
                    max_val = float(max_model.get_value_as_float())
                    if val > max_val:
                        max_model.set_value(val)
                        max_val = val
                    self._set_bound(a, val, max_val)

                def set_max(m, a=axis, min_model=model_min):
                    val = float(m.get_value_as_float())
                    min_val = float(min_model.get_value_as_float())
                    if val < min_val:
                        min_model.set_value(val)
                        min_val = val
                    self._set_bound(a, min_val, val)

                model_min.add_value_changed_fn(set_min)
                model_max.add_value_changed_fn(set_max)

    def _set_center_axis(self, idx: int, value: float):
        center = np.asarray(self._teleop.target_center, dtype=np.float64).copy()
        center[idx] = value
        self._teleop.target_center = center

    def _set_gain_axis(self, idx: int, value: float):
        gain = np.asarray(self._teleop.target_gain, dtype=np.float64).copy()
        gain[idx] = value
        self._teleop.target_gain = gain

    def _set_bound(self, axis: str, lo: float, hi: float):
        bounds = dict(self._teleop.workspace_bounds or {})
        bounds[axis] = (float(lo), float(hi))
        self._teleop.workspace_bounds = bounds

    def _set_alpha(self, value: float):
        self._teleop.alpha = float(value)

    def _reset_axis(self, model, default_value: float, setter, idx: int):
        model.set_value(default_value)
        setter(idx, float(default_value))

    def _reset_bounds(
        self,
        axis: str,
        model_min,
        model_max,
        default_min: float,
        default_max: float,
    ):
        model_min.set_value(float(default_min))
        model_max.set_value(float(default_max))
        self._set_bound(axis, float(default_min), float(default_max))

    def _reset_float(self, model, default_value: float, setter):
        model.set_value(float(default_value))
        setter(float(default_value))

    def _reset_all(self):
        for idx, model in self._center_models.items():
            dv = float(self._default_center[idx])
            model.set_value(dv)
            self._set_center_axis(idx, dv)
        for idx, model in self._gain_models.items():
            dv = float(self._default_gain[idx])
            model.set_value(dv)
            self._set_gain_axis(idx, dv)
        for axis, models in self._bounds_models.items():
            dv = self._default_bounds.get(axis, (0.0, 0.0))
            models[0].set_value(float(dv[0]))
            models[1].set_value(float(dv[1]))
            self._set_bound(axis, float(dv[0]), float(dv[1]))
        if self._alpha_model is not None:
            self._alpha_model.set_value(float(self._default_alpha))
            self._set_alpha(float(self._default_alpha))
