"""
Scene/world helpers for Isaac Sim (環境設定、匯入 URDF 等).

Typical usage::

    from Sim import ensure_basic_lighting, create_world, add_ground_plane, import_robot_from_urdf
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom, UsdLux

__all__ = [
    "ensure_basic_lighting",
    "use_default_lighting",
    "use_default_light_rig",
    "create_world",
    "add_ground_plane",
    "import_robot_from_urdf",
    "wait_for_manual_gui_close",
]

_urdf_interface = _urdf.acquire_urdf_interface()

def create_world(stage_units: float = 1.0, physics_dt: float = 1.0 / 60.0) -> World:
    """Create an Isaac World with optional unit/dt overrides."""
    return World(stage_units_in_meters=stage_units, physics_dt=physics_dt)

def ensure_basic_lighting(
    dome_path: str = "/World/EnvLight",
    sun_path: str = "/World/SunLight",
    dome_intensity: float = 150.0,
    sun_intensity: float = 10000.0,
) -> None:
    """
    Guarantee a minimal lighting rig so imported geometry is visible.

    呼叫時機：在匯入任何幾何或場景前執行一次即可。
    """
    stage = get_current_stage()
    if not stage.GetPrimAtPath(dome_path):
        dome = UsdLux.DomeLight.Define(stage, dome_path)
        dome.CreateIntensityAttr(dome_intensity)
        dome.CreateSpecularAttr(0.1)

    if not stage.GetPrimAtPath(sun_path):
        sun = UsdLux.DistantLight.Define(stage, sun_path)
        sun.CreateIntensityAttr(sun_intensity)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.92))
        xform = UsdGeom.Xformable(sun)
        xform.AddRotateYOp().Set(-45.0)
        xform.AddRotateXOp().Set(-60.0)

def use_default_lighting(
    dome_path: str = "/World/EnvLight",
    sun_path: str = "/World/SunLight",
) -> None:
    """
    Remove custom lights so the viewport uses its default lighting.

    Call this before adding any custom lights if you want the stage defaults.
    """
    stage = get_current_stage()
    if stage.GetPrimAtPath(dome_path):
        stage.RemovePrim(dome_path)
    if stage.GetPrimAtPath(sun_path):
        stage.RemovePrim(sun_path)

def use_default_light_rig(rig_name: str = "Default", *, enable_auto_rig: bool = True) -> None:
    """
    Switch the viewport lighting to the built-in light rig (e.g. "Default").

    This matches the Viewport menu: Light Rigs -> Default.
    """
    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.kit.viewport.menubar.lighting")
    except Exception:
        pass

    # Remove any custom stage lights so the rig is the only illumination source.
    use_default_lighting()

    try:
        import carb
        import omni.usd
        from omni.kit.viewport.menubar.lighting import actions as lighting_actions
        from omni.kit.viewport.menubar.lighting.utility import _get_rig_names_and_paths

        if enable_auto_rig:
            carb.settings.get_settings().set(
                "/persistent/exts/omni.kit.viewport.menubar.lighting/autoLightRig/enabled",
                True,
            )

        rigs = _get_rig_names_and_paths("omni.kit.viewport.menubar.lighting") or []
        rig_names = [name for name, _ in rigs if name]
        if rig_names and rig_name not in rig_names:
            rig_name = rig_names[0]

        carb.settings.get_settings().set(
            "/exts/omni.kit.viewport.menubar.lighting/defaultRig", rig_name
        )

        lighting_actions._set_lighting_mode(
            rig_name, usd_context=omni.usd.get_context()
        )
    except Exception:
        # Fail silently if the viewport lighting extension is unavailable.
        pass


def add_ground_plane(
    world: World,
    path: str = "/World/Ground",
    *,
    size: float = 20.0,
    color: Sequence[float] = (0.9, 0.9, 0.9),
) -> None:
    """Insert a ground plane so the 機械手臂有支撐。"""
    color_array = np.array(color, dtype=np.float32)
    if not world.scene.get_object(path):
        world.scene.add(GroundPlane(path, size=size, color=color_array))


def import_robot_from_urdf(
    world: World,
    urdf_abs_path: str,
    *,
    robot_name: str = "robot",
    fix_base: bool = True,
    merge_fixed_joints: bool = False,
    make_default_prim: bool = True,
) -> Robot:
    """Parse and import a URDF robot, returning the wrapped Robot handle."""
    if not os.path.isabs(urdf_abs_path):
        raise ValueError("URDF path must be absolute to avoid relative stage issues.")

    urdf_dir, urdf_file = os.path.split(urdf_abs_path)
    import_cfg = _urdf.ImportConfig()
    import_cfg.set_fix_base(fix_base)
    import_cfg.set_merge_fixed_joints(merge_fixed_joints)
    import_cfg.set_make_default_prim(make_default_prim)
    import_cfg.set_create_physics_scene(True)

    parsed_robot = _urdf_interface.parse_urdf(urdf_dir, urdf_file, import_cfg)
    root_prim = _urdf_interface.import_robot(
        urdf_dir, urdf_file, parsed_robot, import_cfg, "", True
    )

    return world.scene.add(Robot(prim_path=root_prim, name=robot_name))


def wait_for_manual_gui_close(simulation_app) -> None:
    """Block until the Isaac Sim GUI window is closed."""
    if not simulation_app.is_running(): 
        return
    print("動作結束，請手動關閉 Isaac Sim GUI 視窗以結束程式。")
    while simulation_app.is_running():
        simulation_app.update()
