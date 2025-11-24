"""
Scene/world helpers for Isaac Sim (環境設定、匯入 URDF 等).

Typical usage::

    from Sim import ensure_basic_lighting, create_world, add_ground_plane, import_robot_from_urdf
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import omni.usd
from isaacsim.asset.importer.urdf import _urdf
from omni.isaac.core import World
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.robots import Robot
from pxr import Gf, UsdGeom, UsdLux

__all__ = [
    "ensure_basic_lighting",
    "create_world",
    "add_ground_plane",
    "import_robot_from_urdf",
    "wait_for_manual_gui_close",
]

_urdf_interface = _urdf.acquire_urdf_interface()


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
    stage = omni.usd.get_context().get_stage()
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


def create_world(stage_units: float = 1.0, physics_dt: float = 1.0 / 60.0) -> World:
    """Create an Isaac World with optional unit/dt overrides."""
    return World(stage_units_in_meters=stage_units, physics_dt=physics_dt)


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
