"""
Main entry point that drives the Staubli TX60L arm via reusable helpers.

使用的函式/類別:
- ensure_basic_lighting(): 建立環境光與太陽光，讓匯入的模型不會變黑。
- create_world() & add_ground_plane(): 建立物理世界並鋪地板，避免手臂掉落。
- import_robot_from_urdf(): 將 URDF 轉成場景中的可控 Robot 物件。
- JointMotionController: 提供 clamp、move_smooth、play_keyframes 等控制介面。
- wait_for_manual_gui_close(): 腳本跑完後保持 GUI，方便觀察結果。
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from joint_def import (  # noqa: E402  pylint: disable=wrong-import-position
    Keyframe,
    JointMotionController,
    add_ground_plane,
    create_world,
    ensure_basic_lighting,
    import_robot_from_urdf,
    wait_for_manual_gui_close,
)
from pxr import Gf, UsdGeom
import omni.usd
from typing import Tuple

# === 你的 URDF 絕對路徑（請確認這個檔存在） ===
URDF_ABS = "/home/scl114/Documents/urdf_files_dataset-main/urdf_files/ros-industrial/xacro_generated/staubli/staubli_tx60_support/urdf/tx60l.urdf"


def _build_keyframes(home_pose):
    """Precompute a few joint-space poses for the sample routine."""
    reach_forward = home_pose.copy()
    reach_forward[0] += 0.3
    reach_forward[1] = -0.4
    reach_forward[2] = 0.9
    reach_forward[3] = -0.3
    reach_forward[4] = 1.2
    reach_forward[5] = 0.6

    prep_pick = home_pose.copy()
    prep_pick[0] -= 0.4
    prep_pick[1] = -0.6
    prep_pick[2] = 1.1
    prep_pick[3] = 0.4
    prep_pick[4] = 0.9
    prep_pick[5] = -0.5

    lift_high = home_pose.copy()
    lift_high[1] = -0.3
    lift_high[2] = 1.4
    lift_high[3] = 0.2
    lift_high[4] = 1.3
    lift_high[5] = 0.0

    return [
        Keyframe(pose=home_pose.copy(), duration=0.7),
        Keyframe(pose=reach_forward, duration=1.2),
        Keyframe(pose=prep_pick, duration=1.0),
        Keyframe(pose=lift_high, duration=1.0),
        Keyframe(pose=home_pose.copy(), duration=0.8),
    ]


def _compute_robot_focus(prim_path: str) -> Tuple[Gf.Vec3d, Gf.Vec3d]:
    """
    Estimate a good camera target/offset from the robot bounding box.

    Returns:
        (target, offset): target 是 bbox 中心上方 10% 高的位置，
        offset 則朝負 X、正 Y 方向退後並抬高，避免鏡頭位於地面以下。
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return Gf.Vec3d(0.0, 0.0, 0.8), Gf.Vec3d(-1.8, 0.8, 0.9)

    try:
        bbox_cache = UsdGeom.BBoxCache(0.0, ["default"])
        bbox = bbox_cache.ComputeWorldBound(prim)
        box = bbox.GetBox()
        center = box.GetCenter()
        size = box.GetSize()
        target = center + Gf.Vec3d(0.0, 0.0, max(0.05, size[2] * 0.1))
        offset = Gf.Vec3d(
            -max(1.2, size[0] * 1.5),
            max(0.6, size[1] * 0.7),
            max(0.9, size[2] * 0.9),
        )
        return target, offset
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[camera] Failed to compute bbox ({exc}), using fallback placement.")
        return Gf.Vec3d(0.0, 0.0, 0.8), Gf.Vec3d(-1.8, 0.8, 0.9)


def _place_observer_camera(robot_prim_path: str) -> None:
    """
    Spawn/align a camera near the arm so 觀察角度固定.

    Args:
        robot_prim_path: Robot 的 articulation root，用來估算 bbox。
            若路徑錯誤會退回預設位置。
    """
    stage = omni.usd.get_context().get_stage()
    cam_path = "/World/ArmObserver"
    camera = UsdGeom.Camera.Get(stage, cam_path)
    if not camera:
        camera = UsdGeom.Camera.Define(stage, cam_path)
        camera.CreateFocalLengthAttr(24.0)

    try:
        target, offset = _compute_robot_focus(robot_prim_path)
        cam_pos = target + offset
        transform = Gf.Matrix4d().SetLookAt(cam_pos, target, Gf.Vec3d(0.0, 0.0, 1.0))
        xform = UsdGeom.Xformable(camera)
        ops = xform.GetOrderedXformOps()
        if ops:
            ops[0].Set(transform)
        else:
            xform.AddTransformOp().Set(transform)
        print(f"[camera] Placed observer view at {cam_pos} looking at {target}.")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[camera] Failed to place observer camera: {exc}")


def main():
    ensure_basic_lighting()
    world = create_world()
    add_ground_plane(world)

    robot = import_robot_from_urdf(world, URDF_ABS, robot_name="tx60l")
    world.reset()

    controller = JointMotionController(robot)
    home = controller.home_pose([0.0, -0.8, 1.2, 0.0, 1.4, 0.0])
    _place_observer_camera(robot.prim_path)

    print("[motion] Moving to home pose...")
    controller.move_smooth(world, home, duration=2.0)
    print("[motion] Playing keyframe routine...")
    controller.play_keyframes(world, _build_keyframes(home))

    print("[motion] Oscillating selected joints...")
    controller.oscillate(
        world,
        base_pose=home,
        joints=[0, 1, 3, 5],
        amplitude=0.3,
        frequency=0.35,
        duration=8.0,
    )

    print("[motion] Holding home pose...")
    controller.hold_pose(world, home, duration=0.5)


if __name__ == "__main__":
    try:
        main()
        wait_for_manual_gui_close(simulation_app)
    finally:
        simulation_app.close()
