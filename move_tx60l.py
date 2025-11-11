"""
TX60L motion driver with three runtime modes controlled via keyboard:

A 模式 (按鍵 1): 依序執行預設的多段關節動作並持續循環。
B 模式 (按鍵 2): 急停，立即維持當前姿勢並可隨時回到 A 繼續未完成的循環。
C 模式 (按鍵 3): 回到初始 home 姿勢並維持；若 C 後再切回 A，整個循環會從頭開始。

關閉 Isaac Sim GUI 之前，腳本會持續執行上述狀態機；GUI 關閉才算整個程式結束。
"""

from __future__ import annotations

import enum
from typing import Tuple

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb  # noqa: E402  pylint: disable=wrong-import-position
import omni.appwindow  # noqa: E402  pylint: disable=wrong-import-position
import omni.usd  # noqa: E402  pylint: disable=wrong-import-position
from pxr import Gf, UsdGeom  # noqa: E402  pylint: disable=wrong-import-position

from joint_def import (  # noqa: E402  pylint: disable=wrong-import-position
    Keyframe,
    JointMotionController,
    add_ground_plane,
    create_world,
    ensure_basic_lighting,
    import_robot_from_urdf,
    wait_for_manual_gui_close,
)


class Mode(enum.Enum):
    """High-level operating modes toggled via keyboard."""

    LOOP = "A"
    PAUSE = "B"
    RETURNING = "C_return"
    HOME = "C_hold"


class LoopRoutine:
    """Non-blocking keyframe player that can pause/resume across segments."""

    def __init__(self, controller: JointMotionController, keyframes):
        self._controller = controller
        self._targets = [controller.clamp(k.pose) for k in keyframes]
        self._durations = [max(1e-3, k.duration) for k in keyframes]
        self._segment = 0
        self._elapsed = 0.0
        self._segment_start = controller.current_pose()
        self._active = False

    def reset_cycle(self):
        self._segment = 0
        self._elapsed = 0.0
        self._segment_start = self._controller.current_pose()

    def resume(self):
        self._active = True

    def pause(self):
        self._active = False

    def update(self, dt: float):
        if not self._active or not self._targets:
            return

        target = self._targets[self._segment]
        duration = self._durations[self._segment]
        alpha = min(1.0, self._elapsed / duration)
        pose = [
            self._segment_start[i] + (target[i] - self._segment_start[i]) * alpha
            for i in range(self._controller.dof_count)
        ]
        self._controller.apply_pose(pose)
        self._elapsed += dt

        if self._elapsed >= duration:
            self._segment = (self._segment + 1) % len(self._targets)
            self._elapsed = 0.0
            self._segment_start = self._controller.current_pose()


class ReturnHomeAction:
    """Single-shot blend back to the provided home pose."""

    def __init__(self, controller: JointMotionController, home_pose, duration: float = 2.0):
        self._controller = controller
        self._home = controller.clamp(home_pose)
        self._duration = max(1e-3, duration)
        self._active = False
        self._elapsed = 0.0
        self._start = self._home

    def start(self):
        self._active = True
        self._elapsed = 0.0
        self._start = self._controller.current_pose()

    def stop(self):
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def update(self, dt: float) -> bool:
        if not self._active:
            return True

        alpha = min(1.0, self._elapsed / self._duration)
        pose = [
            self._start[i] + (self._home[i] - self._start[i]) * alpha
            for i in range(self._controller.dof_count)
        ]
        self._controller.apply_pose(pose)
        self._elapsed += dt

        if self._elapsed >= self._duration:
            self._controller.apply_pose(self._home)
            self._active = False
            return True
        return False


class ModeManager:
    """Coordinates keyboard commands with the underlying motion primitives."""

    def __init__(self, controller, home_pose, routine: LoopRoutine):
        self._controller = controller
        self._home_pose = controller.clamp(home_pose)
        self._routine = routine
        self._return_home = ReturnHomeAction(controller, self._home_pose, duration=2.5)
        self._mode = Mode.HOME
        self._hold_pose = self._home_pose[:]
        self._cycle_reset_pending = True

    def handle_mode_request(self, requested: Mode):
        if requested == Mode.LOOP:
            self._enter_loop()
        elif requested == Mode.PAUSE:
            self._enter_pause()
        elif requested == Mode.RETURNING:
            self._enter_return()

    def update(self, dt: float):
        if self._mode == Mode.LOOP:
            self._routine.update(dt)
        elif self._mode == Mode.PAUSE:
            self._controller.apply_pose(self._hold_pose)
        elif self._mode == Mode.RETURNING:
            finished = self._return_home.update(dt)
            if finished:
                self._mode = Mode.HOME
                self._hold_pose = self._home_pose[:]
                print("[mode] 已回到 C 模式：保持 home 姿勢。")

        if self._mode == Mode.HOME:
            self._controller.apply_pose(self._home_pose)

    def _enter_loop(self):
        if self._cycle_reset_pending:
            self._routine.reset_cycle()
        self._return_home.stop()
        self._mode = Mode.LOOP
        self._cycle_reset_pending = False
        self._routine.resume()
        print("[mode] A 模式啟動：開始 / 持續循環動作。")

    def _enter_pause(self):
        self._return_home.stop()
        self._routine.pause()
        self._hold_pose = self._controller.current_pose()
        self._mode = Mode.PAUSE
        print("[mode] B 模式：急停並保持目前姿勢。")

    def _enter_return(self):
        self._routine.pause()
        self._return_home.start()
        self._mode = Mode.RETURNING
        self._cycle_reset_pending = True
        print("[mode] C 模式：回到 home 姿勢並鎖定。")


class KeyboardModeSwitcher:
    """Registers keyboard shortcuts that map to Mode commands."""

    def __init__(self, mode_manager: ModeManager):
        self._mode_manager = mode_manager
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

    def shutdown(self):
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            self._subscription = None
        if self._input is not None:
            if hasattr(carb.input, "release_input_interface"):
                carb.input.release_input_interface(self._input)
            self._input = None

    def _on_keyboard_event(self, event, *_args, **_kwargs) -> bool:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return False

        if event.input == carb.input.KeyboardInput.KEY_1:
            self._mode_manager.handle_mode_request(Mode.LOOP)
            return True
        if event.input == carb.input.KeyboardInput.KEY_2:
            self._mode_manager.handle_mode_request(Mode.PAUSE)
            return True
        if event.input == carb.input.KeyboardInput.KEY_3:
            self._mode_manager.handle_mode_request(Mode.RETURNING)
            return True
        return False


def _build_keyframes(home_pose):
    """Preset poses used for the looping routine."""
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
    routine = LoopRoutine(controller, _build_keyframes(home))

    mode_manager = ModeManager(controller, home, routine)
    keyboard = KeyboardModeSwitcher(mode_manager)
    _place_observer_camera(robot.prim_path)

    print("[controls] 1 → A 模式 | 2 → B 模式 | 3 → C 模式。關閉 GUI 以結束程式。")
    mode_manager.handle_mode_request(Mode.LOOP)  # 預設開啟 A 模式

    try:
        while simulation_app.is_running():
            dt = world.get_physics_dt()
            mode_manager.update(dt)
            world.step(render=True)
    finally:
        keyboard.shutdown()


# === 你的 URDF 絕對路徑（請確認這個檔存在） ===
URDF_ABS = "/home/scl114/Documents/urdf_files_dataset-main/urdf_files/ros-industrial/xacro_generated/staubli/staubli_tx60_support/urdf/tx60l.urdf"


if __name__ == "__main__":
    try:
        main()
        wait_for_manual_gui_close(simulation_app)
    finally:
        simulation_app.close()
