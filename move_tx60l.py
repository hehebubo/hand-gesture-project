"""
TX60L motion driver with ROS 2 Integration (Numpy Fix & IK Placeholder).

[FIX] 環境問題解法：請務必執行 './python.sh -m pip install "numpy<2.0"'
[FIX] 視窗閃退解法：在 finally 區塊強制等待 GUI 關閉。
"""

from __future__ import annotations

import sys
import os
import enum
import traceback  # [NEW] 用於印出詳細錯誤
import numpy as np

# [FIX] 1. 確保當前目錄在 Python 搜尋路徑中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def _ensure_ros_env():
    """
    預設啟用內建 humble ROS 2，避免 extension 啟動時找不到環境變數。
    若使用系統 ROS，請自行在外部設定並移除這段。
    """
    ros_root = "/home/scl114/Documents/isaac-sim/exts/isaacsim.ros2.bridge/humble"
    project_root = os.path.dirname(os.path.abspath(__file__))
    shim_lib = os.path.join(project_root, "lib_shim")
    env = os.environ
    if "ROS_DISTRO" not in env:
        env["ROS_DISTRO"] = "humble"
    if "RMW_IMPLEMENTATION" not in env:
        env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    # AMENT_PREFIX_PATH 用於尋找資源；若外部未設定，指向內建 humble
    if "AMENT_PREFIX_PATH" not in env:
        env["AMENT_PREFIX_PATH"] = ros_root
    # 確保 LD_LIBRARY_PATH 包含 shim (補 liblibrosidl_runtime_c.so) 與內建 humble/lib
    ros_lib = os.path.join(ros_root, "lib")
    desired = [shim_lib, ros_lib]
    existing = [p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    for p in existing:
        if p not in desired:
            desired.append(p)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(desired)

    # 預先 dlopen 必要的 ROS 2 C-lib，避免 extension 找不到
    import ctypes
    for lib_name in ("liblibrosidl_runtime_c.so", "librosidl_runtime_c.so"):
        candidate = os.path.join(shim_lib, lib_name)
        if os.path.exists(candidate):
            try:
                ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                break
            except OSError as exc:  # noqa: PERF203
                print(f"[ROS ENV] preload {lib_name} failed: {exc}")

    # Debug 1 次，確認環境是否帶入
    if not env.get("_ROS_ENV_ECHOED"):
        env["_ROS_ENV_ECHOED"] = "1"
        print("[ROS ENV] ROS_DISTRO=", env.get("ROS_DISTRO"))
        print("[ROS ENV] RMW_IMPLEMENTATION=", env.get("RMW_IMPLEMENTATION"))
        print("[ROS ENV] AMENT_PREFIX_PATH=", env.get("AMENT_PREFIX_PATH"))
        print("[ROS ENV] LD_LIBRARY_PATH=", env.get("LD_LIBRARY_PATH"))

_ensure_ros_env()

from isaacsim import SimulationApp

# 啟動模擬器
simulation_app = SimulationApp({"headless": False})

# 確保啟用內建 ROS2 Bridge extension（使用 OmniGraph）
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
enable_extension("omni.graph.action")

import carb
from pxr import Gf, UsdGeom

# 引入自定義模組
from Sim import (
    add_ground_plane,
    create_world,
    ensure_basic_lighting,
    import_robot_from_urdf,
    wait_for_manual_gui_close,
)
from joint_def import Keyframe, JointMotionController

# ros2_bridge.py 不再使用（改用 OmniGraph ROS2 Bridge）

class Mode(enum.Enum):
    LOOP = "A"
    PAUSE = "B"
    RETURNING = "C_return"
    HOME = "C_hold"
    ROS_CONTROL = "D_ros"

# ... (LoopRoutine, ReturnHomeAction 保持原樣) ...
# ... (為了節省篇幅，這裡省略這兩個類別的定義，請保持原樣) ...
class LoopRoutine:
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
    def __init__(self, controller, home_pose, routine: LoopRoutine):
        self._controller = controller
        self._home_pose = controller.clamp(home_pose)
        self._routine = routine
        self._return_home = ReturnHomeAction(controller, self._home_pose, duration=2.5)
        self._mode = Mode.HOME
        self._hold_pose = self._home_pose[:]
        
        # ROS 2 由 OmniGraph 處理，此處只控制模式切換
        self._ros_enabled = True
        self._ros_sub = None

    def handle_mode_request(self, requested: Mode):
        if requested == Mode.LOOP:
            self._enter_loop()
        elif requested == Mode.PAUSE:
            self._enter_pause()
        elif requested == Mode.RETURNING:
            self._enter_return()
        elif requested == Mode.ROS_CONTROL:
            self._enter_ros_control()

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
        
        elif self._mode == Mode.ROS_CONTROL:
            # ROS 控制改由 OmniGraph 處理，此處不直接讀取 Python rclpy
            pass

        if self._mode == Mode.HOME:
            self._controller.apply_pose(self._home_pose)

    # ... (Enter functions 保持原樣) ...
    def _enter_loop(self):
        self._routine.reset_cycle()
        self._return_home.stop()
        self._mode = Mode.LOOP
        self._routine.resume()
        print("[mode] A: Loop")

    def _enter_pause(self):
        self._return_home.stop()
        self._routine.pause()
        self._hold_pose = self._controller.current_pose()
        self._mode = Mode.PAUSE
        print("[mode] B: Pause")

    def _enter_return(self):
        self._routine.pause()
        self._return_home.start()
        self._mode = Mode.RETURNING
        print("[mode] C: Home")

    def _enter_ros_control(self):
        self._routine.pause()
        self._return_home.stop()
        # OmniGraph 接 ROS 指令，不依賴 Python 訂閱器
        self._mode = Mode.ROS_CONTROL
        print("[mode] D: ROS Control (OmniGraph)")


class KeyboardModeSwitcher:
    # [FIX] 強化版鍵盤監聽
    def __init__(self, mode_manager: ModeManager):
        self._mode_manager = mode_manager
        self._input = carb.input.acquire_input_interface()
        self._subscription = self._input.subscribe_to_input_events(
            self._on_input_event, order=-100
        )
        self._debug_left = 8  # 只在前幾筆事件印詳細欄位

    def shutdown(self):
        if self._subscription is not None:
            self._input.unsubscribe_to_input_events(self._subscription)
            self._subscription = None

    def _on_input_event(self, event, *_args, **_kwargs) -> bool:
        """
        兼容 Isaac Sim 5.0 的 Keyboard 事件結構：
        - 有些事件在 event.keyboard 內，有些直接在 event.input
        - 預設只處理 key press/repeat；鬆鍵會直接略過
        """
        print("[raw-event]", event)

        def _extract_key(evt):
            kb = getattr(evt, "keyboard", None)
            if kb is not None:
                return getattr(kb, "input", None) or getattr(kb, "key", None)
            evtev = getattr(evt, "event", None)
            if evtev is not None:
                return (
                    getattr(evtev, "input", None)
                    or getattr(evtev, "key", None)
                    or getattr(evtev, "character", None)
                )
            return (
                getattr(evt, "input", None)
                or getattr(evt, "key", None)
                or getattr(evt, "character", None)
            )

        def _extract_type(evt):
            kb = getattr(evt, "keyboard", None)
            if kb is not None:
                return getattr(kb, "type", None)
            evtev = getattr(evt, "event", None)
            if evtev is not None:
                return getattr(evtev, "type", None)
            return getattr(evt, "type", None)

        def _extract_value(evt):
            kb = getattr(evt, "keyboard", None)
            if kb is not None:
                return getattr(kb, "value", None)
            evtev = getattr(evt, "event", None)
            if evtev is not None:
                return getattr(evtev, "value", None)
            return getattr(evt, "value", None)

        key = _extract_key(event)
        event_type = _extract_type(event)
        key_value = _extract_value(event)

        # 觀察解析結果
        try:
            kb = getattr(event, "keyboard", None)
            evtev = getattr(event, "event", None)
            if self._debug_left > 0:
                attrs = {}
                for name in ("device", "device_id", "device_type", "flags", "timestamp"):
                    if hasattr(event, name):
                        attrs[name] = getattr(event, name)
                # 簡化的 dir 列表
                attrs_list = [a for a in dir(event) if not a.startswith("_")]
                ev_dir = [a for a in dir(evtev) if not a.startswith("_")] if evtev else None
                print(
                    "[event-debug]",
                    "key=", key,
                    "type=", event_type,
                    "value=", key_value,
                    "kbd.input=", getattr(kb, "input", None) if kb else None,
                    "kbd.key=", getattr(kb, "key", None) if kb else None,
                    "kbd.type=", getattr(kb, "type", None) if kb else None,
                    "kbd.value=", getattr(kb, "value", None) if kb else None,
                    "ev.event=", evtev,
                    "ev.key=", getattr(evtev, "key", None) if evtev else None,
                    "ev.input=", getattr(evtev, "input", None) if evtev else None,
                    "ev.type=", getattr(evtev, "type", None) if evtev else None,
                    "ev.value=", getattr(evtev, "value", None) if evtev else None,
                    "ev.character=", getattr(evtev, "character", None) if evtev else None,
                    "evt.input=", getattr(event, "input", None),
                    "evt.key=", getattr(event, "key", None),
                    "evt.character=", getattr(event, "character", None),
                    "attrs=", attrs,
                    "dir=", attrs_list,
                    "ev.dir=", ev_dir,
                )
                self._debug_left -= 1
        except Exception as exc:
            print("[event-debug] print failed:", exc)

        # 針對不同型別的 key 進行比對（carb enum / int / 字串）
        def _match(k, target_enum):
            if k is None:
                return False
            # carb enum
            if k == target_enum:
                return True
            # 整數（可能是 enum 數值或 ASCII）
            if isinstance(k, int):
                return (
                    k == int(target_enum)
                    or k == ord(str(int(int(target_enum) - int(carb.input.KeyboardInput.KEY_0))))
                )
            # 單一字元字串
            if isinstance(k, str):
                key_str = k.lower()
                target_str = str(target_enum).lower()
                # 直接字串相等或包含關鍵字
                if key_str == target_str or target_str in key_str:
                    return True
                # 數字字元比對
                if len(k) == 1 and k.isdigit():
                    return int(k) == int(target_enum) - int(carb.input.KeyboardInput.KEY_0)
                return False
            return False

        # 只有 key_press/rep 或 value>0 才處理；沒有 type/value 時也照處理
        press_types = {
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        }
        if event_type is not None and event_type not in press_types:
            return True
        if key_value is not None and key_value <= 0:
            return True
        if key is None:
            return True

        if _match(key, carb.input.KeyboardInput.KEY_1):
            self._mode_manager.handle_mode_request(Mode.LOOP)
            print("[key] 1 -> Loop")
            return False
        if _match(key, carb.input.KeyboardInput.KEY_2):
            self._mode_manager.handle_mode_request(Mode.PAUSE)
            print("[key] 2 -> Pause")
            return False
        if _match(key, carb.input.KeyboardInput.KEY_3):
            self._mode_manager.handle_mode_request(Mode.RETURNING)
            print("[key] 3 -> Home")
            return False
        if _match(key, carb.input.KeyboardInput.KEY_4):
            self._mode_manager.handle_mode_request(Mode.ROS_CONTROL)
            print("[key] 4 -> ROS Control")
            return False
        return True

# ... (其餘輔助函數保持原樣) ...
def _build_keyframes(home_pose):
    """
    定義一組簡單的循環動作（相對 home 的偏移），每段約 2 秒：
    - keyframe1: 手腕抬高、手肘微彎
    - keyframe2: 手腕下壓、手肘伸直
    - keyframe3: 手腕回到中間、末端小幅旋轉
    """
    # 相對 home 的關節偏移（rad）
    offsets = [
        [0.0,  0.2, -0.2, 0.0,  0.15, 0.0],   # 抬高
        [0.0, -0.25, 0.25, 0.0, -0.2,  0.0],   # 下壓
        [0.0,  0.0,  0.0,  0.3,  0.0,  0.4],   # 手腕旋轉
    ]
    duration = 2.0

    keyframes = []
    for off in offsets:
        pose = [home_pose[i] + off[i] for i in range(len(home_pose))]
        keyframes.append(Keyframe(pose, duration))
    return keyframes

def _compute_robot_focus(prim_path):
    return Gf.Vec3d(0,0,0), Gf.Vec3d(1,1,1) # 簡化版

def _place_observer_camera(robot_prim_path):
    pass # 簡化版

def build_ros2_joint_command_graph(robot_prim_path: str, topic_name: str = "/joint_command"):
    """
    建立一個 OmniGraph，訂閱 ROS2 JointState，並將指令送到機器人。
    Topic 預設 /joint_command，訊息型別為 sensor_msgs/JointState（預設 OGN 節點）。
    """
    import omni.graph.core as og
    graph_path = "/World/ROS2JointCommandGraph"
    stage = getattr(og.GraphPipelineStage, "SIMULATION", None) or getattr(
        og.GraphPipelineStage, "GRAPH_PIPELINE_STAGE_SIMULATION", None
    )
    try:
        og.Controller.edit(
            {"graph_path": graph_path, "pipeline_stage": stage} if stage else {"graph_path": graph_path},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlayback", "omni.graph.action.OnPlaybackStep"),
                    ("Subscriber", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                    ("Controller", "isaacsim.core.nodes.IsaacArticulationController"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("Subscriber.inputs:topicName", topic_name),
                    ("Controller.inputs:robotPath", robot_prim_path),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlayback.outputs:tick", "Subscriber.inputs:execIn"),
                    ("Subscriber.outputs:execOut", "Controller.inputs:execIn"),
                    ("Subscriber.outputs:positionCommand", "Controller.inputs:positionCommand"),
                    ("Subscriber.outputs:velocityCommand", "Controller.inputs:velocityCommand"),
                    ("Subscriber.outputs:effortCommand", "Controller.inputs:effortCommand"),
                    ("Subscriber.outputs:jointNames", "Controller.inputs:jointNames"),
                ],
            },
        )
        print(f"[ROS] OmniGraph joint command graph created at {graph_path}, topic={topic_name}")
    except Exception as exc:
        print(f"[ROS] Failed to build ROS2 graph: {exc}")

def main():
    ensure_basic_lighting()
    world = create_world()
    add_ground_plane(world)

    try:
        robot = import_robot_from_urdf(world, URDF_ABS, robot_name="tx60l")
        world.reset()
    except Exception as e:
        print(f"[ERROR] 載入機器人失敗: {e}")
        traceback.print_exc() # [FIX] 印出詳細錯誤
        # 如果機器人載入失敗，下面的 controller 會掛掉，直接跳到 finally
        
    try:
        # [CHECK] 如果上面機器人沒載入，robot 變數可能不存在或為 None，這裡會報錯
        # 但我們已經在上面 catch 了，如果 robot 沒宣告，下面會報 UnboundLocalError
        
        controller = JointMotionController(robot)
        home = controller.home_pose([0.0, -0.8, 1.2, 0.0, 1.4, 0.0])
        
        routine = LoopRoutine(controller, _build_keyframes(home))

        # 建立 ROS2 OmniGraph（JointState -> ArticulationController）
        build_ros2_joint_command_graph(robot.prim_path)

        mode_manager = ModeManager(controller, home, routine)
        keyboard = KeyboardModeSwitcher(mode_manager)
        
        print("\n[controls] 1:Loop | 2:Pause | 3:Home | 4:ROS Control")
        mode_manager.handle_mode_request(Mode.LOOP)

        while simulation_app.is_running():
            simulation_app.update()
            dt = world.get_physics_dt()
            mode_manager.update(dt)
            world.step(render=True)
            
    except Exception as e:
        print(f"\n[CRITICAL ERROR] 主迴圈發生錯誤: {e}")
        traceback.print_exc()
        
    finally:
        if 'keyboard' in locals():
            keyboard.shutdown()
        print("[main] 程式結束，準備進入等待關閉模式...")
        
        # [FIX] 關鍵修改：無論如何（包含報錯），都呼叫 wait_for_manual_gui_close
        # 這樣視窗不會直接消失，讓你看到錯誤訊息
        wait_for_manual_gui_close(simulation_app)

# === 你的 URDF 絕對路徑 ===
URDF_ABS = "/home/scl114/Documents/isaac-sim/simPythonProject/hand-gesture-project/urdf_fixed/tx60l_fixed.urdf"

if __name__ == "__main__":
    main()
    simulation_app.close()
