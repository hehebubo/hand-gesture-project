"""
TX60L motion driver with ROS 2 Integration (Numpy Fix & IK Placeholder).

[FIX] 環境問題解法：請務必執行 './python.sh -m pip install "numpy<2.0"'
[FIX] 視窗閃退解法：在 finally 區塊強制等待 GUI 關閉。
"""

from __future__ import annotations

import sys
import os
import importlib.util
import traceback  # [NEW] 用於印出詳細錯誤
import numpy as np

# [FIX] 1. 確保當前目錄在 Python 搜尋路徑中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def _resolve_bridge_lib(ros_distro: str) -> str | None:
    spec = importlib.util.find_spec("isaacsim")
    pkg_dir = None
    if spec and spec.submodule_search_locations:
        pkg_dir = spec.submodule_search_locations[0]
    elif spec and spec.origin:
        pkg_dir = os.path.dirname(spec.origin)
    if pkg_dir:
        candidate = os.path.join(
            pkg_dir,
            "exts",
            "isaacsim.ros2.bridge",
            ros_distro,
            "lib",
        )
        if os.path.isdir(candidate):
            return candidate

    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    fallback = os.path.join(
        sys.prefix,
        "lib",
        py_ver,
        "site-packages",
        "isaacsim",
        "exts",
        "isaacsim.ros2.bridge",
        ros_distro,
        "lib",
    )
    return fallback if os.path.isdir(fallback) else None


def _ensure_ros2_bridge_env() -> None:
    """Set ROS2 bridge env vars early so dlopen can find bundled RMW libs."""
    ros_distro = os.environ.get("ROS_DISTRO") or "humble"
    os.environ.setdefault("ROS_DISTRO", ros_distro)
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    bridge_lib = _resolve_bridge_lib(ros_distro)
    if bridge_lib:
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if bridge_lib not in ld_path.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{ld_path}:{bridge_lib}" if ld_path else bridge_lib


_ensure_ros2_bridge_env()

from isaacsim import SimulationApp

# 啟動模擬器
simulation_app = SimulationApp({"headless": False})

# 確保啟用內建 ROS2 Bridge extension（使用 OmniGraph）
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
enable_extension("omni.graph.action")

import omni.usd
from pxr import Gf

# 引入自定義模組
from Sim import (
    add_ground_plane,
    create_world,
    ensure_basic_lighting,
    import_robot_from_urdf,
    wait_for_manual_gui_close,
)
from joint_def import Keyframe, JointMotionController
from robot_modules.controller import TeleopController
from robot_modules.ik_solver import (
    LulaIKBridge,
    find_prim_by_suffix,
    get_world_T,
    rotmat_to_quat_xyzw,
)
from robot_modules.ros_interface import (
    HandTargetIKBridge,
    build_ros2_hand_target_graph,
    build_ros2_joint_command_graph,
)
from robot_modules.state_machine import (
    KeyboardModeSwitcher,
    LoopRoutine,
    Mode,
    ModeManager,
)

# ros2_bridge.py 不再使用（改用 OmniGraph ROS2 Bridge）

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
        
        # === IK / ROS2 hand pose setup ===
        LULA_YAML = "/home/scl114/Documents/isaac-sim/lula_configs/tx60l/tx60l.yaml"
        hand_graph = build_ros2_hand_target_graph("/hand_target")

        stage = omni.usd.get_context().get_stage()
        # --- hard stop: 不再處理 tool0 / flange，EE 直接用 link_6 ---
        # 有些情況 robot.prim_path 不是 /World 開頭，這裡順手做個修正
        robot_root = robot.prim_path
        if not stage.GetPrimAtPath(robot_root).IsValid():
            if not robot_root.startswith("/World"):
                robot_root = "/World" + robot_root
        print("[debug] robot_root =", robot_root)

        LINK6_PATH = find_prim_by_suffix(
            stage, robot_root, ["link_6", "link6", "link-6", "flange", "tool0", "ee", "end_effector"]
        )
        print("[debug] EE prim found =", LINK6_PATH)

        ee_name = LINK6_PATH.split("/")[-1]
        ik = LulaIKBridge(
            robot,
            LULA_YAML,
            URDF_ABS,
            ee_frame_name=ee_name,
            debug=False,
            allow_best_effort=True,
        )
        print("[IK] Lula IK ready. EE=", ee_name)

        Tw_link6 = get_world_T(stage, LINK6_PATH)
        link6_quat = rotmat_to_quat_xyzw(Tw_link6[:3, :3])
        link6_z = float(Tw_link6[2, 3])
        print("[debug] link6_quat =", link6_quat)
        print("[debug] link6_z =", link6_z)

        # [MODIFIED] use metric coordinates from hand.py in robot frame
        teleop_ctrl = TeleopController(
            target_center=(0.0, 0.0, 0.0),
            target_gain=(1.0, 1.0, 1.0),
            workspace_bounds={
                "x": (0.10, 0.80),
                "y": (-0.30, 0.30),
                "z": (0.15, 0.80),
            },
            hand_pos_is_0_1=False,
            alpha=0.15,
        )
        t_acc = 0.0

        controller = JointMotionController(robot)
        home = controller.home_pose([0.0, -0.8, 1.2, 0.0, 1.4, 0.0])
        
        routine = LoopRoutine(controller, _build_keyframes(home))

        hand_bridge = HandTargetIKBridge(
            robot,
            controller,
            LINK6_PATH,
            graph_path=hand_graph,
            prefer_rclpy=True,
            publish_ik=True,
            publish_topic="/ik_joint_states",
            debug=True,
            debug_every=10,
        )
        hand_bridge.set_ik_solver(ik)

        # 建立 ROS2 OmniGraph（JointState -> ArticulationController）
        build_ros2_joint_command_graph(robot.prim_path)

        mode_manager = ModeManager(controller, home, routine)
        keyboard = KeyboardModeSwitcher(mode_manager)
        
        print("\n[controls] 1:Loop | 2:Pause | 3:Home | 4:ROS Control")
        mode_manager.handle_mode_request(Mode.LOOP)
        prev_mode = None

        while simulation_app.is_running():
            simulation_app.update()
            dt = world.get_physics_dt()
            mode_manager.update(dt)
            current_mode = mode_manager.current_mode
            if current_mode != prev_mode:
                if current_mode == Mode.ROS_CONTROL:
                    teleop_ctrl.reset_filter()
                    hand_bridge.reset()
                prev_mode = current_mode
            if current_mode == Mode.ROS_CONTROL:
                t_acc += dt
                if t_acc >= 1.0 / 30.0:
                    t_acc = 0.0
                    hand_bridge.update(teleop_ctrl=teleop_ctrl)
            else:
                t_acc = 0.0
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
