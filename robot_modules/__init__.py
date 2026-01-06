# robot_modules/__init__.py

# 使用相對路徑 (.) 匯入，這樣不管資料夾搬去哪都不會壞
from .controller import TeleopController
from .ik_solver import LulaIKBridge, rotmat_to_quat_xyzw, get_world_T
from .ros_interface import (
    HandTargetIKBridge,
    build_ros2_hand_target_graph,
    build_ros2_joint_command_graph,
)
from .state_machine import ModeManager, KeyboardModeSwitcher, Mode, LoopRoutine

# 您也可以定義 __all__，這是一種好習慣，但在這裡非必要
__all__ = [
    "TeleopController",
    "LulaIKBridge",
    "rotmat_to_quat_xyzw",
    "get_world_T",
    "HandTargetIKBridge",
    "build_ros2_hand_target_graph",
    "HandVelocityIKBridge",
    "build_ros2_hand_cmd_graph",
    "build_ros2_joint_command_graph",
    "ModeManager",
    "KeyboardModeSwitcher",
    "Mode",
    "LoopRoutine",
]

# Backward-compatible aliases (old name references).
HandVelocityIKBridge = HandTargetIKBridge
build_ros2_hand_cmd_graph = build_ros2_hand_target_graph
