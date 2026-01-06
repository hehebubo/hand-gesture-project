"""
robot_modules/ros_interface.py
[Critical Fix] Remove queueSize from Pose subscriber, enable detailed error output
"""
from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

import importlib.util
import numpy as np
import omni.graph.core as og
import omni.usd
import os
import random
import sys

if TYPE_CHECKING:
    from joint_def import JointMotionController
    from robot_modules.controller import TeleopController


_OG_POSE_UNAVAILABLE = False


def _clip_workspace(pos: np.ndarray, bounds) -> np.ndarray:
    if bounds is None:
        return pos
    for idx, axis in enumerate(("x", "y", "z")):
        axis_bounds = bounds.get(axis) if isinstance(bounds, dict) else None
        if axis_bounds is None:
            continue
        pos[idx] = np.clip(pos[idx], axis_bounds[0], axis_bounds[1])
    return pos


def _normalize_pose(pos, quat) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        pos_np = np.asarray(pos, dtype=np.float64).reshape(-1)
        quat_np = np.asarray(quat, dtype=np.float64).reshape(-1)
        if pos_np.size != 3 or quat_np.size != 4:
            return None, None
        if not np.all(np.isfinite(pos_np)) or not np.all(np.isfinite(quat_np)):
            return None, None
        if np.allclose(pos_np, 0) and np.allclose(quat_np, 0):
            return None, None
        return pos_np, quat_np
    except Exception:
        return None, None


def _read_pose_from_og(node_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Helper to read position and orientation from an OG node."""
    def get(name):
        try:
            attr = og.Controller.attribute(f"{node_path}.{name}")
            return og.Controller.get(attr)
        except Exception:
            return None

    pos_fields = [
        get("outputs:pose:position:x"),
        get("outputs:pose:position:y"),
        get("outputs:pose:position:z"),
    ]
    quat_fields = [
        get("outputs:pose:orientation:x"),
        get("outputs:pose:orientation:y"),
        get("outputs:pose:orientation:z"),
        get("outputs:pose:orientation:w"),
    ]

    if all(v is not None for v in pos_fields + quat_fields):
        pos_np, quat_np = _normalize_pose(pos_fields, quat_fields)
        if pos_np is not None:
            return pos_np, quat_np

    pos = get("outputs:position") or get("outputs:location") or get("outputs:pose")
    if pos is None:
        pos = get("outputs:translation")

    quat = get("outputs:orientation") or get("outputs:rotation")

    if pos is None or quat is None:
        return None, None

    return _normalize_pose(pos, quat)


def _ensure_internal_rclpy_on_path(ros_distro: str) -> None:
    if importlib.util.find_spec("rclpy"):
        return
    spec = importlib.util.find_spec("isaacsim")
    pkg_dir = None
    if spec and spec.submodule_search_locations:
        pkg_dir = spec.submodule_search_locations[0]
    elif spec and spec.origin:
        pkg_dir = os.path.dirname(spec.origin)
    if not pkg_dir:
        return
    candidate = os.path.join(
        pkg_dir,
        "exts",
        "isaacsim.ros2.bridge",
        ros_distro,
        "rclpy",
    )
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)


def build_ros2_hand_target_graph(topic_name: str = "/hand_target") -> str:
    """
    Build OmniGraph nodes for hand target pose.
    [Fix] Removed inputs:queueSize to avoid missing-attr failures.
    """
    global _OG_POSE_UNAVAILABLE
    if _OG_POSE_UNAVAILABLE:
        return ""

    base_id = random.randint(1000, 9999)

    candidates = [
        "isaacsim.ros2.bridge.ROS2Subscriber",
        "isaacsim.ros2.nodes.ROS2Subscriber",
        "omni.isaac.ros2_bridge.ROS2Subscriber",
    ]

    keys = og.Controller.Keys
    success_path = ""

    print(f"[ROS-OG] Building ROS2 Hand Target Graph (topic: {topic_name})")

    for i, node_type in enumerate(candidates):
        current_path = f"/World/ROS2HandTargetGraph_{base_id}_try{i}"

        try:
            stage = omni.usd.get_context().get_stage()
            if stage.GetPrimAtPath(current_path).IsValid():
                stage.RemovePrim(current_path)

            og.Controller.edit(
                {"graph_path": current_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnPlayback", "omni.graph.action.OnPlaybackTick"),
                        ("SubPose", node_type),
                    ],
                    keys.SET_VALUES: [
                        ("SubPose.inputs:topicName", topic_name),
                    ],
                    keys.CONNECT: [
                        ("OnPlayback.outputs:tick", "SubPose.inputs:execIn"),
                    ],
                },
            )
            sub_path = f"{current_path}/SubPose"
            pkg_set = _set_first_existing_attr(
                og,
                sub_path,
                ["inputs:messagePackage", "inputs:msgPackage", "inputs:message_package"],
                "geometry_msgs",
                label="message package",
            )
            name_set = _set_first_existing_attr(
                og,
                sub_path,
                ["inputs:messageName", "inputs:msgName", "inputs:message_name"],
                "Pose",
                label="message name",
            )
            if not (pkg_set or name_set):
                _set_first_existing_attr(
                    og,
                    sub_path,
                    ["inputs:messageType", "inputs:msgType", "inputs:messageTypeName", "inputs:msgTypeName"],
                    "geometry_msgs/msg/Pose",
                    label="message type",
                )
            print("[ROS-OG] Graph created")
            print(f"  - path: {current_path}")
            print(f"  - node: {node_type}")
            success_path = current_path
            break

        except Exception as exc:
            print(f"[ROS-Debug] Failed {node_type}: {exc}")
            try:
                omni.usd.get_context().get_stage().RemovePrim(current_path)
            except Exception:
                pass
            continue

    if not success_path:
        print("[ROS-OG] CRITICAL: All candidate node types failed.")
        _OG_POSE_UNAVAILABLE = True
        return ""

    return success_path


class HandTargetIKBridge:
    def __init__(
        self,
        robot,
        controller: "JointMotionController",
        ee_prim_path: str,
        graph_path: str,
        topic_name: str = "/hand_target",
        enable_rclpy_fallback: bool = True,
        prefer_rclpy: bool = False,
        publish_ik: bool = False,
        publish_topic: str = "/ik_joint_states",
        debug: bool = False,
        debug_every: int = 10,
        deadband: float = 0.015,
        low_pass_alpha: float = 0.05,
        max_step: float = 0.0015,
        workspace_bounds: Optional[dict] = None,
    ):
        self.robot = robot
        self._controller = controller
        self.ee_prim_path = ee_prim_path
        self.graph_path = graph_path
        self.node_path = f"{graph_path}/SubPose"
        self._topic_name = topic_name
        self._use_rclpy = False
        self._rclpy_sub = None
        self._allow_rclpy_fallback = bool(enable_rclpy_fallback)
        self._rclpy_attempted = False
        self._prefer_rclpy = bool(prefer_rclpy)
        self._publish_ik = bool(publish_ik)
        self._publish_topic = publish_topic
        self._ik_pub = None
        self._debug = bool(debug)
        self._debug_every = max(1, int(debug_every))
        self._debug_count = 0
        self._last_seed: Optional[np.ndarray] = None

        self.art_ctrl = (
            getattr(robot, "get_articulation_controller", lambda: None)()
            or getattr(robot, "articulation_controller", None)
        )
        self._ik = None
        self.deadband = float(deadband)
        self.low_pass_alpha = float(low_pass_alpha)
        self.max_step = float(max_step)
        self.workspace_bounds = workspace_bounds
        self._last_target: Optional[np.ndarray] = None
        self._filtered_target: Optional[np.ndarray] = None
        self._prev_cmd_target: Optional[np.ndarray] = None
        if (self._prefer_rclpy or not self.graph_path) and self._allow_rclpy_fallback:
            self._init_rclpy_fallback()
        if self._publish_ik:
            self._init_ik_publisher()

    def set_ik_solver(self, ik_solver_instance):
        self._ik = ik_solver_instance

    def _init_rclpy_fallback(self):
        if self._rclpy_attempted:
            return
        self._rclpy_attempted = True
        try:
            ros_distro = os.environ.get("ROS_DISTRO") or "humble"
            _ensure_internal_rclpy_on_path(ros_distro)
            self._rclpy_sub = _RclpyHandPoseSubscriber(self._topic_name)
            self._use_rclpy = True
            print("[ROS] Using rclpy fallback for hand target.")
        except Exception as exc:
            print(f"[ROS] rclpy fallback unavailable: {exc}")

    def _init_ik_publisher(self):
        if self._ik_pub is not None:
            return
        try:
            ros_distro = os.environ.get("ROS_DISTRO") or "humble"
            _ensure_internal_rclpy_on_path(ros_distro)
            joint_names = []
            if self._controller is not None and hasattr(self._controller, "dof_names"):
                joint_names = list(self._controller.dof_names)
            elif getattr(self.robot, "dof_names", None):
                joint_names = list(self.robot.dof_names)
            self._ik_pub = _RclpyJointStatePublisher(self._publish_topic, joint_names)
            print(f"[ROS] Publishing IK joint states to {self._publish_topic}")
        except Exception as exc:
            print(f"[ROS] IK publisher unavailable: {exc}")
            self._ik_pub = None

    def update(self, teleop_ctrl=None, fixed_quat=None, fixed_z: Optional[float] = None, seed=None):
        if not self._ik:
            return
        pos = None
        quat = None
        self._debug_count += 1
        do_log = self._debug and (self._debug_count % self._debug_every == 0)
        if self._prefer_rclpy:
            if not self._use_rclpy and self._allow_rclpy_fallback:
                self._init_rclpy_fallback()
            if self._use_rclpy and self._rclpy_sub:
                self._rclpy_sub.spin_once(timeout_sec=0.0)
                pos, quat = self._rclpy_sub.latest()
            if pos is None and self.graph_path:
                pos, quat = _read_pose_from_og(self.node_path)
        else:
            if self.graph_path:
                pos, quat = _read_pose_from_og(self.node_path)
            if pos is None and self._allow_rclpy_fallback:
                self._init_rclpy_fallback()
                if self._use_rclpy and self._rclpy_sub:
                    self._rclpy_sub.spin_once(timeout_sec=0.0)
                    pos, quat = self._rclpy_sub.latest()
        if pos is None:
            return
        if do_log:
            raw_pos = np.array2string(np.asarray(pos, dtype=np.float64), precision=4, suppress_small=True)
            raw_quat = np.array2string(np.asarray(quat, dtype=np.float64), precision=4, suppress_small=True)
            print(f"[ROS-IK] raw_pos={raw_pos} raw_quat={raw_quat}")

        target_pos = teleop_ctrl.process(pos) if teleop_ctrl else pos
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(-1)
        if target_pos.size < 3 or not np.all(np.isfinite(target_pos[:3])):
            return
        target_pos = target_pos[:3]
        if fixed_z is not None and np.isfinite(fixed_z):
            target_pos[2] = float(fixed_z)

        if self.deadband > 0.0 and self._last_target is not None:
            if np.linalg.norm(target_pos - self._last_target) < self.deadband:
                target_pos = self._last_target.copy()
        self._last_target = target_pos.copy()

        if 0.0 < self.low_pass_alpha < 1.0:
            if self._filtered_target is None:
                self._filtered_target = target_pos.copy()
            else:
                self._filtered_target = (
                    self.low_pass_alpha * target_pos
                    + (1.0 - self.low_pass_alpha) * self._filtered_target
                )
            target_pos = self._filtered_target.copy()

        if self._prev_cmd_target is None:
            self._prev_cmd_target = target_pos.copy()
        elif self.max_step > 0.0:
            delta = target_pos - self._prev_cmd_target
            dist = np.linalg.norm(delta)
            if dist > self.max_step:
                target_pos = self._prev_cmd_target + delta / dist * self.max_step

        bounds = self.workspace_bounds
        if bounds is None and teleop_ctrl is not None:
            bounds = getattr(teleop_ctrl, "workspace_bounds", None)
        target_pos = _clip_workspace(target_pos, bounds)
        self._prev_cmd_target = target_pos.copy()

        quat_use = fixed_quat if fixed_quat is not None else quat

        if seed is None:
            seed = self._last_seed
        if seed is None and self._controller:
            try:
                seed = np.array(self._controller.current_pose(), dtype=np.float64)
            except Exception:
                pass
        if do_log:
            tgt = np.array2string(target_pos, precision=4, suppress_small=True)
            quat_fmt = np.array2string(np.asarray(quat_use, dtype=np.float64), precision=4, suppress_small=True)
            seed_flag = "set" if seed is not None else "none"
            print(f"[ROS-IK] target_pos={tgt} quat={quat_fmt} seed={seed_flag}")

        try:
            res = self._ik.solve(target_pos, quat_use, seed=seed)
        except TypeError:
            res = self._ik.solve(target_pos, quat_use)

        action = None
        success = True
        if isinstance(res, tuple):
            if len(res) >= 2:
                first, second = res[0], res[1]
                if isinstance(first, (bool, np.bool_)) and not isinstance(second, (bool, np.bool_)):
                    success = bool(first)
                    action = second
                elif isinstance(second, (bool, np.bool_)) and not isinstance(first, (bool, np.bool_)):
                    success = bool(second)
                    action = first
                else:
                    action = first
            elif res:
                action = res[0]
        else:
            action = res

        if (not success) or action is None:
            if do_log:
                print(f"[ROS-IK] solve failed (success={success})")
            return

        jp = getattr(action, "joint_positions", None)
        if jp is not None:
            joint_positions = np.asarray(jp, dtype=np.float64).reshape(-1)
            if joint_positions.size == 0 or not np.all(np.isfinite(joint_positions)):
                return
            if do_log:
                jp_fmt = np.array2string(joint_positions, precision=4, suppress_small=True)
                print(f"[ROS-IK] joint_positions={jp_fmt}")
            if self._ik_pub:
                self._ik_pub.publish(joint_positions)

        if self.art_ctrl:
            self.art_ctrl.apply_action(action)
            if jp is not None:
                self._last_seed = joint_positions.copy()

    def reset(self):
        self._last_seed = None
        self._last_target = None
        self._filtered_target = None
        self._prev_cmd_target = None


class _RclpyHandPoseSubscriber:
    def __init__(self, topic: str = "/hand_target"):
        self._topic = topic
        self._latest: Optional[Tuple[np.ndarray, np.ndarray]] = None

        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Pose

        if not rclpy.ok():
            rclpy.init(args=None)

        class _N(Node):
            pass

        self._node = _N("isaac_hand_target_sub")
        self._sub_pose = self._node.create_subscription(
            Pose, topic, self._cb_pose, 10
        )

    def _cb_pose(self, msg):
        p = msg.position
        o = msg.orientation
        p = np.array([p.x, p.y, p.z], dtype=np.float64)
        q = np.array([o.x, o.y, o.z, o.w], dtype=np.float64)
        self._latest = (p, q)

    def spin_once(self, timeout_sec: float = 0.0):
        import rclpy
        rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._latest is None:
            return None, None
        return self._latest[0], self._latest[1]


class _RclpyJointStatePublisher:
    def __init__(self, topic: str, joint_names):
        self._topic = topic
        self._joint_names = list(joint_names) if joint_names else []
        self._warned_name_mismatch = False

        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        if not rclpy.ok():
            rclpy.init(args=None)

        class _N(Node):
            pass

        self._node = _N("isaac_ik_joint_state_pub")
        self._pub = self._node.create_publisher(JointState, topic, 10)
        self._msg_type = JointState

    def publish(self, joint_positions: np.ndarray) -> None:
        msg = self._msg_type()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        if self._joint_names and len(self._joint_names) == len(joint_positions):
            msg.name = list(self._joint_names)
        elif self._joint_names and not self._warned_name_mismatch:
            print("[ROS] Joint name count mismatch; publishing positions only.")
            self._warned_name_mismatch = True
        msg.position = [float(x) for x in joint_positions.tolist()]
        self._pub.publish(msg)


def _set_first_existing_attr(og_core, base_node_path: str, candidates, value, label: str = "target prim") -> bool:
    for attr_name in candidates:
        full = f"{base_node_path}.{attr_name}"
        try:
            attr = og_core.Controller.attribute(full)
            if attr:
                og_core.Controller.set(attr, value)
                print(f"[ROS] Set {attr_name} = {value}")
                return True
        except Exception:
            continue
    print(f"[ROS] Warning: Could not set {label} on {base_node_path}")
    return False


def build_ros2_joint_command_graph(robot_prim_path: str, topic_name: str = "/joint_command"):
    graph_path = "/World/ROS2JointCommandGraph"

    try:
        if hasattr(og.Controller, "graph"):
            if og.Controller.graph(graph_path):
                return
    except Exception:
        pass

    candidates = [
        "isaacsim.ros2.bridge.ROS2SubscribeJointState",
        "omni.isaac.ros2_bridge.ROS2SubscribeJointState",
    ]

    keys = og.Controller.Keys

    for node_type in candidates:
        try:
            stage = omni.usd.get_context().get_stage()
            if stage.GetPrimAtPath(graph_path).IsValid():
                stage.RemovePrim(graph_path)

            og.Controller.edit(
                {"graph_path": graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnPlayback", "omni.graph.action.OnPlaybackTick"),
                        ("Subscriber", node_type),
                        ("Controller", "isaacsim.core.nodes.IsaacArticulationController"),
                    ],
                    keys.SET_VALUES: [
                        ("Subscriber.inputs:topicName", topic_name),
                    ],
                    keys.CONNECT: [
                        ("OnPlayback.outputs:tick", "Subscriber.inputs:execIn"),
                        ("Subscriber.outputs:execOut", "Controller.inputs:execIn"),
                        ("Subscriber.outputs:positionCommand", "Controller.inputs:positionCommand"),
                        ("Subscriber.outputs:velocityCommand", "Controller.inputs:velocityCommand"),
                        ("Subscriber.outputs:effortCommand", "Controller.inputs:effortCommand"),
                        ("Subscriber.outputs:jointNames", "Controller.inputs:jointNames"),
                    ],
                },
            )

            _set_first_existing_attr(
                og,
                graph_path + "/Controller",
                ["inputs:robotPath", "inputs:targetPrim", "inputs:robotPrim"],
                robot_prim_path,
            )
            print(f"[ROS-OG] Joint Graph created ({node_type})")
            return
        except Exception:
            continue

    print("[ROS] Failed to build joint graph.")
