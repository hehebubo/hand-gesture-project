"""
Joint-space utilities for TX60L and other articulated arms.

Typical usage (環境建立請改用 Sim.py)::

    from joint_def import get_preset_pose, JointMotionController, Keyframe
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction

__all__ = [
    "get_preset_pose",
    "Keyframe",
    "JointMotionController",
]

# Common joint-space presets (unit: rad). Extend as needed.
PRESET_POSES = {
    "tx60l_home": [0.0, -0.8, 1.2, 0.0, 1.4, 0.0],
    # 適合作為復歸姿勢：手肘略低、手腕打開，較不會碰撞。
    "tx60l_reset": [0.15, -1.05, 1.45, 0.0, 1.05, 0.0],
}


def get_preset_pose(
    name: str = "tx60l_home",
    *,
    fallback: Sequence[float] | None = None,
) -> List[float]:
    """
    Retrieve a named preset joint pose for快速復歸或起始姿態。

    Args:
        name: 在 PRESET_POSES 中註冊的名稱，如 "tx60l_home" 或 "tx60l_reset"。
        fallback: 查無對應名稱時改回的姿態；缺省時會拋出 ValueError。
            設定與預設姿差距過大可能造成瞬間跳躍，建議仍搭配 controller.clamp。
    Returns:
        List[float]: 指定姿態的複本，可直接給 JointMotionController.clamp 使用。
    """
    pose = PRESET_POSES.get(name)
    if pose is None:
        if fallback is None:
            raise ValueError(f"Unknown preset pose '{name}'.")
        pose = list(fallback)
    return list(pose)


@dataclass
class Keyframe:
    """
    Simple container describing a target joint pose and播放時間.

    Args:
        pose: 關節角列表（rad）。長度不足會在 controller 端自動 padding。
        duration: 從當前姿勢平滑走到目標所花的秒數；時間過短會造成動作突然、
            過長則使動作顯得拖慢。
    用途:
        以關節空間 keyframes 方式定義播放序列，保持腳本結構清晰。
    """

    pose: Sequence[float]
    duration: float


class JointMotionController:
    """
    Wrap Isaac Sim Robot 以提供常用的關節空間操作。

    功能:
        - clamp(): 依機器人關節限制截斷角度。
        - move_smooth(): 由目前姿勢平滑走到目標姿勢。
        - hold_pose(): 維持某姿勢一定時間，確保模型不下墜。
        - play_keyframes(): 依序播放多個 Keyframe。
        - oscillate(): 讓多個關節做正弦擺動，方便檢查動作或展示。
    """

    def __init__(self, robot: Robot):
        self._robot = robot
        self.dof_names = robot.dof_names
        props = robot.dof_properties
        self._lower = props["lower"].tolist()
        self._upper = props["upper"].tolist()
        self.dof_count = len(self.dof_names)

    def _pad(self, values: Sequence[float]) -> List[float]:
        padded = list(values[: self.dof_count])
        if len(padded) < self.dof_count:
            padded += [0.0] * (self.dof_count - len(padded))
        return padded

    def clamp(self, values: Sequence[float]) -> List[float]:
        """
        Clamp a candidate joint vector to the robot limits.

        Args:
            values: 任何長度的角度列表。
        Returns:
            list: 已截斷且長度符合機器人 DOF 的角度。
        用途:
            撰寫動作時避免超出機構極限導致模擬發散。
        """
        padded = self._pad(values)
        return [
            max(self._lower[i], min(self._upper[i], padded[i]))
            for i in range(self.dof_count)
        ]

    def current_pose(self) -> List[float]:
        """
        Read current joint angles as a list.

        用途: 排程下一個動作前先取得瞬時狀態，例如閉迴路補償。
        """
        return self._robot.get_joint_positions().tolist()

    def home_pose(self, fallback: Sequence[float] | None = None) -> List[float]:
        """
        Return a sanitized home pose, optionally using a provided fallback.

        Args:
            fallback: 自訂的 home；若缺省則使用全 0。
                角度與實際安全姿勢差太多會造成回 home 時需要較大移動距離。
        用途:
            快速取得安全起始點，可再搭配 get_preset_pose("tx60l_reset")。
        """
        if fallback is None:
            fallback = [0.0] * self.dof_count
        return self.clamp(fallback)

    def apply_pose(self, pose: Sequence[float]) -> None: 
        """Apply a joint pose immediately (no stepping or interpolation)."""
        self._robot.apply_action(ArticulationAction(joint_positions=self.clamp(pose)))
    

    def move_smooth(
        self,
        world: World,
        target: Sequence[float],
        *,
        duration: float = 2.0,
        render: bool = True,
    ) -> None:
        """
        Blend from the current pose to the target pose over the chosen duration.

        Args:
            world: 同步的 World。
            target: 期望角度。
            duration: 走完動作的秒數；過小會導致行程過快、可能超出馬達能力，
                過大則動作拖慢但更平滑。
            render: 是否同步渲染；設為 False 可加速純模擬，但就看不到畫面。
        用途:
            比直接設定 joint positions 更平滑，不會造成瞬間跳躍。
        """
        dt = world.get_physics_dt()
        steps = max(1, int(duration / dt))
        start = self.current_pose()
        goal = self.clamp(target)

        for step in range(1, steps + 1):
            alpha = step / steps
            pose = [
                start[i] + (goal[i] - start[i]) * alpha for i in range(self.dof_count)
            ]
            self._robot.apply_action(ArticulationAction(joint_positions=self.clamp(pose)))
            world.step(render=render)

    def hold_pose(
        self,
        world: World,
        pose: Sequence[float] | None = None,
        *,
        duration: float = 1.0,
        render: bool = True,
    ) -> None:
        """
        Maintain a pose for a fixed duration.

        Args:
            pose: 要維持的姿勢；缺省則使用目前姿勢。
            duration: 維持秒數；時間過短仍可能在落下前放開，過長則浪費模擬時間。
        用途:
            停動後鎖住關節，避免因重力造成自由落體。
        """
        target = self.clamp(pose or self.current_pose())
        dt = world.get_physics_dt()
        steps = max(1, int(duration / dt))
        for _ in range(steps):
            self._robot.apply_action(ArticulationAction(joint_positions=target))
            world.step(render=render)

    def play_keyframes(
        self,
        world: World,
        keyframes: Iterable[Keyframe],
        *,
        render: bool = True,
    ) -> None:
        """
        Run a sequence of Keyframe objects back-to-back.

        Args:
            keyframes: Keyframe iterable。
            render: 是否同步渲染。
        用途:
            以劇本形式安排多個姿勢，便於複用及維護。
        """
        for keyframe in keyframes:
            self.move_smooth(
                world, keyframe.pose, duration=keyframe.duration, render=render
            )

    def oscillate(
        self,
        world: World,
        *,
        base_pose: Sequence[float] | None = None,
        joints: Sequence[int] | None = None,
        amplitude: float = 0.25,
        frequency: float = 0.5,
        duration: float = 6.0,
        phase_offset: float = 0.2,
        render: bool = True,
    ) -> None:
        """
        Drive the selected joints with sine waves for a fixed duration.

        Args:
            base_pose: 正弦疊加的基準姿勢。
            joints: 需要擺動的關節 index。
            amplitude: 正弦擺動幅度（rad）；過大會打到極限或造成自碰撞，過小則肉眼難辨。
            frequency: 擺動頻率（Hz）；過高會讓模擬需要更細時間步，過低則動作遲滯。
            duration: 播放秒數；過長會持續占用模擬時間。
            phase_offset: 每顆關節的相位差；過大（>π）會造成相位跳躍，0 則同步擺動。
        用途:
            展示與測試關節聯動，或模擬簡單掃描動作。
        """
        dt = world.get_physics_dt()
        steps = max(1, int(duration / dt))
        base = self.clamp(base_pose or self.current_pose())
        candidate_joints = list(joints) if joints else list(range(min(3, self.dof_count)))

        for step in range(steps):
            t = step * dt
            pose = base.copy()
            for idx, joint in enumerate(candidate_joints):
                if joint >= self.dof_count:
                    continue
                pose[joint] = base[joint] + amplitude * math.sin(
                    2.0 * math.pi * frequency * t + phase_offset * idx
                )
            self._robot.apply_action(ArticulationAction(joint_positions=self.clamp(pose)))
            world.step(render=render)
