from __future__ import annotations

from typing import Optional

import numpy as np
from pxr import Gf, UsdGeom

from isaacsim.core.utils.types import ArticulationAction


def _gf_matrix_to_np(m: Gf.Matrix4d) -> np.ndarray:
    return np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)


def _np_to_gf_matrix(a: np.ndarray) -> Gf.Matrix4d:
    m = Gf.Matrix4d()
    for i in range(4):
        for j in range(4):
            m[i][j] = float(a[i, j])
    return m


def pose_to_mat4(pos_xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Return 4x4 homogeneous matrix from position + quat(x,y,z,w)."""
    x, y, z, w = [float(v) for v in quat_xyzw]
    q = Gf.Quatd(w, Gf.Vec3d(x, y, z))
    r = Gf.Matrix3d(q)
    t = Gf.Vec3d(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2]))
    m = Gf.Matrix4d(1.0)
    m.SetRotate(r)
    m.SetTranslate(t)
    return _gf_matrix_to_np(m)


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to quaternion (x, y, z, w).
    Robust for numerical issues.
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)

    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def find_prim_by_suffix(stage, _root_path: str, suffix_candidates) -> str:
    suffix_candidates = [s.lower() for s in suffix_candidates]
    hits = []
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        pl = p.lower()
        if any(pl.endswith("/" + s) or pl.endswith(s) for s in suffix_candidates):
            hits.append(p)

    if not hits:
        raise RuntimeError(
            f"Could not find prim with suffix in {suffix_candidates} anywhere in stage"
        )
    hits.sort(key=len)
    return hits[0]


def get_world_T(stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim not found or invalid: {prim_path}")
    cache = UsdGeom.XformCache()
    m = cache.GetLocalToWorldTransform(prim)
    return _gf_matrix_to_np(Gf.Matrix4d(m))


def compute_link6_tool0_offset(stage, link6_path: str, tool0_path: str) -> np.ndarray:
    """Return T_link6_tool0 (link6->tool0) in rest pose."""
    Tw_link6 = get_world_T(stage, link6_path)
    Tw_tool0 = get_world_T(stage, tool0_path)
    return inv_T(Tw_link6) @ Tw_tool0


class LulaIKBridge:
    """
    Loads Lula kinematics from your generated YAML, solves IK to EE (link_6),
    and returns joint positions for controller.apply_pose().
    """

    def __init__(
        self,
        robot,
        lula_yaml_path: str,
        urdf_path: str,
        ee_frame_name: str = "link_6",
        debug: bool = False,
        allow_best_effort: bool = False,
    ):
        self.robot = robot
        self.lula_yaml_path = lula_yaml_path
        self.urdf_path = urdf_path
        self.ee_frame_name = ee_frame_name
        self.debug = debug
        self.allow_best_effort = allow_best_effort
        self._fail_count = 0
        self._solver = self._create_solver()

    def _create_solver(self):
        errors = []
        articulation = (
            getattr(self.robot, "_articulation", None)
            or getattr(self.robot, "articulation", None)
            or self.robot
        )

        try:
            from isaacsim.robot_motion.motion_generation.articulation_kinematics_solver import (
                ArticulationKinematicsSolver,
            )
            from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver
            try:
                lula = LulaKinematicsSolver(
                    urdf_path=self.urdf_path,
                    robot_description_path=self.lula_yaml_path,
                )
            except TypeError:
                lula = LulaKinematicsSolver(self.urdf_path, self.lula_yaml_path)
            return ArticulationKinematicsSolver(articulation, lula, self.ee_frame_name)
        except Exception as e:
            errors.append(("isaacsim.robot_motion.motion_generation.lula.kinematics", repr(e)))

        try:
            from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver
            from isaacsim.robot_motion.motion_generation.lula import LulaKinematicsSolver
            try:
                lula = LulaKinematicsSolver(
                    urdf_path=self.urdf_path,
                    robot_description_path=self.lula_yaml_path,
                )
            except TypeError:
                lula = LulaKinematicsSolver(self.urdf_path, self.lula_yaml_path)
            return ArticulationKinematicsSolver(articulation, lula, self.ee_frame_name)
        except Exception as e:
            errors.append(("isaacsim.robot_motion.motion_generation.lula", repr(e)))

        try:
            from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
            try:
                lula = LulaKinematicsSolver(
                    urdf_path=self.urdf_path,
                    robot_description_path=self.lula_yaml_path,
                )
            except TypeError:
                lula = LulaKinematicsSolver(self.urdf_path, self.lula_yaml_path)
            return ArticulationKinematicsSolver(articulation, lula, self.ee_frame_name)
        except Exception as e:
            errors.append(("omni.isaac.motion_generation", repr(e)))

        print("[IK] Failed to create Lula IK solver. Tried candidates:")
        for name, err in errors:
            print(f"  - {name}: {err}")
        raise RuntimeError("Could not create Lula IK solver. See logs above.")

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
        seed: Optional[np.ndarray] = None,
    ) -> Optional[ArticulationAction]:
        """
        target_pos: (3,) target position of EE (link_6) in world.
        target_quat_xyzw: (4,) quaternion target in x,y,z,w.
        returns: ArticulationAction or None
        """
        try:
            pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
            quat_xyzw = np.asarray(target_quat_xyzw, dtype=np.float64).reshape(4)
            n = np.linalg.norm(quat_xyzw)
            if n < 1e-12:
                quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            else:
                quat_xyzw = quat_xyzw / n

            if hasattr(self._solver, "compute_inverse_kinematics"):
                if seed is None:
                    res = self._solver.compute_inverse_kinematics(pos, quat_xyzw)
                else:
                    try:
                        res = self._solver.compute_inverse_kinematics(pos, quat_xyzw, seed)
                    except TypeError:
                        res = self._solver.compute_inverse_kinematics(pos, quat_xyzw)
            else:
                if seed is None:
                    res = self._solver.compute_ik(pos, quat_xyzw)
                else:
                    try:
                        res = self._solver.compute_ik(pos, quat_xyzw, seed)
                    except TypeError:
                        res = self._solver.compute_ik(pos, quat_xyzw)

            if self.debug:
                print("[IK debug] res type=", type(res), "res=", res)

            if res is None:
                return None

            if isinstance(res, tuple) and len(res) >= 2:
                first, second = res[0], res[1]
                if isinstance(first, ArticulationAction) and isinstance(second, (bool, np.bool_)):
                    if second:
                        self._fail_count = 0
                        return first
                    self._fail_count += 1
                    return first if self.allow_best_effort else None
                if isinstance(second, ArticulationAction) and isinstance(first, (bool, np.bool_)):
                    if first:
                        self._fail_count = 0
                        return second
                    self._fail_count += 1
                    return second if self.allow_best_effort else None
                if isinstance(first, (bool, np.bool_)):
                    if not first:
                        self._fail_count += 1
                        return None
                    arr = np.asarray(second, dtype=np.float64).reshape(-1)
                    self._fail_count = 0
                    return ArticulationAction(joint_positions=arr.tolist()) if arr.size > 1 else None
                if isinstance(second, (bool, np.bool_)):
                    if not second:
                        self._fail_count += 1
                        return None
                    arr = np.asarray(first, dtype=np.float64).reshape(-1)
                    self._fail_count = 0
                    return ArticulationAction(joint_positions=arr.tolist()) if arr.size > 1 else None
                for item in res:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        arr = np.asarray(item, dtype=np.float64).reshape(-1)
                        if arr.size > 1:
                            self._fail_count = 0
                            return ArticulationAction(joint_positions=arr.tolist())
                return None

            if isinstance(res, ArticulationAction):
                self._fail_count = 0
                return res

            if isinstance(res, dict):
                for k in ("joint_positions", "joints", "positions", "solution"):
                    if k in res:
                        arr = np.asarray(res[k], dtype=np.float64).reshape(-1)
                        self._fail_count = 0
                        return ArticulationAction(joint_positions=arr.tolist()) if arr.size > 1 else None
                return None

            arr = np.asarray(res, dtype=np.float64).reshape(-1)
            if arr.size <= 1:
                return None
            self._fail_count = 0
            return ArticulationAction(joint_positions=arr.tolist())

        except Exception as e:
            if self.debug:
                print(f"[IK] solve failed: {e}")
            return None
