# move_tx60l_from_urdf.py
# 從本地 URDF 匯入 Staubli TX60L，進場後以關節角控制；程式會跟 GUI 生命週期一起跑，關 GUI 才會結束。

import os, math
import numpy as np
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
from pxr import Gf, UsdGeom, UsdLux
import omni.usd

# ---- Isaac Sim Core (沿用舊命名空間，會有 deprecate 警告但可用) ----
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.objects.ground_plane import GroundPlane

# ---- URDF Importer (新 API) ----
from isaacsim.asset.importer.urdf import _urdf  # 官方文件示例用法
urdf_interface = _urdf.acquire_urdf_interface()
import_cfg = _urdf.ImportConfig()
import_cfg.set_merge_fixed_joints(False)     # 不合併 fixed joints（保留原關節）
import_cfg.set_fix_base(True)                # 固定底座
import_cfg.set_make_default_prim(True)       # 設定為 default prim
import_cfg.set_create_physics_scene(True)    # 若空場景，自動建立 physics scene

# === 你的 URDF 絕對路徑（請確認這個檔存在） ===
URDF_ABS = "/home/scl114/Documents/urdf_files_dataset-main/urdf_files/ros-industrial/xacro_generated/staubli/staubli_tx60_support/urdf/tx60l.urdf"
URDF_DIR, URDF_FILE = os.path.split(URDF_ABS)

# 解析並匯入到當前開啟的 Stage（in-memory）
parsed_robot = urdf_interface.parse_urdf(URDF_DIR, URDF_FILE, import_cfg)
# getArticulationRoot=True 會回傳 articulation root 的 prim path，方便後續掛 Robot 控制
tx_root_prim = urdf_interface.import_robot(URDF_DIR, URDF_FILE, parsed_robot, import_cfg, "", True)

# ---- 建世界與地板 ----
world = World(stage_units_in_meters=1.0)
ground_color = np.array([0.9, 0.9, 0.9], dtype=np.float32)
world.scene.add(GroundPlane("/World/Ground", size=20, color=ground_color))


def ensure_basic_lighting():
    """Add simple dome + distant lights so the scene isn't black."""
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/World/EnvLight"):
        dome = UsdLux.DomeLight.Define(stage, "/World/EnvLight")
        dome.CreateIntensityAttr(150.0)  # 全天域環境光的強度（數值越大越亮）
        dome.CreateSpecularAttr(0.1)     # 控制反射高光占比（越大越亮、越鏡面）
    if not stage.GetPrimAtPath("/World/SunLight"):
        sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
        sun.CreateIntensityAttr(10000.0)        # 平行光（太陽光）強度
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.92))  # 太陽光顏色，偏暖白
        xform = UsdGeom.Xformable(sun)
        xform.AddRotateYOp().Set(-45.0)  # 先繞 Y 旋轉，決定水平入射方向
        xform.AddRotateXOp().Set(-60.0)  # 再繞 X 旋轉，決定入射高度角


ensure_basic_lighting()

# ---- 用剛匯入的 prim 當 Robot 控制目標 ----
tx60l = world.scene.add(Robot(prim_path=tx_root_prim, name="tx60l"))
world.reset()

# 讀 DOF 與限制
dof_names = tx60l.dof_names
dof_props = tx60l.dof_properties
lower = dof_props["lower"].tolist()
upper = dof_props["upper"].tolist()

def clamp(q): # 將關節角 q 限制在 lower ~ upper 範圍內
    return [max(lower[i], min(upper[i], q[i])) for i in range(len(q))]

# 設定一個 home（6 軸；單位 rad），若模型多 DOF 就截斷
home = [0.0, -0.8, 1.2, 0.0, 1.4, 0.0]
home = home[:len(dof_names)]
home = clamp(home)


def wait_for_manual_gui_close(): #此函式用來保持 GUI 開啟
    """Keep GUI alive after script ends so the user can close it manually."""
    if not simulation_app.is_running():
        return
    print("動作結束，請手動關閉 Isaac Sim GUI 視窗以結束程式。")
    while simulation_app.is_running():
        simulation_app.update()


# 先平滑回 home（約 2 秒），再讓關節 1 做正弦擺動；整體直到關 GUI 才結束
phase = "go_home"
t = 0.0
step_counter = 0
steps_to_home = 2 * 60  # 2 秒（假設 60FPS）

try:
    while simulation_app.is_running():
        if phase == "go_home":
            cur = tx60l.get_joint_positions().tolist()
            alpha = min(1.0, step_counter / max(1, steps_to_home))
            tgt = [cur[i] + (home[i] - cur[i]) * alpha for i in range(len(home))]
            tx60l.apply_action(ArticulationAction(joint_positions=clamp(tgt)))
            step_counter += 1
            if alpha >= 1.0:
                phase = "sine"
                t = 0.0
        else:
            q = tx60l.get_joint_positions().tolist()
            if len(q) >= 1:
                q[0] = home[0] + 0.3 * math.sin(t)
            tx60l.apply_action(ArticulationAction(joint_positions=clamp(q)))
            t += 0.03

        world.step(render=True)
finally:
    wait_for_manual_gui_close()
    simulation_app.close()
