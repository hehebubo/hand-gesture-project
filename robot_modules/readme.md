project_root/
│
├── move_tx60l.py                # 主程式
├── Sim.py                 # (既有) 環境設置
├── joint_def.py           # (既有) 關節動作定義
├── hand.py                # (既有) 視覺端
│
└── robot_modules/         # [修改] 改用這個名字，避開系統關鍵字
    ├── __init__.py        # 必須有
    ├── ik_solver.py       # Lula IK 相關
    ├── ros_interface.py   # ROS2 通訊相關
    ├── state_machine.py   # 狀態機
    └── controller.py      # 您的 TeleopController (映射與濾波)