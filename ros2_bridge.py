import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as R # 用於將矩陣轉為四元數

# 定義 Topic 名稱，兩邊要一致
TOPIC_NAME = '/hand_target'

class HandGesturePublisher(Node):
    """
    用於 MediaPipe 端：將計算好的座標發布為 ROS 2 Topic
    替代原本的 HandGestureSender
    """
    def __init__(self):
        # 檢查 ROS 2 是否已初始化，避免重複 init 報錯
        if not rclpy.ok():
            rclpy.init()
            
        super().__init__('hand_gesture_publisher')
        
        # 建立發布者: 發布 PoseStamped 類型的訊息
        self.publisher_ = self.create_publisher(PoseStamped, TOPIC_NAME, 10)
        print(f"[ROS2] Publisher node started on topic: {TOPIC_NAME}")

    def send_target(self, position, rotation_matrix=None):
        """
        發送目標數據
        position: [x, y, z] (單位: 公尺)
        rotation_matrix: 3x3 numpy array (選填)
        """
        msg = PoseStamped()
        
        # 1. 填入 Header (時間戳記與座標系ID)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link" # 假設相機座標系名稱

        # 2. 填入位置 (Position)
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])

        # 3. 填入姿態 (Orientation) - 矩陣轉四元數
        if rotation_matrix is not None:
            try:
                # 使用 scipy 將 3x3 旋轉矩陣轉為四元數 [x, y, z, w]
                quat = R.from_matrix(rotation_matrix).as_quat()
                msg.pose.orientation.x = quat[0]
                msg.pose.orientation.y = quat[1]
                msg.pose.orientation.z = quat[2]
                msg.pose.orientation.w = quat[3]
            except Exception as e:
                print(f"Rotation conversion error: {e}")
                msg.pose.orientation.w = 1.0 # 發生錯誤時給預設值
        else:
            # 如果沒給旋轉，預設為單位四元數 (不旋轉)
            msg.pose.orientation.w = 1.0

        # 發布訊息
        self.publisher_.publish(msg)

class IsaacSimSubscriber(Node):
    """
    用於 Isaac Sim 端：接收 ROS 2 Topic
    替代原本的 IsaacSimReceiver
    """
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
            
        super().__init__('isaac_hand_subscriber')
        
        # 建立訂閱者
        self.subscription = self.create_subscription(
            PoseStamped,
            TOPIC_NAME,
            self.listener_callback,
            10)
        
        self.latest_data = None
        print(f"[ROS2] Subscriber node started on topic: {TOPIC_NAME}")

    def listener_callback(self, msg):
        """
        當收到新訊息時會自動觸發此函數
        """
        # 將 ROS 訊息轉回我們習慣的 Dictionary 格式
        self.latest_data = {
            "x": msg.pose.position.x,
            "y": msg.pose.position.y,
            "z": msg.pose.position.z,
            # 如果需要旋轉，也可以在這裡解出來
            "orientation": {
                "x": msg.pose.orientation.x,
                "y": msg.pose.orientation.y,
                "z": msg.pose.orientation.z,
                "w": msg.pose.orientation.w
            }
        }

    def get_latest_target(self):
        """
        被 Isaac Sim 主迴圈呼叫
        重要：這裡必須執行 spin_once 來觸發 ROS 的 callback 機制
        """
        # spin_once(timeout_sec=0) 讓它只檢查一次，不會卡住主程式
        rclpy.spin_once(self, timeout_sec=0)
        return self.latest_data