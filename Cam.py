"""
Webcam-based hand landmark viewer using OpenCV + MediaPipe.

執行後會開啟攝影機畫面，並以 Mediapipe Hands 偵測並標註每隻手的21個關鍵點與連線。
按下 q 或 ESC 結束。
"""

from __future__ import annotations

import sys
from typing import Optional, Tuple

import cv2
import mediapipe as mp


DrawingSpec = Tuple[int, int, int]


def _draw_hand_landmarks(
    frame_bgr,
    results,
    *,
    landmark_color: DrawingSpec = (0, 255, 0),
    text_color: DrawingSpec = (255, 255, 255),
) -> None:
    """Draw MediaPipe hand landmarks and their indices on the provided BGR frame."""
    if not results.multi_hand_landmarks:
        return

    h, w, _ = frame_bgr.shape
    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    for hand_landmarks in results.multi_hand_landmarks:
        drawing_utils.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style(),
        )
        # Overlay index number on each landmark for easier reference.
        for idx, lm in enumerate(hand_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_bgr, (cx, cy), 3, landmark_color, -1)
            cv2.putText(
                frame_bgr,
                str(idx),
                (cx + 3, cy - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                text_color,
                1,
                cv2.LINE_AA,
            )


def run_camera(
    camera_index: int = 0,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """
    Open a webcam stream and display annotated hand landmarks.

    Args:
        camera_index: cv2.VideoCapture index, 預設 0。
        width/height: 期望的畫面尺寸；None 則使用攝影機預設。
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[Cam] 無法開啟攝影機 (index={camera_index}).", file=sys.stderr)
        return

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    hands = mp.solutions.hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Cam] 讀取影像失敗，結束。", file=sys.stderr)
                break

            # Mirror the view for a selfie-like experience.
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            _draw_hand_landmarks(frame, results)

            cv2.imshow("Hand Landmarks (press q/ESC to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 27 = ESC
                break
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera()
