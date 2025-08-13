import cv2
import mediapipe as mp
import math
import json
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 中文字型
font_path = "C:/Windows/Fonts/msjh.ttc"
font = ImageFont.truetype(font_path, 24)

# 動作幅度門檻
LOW_DEPTH = 0.12   # 深度不足
HIGH_DEPTH = 0.42  # 蹲太低

stage = None
count = 0
start_hip_y = None
min_hip_y_in_rep = None
feedback = ""

# JSON 紀錄
log_data = []

# 載入影片
video_path = r"C:\Users\User\Desktop\AI專題\訓練資料\air squat.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=0.6, fy=0.6)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark

        # 取右側臀部座標（可改左側 index=23）
        hip_y = landmarks[24].y

        # 初始化起始站立高度
        if start_hip_y is None:
            start_hip_y = hip_y

        # 計算下蹲比例（相對站立高度）
        drop_ratio = (hip_y - start_hip_y) / start_hip_y

        # 狀態切換
        if drop_ratio > 0.12 and stage != "down":  # 開始下蹲
            stage = "down"
            min_hip_y_in_rep = hip_y

        elif stage == "down" and drop_ratio < 0.05:  # 回到站立
            stage = "up"
            count += 1

            # 計算深蹲深度
            depth_ratio = (min_hip_y_in_rep - start_hip_y) / start_hip_y
            if depth_ratio < LOW_DEPTH:
                feedback = "深度不足"
            elif depth_ratio > HIGH_DEPTH:
                feedback = "蹲太低"
            else:
                feedback = "標準"

            # 存 JSON（每次深蹲一筆）
            log_data.append({
                "timestamp": datetime.now().isoformat(),
                "rep": count,
                "drop_ratio": round(depth_ratio, 3),
                "feedback": feedback
            })

            min_hip_y_in_rep = None

        # 更新最低點
        if stage == "down" and (min_hip_y_in_rep is None or hip_y > min_hip_y_in_rep):
            min_hip_y_in_rep = hip_y

        # 顯示文字
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((10, 30), f"次數: {count}", font=font, fill=(0, 255, 0))
        draw.text((10, 60), f"狀態: {stage}", font=font, fill=(0, 255, 255))
        draw.text((10, 90), f"回饋: {feedback}", font=font, fill=(255, 0, 0))
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("Squat Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 輸出 JSON
with open("squat_data.json", "w", encoding="utf-8") as f:
    json.dump(log_data, f, ensure_ascii=False, indent=2)

print("JSON 儲存完成")
