import cv2
import mediapipe as mp
import math
import json
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import time
from collections import deque

# 初始化 MediaPipe 和姿勢偵測模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 中文字型路徑，請根據你的系統設定
font_path = "C:/Windows/Fonts/msjh.ttc"
font = ImageFont.truetype(font_path, 24)

# 動作狀態變數
stage = "up" 
count = 0
feedback = "準備好..."
current_rep_errors = []

# 追蹤角度變化所需的變數
down_start_time = None
max_depth_hip_angle = 180
max_depth_knee_angle = 180
max_knee_foot_displacement = 0 

# 新增：用於角度平滑的緩衝區
# 緩衝區可以儲存最近的N個角度值，用來計算平均，忽略單一幀的偵測誤差
HIP_ANGLE_BUFFER_SIZE = 5
KNEE_ANGLE_BUFFER_SIZE = 5
hip_angle_buffer = deque(maxlen=HIP_ANGLE_BUFFER_SIZE)
knee_angle_buffer = deque(maxlen=KNEE_ANGLE_BUFFER_SIZE)

# 新增的最小深度和最小時間限制
MIN_SQUAT_DEPTH = 140  # 髖關節角度小於此值才計數
MIN_SQUAT_DURATION = 0.5 # 下蹲時間大於此值才計數

# JSON 紀錄列表
log_data = []

# 輔助函式：計算兩向量之間的夾角
def calculate_angle(a, b, c):
    """
    計算由三個點形成的夾角，單位為度。
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# 載入影片 (或使用攝影機)
video_path = r"C:\Users\User\Desktop\AI專題\訓練資料\徒手深蹲.mp4"
cap = cv2.VideoCapture(video_path)

# 取得影片的原始長寬並設定顯示尺寸
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if original_width > original_height:
    display_width = 800
    display_height = int(display_width * original_height / original_width)
else:
    display_height = 800
    display_width = int(display_height * original_width / original_height)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (display_width, display_height))

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        
        # 判斷使用者面向的方向（左或右）
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        if left_shoulder[0] > right_shoulder[0]:
            side = "right"
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        else:
            side = "left"
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        # 計算原始關節角度
        raw_knee_angle = calculate_angle(hip, knee, ankle)
        raw_hip_angle = calculate_angle(shoulder, hip, knee)
        
        # 將原始角度加入緩衝區，用於平滑處理
        hip_angle_buffer.append(raw_hip_angle)
        knee_angle_buffer.append(raw_knee_angle)

        # 計算平滑後的角度
        hip_angle = np.mean(hip_angle_buffer)
        knee_angle = np.mean(knee_angle_buffer)
        
        # 在畫面上顯示角度
        cv2.putText(frame, str(int(knee_angle)), tuple(np.multiply(knee, [display_width, display_height]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(int(hip_angle)), tuple(np.multiply(hip, [display_width, display_height]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # ==================== 動作階段判斷邏輯 ====================

        # 1. 從站立 (up) 進入下蹲 (down)
        # 當髖關節角度小於165度時，視為開始下蹲
        if stage == "up" and hip_angle < 165:
            stage = "down"
            feedback = "正在下蹲..."
            current_rep_errors = []
            max_depth_hip_angle = 180
            max_depth_knee_angle = 180
            down_start_time = time.time()
            last_hip_angle = hip_angle
            last_knee_angle = knee_angle
            max_knee_foot_displacement = 0
        
        # 2. 下蹲過程中 (down) 的錯誤判斷
        if stage == "down":
            # 持續追蹤下蹲過程中的最深角度
            if hip_angle < max_depth_hip_angle:
                max_depth_hip_angle = hip_angle
            if knee_angle < max_depth_knee_angle:
                max_depth_knee_angle = knee_angle
            
            # 追蹤膝蓋與腳尖的水平位移，記錄最大超前值
            current_displacement = knee[0] - foot_index[0]
            if side == "left":
                current_displacement = foot_index[0] - knee[0]
            
            if current_displacement > max_knee_foot_displacement:
                max_knee_foot_displacement = current_displacement

            # 判斷「臀部推太多」或「膝蓋推太多」
            if last_hip_angle is not None and last_knee_angle is not None:
                hip_angle_change = abs(hip_angle - last_hip_angle)
                knee_angle_change = abs(knee_angle - last_knee_angle)

                # 臀部推太多：髖關節角度變化大(>15)，但膝蓋角度變化小(<10)
                if hip_angle_change > 15 and knee_angle_change < 10 and "臀部推太多" not in current_rep_errors:
                    current_rep_errors.append("臀部推太多")
                # 膝蓋推太多：膝蓋角度變化大(>15)，但髖關節角度變化小(<10)
                elif knee_angle_change > 15 and hip_angle_change < 10 and "膝蓋推太多" not in current_rep_errors:
                    current_rep_errors.append("膝蓋推太多")

            # 判斷「膝蓋超前腳尖」
            # 膝蓋的X座標與腳尖的X座標相差超過0.06（約畫面6%）
            if (side == "right" and knee[0] > foot_index[0] + 0.06) or (side == "left" and knee[0] < foot_index[0] - 0.06):
                if "膝蓋超前腳尖" not in current_rep_errors:
                    current_rep_errors.append("膝蓋超前腳尖")
            
            # 判斷「深度足夠」
            if hip_angle < 100 and "深度足夠" not in current_rep_errors:
                current_rep_errors.append("深度足夠")

            # 在螢幕上顯示即時回饋
            negative_feedback_onscreen = [err for err in current_rep_errors if err not in ["深度足夠"]]
            if negative_feedback_onscreen:
                feedback = ", ".join(negative_feedback_onscreen)
            else:
                feedback = "正在下蹲..."
            
            last_hip_angle = hip_angle
            last_knee_angle = knee_angle

        # 3. 從下蹲 (down) 回到站立 (up)，並進行計數
        # 當髖關節角度回到165度以上時，視為完成一次深蹲
        if stage == "down" and hip_angle > 165:
            # 必須滿足最小深度和最小時間，才計為一次完整深蹲
            if max_depth_hip_angle < MIN_SQUAT_DEPTH and (time.time() - down_start_time) > MIN_SQUAT_DURATION:
                count += 1
                
                # 整理最終回饋
                negative_feedback_list = [err for err in current_rep_errors if err not in ["深度足夠"]]
                if negative_feedback_list:
                    final_feedback = ", ".join(negative_feedback_list)
                else:
                    if "深度足夠" in current_rep_errors:
                        final_feedback = "標準"
                    else:
                        final_feedback = "深度不足"
                
                # 新增欄位來解釋膝蓋位移量，使其更易於理解
                displacement_note = "正值代表膝蓋超前腳尖，負值或0代表對齊。"
                if side == "left":
                    displacement_note = "正值代表膝蓋超前腳尖，負值或0代表對齊。"

                # 將本次深蹲的數據記錄到 JSON 列表中
                log_data.append({
                    "timestamp": datetime.now().isoformat(),
                    "rep": count,
                    "final_feedback": final_feedback,
                    "all_errors": current_rep_errors,
                    "max_depth_hip_angle": round(max_depth_hip_angle, 2),
                    "max_depth_knee_angle": round(max_depth_knee_angle, 2),
                    "duration_seconds": round(time.time() - down_start_time, 2),
                    "max_knee_foot_displacement": round(max_knee_foot_displacement, 2),
                    "knee_foot_displacement_note": displacement_note
                })
                
                stage = "up"
                feedback = f"第 {count} 次：{final_feedback}"
            
            else:
                # 若深度或時間不達標，則不計次數
                stage = "up"
                feedback = "深度或時間不足，不計數"
        
        # 在螢幕上顯示資訊
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

# 將所有數據寫入 JSON 檔案
with open("squat_data.json", "w", encoding="utf-8") as f:
    json.dump(log_data, f, ensure_ascii=False, indent=2)

print("JSON 儲存完成")
