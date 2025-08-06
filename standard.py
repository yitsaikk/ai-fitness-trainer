#利用mediapipe讀取影片中 深蹲時膝蓋的角度

import cv2
import mediapipe as mp
import math

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 影片路徑
video_path = "D:\\fitness\\exampleVideo\\airsquat.mp4"
cap = cv2.VideoCapture(video_path)

# 計算角度的函式
def calculate_angle(a, b, c):
    ba = [a[0]-b[0], a[1]-b[1]]
    bc = [c[0]-b[0], c[1]-b[1]]
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    norm_ba = math.hypot(*ba)
    norm_bc = math.hypot(*bc)
    angle = math.acos(dot / (norm_ba * norm_bc))
    return round(math.degrees(angle), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 將畫面縮小 60%
    frame = cv2.resize(frame, None, fx=0.6, fy=0.6)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 取得右腿關鍵點：髖(24)、膝(26)、踝(28)
        landmarks = results.pose_landmarks.landmark
        hip = [landmarks[24].x * frame.shape[1], landmarks[24].y * frame.shape[0]]
        knee = [landmarks[26].x * frame.shape[1], landmarks[26].y * frame.shape[0]]
        ankle = [landmarks[28].x * frame.shape[1], landmarks[28].y * frame.shape[0]]

        angle = calculate_angle(hip, knee, ankle)
        cv2.putText(frame, f'Knee Angle: {angle}', tuple(map(int, knee)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Pose with Knee Angle", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
