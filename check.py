#使用mediapipe偵測train裡面的圖片

import cv2
import mediapipe as mp
import math

# 計算三點夾角的函式
def calculate_angle(a, b, c):
    """a, b, c 為 (x, y) 座標點，計算角度為 abc 的夾角（單位：度）"""
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])

    dot_product = ab[0] * cb[0] + ab[1] * cb[1]
    norm_ab = math.hypot(*ab)
    norm_cb = math.hypot(*cb)

    if norm_ab == 0 or norm_cb == 0:
        return None  # 防止除以0

    angle_rad = math.acos(dot_product / (norm_ab * norm_cb))
    return math.degrees(angle_rad)

# MediaPipe 初始化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# 測試圖片路徑（請換成你 Train 裡的某張圖片）
img_path = "D:\fitness\FitnessApp\train\yt-IODxDxX7oi4-0108_jpg.rf.0888418a8faba86b2a7204de977fb35d.jpg"

img = cv2.imread(img_path)
if img is None:
    print("❌ 圖片讀取失敗")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = pose.process(img_rgb)

if results.pose_landmarks:
    # 繪製姿勢點
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 取出 landmarks
    landmarks = results.pose_landmarks.landmark

    # 取得三個點來算角度（例如左手臂：肩膀、手肘、手腕）
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow    = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_wrist    = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

    # 轉為 2D 座標（相對於圖片尺寸）
    h, w = img.shape[:2]
    a = (left_shoulder.x * w, left_shoulder.y * h)
    b = (left_elbow.x * w, left_elbow.y * h)
    c = (left_wrist.x * w, left_wrist.y * h)

    angle = calculate_angle(a, b, c)
    print(f"📐 左手肘夾角：{angle:.2f} 度")

    # 顯示圖片
    cv2.imshow("Pose", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("⚠️ 沒有偵測到姿勢")
