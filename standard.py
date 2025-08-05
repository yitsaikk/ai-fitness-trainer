import cv2
import mediapipe as mp
import numpy as np
import math

#初始化
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    # 計算角度（單位：度）
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.7,
                  min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 轉成 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 偵測人體姿勢
        results = pose.process(image)

        # 回到 BGR 模式以供顯示
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 如果偵測到人體
        if results.pose_landmarks:
            # 繪製骨架
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 取得人體 landmark 座標
            landmarks = results.pose_landmarks.landmark

            # 取得右肩、右手肘、右手腕的位置
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # 計算手肘角度
            angle = calculate_angle(shoulder, elbow, wrist)

            # 顯示角度
            cv2.putText(image, str(int(angle)),
                        tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 顯示畫面
        cv2.imshow('Pose Angle Detection', image)
        cv2.imshow('Pose Angle Detection', image)

        # 按下 q 離開
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 結束攝影機
    cap.release()
    cv2.destroyAllWindows()



cap.release()
cv2.destroyAllWindows()