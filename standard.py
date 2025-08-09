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
out_txt = "knee_angles.txt"
scale = 0.6
frame_idx = 0 #幀數編號 預設為0，用來影片當前處理到的第幾幀

# 計算角度的函式
def calculate_angle(a, b, c):
    ba = [a[0]-b[0], a[1]-b[1]]
    bc = [c[0]-b[0], c[1]-b[1]]
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    norm_ba = math.hypot(*ba)
    norm_bc = math.hypot(*bc)
    angle = math.acos(dot / (norm_ba * norm_bc))
    return round(math.degrees(angle), 2)

if not cap.isOpened():
    print("無法開啟影片")

#影片幀數擷取 或 預設為30
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

with open(out_txt, "w",encoding="utf-8") as f:
    f.write("# frame\tseconds\tright_knee\tleft_knee\n")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 將畫面縮小 60%
    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2] #取得影像的高度跟寬度
    results = pose.process(img_rgb)
    r_knee = l_knee =None #右膝 左膝角度預設0

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # 取得右腿關鍵點：髖(24)、膝(26)、踝(28)
        landmarks = results.pose_landmarks.landmark

        def pt(i) :return(landmarks[i].x*w, landmarks[i].y*h)

        r_hip, r_knee, r_ankle = pt(24), pt(26), pt(28)
        l_hip, l_knee, l_ankle = pt(23), pt(25), pt(27)
        r_knee = calculate_angle(r_hip, r_knee, r_ankle)
        l_knee = calculate_angle(l_hip, l_knee, l_ankle)
        
        seconds = round(frame_idx / fps, 3)
        with open(out_txt, "a", encoding="utf-8") as f:
            f.write(f"{frame_idx}\t{seconds}\t{r_knee if r_knee is not None else 'NA'}\t{l_knee if l_knee is not None else 'NA'}\n")



       #cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Pose with Knee Angle", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
