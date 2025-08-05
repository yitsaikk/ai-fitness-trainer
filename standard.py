import cv2
import mediapipe as mp
import math

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 影片路徑
video_path = "https://rr4---sn-vgqsrnld.googlevideo.com/videoplayback?expire=1754396991&ei=36SRaPXyLYjbp-oPzNPbmQw&ip=89.117.115.104&id=o-AO13Mwuv9Lct_vER9M4ieVytiHKgqj4SCChO1XRYWPDk&itag=137&aitags=133%2C134%2C135%2C136%2C137%2C160%2C242%2C243%2C244%2C247%2C248%2C278%2C394%2C395%2C396%2C397%2C398%2C399&source=youtube&requiressl=yes&xpc=EgVo2aDSNQ%3D%3D&siu=1&bui=AY1jyLO-SNKiEssiQpQ1Zqgvr1QQ2L48vCXOKaySwev5GGpTZXw-omS6bKJzxy_rWD1Yo6dT9Q&spc=l3OVKYe7KIRF&vprv=1&svpuc=1&mime=video%2Fmp4&ns=AJB2RquilrQWqe5gJJiSGXoQ&rqh=1&gir=yes&clen=2206156&dur=60.059&lmt=1748065061554489&keepalive=yes&c=TVHTML5_SIMPLY_EMBEDDED_PLAYER&sefc=1&txp=4432534&n=cbBIfMrOL63UrA&sparams=expire%2Cei%2Cip%2Cid%2Caitags%2Csource%2Crequiressl%2Cxpc%2Csiu%2Cbui%2Cspc%2Cvprv%2Csvpuc%2Cmime%2Cns%2Crqh%2Cgir%2Cclen%2Cdur%2Clmt&sig=AJfQdSswRgIhAMBZrEyJGpr-zzGgLYnKQ0UeoSWSyLWCxXHldNgC31pPAiEA8h4s23_HF8MmBzWaTP6h5i2NOBDTk8rWZBKYARvExMg%3D&title=The+Air+Squat&redirect_counter=1&cm2rm=sn-4g5erl7e&rrc=80&fexp=24350590,24350737,24350827,24351316,24351318,24351528,24352220,24352460,24352468,24352517,24352519,24352559,24352568,24352573,24352961&req_id=a7235920e7eca3ee&cms_redirect=yes&cmsv=e&met=1754375401,&mh=w2&mip=36.224.98.232&mm=34&mn=sn-vgqsrnld&ms=ltu&mt=1754373598&mv=D&mvi=4&pl=0&rms=ltu,au&lsparams=met,mh,mip,mm,mn,ms,mv,mvi,pl,rms&lsig=APaTxxMwRAIgQ89Us18IkSlI8v8aOWtzLXfMRWRJ627k3GCoT15ckasCIAwKMklOFhu4WGuN6eX8YVx8_ypqNfWGtBnOv3-Qhm_2"  # ← 改成你的影片檔名
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
