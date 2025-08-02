import cv2
import numpy as np

import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='model/blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.IMAGE)

with FaceDetector.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()             # 讀取影片的每一幀
        w = frame.shape[1]
        h = frame.shape[0]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            lx = bbox.origin_x
            ly = bbox.origin_y
            width = bbox.width
            height = bbox.height
            cv2.rectangle(frame,(lx,ly),(lx+width,ly+height),(0,0,255),5)
            for keyPoint in detection.keypoints:
                print(keyPoint, w, h)
                cx = int(keyPoint.x*w)
                cy = int(keyPoint.y*h)
                print(cx, cy)
                cv2.circle(frame,(cx,cy),10,(0,0,255),-1) 
            print(bbox)
        if not ret:
            print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
            break
        cv2.imshow('oxxostudio', frame)     # 如果讀取成功，顯示該幀的畫面
        if cv2.waitKey(10) == ord('q'):      # 每一毫秒更新一次，直到按下 q 結束
            break
    cap.release()                           # 所有作業都完成後，釋放資源
    cv2.destroyAllWindows()                 # 結束所有視窗
    