import cv2
import mediapipe as mp
import pandas as pd
import os
import csv

# 設定資料夾與檔名
csv_input = "D:/fitness/FitnessApp/train/_classes.csv"
image_folder = "D:/fitness/FitnessApp/train"

output_csv = f"{image_folder}_pose_keypoints.csv"

# 讀取標籤檔
print("👉 開始讀取 CSV 檔案...")

df = pd.read_csv(csv_input)  # tab 分隔
df.columns = df.columns.str.strip()          # ← 把欄位名稱前後空白清掉
df['label'] = df.iloc[:, 1:].idxmax(axis=1)  # 建立單欄標籤

print(df.head())
print("欄位名稱：", df.columns)

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# 建立輸出 CSV
header = []
for i in range(33):
    header += [f'{i}_x', f'{i}_y', f'{i}_z', f'{i}_visibility']
header.append('label')

with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for index, row in df.iterrows():
        filename = row['filename']
        label = row['label']
        img_path = os.path.join(image_folder, filename)

        if not os.path.exists(img_path):
            print(f"[警告] 找不到圖檔：{img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[錯誤] 圖片無法讀取：{img_path}")
            continue

        # 將圖片轉成 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            row_data = []
            for lm in results.pose_landmarks.landmark:
                row_data += [lm.x, lm.y, lm.z, lm.visibility]
            row_data.append(label)
            writer.writerow(row_data)
        else:
            print(f"[略過] 無偵測到姿勢：{filename}")

print(f"[完成] 關鍵點資料已輸出至 {output_csv}")
