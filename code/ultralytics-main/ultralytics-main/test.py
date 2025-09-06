import cv2
import os
import pandas as pd
import sys

root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_path)
from ultralytics import YOLO

# 加载模型
# model = YOLO('back_27points.pt')  # 加载旧权重
model = YOLO('C:\\Users\\Dobot\\Desktop\\ultralytics-main\\ultralytics-main\\runs\\train\\exp8\\weights\\best.pt')  # 新权重
# model = YOLO('runs/train/exp8/weights/best.pt')

# image_folder = 'E:/dataset/auto_label'
output_folder = "./predict-pose"
image_name = 'white_dog.jpg'
img_path = "C:\\Users\\Dobot\\Desktop\\for_test"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#img = cv2.imread(os.path.join(img_path, image_name))\
img =cv2.imread(os.path.join(img_path, image_name))

# 进行推理
results = model(img)[0]
print(f"Processing {image_name}")

# 保存推理结果到输出文件夹
output_path = os.path.join(output_folder, f"result_{image_name}")
results.save(output_path)  # 保存推理结果到输出文件夹v