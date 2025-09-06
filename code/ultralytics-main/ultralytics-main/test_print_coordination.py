import cv2
import os
import numpy as np
from ultralytics import YOLO

# 加载模型
model = YOLO('C:\\Users\\Dobot\\Desktop\\ultralytics-main\\ultralytics-main\\runs\\train\\exp7\\weights\\best.pt')

# 设置路径
output_folder = "./predict-pose"
image_name = 'man_front_squat.jpg'
img_path = "C:\\Users\\Dobot\\Desktop\\for_test"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取图像
img = cv2.imread(os.path.join(img_path, image_name))
if img is None:
    print(f"Error: Could not read image {image_name}")
    exit()

# 进行推理
results = model(img)[0]
print(f"\nProcessing {image_name}")

# 获取关键点数据
keypoints = results.keypoints.xy.cpu().numpy()  # 获取所有检测到的关键点 [N,12,2]
boxes = results.boxes.xyxy.cpu().numpy()  # 获取边界框 [N,4]

# 打印关键点坐标
print("\nDetected Keypoints Coordinates:")
for i, (box, kps) in enumerate(zip(boxes, keypoints)):
    print(f"\nPerson {i + 1}:")
    print("Bounding Box:", box)
    for j, (x, y) in enumerate(kps):
        print(f"Keypoint {j + 1}: x={x:.2f}, y={y:.2f}")

# 定义关键点连接线（根据你的12个关键点调整）
skeleton = [
    (0, 1), (1, 2), (2, 3),  # 示例连接关系，根据你的关键点实际含义修改
    (4, 5), (5, 6),
    (7, 8), (8, 9),
    (10, 11)
]

# 定义关键点颜色和连线颜色
kp_color = (0, 255, 0)  # 绿色关键点
line_color = (0, 0, 255)  # 红色连线
box_color = (255, 0, 0)  # 蓝色边界框

# 在图像上绘制结果
for box, kps in zip(boxes, keypoints):
    # 绘制边界框
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

    # 绘制关键点连线
    for start, end in skeleton:
        if start < len(kps) and end < len(kps):
            start_point = tuple(map(int, kps[start]))
            end_point = tuple(map(int, kps[end]))
            cv2.line(img, start_point, end_point, line_color, 2)

    # 绘制关键点
    for kp in kps:
        x, y = map(int, kp)
        cv2.circle(img, (x, y), 5, kp_color, -1)

# 保存结果
output_path = os.path.join(output_folder, f"result_{image_name}")
cv2.imwrite(output_path, img)

# 显示结果（可选）
cv2.imshow('Pose Estimation', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"\nResults saved to {output_path}")