import cv2
import numpy as np

# 1. 读取 .jpeg 图像（假设尺寸 640×640，RGB 三通道）
img = cv2.imread("2dogs.jpeg")  # 读取为 BGR 格式（OpenCV 默认）
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为 RGB（与模型训练时一致）

# 2. 预处理（与模型训练时的预处理完全一致！）
img = cv2.resize(img, (640, 640))  # 调整尺寸
img = img / 255.0  # 归一化到 [0,1]（根据模型需求，可能还有 mean/std 减均值除方差）
img = img.transpose(2, 0, 1)  # 转置为 CHW 格式（模型输入通常是 [batch, channel, height, width]）
img = np.expand_dims(img, axis=0).astype(np.float32)  # 增加 batch 维度（1×3×640×640）

# 3. 保存为 .npy 文件
np.save("2dogs.npy", img)