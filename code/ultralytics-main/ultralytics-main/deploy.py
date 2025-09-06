#!/user/bin/env python
# Copyright (c) 2024，WuChao D-Robotics.
# 关键点检测模型板端部署代码（适配热图输出模型）

import cv2
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn  # RDK板端BPU接口
from time import time
import argparse
import logging

# -------------------------- 日志配置 --------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("Keypoint_Detection")

# -------------------------- 全局参数 --------------------------
# 关键点名称（根据你的24个关键点定义修改）
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head_top", "neck", "spine", "waist", "left_hand", "right_hand", "tail"
]

# 关键点颜色（24种颜色，用于可视化）
KEYPOINT_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 255, 128), (255, 128, 255),
    (128, 255, 255), (192, 192, 192), (255, 192, 128), (128, 192, 255)
]


# -------------------------- 基础模型类 --------------------------
class BaseModel:
    def __init__(self, model_file: str) -> None:
        """加载BPU量化模型，初始化输入输出参数"""
        try:
            begin_time = time()
            self.model = dnn.load(model_file)  # 加载bin模型
            logger.debug(f"Load model time: {1000 * (time() - begin_time):.2f} ms")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {model_file}")
            logger.error(e)
            exit(1)

        # 打印输入输出信息
        logger.info("\033[1;32m-> Input Tensors\033[0m")
        for i, inp in enumerate(self.model[0].inputs):
            self.input_shape = inp.properties.shape  # [1, 3, H, W]
            logger.info(f"Input[{i}]: name={inp.name}, shape={inp.properties.shape}, dtype={inp.properties.dtype}")
        self.input_h, self.input_w = self.input_shape[2], self.input_shape[3]  # 模型输入尺寸（H, W）

        logger.info("\033[1;32m-> Output Tensors\033[0m")
        for i, out in enumerate(self.model[0].outputs):
            self.output_shape = out.properties.shape  # [1, 24, H_heatmap, W_heatmap]
            logger.info(f"Output[{i}]: name={out.name}, shape={out.properties.shape}, dtype={out.properties.dtype}")
        self.heatmap_h, self.heatmap_w = self.output_shape[2], self.output_shape[3]  # 热图尺寸（H, W）

    def bgr2nv12(self, bgr_img: np.ndarray) -> np.ndarray:
        """BGR图像转NV12格式（模型输入要求）"""
        begin_time = time()
        # 缩放图像到模型输入尺寸
        img_resized = cv2.resize(bgr_img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        # BGR转YUV I420，再转换为NV12
        height, width = img_resized.shape[0], img_resized.shape[1]
        yuv420p = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YUV_I420).reshape((height * width * 3 // 2,))
        y = yuv420p[:height * width]
        uv_planar = yuv420p[height * width:].reshape((2, height // 2 * width // 2))
        uv_packed = uv_planar.transpose((1, 0)).reshape((height // 2 * width // 2 * 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        logger.debug(f"BGR to NV12 time: {1000 * (time() - begin_time):.2f} ms")
        return nv12

    def forward(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        """模型推理（输入NV12数据，输出热图）"""
        begin_time = time()
        outputs = self.model[0].forward(input_tensor)  # 推理
        # 转换为numpy数组（含反量化）
        outputs_np = [out.buffer.astype(np.float32) * out.properties.scale_data for out in outputs]
        logger.debug(f"Inference time: {1000 * (time() - begin_time):.2f} ms")
        return outputs_np


# -------------------------- 关键点检测类 --------------------------
class KeypointDetector(BaseModel):
    def __init__(self, model_file: str, vis_threshold: float = 0.1) -> None:
        """初始化关键点检测器"""
        super().__init__(model_file)
        self.vis_threshold = vis_threshold  # 可见性阈值（热图峰值>此值视为有效点）
        self.num_keypoints = self.output_shape[1]  # 关键点数量（24）

    def heatmap_to_keypoints(self, heatmap: np.ndarray, original_size: tuple) -> np.ndarray:
        """热图转换为原图关键点坐标（带预处理逆映射）"""
        original_h, original_w = original_size  # 原图尺寸
        keypoints = np.zeros((self.num_keypoints, 3), dtype=np.float32)  # (24, 3): (x, y, visibility)

        # 遍历每个关键点的热图
        for kpt_idx in range(self.num_keypoints):
            kpt_heatmap = heatmap[0, kpt_idx]  # 第kpt_idx个关键点的热图 (H_heatmap, W_heatmap)

            # 找热图峰值（排除边缘5%区域，减少噪声干扰）
            mask = np.zeros_like(kpt_heatmap)
            h, w = kpt_heatmap.shape
            mask[int(0.05 * h):int(0.95 * h), int(0.05 * w):int(0.95 * w)] = 1
            masked_heatmap = kpt_heatmap * mask
            peak_value = np.max(masked_heatmap)
            if peak_value < self.vis_threshold:
                keypoints[kpt_idx] = [0, 0, 0]  # 无效点：(0,0,0)
                continue

            # 峰值坐标 (h_peak, w_peak)
            peak_h, peak_w = np.unravel_index(np.argmax(masked_heatmap), masked_heatmap.shape)

            # 热图坐标 → 原图坐标（逆预处理映射）
            # 1. 热图归一化坐标 (0~1)
            norm_x = peak_w / (w - 1)  # 热图宽度方向
            norm_y = peak_h / (h - 1)  # 热图高度方向

            # 2. 模型输入图像坐标（输入尺寸：input_h x input_w）
            input_x = norm_x * self.input_w
            input_y = norm_y * self.input_h

            # 3. 原图坐标（输入图像→原图：逆resize）
            orig_x = input_x * (original_w / self.input_w)
            orig_y = input_y * (original_h / self.input_h)

            keypoints[kpt_idx] = [orig_x, orig_y, 1.0]  # 有效点：(x,y,1)

        return keypoints

    def postprocess(self, outputs: list[np.ndarray], original_size: tuple) -> np.ndarray:
        """后处理：从热图提取关键点坐标"""
        begin_time = time()
        heatmap = outputs[0]  # 热图输出（假设第一个输出是热图）
        keypoints = self.heatmap_to_keypoints(heatmap, original_size)
        logger.debug(f"Postprocess time: {1000 * (time() - begin_time):.2f} ms")
        return keypoints


# -------------------------- 可视化函数 --------------------------
def draw_keypoints(img: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """在图像上绘制关键点和连接关系"""
    img_copy = img.copy()
    h, w = img.shape[:2]

    # 绘制关键点
    for kpt_idx in range(keypoints.shape[0]):
        x, y, vis = keypoints[kpt_idx]
        if vis < 0.5:  # 无效点不绘制
            continue
        x = int(np.clip(x, 0, w - 1))  # 边界裁剪
        y = int(np.clip(y, 0, h - 1))
        color = KEYPOINT_COLORS[kpt_idx % len(KEYPOINT_COLORS)]
        # 绘制关键点圆圈
        cv2.circle(img_copy, (x, y), 6, color, -1)  # 实心圆
        cv2.circle(img_copy, (x, y), 8, (255, 255, 255), 2)  # 白色边框

        # 绘制关键点名称（可选）
        name = KEYPOINT_NAMES[kpt_idx] if kpt_idx < len(KEYPOINT_NAMES) else f"kpt{kpt_idx}"
        cv2.putText(img_copy, name[:3], (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 绘制骨骼连接（根据你的关键点连接定义修改）
    skeleton_connections = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 肩-肘-腕
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),  # 肩-髋-膝-踝
        (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)  # 面部关键点
    ]
    for (k1, k2) in skeleton_connections:
        x1, y1, v1 = keypoints[k1]
        x2, y2, v2 = keypoints[k2]
        if v1 < 0.5 or v2 < 0.5:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        color = KEYPOINT_COLORS[k1 % len(KEYPOINT_COLORS)]
        cv2.line(img_copy, (x1, y1), (x2, y2), color, 3)

    return img_copy


# -------------------------- 主函数 --------------------------
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help="BPU量化模型路径（*.bin）")
    parser.add_argument('--test-img', type=str, default='test.jpg', help="测试图像路径")
    parser.add_argument('--save-path', type=str, default='img_onnx.jpg', help="结果保存路径")
    parser.add_argument('--vis-threshold', type=float, default=0.1, help="关键点可见性阈值（0~1）")
    opt = parser.parse_args()
    logger.info(opt)

    # 1. 加载模型
    detector = KeypointDetector(opt.model_path, vis_threshold=opt.vis_threshold)

    # 2. 读取图像并获取原图尺寸
    img = cv2.imread(opt.test_img)
    if img is None:
        logger.error(f"❌ 无法读取图像: {opt.test_img}")
        exit(1)
    original_h, original_w = img.shape[:2]
    logger.info(f"原图尺寸: (H={original_h}, W={original_w})")

    # 3. 图像预处理（BGR转NV12）
    input_nv12 = detector.bgr2nv12(img)

    # 4. 模型推理（输出热图）
    outputs = detector.forward(input_nv12)

    # 5. 后处理（热图转关键点坐标）
    keypoints = detector.postprocess(outputs, (original_h, original_w))

    # 6. 可视化结果
    result_img = draw_keypoints(img, keypoints)

    # 7. 保存结果
    cv2.imwrite(opt.save_path, result_img)
    logger.info(f"\033[1;32m结果已保存至: {opt.save_path}\033[0m")

    # 8. 打印关键点坐标
    logger.info("\033[1;32m关键点坐标（x, y, visibility）:\033[0m")
    for kpt_idx in range(keypoints.shape[0]):
        x, y, vis = keypoints[kpt_idx]
        if vis > 0.5:
            logger.info(f"{KEYPOINT_NAMES[kpt_idx]:<12} (x={x:.1f}, y={y:.1f})")


if __name__ == "__main__":
    main()