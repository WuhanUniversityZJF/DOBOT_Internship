import cv2
import numpy as np
import onnxruntime as ort


class YOLOv11PoseDetector:
    def __init__(self, model_path, conf_thres=0.4, iou_thres=0.5, kpt_thres=0.3):
        """初始化YOLOv11姿态检测器
        Args:
            model_path: ONNX模型路径
            conf_thres: 置信度阈值（过滤低置信度检测）
            iou_thres: NMS的IOU阈值（去重重叠框）
            kpt_thres: 关键点置信度阈值（过滤低置信度关键点）
        """
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 超参数设置（关键！解决重叠检测问题）
        self.conf_thres = conf_thres  # 建议值：0.4~0.6
        self.iou_thres = iou_thres  # 建议值：0.4~0.5
        self.kpt_thres = kpt_thres

        # 人体骨架连接定义
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                         [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                         [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    def preprocess(self, img):
        """图像预处理：保持长宽比+填充+归一化"""
        self.orig_h, self.orig_w = img.shape[:2]

        # 计算缩放比例（确保长边不超过640）
        scale = min(640 / self.orig_w, 640 / self.orig_h)
        new_w, new_h = int(round(self.orig_w * scale)), int(round(self.orig_h * scale))

        # 缩放图像
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 计算填充（左右/上下对称填充）
        self.pad_x = (640 - new_w) // 2
        self.pad_y = (640 - new_h) // 2
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)  # 灰底填充
        padded[self.pad_y: self.pad_y + new_h, self.pad_x: self.pad_x + new_w] = resized

        # 记录缩放参数（用于后续坐标转换）
        self.scale_factor = scale

        # 转换为模型输入格式 (NCHW + 归一化到0-1)
        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)[None]  # (1, 3, 640, 640)
        return input_tensor

    def decode_prediction(self, pred):
        """解码模型输出：从原始预测中提取框、分数和关键点"""
        # 模型输出格式: (1, 56, 8400) -> 转置为 (8400, 56)
        pred = pred.transpose(1, 0)  # (8400, 56)

        # 1. 过滤低置信度检测
        valid_mask = pred[:, 4] > self.conf_thres
        pred = pred[valid_mask]
        if len(pred) == 0:
            return [], [], []

        # 2. 提取框坐标 (中心x, 中心y, 宽, 高)
        boxes = pred[:, :4].copy()
        # 转换为 (x1, y1, x2, y2) 格式（左上角+右下角）
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2

        # 3. 提取置信度和关键点
        scores = pred[:, 4]
        kpts = pred[:, 5:].reshape(-1, 17, 3)  # (N, 17, 3) 其中3: (x, y, conf)

        return boxes, scores, kpts

    def apply_nms(self, boxes, scores, kpts):
        """非极大值抑制（NMS）：去除重叠检测框"""
        if len(boxes) == 0:
            return [], [], []

        # 转换框格式为整数（NMS需要）
        boxes_int = boxes.astype(np.int32)

        # OpenCV NMS需要的输入格式：boxes (x1,y1,x2,y2) + scores
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_int.tolist(),
            scores=scores.tolist(),
            score_threshold=self.conf_thres,
            nms_threshold=self.iou_thres
        )

        # 提取NMS后的结果
        if len(indices) == 0:
            return [], [], []

        indices = indices.flatten()  # 展平索引
        boxes = boxes[indices]
        scores = scores[indices]
        kpts = kpts[indices]

        return boxes, scores, kpts

    def scale_coords(self, coords):
        """将坐标从640x640图像转换回原图尺寸"""
        # 减去填充 + 除以缩放因子
        coords[:, [0, 2]] = (coords[:, [0, 2]] - self.pad_x) / self.scale_factor  # x坐标（x1, x2）
        coords[:, [1, 3]] = (coords[:, [1, 3]] - self.pad_y) / self.scale_factor  # y坐标（y1, y2）
        return coords

    def scale_keypoints(self, kpts):
        """将关键点从640x640图像转换回原图尺寸"""
        kpts[..., 0] = (kpts[..., 0] - self.pad_x) / self.scale_factor  # x坐标
        kpts[..., 1] = (kpts[..., 1] - self.pad_y) / self.scale_factor  # y坐标
        return kpts

    def draw_results(self, img, boxes, scores, kpts):
        """绘制检测结果：框+关键点+骨架"""
        for i, (box, score, kpt) in enumerate(zip(boxes, scores, kpts)):
            # 1. 绘制检测框（绿色）
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 2. 绘制置信度标签
            label = f"Person {i + 1}: {score:.2f}"
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 3. 绘制关键点（红色）
            for j in range(17):
                x, y, conf = kpt[j]
                if conf > self.kpt_thres:
                    cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)  # 实心圆

            # 4. 绘制骨架连线（青色）
            for (p1, p2) in self.skeleton:
                # 关键点索引从1开始，需减1
                x1_kp, y1_kp, conf1 = kpt[p1 - 1]
                x2_kp, y2_kp, conf2 = kpt[p2 - 1]
                if conf1 > self.kpt_thres and conf2 > self.kpt_thres:
                    cv2.line(img, (int(x1_kp), int(y1_kp)), (int(x2_kp), int(y2_kp)),
                             (255, 255, 0), 2)  # 青色连线
        return img

    def detect(self, img_path):
        """完整检测流程：预处理→推理→后处理→可视化"""
        # 1. 读取图像
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法加载图像: {img_path}")

        # 2. 预处理
        input_tensor = self.preprocess(img)

        # 3. 模型推理
        pred = self.session.run([self.output_name], {self.input_name: input_tensor})[0][0]  # (56, 8400)

        # 4. 解码预测结果
        boxes, scores, kpts = self.decode_prediction(pred)
        if len(boxes) == 0:
            print("未检测到人体")
            return img

        # 5. NMS去重（关键！解决重叠检测问题）
        boxes, scores, kpts = self.apply_nms(boxes, scores, kpts)
        if len(boxes) == 0:
            print("NMS后未保留检测结果")
            return img

        # 6. 坐标转换（从640x640映射回原图）
        boxes = self.scale_coords(boxes)
        kpts = self.scale_keypoints(kpts)

        # 7. 绘制结果
        result_img = self.draw_results(img.copy(), boxes, scores, kpts)

        # 8. 保存并显示结果
        cv2.imwrite("img_onnx.jpg", result_img)
        print(f"检测完成，结果已保存到 result.jpg (共检测到 {len(boxes)} 人)")

        # 显示结果（按任意键关闭窗口）
        cv2.imshow("YOLOv11 Pose Detection", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return result_img


if __name__ == "__main__":
    # 初始化检测器（调整阈值解决重叠问题）
    detector = YOLOv11PoseDetector(
        model_path="yolo11n-pose.onnx",
        conf_thres=0.5,  # 提高置信度阈值（如0.5）减少低质量检测
        iou_thres=0.4  # 降低IOU阈值（如0.4）增强去重效果
    )

    # 检测单张图片（替换为你的图片路径）
    detector.detect("img_onnx.png")