# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# YOLOv11-Pose 可视化脚本
# 在 COCO val2017 上推理并把结果画到图片
# 结果保存在 ./onnx_vis_results/
# """
#
# import os
# import cv2
# import json
# import numpy as np
# from pathlib import Path
# from pycocotools.coco import COCO
# from tqdm import tqdm
# import onnxruntime as ort
#
#
# class YOLOv11PoseDetector:
#     def __init__(self, model_path, conf_thres=0.4, iou_thres=0.5, kpt_thres=0.3):
#         self.session = ort.InferenceSession(model_path)
#         self.input_name = self.session.get_inputs()[0].name
#         self.output_name = self.session.get_outputs()[0].name
#         self.conf_thres = conf_thres
#         self.iou_thres = iou_thres
#         self.kpt_thres = kpt_thres
#         # COCO 17 keypoints skeleton
#         self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
#                          [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
#                          [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
#
#     # ------------ 前处理 ------------
#     def preprocess(self, img):
#         self.orig_h, self.orig_w = img.shape[:2]
#         scale = min(640 / self.orig_w, 640 / self.orig_h)
#         new_w, new_h = int(self.orig_w * scale), int(self.orig_h * scale)
#         resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#         self.pad_x, self.pad_y = (640 - new_w) // 2, (640 - new_h) // 2
#         padded = np.full((640, 640, 3), 114, dtype=np.uint8)
#         padded[self.pad_y:self.pad_y + new_h, self.pad_x:self.pad_x + new_w] = resized
#         self.scale_factor = scale
#         return (padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
#
#     # ------------ 后处理 ------------
#     def decode_prediction(self, pred):
#         pred = pred.transpose(1, 0)
#         valid_mask = pred[:, 4] > self.conf_thres
#         pred = pred[valid_mask]
#         if len(pred) == 0:
#             return np.empty((0, 4)), np.empty((0,)), np.empty((0, 17, 3))
#         boxes = pred[:, :4].copy()
#         # xywh -> xyxy
#         boxes[:, 0] -= boxes[:, 2] / 2
#         boxes[:, 1] -= boxes[:, 3] / 2
#         boxes[:, 2] += boxes[:, 0]
#         boxes[:, 3] += boxes[:, 1]
#         return boxes, pred[:, 4], pred[:, 5:].reshape(-1, 17, 3)
#
#     def apply_nms(self, boxes, scores, kpts):
#         if len(boxes) == 0:
#             return boxes, scores, kpts
#         indices = cv2.dnn.NMSBoxes(
#             boxes.astype(np.int32).tolist(),
#             scores.tolist(),
#             self.conf_thres,
#             self.iou_thres
#         ).flatten()
#         return boxes[indices], scores[indices], kpts[indices]
#
#     def scale_coords(self, coords):
#         coords[:, [0, 2]] = (coords[:, [0, 2]] - self.pad_x) / self.scale_factor
#         coords[:, [1, 3]] = (coords[:, [1, 3]] - self.pad_y) / self.scale_factor
#         return coords
#
#     def scale_keypoints(self, kpts):
#         kpts[..., 0] = (kpts[..., 0] - self.pad_x) / self.scale_factor
#         kpts[..., 1] = (kpts[..., 1] - self.pad_y) / self.scale_factor
#         return kpts
#
#
# # -------------------------------------------------
# # 可视化函数
# # -------------------------------------------------
# def draw_pose(img, box, kpts, skeleton, kpt_thres=0.3):
#     x1, y1, x2, y2 = box.astype(int)
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     # 画点
#     for x, y, conf in kpts:
#         if conf > kpt_thres:
#             cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
#     # 画骨架
#     for sk in skeleton:
#         x1_kpt, y1_kpt, c1 = kpts[sk[0] - 1]
#         x2_kpt, y2_kpt, c2 = kpts[sk[1] - 1]
#         if c1 > kpt_thres and c2 > kpt_thres:
#             cv2.line(img, (int(x1_kpt), int(y1_kpt)),
#                      (int(x2_kpt), int(y2_kpt)), (255, 0, 0), 2)
#     return img
#
#
# # -------------------------------------------------
# # 主流程
# # -------------------------------------------------
# def main():
#     img_dir = Path("val2017")
#     out_dir = Path("onnx_vis_results")
#     out_dir.mkdir(exist_ok=True)
#
#     detector = YOLOv11PoseDetector(
#         model_path="yolo11n-pose.onnx",
#         conf_thres=0.5,
#         iou_thres=0.45,
#         kpt_thres=0.3
#     )
#
#     coco = COCO("annotations/person_keypoints_val2017.json")
#     # 可视化前 N 张图，可改成全部
#     img_ids = coco.getImgIds()[:300]
#
#     for img_id in tqdm(img_ids, desc="Visualizing"):
#         img_info = coco.loadImgs(img_id)[0]
#         img_path = img_dir / img_info["file_name"]
#         img = cv2.imread(str(img_path))
#         if img is None:
#             continue
#
#         # 推理
#         tensor = detector.preprocess(img)
#         pred = detector.session.run(
#             [detector.output_name],
#             {detector.input_name: tensor}
#         )[0][0]
#
#         boxes, scores, kpts = detector.decode_prediction(pred)
#         boxes, scores, kpts = detector.apply_nms(boxes, scores, kpts)
#         if len(boxes) == 0:
#             continue
#
#         boxes = detector.scale_coords(boxes)
#         kpts = detector.scale_keypoints(kpts)
#
#         # 逐人绘制
#         vis = img.copy()
#         for box, score, kpt in zip(boxes, scores, kpts):
#             vis = draw_pose(vis, box, kpt, detector.skeleton, detector.kpt_thres)
#
#         save_path = out_dir / img_info["file_name"]
#         cv2.imwrite(str(save_path), vis)
#
#     print(f"可视化完成，结果保存在: {out_dir.resolve()}")
#
#
# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11-Pose 可视化（预测 vs 真实）
蓝色：预测
绿色：真实
依赖：
pip install opencv-python pycocotools tqdm onnxruntime
"""

import os
import cv2
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from tqdm import tqdm
import onnxruntime as ort


# ====================== 工具函数 ======================
def compute_iou(box1, box2):
    """计算两个 xyxy 框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union else 0


# ====================== 检测器 ======================
class YOLOv11PoseDetector:
    def __init__(self, model_path, conf_thres=0.4, iou_thres=0.5, kpt_thres=0.3):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.kpt_thres = kpt_thres
        # COCO 17 关键点骨架
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                         [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                         [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    # ------------ 前处理 ------------
    def preprocess(self, img):
        self.orig_h, self.orig_w = img.shape[:2]
        scale = min(640 / self.orig_w, 640 / self.orig_h)
        new_w, new_h = int(self.orig_w * scale), int(self.orig_h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        self.pad_x, self.pad_y = (640 - new_w) // 2, (640 - new_h) // 2
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        padded[self.pad_y:self.pad_y + new_h, self.pad_x:self.pad_x + new_w] = resized
        self.scale_factor = scale
        return (padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    # ------------ 后处理 ------------
    def decode_prediction(self, pred):
        pred = pred.transpose(1, 0)
        valid_mask = pred[:, 4] > self.conf_thres
        pred = pred[valid_mask]
        if len(pred) == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0, 17, 3))
        boxes = pred[:, :4].copy()
        # xywh -> xyxy
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        return boxes, pred[:, 4], pred[:, 5:].reshape(-1, 17, 3)

    def apply_nms(self, boxes, scores, kpts):
        if len(boxes) == 0:
            return boxes, scores, kpts
        indices = cv2.dnn.NMSBoxes(
            boxes.astype(np.int32).tolist(),
            scores.tolist(),
            self.conf_thres,
            self.iou_thres
        ).flatten()
        return boxes[indices], scores[indices], kpts[indices]

    def scale_coords(self, coords):
        coords[:, [0, 2]] = (coords[:, [0, 2]] - self.pad_x) / self.scale_factor
        coords[:, [1, 3]] = (coords[:, [1, 3]] - self.pad_y) / self.scale_factor
        return coords

    def scale_keypoints(self, kpts):
        kpts[..., 0] = (kpts[..., 0] - self.pad_x) / self.scale_factor
        kpts[..., 1] = (kpts[..., 1] - self.pad_y) / self.scale_factor
        return kpts


# ====================== 可视化 ======================
def draw_pose_with_gt(img, pred_box, pred_kpts, gt_kpts,
                      skeleton, kpt_thres=0.3):
    """
    同时绘制预测（蓝）和真实（绿）关键点
    pred_box: (4,) xyxy
    pred_kpts: (17,3)  x,y,conf
    gt_kpts:   (17,3)  x,y,v
    """
    # 1) 预测框
    x1, y1, x2, y2 = pred_box.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 2) 预测点 & 骨架（蓝色）
    for x, y, conf in pred_kpts:
        if conf > kpt_thres:
            cv2.circle(img, (int(x), int(y)), 4, (255, 0, 0), -1)
    for sk in skeleton:
        x1_p, y1_p, c1 = pred_kpts[sk[0] - 1]
        x2_p, y2_p, c2 = pred_kpts[sk[1] - 1]
        if c1 > kpt_thres and c2 > kpt_thres:
            cv2.line(img, (int(x1_p), int(y1_p)),
                     (int(x2_p), int(y2_p)), (255, 0, 0), 2)

    # 3) 真实点 & 骨架（绿色）
    for x, y, v in gt_kpts:
        if v > 0:
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
    for sk in skeleton:
        x1_g, y1_g, v1 = gt_kpts[sk[0] - 1]
        x2_g, y2_g, v2 = gt_kpts[sk[1] - 1]
        if v1 > 0 and v2 > 0:
            cv2.line(img, (int(x1_g), int(y1_g)),
                     (int(x2_g), int(y2_g)), (0, 255, 0), 2)
    return img


# ====================== 主程序 ======================
def main():
    img_dir = Path("val2017")
    out_dir = Path("onnx_vis_results")
    out_dir.mkdir(exist_ok=True)

    detector = YOLOv11PoseDetector(
        model_path="yolo11n-pose.onnx",
        conf_thres=0.5,
        iou_thres=0.45,
        kpt_thres=0.3
    )

    coco = COCO("annotations/person_keypoints_val2017.json")
    img_ids = coco.getImgIds()[:300]  # 可视化的图片数量

    for img_id in tqdm(img_ids, desc="Visualizing"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = img_dir / img_info["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 推理
        tensor = detector.preprocess(img)
        pred = detector.session.run(
            [detector.output_name],
            {detector.input_name: tensor}
        )[0][0]

        boxes, scores, kpts_pred = detector.decode_prediction(pred)
        boxes, scores, kpts_pred = detector.apply_nms(boxes, scores, kpts_pred)
        boxes = detector.scale_coords(boxes)
        kpts_pred = detector.scale_keypoints(kpts_pred)

        # 读取所有真实标注
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=coco.getCatIds(['person']), iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        vis = img.copy()
        for box, score, kpt_pred in zip(boxes, scores, kpts_pred):
            # 简单匹配：选最大 IoU 的 GT
            best_iou, best_gt = 0, None
            pred_xyxy = box
            for ann in anns:
                gt_xywh = np.array(ann['bbox'])
                gt_xyxy = np.array([gt_xywh[0], gt_xywh[1],
                                    gt_xywh[0] + gt_xywh[2],
                                    gt_xywh[1] + gt_xywh[3]])
                iou = compute_iou(pred_xyxy, gt_xyxy)
                if iou > best_iou:
                    best_iou, best_gt = iou, ann
            if best_gt is None:
                continue

            gt_keypoints = np.array(best_gt['keypoints']).reshape(-1, 3)
            vis = draw_pose_with_gt(vis, box, kpt_pred, gt_keypoints,
                                    detector.skeleton, detector.kpt_thres)

        save_path = out_dir / img_info["file_name"]
        cv2.imwrite(str(save_path), vis)

    print(f"可视化完成，结果保存在: {out_dir.resolve()}")


if __name__ == "__main__":
    main()