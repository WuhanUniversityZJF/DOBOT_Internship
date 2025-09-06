#!/usr/bin/env python
# Copyright (c) 2024, WuChao D-Robotics.
# Licensed under the Apache License, Version 2.0.

import os
import cv2
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from hobot_dnn import pyeasy_dnn as dnn
import logging
from time import time
from scipy.special import softmax
import shutil

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] [%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO_EVAL")

# COCO关键点连接关系
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]


class KeypointVisualizer:
    """关键点可视化工具"""

    def __init__(self, output_dir="visualization"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 颜色配置
        self.colors = {
            'gt': (0, 255, 0),  # 绿色 - 真实关键点
            'pred': (0, 0, 255),  # 红色 - 预测关键点
            'line': (255, 0, 0)  # 蓝色 - 连接线
        }

    def draw_skeleton(self, img, kpts, color, thickness=2):
        """绘制关键点骨架"""
        for i, (start, end) in enumerate(COCO_SKELETON):
            start_idx, end_idx = start - 1, end - 1
            if (kpts[start_idx, 2] > 0 and kpts[end_idx, 2] > 0):
                start_pt = tuple(kpts[start_idx, :2].astype(int))
                end_pt = tuple(kpts[end_idx, :2].astype(int))
                cv2.line(img, start_pt, end_pt, color, thickness)

    def draw_keypoints(self, img, kpts, color, radius=5):
        """绘制关键点"""
        for i, kpt in enumerate(kpts):
            if kpt[2] > 0:  # 只绘制可见关键点
                center = tuple(kpt[:2].astype(int))
                cv2.circle(img, center, radius, color, -1)
                cv2.putText(img, str(i + 1), (center[0] + 5, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def draw_connections(self, img, pred_kpts, gt_kpts, thickness=1):
        """绘制预测与真实关键点的连接线"""
        for i in range(len(pred_kpts)):
            if pred_kpts[i, 2] > 0 and gt_kpts[i, 2] > 0:
                pred_pt = tuple(pred_kpts[i, :2].astype(int))
                gt_pt = tuple(gt_kpts[i, :2].astype(int))
                cv2.line(img, pred_pt, gt_pt, self.colors['line'], thickness)

    def visualize(self, img, pred_kpts, gt_kpts, img_name):
        """生成可视化结果并保存"""
        vis_img = img.copy()

        # 绘制骨架
        self.draw_skeleton(vis_img, gt_kpts, self.colors['gt'])
        self.draw_skeleton(vis_img, pred_kpts, self.colors['pred'])

        # 绘制关键点
        self.draw_keypoints(vis_img, gt_kpts, self.colors['gt'])
        self.draw_keypoints(vis_img, pred_kpts, self.colors['pred'])

        # 绘制连接线
        self.draw_connections(vis_img, pred_kpts, gt_kpts)

        # 保存结果
        output_path = os.path.join(self.output_dir, f"vis_{img_name}")
        cv2.imwrite(output_path, vis_img)
        return output_path


class YOLO11_Pose:
    def __init__(self, model_file: str, conf_thres: float = 0.25, iou_thres: float = 0.45):
        try:
            self.quantize_model = dnn.load(model_file)
            logger.info("Model loaded successfully")

            self.input_shape = self.quantize_model[0].inputs[0].properties.shape[2:]
            self.output_names = [out.name for out in self.quantize_model[0].outputs]

            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
            self.conf_inverse = -np.log(1 / conf_thres - 1)

            self.weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, np.newaxis, :]

            self.s_anchor = np.stack([
                np.tile(np.linspace(0.5, 79.5, 80), reps=80),
                np.repeat(np.arange(0.5, 80.5, 1), 80)
            ], axis=0).transpose(1, 0)

            self.m_anchor = np.stack([
                np.tile(np.linspace(0.5, 39.5, 40), reps=40),
                np.repeat(np.arange(0.5, 40.5, 1), 40)
            ], axis=0).transpose(1, 0)

            self.l_anchor = np.stack([
                np.tile(np.linspace(0.5, 19.5, 20), reps=20),
                np.repeat(np.arange(0.5, 20.5, 1), 20)
            ], axis=0).transpose(1, 0)

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def bgr2nv12(self, bgr_img: np.ndarray) -> np.ndarray:
        height, width = bgr_img.shape[:2]
        area = height * width
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        return nv12

    def forward(self, input_tensor: np.ndarray):
        return self.quantize_model[0].forward(input_tensor)

    def c2numpy(self, outputs):
        return [output.buffer for output in outputs]

    def postProcess(self, outputs: list[np.ndarray]) -> tuple:
        try:
            s_clses = outputs[0].reshape(-1, 1)
            s_bboxes = outputs[1].reshape(-1, 64)
            s_kpts = outputs[2].reshape(-1, 51)

            m_clses = outputs[3].reshape(-1, 1)
            m_bboxes = outputs[4].reshape(-1, 64)
            m_kpts = outputs[5].reshape(-1, 51)

            l_clses = outputs[6].reshape(-1, 1)
            l_bboxes = outputs[7].reshape(-1, 64)
            l_kpts = outputs[8].reshape(-1, 51)

            # Classify
            s_max_scores = np.max(s_clses, axis=1)
            s_valid_indices = np.flatnonzero(s_max_scores >= self.conf_inverse)
            s_ids = np.argmax(s_clses[s_valid_indices], axis=1)
            s_scores = s_max_scores[s_valid_indices]

            m_max_scores = np.max(m_clses, axis=1)
            m_valid_indices = np.flatnonzero(m_max_scores >= self.conf_inverse)
            m_ids = np.argmax(m_clses[m_valid_indices], axis=1)
            m_scores = m_max_scores[m_valid_indices]

            l_max_scores = np.max(l_clses, axis=1)
            l_valid_indices = np.flatnonzero(l_max_scores >= self.conf_inverse)
            l_ids = np.argmax(l_clses[l_valid_indices], axis=1)
            l_scores = l_max_scores[l_valid_indices]

            # Apply sigmoid
            s_scores = 1 / (1 + np.exp(-s_scores))
            m_scores = 1 / (1 + np.exp(-m_scores))
            l_scores = 1 / (1 + np.exp(-l_scores))

            # Dequantize
            s_bbox_scale = self.quantize_model[0].outputs[1].properties.scale_data.astype(np.float32)
            m_bbox_scale = self.quantize_model[0].outputs[4].properties.scale_data.astype(np.float32)
            l_bbox_scale = self.quantize_model[0].outputs[7].properties.scale_data.astype(np.float32)

            s_bboxes_float32 = s_bboxes[s_valid_indices].astype(np.float32) * s_bbox_scale
            m_bboxes_float32 = m_bboxes[m_valid_indices].astype(np.float32) * m_bbox_scale
            l_bboxes_float32 = l_bboxes[l_valid_indices].astype(np.float32) * l_bbox_scale

            # Process bboxes
            s_ltrb = np.sum(softmax(s_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
            s_anchor = self.s_anchor[s_valid_indices]
            s_x1y1 = s_anchor - s_ltrb[:, :2]
            s_x2y2 = s_anchor + s_ltrb[:, 2:]
            s_dbboxes = np.hstack([s_x1y1, s_x2y2]) * 8.0

            m_ltrb = np.sum(softmax(m_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
            m_anchor = self.m_anchor[m_valid_indices]
            m_x1y1 = m_anchor - m_ltrb[:, :2]
            m_x2y2 = m_anchor + m_ltrb[:, 2:]
            m_dbboxes = np.hstack([m_x1y1, m_x2y2]) * 16.0

            l_ltrb = np.sum(softmax(l_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
            l_anchor = self.l_anchor[l_valid_indices]
            l_x1y1 = l_anchor - l_ltrb[:, :2]
            l_x2y2 = l_anchor + l_ltrb[:, 2:]
            l_dbboxes = np.hstack([l_x1y1, l_x2y2]) * 32.0

            # Process keypoints
            s_kpts = s_kpts[s_valid_indices].reshape(-1, 17, 3)
            s_kpts_xy = (s_kpts[:, :, :2] * 2.0 + (self.s_anchor[s_valid_indices][:, np.newaxis] - 0.5)) * 8.0
            s_kpts_score = s_kpts[:, :, 2:3]

            m_kpts = m_kpts[m_valid_indices].reshape(-1, 17, 3)
            m_kpts_xy = (m_kpts[:, :, :2] * 2.0 + (self.m_anchor[m_valid_indices][:, np.newaxis] - 0.5)) * 16.0
            m_kpts_score = m_kpts[:, :, 2:3]

            l_kpts = l_kpts[l_valid_indices].reshape(-1, 17, 3)
            l_kpts_xy = (l_kpts[:, :, :2] * 2.0 + (self.l_anchor[l_valid_indices][:, np.newaxis] - 0.5)) * 32.0
            l_kpts_score = l_kpts[:, :, 2:3]

            # Concatenate
            dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0)
            scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
            ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)
            kpts_xy = np.concatenate((s_kpts_xy, m_kpts_xy, l_kpts_xy), axis=0)
            kpts_score = np.concatenate((s_kpts_score, m_kpts_score, l_kpts_score), axis=0)

            # NMS
            if len(dbboxes) > 0:
                indices = cv2.dnn.NMSBoxes(
                    dbboxes.tolist(),
                    scores.tolist(),
                    self.conf_thres,
                    self.iou_thres
                ).flatten()

                if len(indices) > 0:
                    return (
                        ids[indices],
                        scores[indices],
                        dbboxes[indices],
                        kpts_xy[indices],
                        kpts_score[indices]
                    )

            return [], [], [], [], []

        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            return [], [], [], [], []


class COCOEvaluator:
    def __init__(self, model_path, coco_ann_path, img_dir, num_images=100, kpt_threshold=0.5):
        self.model = YOLO11_Pose(model_path, conf_thres=0.1, iou_thres=0.3)
        self.kpt_threshold = kpt_threshold
        self.visualizer = KeypointVisualizer("visualization_results")

        if not os.path.exists(coco_ann_path):
            raise FileNotFoundError(f"COCO annotations not found: {coco_ann_path}")

        self.coco_gt = COCO(coco_ann_path)
        self.img_dir = img_dir
        cat_ids = self.coco_gt.getCatIds(['person'])
        self.img_ids = self.coco_gt.getImgIds(catIds=cat_ids)[:num_images]

        logger.info(f"Loaded {len(self.img_ids)} images for evaluation")

    def evaluate(self):
        results = []
        stats = {
            'total_kpts': 0,
            'correct_kpts': 0,
            'total_dets': 0,
            'correct_dets': 0
        }

        for img_id in tqdm(self.img_ids, desc="Evaluating"):
            img_info = self.coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])

            if not os.path.exists(img_path):
                logger.warning(f"Image not found: {img_path}")
                continue

            try:
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Failed to read image: {img_path}")
                    continue

                # Get ground truth
                ann_ids = self.coco_gt.getAnnIds(imgIds=img_id, catIds=[1])
                annotations = self.coco_gt.loadAnns(ann_ids)

                # Model inference
                input_tensor = self.model.bgr2nv12(img)
                outputs = self.model.c2numpy(self.model.forward(input_tensor))
                ids, scores, bboxes, kpts_xy, kpts_score = self.model.postProcess(outputs)

                # Process predictions
                pred_kpts_list = []
                for i in range(len(kpts_xy)):
                    kpt_xy = kpts_xy[i]
                    kpt_conf = kpts_score[i]
                    mask = kpt_conf[:, 0] > self.kpt_threshold
                    pred_kpts = np.concatenate([
                        kpt_xy,
                        mask.astype(np.float32).reshape(-1, 1)
                    ], axis=1)
                    pred_kpts_list.append(pred_kpts)

                # Evaluate each ground truth
                for ann in annotations:
                    if 'keypoints' not in ann:
                        continue

                    gt_kpts = np.array(ann['keypoints']).reshape(-1, 3)
                    gt_bbox = ann['bbox']

                    # Find best matching prediction
                    best_match = None
                    best_iou = 0
                    for i, box in enumerate(bboxes):
                        iou = self.calculate_iou(box, [
                            gt_bbox[0],
                            gt_bbox[1],
                            gt_bbox[0] + gt_bbox[2],
                            gt_bbox[1] + gt_bbox[3]
                        ])
                        if iou > best_iou:
                            best_iou = iou
                            best_match = i

                    # Visualize if we have a match
                    if best_match is not None and best_iou > 0.3:
                        pred_kpts = pred_kpts_list[best_match]
                        self.visualizer.visualize(
                            img,
                            pred_kpts,
                            gt_kpts,
                            os.path.basename(img_path)
                        )

                        # Count correct keypoints (within 10 pixels)
                        dist = np.linalg.norm(pred_kpts[:, :2] - gt_kpts[:, :2], axis=1)
                        visible = gt_kpts[:, 2] > 0
                        correct = (dist[visible] < 10).sum()

                        stats['correct_kpts'] += correct
                        stats['total_kpts'] += visible.sum()
                        stats['correct_dets'] += 1 if best_iou > 0.5 else 0

                    stats['total_dets'] += 1

                # Prepare COCO format results
                for i, (box, score) in enumerate(zip(bboxes, scores)):
                    kpt_xy = kpts_xy[i] if i < len(kpts_xy) else np.zeros((17, 2))
                    kpt_score = kpts_score[i] if i < len(kpts_score) else np.zeros(17)

                    kpt_coco = []
                    for j in range(17):
                        x, y = kpt_xy[j]
                        conf = 2 if kpt_score[j, 0] > self.model.conf_inverse else 0
                        kpt_coco.extend([float(x), float(y), conf])

                    results.append({
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                        "score": float(score),
                        "keypoints": kpt_coco,
                        "area": float((box[2] - box[0]) * (box[3] - box[1]))
                    })

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue

        # Print statistics
        if stats['total_kpts'] > 0:
            kpt_acc = stats['correct_kpts'] / stats['total_kpts']
            logger.info(f"Keypoint Accuracy (@10px): {kpt_acc:.4f}")

        if stats['total_dets'] > 0:
            det_acc = stats['correct_dets'] / stats['total_dets']
            logger.info(f"Detection Accuracy (IoU@0.5): {det_acc:.4f}")

        return results

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter_area / (box1_area + box2_area - inter_area)

    def run(self):
        results = self.evaluate()

        if not results:
            logger.error("No detections generated")
            return

        # Save predictions
        with open("predictions.json", "w") as f:
            json.dump(results, f)
            logger.info(f"Predictions saved to predictions.json")

        # COCO evaluation
        try:
            coco_dt = self.coco_gt.loadRes("predictions.json")
            coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints')
            coco_eval.params.imgIds = self.img_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        except Exception as e:
            logger.error(f"COCO evaluation failed: {e}")


if __name__ == "__main__":
    try:
        model_path = "/home/sunrise/Desktop/yolo11n_pose_bayese_640x640_nv12_modified.bin"
        coco_ann_path = "/home/sunrise/Desktop/person_keypoints_val2017.json"
        img_dir = "/home/sunrise/Desktop/val2017"

        # Clean previous results
        if os.path.exists("visualization_results"):
            shutil.rmtree("visualization_results")

        evaluator = COCOEvaluator(
            model_path=model_path,
            coco_ann_path=coco_ann_path,
            img_dir=img_dir,
            num_images=5000,
            kpt_threshold=0.3
        )
        evaluator.run()

    except Exception as e:
        logger.error(f"Runtime error: {e}")