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

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] [%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO_EVAL")


class YOLO11_Pose:
    def __init__(self, model_file: str, conf_thres: float = 0.25, iou_thres: float = 0.45):
        """Initialize YOLOv11 Pose Estimator

        Args:
            model_file: Path to quantized .bin model
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
        """
        try:
            # Load model
            self.quantize_model = dnn.load(model_file)
            logger.info("Model loaded successfully")

            # Get model input/output info
            self.input_shape = self.quantize_model[0].inputs[0].properties.shape[2:]  # (H, W)
            self.output_names = [out.name for out in self.quantize_model[0].outputs]

            # Initialize parameters
            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
            self.conf_inverse = -np.log(1 / conf_thres - 1)  # Sigmoid inverse

            # DFL coefficients
            self.weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, np.newaxis, :]

            # Anchors initialization
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

            logger.info(f"Input shape: {self.input_shape}")
            logger.info(f"Output layers: {self.output_names}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def bgr2nv12(self, bgr_img: np.ndarray) -> np.ndarray:
        """Convert BGR image to NV12 format"""
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
        """Perform inference"""
        return self.quantize_model[0].forward(input_tensor)

    def c2numpy(self, outputs):
        """Convert outputs to numpy arrays"""
        return [output.buffer for output in outputs]

    def postProcess(self, outputs: list[np.ndarray]) -> tuple:
        """Post-process detection results

        Returns:
            tuple: (ids, scores, bboxes, kpts_xy, kpts_score)
        """
        try:
            # Reorder outputs
            s_clses = outputs[0].reshape(-1, 1)  # (80*80, 1)
            s_bboxes = outputs[1].reshape(-1, 64)  # (80*80, 64)
            s_kpts = outputs[2].reshape(-1, 51)  # (80*80, 51)

            m_clses = outputs[3].reshape(-1, 1)
            m_bboxes = outputs[4].reshape(-1, 64)
            m_kpts = outputs[5].reshape(-1, 51)

            l_clses = outputs[6].reshape(-1, 1)
            l_bboxes = outputs[7].reshape(-1, 64)
            l_kpts = outputs[8].reshape(-1, 51)

            # Classify: Filter by confidence
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

            # Apply sigmoid to scores
            s_scores = 1 / (1 + np.exp(-s_scores))
            m_scores = 1 / (1 + np.exp(-m_scores))
            l_scores = 1 / (1 + np.exp(-l_scores))

            # Get scale data for dequantization
            s_bbox_scale = self.quantize_model[0].outputs[1].properties.scale_data.astype(np.float32)
            m_bbox_scale = self.quantize_model[0].outputs[4].properties.scale_data.astype(np.float32)
            l_bbox_scale = self.quantize_model[0].outputs[7].properties.scale_data.astype(np.float32)

            # Dequantize bboxes
            s_bboxes_float32 = s_bboxes[s_valid_indices].astype(np.float32) * s_bbox_scale
            m_bboxes_float32 = m_bboxes[m_valid_indices].astype(np.float32) * m_bbox_scale
            l_bboxes_float32 = l_bboxes[l_valid_indices].astype(np.float32) * l_bbox_scale

            # Process bboxes (dist to bbox)
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

            # Concatenate results from all scales
            dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0)
            scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
            ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)
            kpts_xy = np.concatenate((s_kpts_xy, m_kpts_xy, l_kpts_xy), axis=0)
            kpts_score = np.concatenate((s_kpts_score, m_kpts_score, l_kpts_score), axis=0)

            # Apply NMS
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
    def __init__(self, model_path, coco_ann_path, img_dir, num_images=100):
        """COCO Keypoints Evaluator

        Args:
            model_path: Path to quantized .bin model
            coco_ann_path: Path to COCO annotations
            img_dir: Directory containing COCO images
            num_images: Number of images to evaluate
        """
        # Initialize model with lower thresholds
        self.model = YOLO11_Pose(model_path, conf_thres=0.1, iou_thres=0.3)

        # Validate paths
        if not os.path.exists(coco_ann_path):
            raise FileNotFoundError(f"COCO annotations not found: {coco_ann_path}")

        self.coco_gt = COCO(coco_ann_path)
        self.img_dir = img_dir

        # Get image IDs containing persons
        cat_ids = self.coco_gt.getCatIds(['person'])
        self.img_ids = self.coco_gt.getImgIds(catIds=cat_ids)[:num_images]

        logger.info(f"Loaded {len(self.img_ids)} images for evaluation")

    def evaluate(self):
        """Run evaluation on COCO dataset"""
        results = []
        no_detections = 0
        invalid_images = 0

        for img_id in tqdm(self.img_ids, desc="Evaluating"):
            img_info = self.coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])

            # Validate image path
            if not os.path.exists(img_path):
                logger.warning(f"Image not found: {img_path}")
                invalid_images += 1
                continue

            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Failed to read image: {img_path}")
                    invalid_images += 1
                    continue

                # Preprocess and inference
                input_tensor = self.model.bgr2nv12(img)
                outputs = self.model.c2numpy(self.model.forward(input_tensor))

                # Post-process
                ids, scores, bboxes, kpts_xy, kpts_score = self.model.postProcess(outputs)

                if len(ids) == 0:
                    no_detections += 1
                    continue

                # Convert to COCO format
                for i, (box, score) in enumerate(zip(bboxes, scores)):
                    x1, y1, x2, y2 = box
                    kpt_xy = kpts_xy[i] if i < len(kpts_xy) else np.zeros((17, 2))
                    kpt_score = kpts_score[i] if i < len(kpts_score) else np.zeros(17)

                    # Format keypoints as [x,y,v] where v=2:visible, 0:not visible
                    kpt_coco = []
                    for j in range(17):
                        x, y = kpt_xy[j]
                        conf = 2 if kpt_score[j] > self.model.conf_inverse else 0
                        kpt_coco.extend([float(x), float(y), conf])

                    results.append({
                        "image_id": img_id,
                        "category_id": 1,  # person
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                        "keypoints": kpt_coco,
                        "area": float((x2 - x1) * (y2 - y1))
                    })

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue

        logger.info(f"Evaluation complete - No detections: {no_detections}, Invalid images: {invalid_images}")
        return results

    def run(self):
        """Run full evaluation pipeline"""
        # Generate predictions
        results = self.evaluate()

        if not results:
            logger.error("No detections generated, check model and input data")
            return

        # Save predictions
        with open("predictions.json", "w") as f:
            json.dump(results, f)
            logger.info(f"Predictions saved to predictions.json ({len(results)} detections)")

        # Load and evaluate
        try:
            coco_dt = self.coco_gt.loadRes("predictions.json")
            coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints')
            coco_eval.params.imgIds = self.img_ids

            # Run evaluation
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            # Print metrics
            print("\nKey Metrics:")
            print(f"AP@0.5:0.95 = {coco_eval.stats[0]:.3f}")
            print(f"AP@0.5       = {coco_eval.stats[1]:.3f}")
            print(f"AR@0.5:0.95 = {coco_eval.stats[6]:.3f}")

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            if os.path.exists("predictions.json"):
                with open("predictions.json") as f:
                    logger.info(f"Predictions content: {f.read(500)}...")


if __name__ == "__main__":
    try:
        # Configure paths (modify as needed)
        model_path = "/home/sunrise/Desktop/yolo11n_pose_bayese_640x640_nv12_modified.bin"
        coco_ann_path = "/home/sunrise/Desktop/person_keypoints_val2017.json"
        img_dir = "/home/sunrise/Desktop/val2017"

        # Validate paths
        for path in [model_path, coco_ann_path, img_dir]:
            if not os.path.exists(path):
                logger.error(f"Path not found: {path}")

        # Run evaluation
        evaluator = COCOEvaluator(
            model_path=model_path,
            coco_ann_path=coco_ann_path,
            img_dir=img_dir,
            num_images=100  # Evaluate first 100 images
        )
        evaluator.run()

    except Exception as e:
        logger.error(f"Runtime error: {e}")