import os
import cv2
import json
import numpy as np
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


class YOLOv11PoseDetector:
    def __init__(self, model_path, conf_thres=0.4, iou_thres=0.5, kpt_thres=0.3):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.kpt_thres = kpt_thres
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                         [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                         [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

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

    def decode_prediction(self, pred):
        pred = pred.transpose(1, 0)
        valid_mask = pred[:, 4] > self.conf_thres
        pred = pred[valid_mask]
        if len(pred) == 0:
            return [], [], []
        boxes = pred[:, :4].copy()
        boxes[:, 0] -= boxes[:, 2] / 2  # x1
        boxes[:, 1] -= boxes[:, 3] / 2  # y1
        boxes[:, 2] += boxes[:, 0]  # x2
        boxes[:, 3] += boxes[:, 1]  # y2
        return boxes, pred[:, 4], pred[:, 5:].reshape(-1, 17, 3)

    def apply_nms(self, boxes, scores, kpts):
        if len(boxes) == 0:
            return [], [], []
        indices = cv2.dnn.NMSBoxes(
            boxes.astype(np.int32).tolist(),
            scores.tolist(),
            self.conf_thres,
            self.iou_thres
        )
        return (boxes[indices], scores[indices], kpts[indices]) if len(indices) > 0 else ([], [], [])

    def scale_coords(self, coords):
        coords[:, [0, 2]] = (coords[:, [0, 2]] - self.pad_x) / self.scale_factor
        coords[:, [1, 3]] = (coords[:, [1, 3]] - self.pad_y) / self.scale_factor
        return coords

    def scale_keypoints(self, kpts):
        kpts[..., 0] = (kpts[..., 0] - self.pad_x) / self.scale_factor
        kpts[..., 1] = (kpts[..., 1] - self.pad_y) / self.scale_factor
        return kpts


class COCOEvaluator:
    def __init__(self, detector, coco_ann_path, img_dir, num_images=100):
        self.detector = detector
        self.coco_gt = COCO(coco_ann_path)
        self.img_dir = img_dir
        self.img_ids = self.coco_gt.getImgIds(catIds=self.coco_gt.getCatIds(['person']))[:num_images]

    def evaluate(self):
        results = []
        for img_id in tqdm(self.img_ids, desc="Evaluating"):
            img_info = self.coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                input_tensor = self.detector.preprocess(img)
                pred = self.detector.session.run(
                    [self.detector.output_name],
                    {self.detector.input_name: input_tensor}
                )[0][0]

                boxes, scores, kpts = self.detector.decode_prediction(pred)
                boxes, scores, kpts = self.detector.apply_nms(boxes, scores, kpts)

                if len(boxes) == 0:
                    continue

                boxes = self.detector.scale_coords(boxes)
                kpts = self.detector.scale_keypoints(kpts)

                for box, score, kpt in zip(boxes, scores, kpts):
                    x1, y1, x2, y2 = box
                    kpt_coco = []
                    for i in range(17):
                        x, y, conf = kpt[i]
                        v = 2 if conf > self.detector.kpt_thres else 0
                        kpt_coco.extend([float(x), float(y), v])

                    results.append({
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                        "keypoints": kpt_coco,
                        "area": float((x2 - x1) * (y2 - y1))
                    })

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        return results

    def run(self):
        results = self.evaluate()
        with open("predictions.json", "w") as f:
            json.dump(results, f)

        coco_dt = self.coco_gt.loadRes("predictions.json")
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints')
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        print("\n关键指标:")
        print(f"AP@0.5:0.95 = {coco_eval.stats[0]:.3f}")
        print(f"AP@0.5       = {coco_eval.stats[1]:.3f}")
        print(f"AR@0.5:0.95 = {coco_eval.stats[6]:.3f}")


if __name__ == "__main__":
    # 初始化检测器
    detector = YOLOv11PoseDetector(
        model_path="yolo11n-pose.onnx",
        conf_thres=0.5,
        iou_thres=0.45
    )

    # 运行评估
    evaluator = COCOEvaluator(
        detector=detector,
        coco_ann_path="annotations/person_keypoints_val2017.json",
        img_dir="val2017",
        num_images=5000
    )
    evaluator.run()