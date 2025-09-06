import numpy as np
import cv2
import skimage
from skimage.transform import resize
from horizon_tc_ui import HB_ONNXRuntime
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------- 配置参数（根据训练和数据调整） --------------------------
MODEL_PATH = "dog_pose_quantized_model.onnx"  # 模型路径
TEST_DATASET_DIR = "test_dataset_02/test_dataset"  # 测试集目录
NUM_KEYPOINTS = 24  # 关键点数量
INPUT_SHAPE = (640, 640)  # 模型输入尺寸 (H, W)
HEATMAP_NODE_NAME = "onnx::Shape_640"  # 热图输出节点（80x80分辨率）
KEYPOINT_CHANNELS = 24  # 关键点通道数（前24通道为关键点）

# 预处理参数（必须与训练时完全一致！）
TRAIN_SHORT_SIZE = 256  # 训练时短边缩放尺寸
TRAIN_CROP_SIZE = 224   # 训练时中心裁剪尺寸

# 可见性参数
GT_VIS_VALUE = 2.0      # 标注中可见性有效值（=2.0为有效）
PRED_VIS_THRESHOLD = 0.1  # 预测可见性阈值（热图峰值>0.1为有效）

DEBUG = False  # 调试模式：打印详细坐标映射过程


# -------------------------- 图像预处理（记录参数用于逆映射） --------------------------
def your_custom_data_prepare(image_path):
    """预处理：短边缩放→中心裁剪→resize至输入尺寸，并记录中间参数"""
    # 读取原图（RGB格式，float32，[0,1]）
    image = skimage.img_as_float(skimage.io.imread(image_path))
    original_h, original_w = image.shape[:2]  # 原图尺寸 (H, W)
    
    # -------------------------- 严格复现训练时的预处理步骤 --------------------------
    # 1. 短边缩放至 TRAIN_SHORT_SIZE（256）
    scale = TRAIN_SHORT_SIZE / min(original_h, original_w)  # 缩放比例
    scaled_h, scaled_w = int(original_h * scale), int(original_w * scale)  # 缩放后尺寸
    image_scaled = resize(image, (scaled_h, scaled_w), anti_aliasing=True)  # 短边缩放
    
    # 2. 中心裁剪至 TRAIN_CROP_SIZE（224x224）
    start_h = (scaled_h - TRAIN_CROP_SIZE) // 2  # 裁剪区域左上角y坐标
    start_w = (scaled_w - TRAIN_CROP_SIZE) // 2  # 裁剪区域左上角x坐标
    image_cropped = image_scaled[start_h:start_h+TRAIN_CROP_SIZE, start_w:start_w+TRAIN_CROP_SIZE, :]  # 中心裁剪
    
    # 3. Resize至模型输入尺寸（640x640）
    image_input = resize(image_cropped, (INPUT_SHAPE[0], INPUT_SHAPE[1]), anti_aliasing=True)
    
    # 4. 通道和布局处理（NHWC格式，int8量化）
    image_input = image_input[np.newaxis, ...]  # 添加batch维度：(1, 640, 640, 3)
    image_input = (image_input * 255) - 128     # 量化：[0,1]→[0,255]→[-128, 127]（int8）
    
    # 记录预处理参数（用于后续逆映射）
    preprocess_params = {
        "original_size": (original_h, original_w),  # 原图尺寸
        "scale": scale,                             # 短边缩放比例
        "start_h": start_h,                         # 裁剪区域左上角y
        "start_w": start_w,                         # 裁剪区域左上角x
        "crop_size": TRAIN_CROP_SIZE                # 裁剪尺寸（224）
    }
    if DEBUG:
        print(f"\n预处理参数: {preprocess_params}")
    
    return image_input.astype(np.int8), preprocess_params


# -------------------------- 标注解析（提取真实关键点） --------------------------
def parse_annotation(annotation_path, original_size):
    """解析标注文件：提取24个关键点的像素坐标（x,y,visibility）"""
    original_h, original_w = original_size
    with open(annotation_path, 'r') as f:
        data = list(map(float, f.readline().strip().split()))  # 标注数据（一行式）
    
    # 标注格式：[bbox_x, bbox_y, bbox_w, bbox_h, ..., x0, y0, vis0, x1, y1, vis1, ...]
    # 假设关键点从第5个值开始（索引4），每个关键点3个值（x_norm, y_norm, vis）
    KEYPOINT_START_INDEX = 5
    keypoint_data = data[KEYPOINT_START_INDEX : KEYPOINT_START_INDEX + NUM_KEYPOINTS*3]
    keypoints = []
    
    for i in range(NUM_KEYPOINTS):
        x_norm = keypoint_data[i*3]      # 归一化x（0~1，相对于原图）
        y_norm = keypoint_data[i*3 + 1]  # 归一化y（0~1，相对于原图）
        visibility = keypoint_data[i*3 + 2]  # 可见性（2.0=可见，0=不可见）
        
        # 过滤无效关键点（可见性≠2.0或坐标为0）
        if visibility != GT_VIS_VALUE or x_norm <= 0 or y_norm <= 0:
            keypoints.append([0.0, 0.0, 0.0])  # 无效点标记为(0,0,0)
            continue
        
        # 归一化坐标→原图像素坐标
        x_pixel = x_norm * original_w
        y_pixel = y_norm * original_h
        keypoints.append([x_pixel, y_pixel, visibility])
    
    return np.array(keypoints, dtype=np.float32)  # (24, 3) → (x,y,visibility)


# -------------------------- 热图转关键点（完整逆映射） --------------------------
def heatmap_to_keypoints(heatmap, preprocess_params):
    """热图坐标→原图坐标（通过预处理逆操作精确映射）"""
    # 解析预处理参数
    original_h, original_w = preprocess_params["original_size"]
    scale = preprocess_params["scale"]
    start_h, start_w = preprocess_params["start_h"], preprocess_params["start_w"]
    crop_size = preprocess_params["crop_size"]
    heatmap_h, heatmap_w = heatmap.shape[2], heatmap.shape[3]  # 热图尺寸（80,80）
    
    keypoints = []
    for kpt_idx in range(KEYPOINT_CHANNELS):
        # 1. 提取第kpt_idx个关键点的热图
        kpt_heatmap = heatmap[0, kpt_idx]  # 热图形状：(80,80)
        
        # 2. 找热图峰值坐标（h,w）和峰值强度
        peak_h, peak_w = np.unravel_index(np.argmax(kpt_heatmap), kpt_heatmap.shape)
        peak_value = np.max(kpt_heatmap)  # 峰值强度（用于可见性判断）
        
        # 3. 热图坐标→原图坐标（完整逆映射过程）
        # 步骤1：热图归一化坐标（0~1）
        x_norm_heatmap = peak_w / (heatmap_w - 1)  # 避免除零（80-1=79）
        y_norm_heatmap = peak_h / (heatmap_h - 1)
        
        # 步骤2：热图坐标→模型输入图像坐标（640x640）
        x_input = x_norm_heatmap * INPUT_SHAPE[1]  # 输入图像x（W=640）
        y_input = y_norm_heatmap * INPUT_SHAPE[0]  # 输入图像y（H=640）
        
        # 步骤3：输入图像坐标→裁剪图像坐标（224x224）
        # 输入图像是裁剪图像resize的结果，逆操作需乘裁剪尺寸/输入尺寸
        x_crop = x_input * (crop_size / INPUT_SHAPE[1])
        y_crop = y_input * (crop_size / INPUT_SHAPE[0])
        
        # 步骤4：裁剪图像坐标→缩放后图像坐标（短边256）
        # 裁剪图像是从缩放后图像中裁剪的，需加上裁剪起始坐标
        x_scaled = start_w + x_crop
        y_scaled = start_h + y_crop
        
        # 步骤5：缩放后图像坐标→原图像坐标（逆短边缩放）
        x_original = x_scaled / scale  # 除以缩放比例
        y_original = y_scaled / scale
        
        # 调试模式：打印坐标映射过程
        if DEBUG and kpt_idx == 0:  # 仅打印第一个关键点的映射过程
            print(f"关键点0映射过程：\n"
                  f"热图峰值坐标: (h={peak_h}, w={peak_w}) → "
                  f"热图归一化坐标: (x={x_norm_heatmap:.2f}, y={y_norm_heatmap:.2f}) → "
                  f"输入图像坐标: (x={x_input:.0f}, y={y_input:.0f}) → "
                  f"裁剪图像坐标: (x={x_crop:.0f}, y={y_crop:.0f}) → "
                  f"缩放后坐标: (x={x_scaled:.0f}, y={y_scaled:.0f}) → "
                  f"原图坐标: (x={x_original:.0f}, y={y_original:.0f})")
        
        # 4. 可见性判断（热图峰值>阈值为有效）
        visibility = 1.0 if peak_value > PRED_VIS_THRESHOLD else 0.0
        
        keypoints.append([x_original, y_original, visibility])
    
    return np.array(keypoints, dtype=np.float32)  # (24, 3) → (x,y,visibility)


# -------------------------- 精度指标计算（修复可见性掩码） --------------------------
def compute_mpjpe(pred_keypoints, gt_keypoints):
    """平均关节点误差（MPJPE）：仅计算双方均有效的关键点"""
    # 真实关键点有效掩码：visibility=GT_VIS_VALUE且坐标非(0,0)
    gt_valid_mask = (gt_keypoints[:, 2] == GT_VIS_VALUE) & \
                    (gt_keypoints[:, 0] > 0) & (gt_keypoints[:, 1] > 0)
    
    # 预测关键点有效掩码：visibility=1.0且坐标非(0,0)
    pred_valid_mask = (pred_keypoints[:, 2] == 1.0) & \
                      (pred_keypoints[:, 0] > 0) & (pred_keypoints[:, 1] > 0)
    
    # 双方均有效的掩码
    valid_mask = gt_valid_mask & pred_valid_mask
    if np.sum(valid_mask) == 0:
        return 0.0  # 无有效点，返回0
    
    # 计算有效关键点的平均距离
    pred_valid = pred_keypoints[valid_mask][:, :2]  # (N, 2)
    gt_valid = gt_keypoints[valid_mask][:, :2]      # (N, 2)
    return np.mean(np.linalg.norm(pred_valid - gt_valid, axis=1))


def compute_pck(pred_keypoints, gt_keypoints, threshold=0.05):
    """关键点准确率（PCK@threshold）：误差小于阈值的比例"""
    # 双方均有效掩码（同MPJPE）
    gt_valid_mask = (gt_keypoints[:, 2] == GT_VIS_VALUE) & (gt_keypoints[:, 0] > 0) & (gt_keypoints[:, 1] > 0)
    pred_valid_mask = (pred_keypoints[:, 2] == 1.0) & (pred_keypoints[:, 0] > 0) & (pred_keypoints[:, 1] > 0)
    valid_mask = gt_valid_mask & pred_valid_mask
    if np.sum(valid_mask) == 0:
        return 0.0
    
    # 计算有效关键点的距离和阈值（阈值=原图高度×threshold）
    pred_valid = pred_keypoints[valid_mask][:, :2]
    gt_valid = gt_keypoints[valid_mask][:, :2]
    distances = np.linalg.norm(pred_valid - gt_valid, axis=1)
    threshold_pixel = threshold * gt_keypoints.shape[0]  # 原图高度×0.05
    
    # 计算PCK比例
    return np.mean((distances < threshold_pixel).astype(np.float32))


# -------------------------- 可视化（对比预测和真实关键点） --------------------------
def visualize_results(image_path, pred_keypoints, gt_keypoints, save_path):
    """可视化预测（红）和真实（绿）关键点，保存图像"""
    # 读取原图
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB格式显示
    
    # 绘制真实关键点（绿色，仅可见点）
    for x, y, vis in gt_keypoints:
        if vis == GT_VIS_VALUE:  # 可见性=2.0为有效
            cv2.circle(image, (int(x), int(y)), 6, (0, 255, 0), -1)  # 绿色实心圆，半径6
    
    # 绘制预测关键点（红色，仅可见点）
    for x, y, vis in pred_keypoints:
        if vis == 1.0:  # 可见性=1.0为有效
            cv2.circle(image, (int(x), int(y)), 6, (255, 0, 0), -1)  # 红色实心圆，半径6
    
    # 保存可视化结果
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(f"Pred (Red) vs GT (Green) | MPJPE: {compute_mpjpe(pred_keypoints, gt_keypoints):.1f}px")
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    if DEBUG:
        print(f"可视化结果保存至: {save_path}")


# -------------------------- 主函数：批量评估 --------------------------
def evaluate_model():
    # 1. 加载模型并检查输入输出节点
    sess = HB_ONNXRuntime(model_file=MODEL_PATH)
    input_name = sess.input_names[0]
    print(f"模型输入节点: {input_name}, 热图输出节点: {HEATMAP_NODE_NAME}")
    
    # 2. 获取所有图像-标注对路径
    image_paths = []
    anno_paths = []
    for filename in os.listdir(TEST_DATASET_DIR):
        if filename.endswith(".jpg"):
            img_path = os.path.join(TEST_DATASET_DIR, filename)
            anno_path = os.path.join(TEST_DATASET_DIR, filename.replace(".jpg", ".txt"))
            if os.path.exists(anno_path):
                image_paths.append(img_path)
                anno_paths.append(anno_path)
    print(f"找到 {len(image_paths)} 个有效图像-标注对")
    if not image_paths:
        raise ValueError("未找到有效图像-标注对，请检查TEST_DATASET_DIR路径")
    
    # 3. 批量评估
    mpjpe_list = []
    pck_list = []
    for img_idx, (img_path, anno_path) in enumerate(tqdm(zip(image_paths, anno_paths), total=len(image_paths))):
        # a. 预处理：获取输入数据和预处理参数
        input_data, preprocess_params = your_custom_data_prepare(img_path)
        original_size = preprocess_params["original_size"]
        
        # b. 推理热图节点
        feed_dict = {input_name: input_data}
        outputs = sess.run([HEATMAP_NODE_NAME], feed_dict, input_type="feature")
        heatmap = outputs[0]  # 热图形状: [1, 65, 80, 80]（前24通道为关键点）
        
        # c. 热图转关键点（逆映射至原图）
        pred_keypoints = heatmap_to_keypoints(heatmap, preprocess_params)
        
        # d. 解析标注
        gt_keypoints = parse_annotation(anno_path, original_size)
        
        # e. 计算精度指标
        mpjpe = compute_mpjpe(pred_keypoints, gt_keypoints)
        pck = compute_pck(pred_keypoints, gt_keypoints)
        if mpjpe > 0:  # 过滤无有效点的样本
            mpjpe_list.append(mpjpe)
            pck_list.append(pck)
        
        # f. 可视化前5张结果
        if img_idx < 5:
            visualize_results(img_path, pred_keypoints, gt_keypoints, f"eval_vis_{img_idx}.png")
        
        # 调试模式：仅运行前5张图像
        if DEBUG and img_idx >= 4:
            print("\n调试模式：已停止后续评估")
            break
    
    # 4. 输出最终评估结果
    print("\n===== 精度评估结果 =====")
    print(f"有效测试样本数: {len(mpjpe_list)}")
    print(f"平均MPJPE (像素): {np.mean(mpjpe_list):.2f}")
    print(f"平均PCK@0.05: {np.mean(pck_list):.2%}")
    print("=======================")


if __name__ == "__main__":
    evaluate_model()
