#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import argparse
import numpy as np
from ultralytics import YOLO

def convert_yolov11_pose_to_onnx(pth_path, onnx_path, opset_version=11, simplify=True):
    """
    将 YOLOv11-Pose 的 .pth 模型转换为 ONNX 格式
    :param pth_path: 输入的 .pth 模型路径
    :param onnx_path: 输出的 .onnx 文件路径
    :param opset_version: ONNX opset 版本（建议 11）
    :param simplify: 是否简化 ONNX 模型（去除冗余节点）
    """
    # 1. 加载 YOLOv11-Pose 模型
    model = YOLO(pth_path, task='pose')  # 显式指定任务为 'pose'
    model.fuse()  # 融合 Conv+BN 层
    model.eval()

    # 2. 准备虚拟输入（YOLOv11-Pose 的默认输入尺寸为 640x640）
    dummy_input = torch.randn(1, 3, 640, 640).to(next(model.parameters()).device)

    # 3. 导出 ONNX 模型
    torch.onnx.export(
        model.model,  # 直接导出底层 PyTorch 模型
        dummy_input,
        onnx_path,
        input_names=["images"],
        output_names=["output0"],  # YOLO 输出节点名
        opset_version=opset_version,
        dynamic_axes={
            'images': {0: 'batch'},  # 仅允许 batch 维度动态
            'output0': {0: 'batch'}
        } if opset_version >= 11 else None,  # 地平线芯片需固定维度时设为 None
        do_constant_folding=True,
        verbose=False
    )

    # 4. 简化 ONNX 模型（可选）
    if simplify:
        try:
            import onnxsim
            import onnx
            onnx_model = onnx.load(onnx_path)
            simplified_model, check = onnxsim.simplify(onnx_model)
            assert check, "Simplified ONNX model validation failed"
            onnx.save(simplified_model, onnx_path)
        except ImportError:
            print("未安装 onnxsim，跳过简化步骤。运行: pip install onnxsim")

    # 5. 验证 ONNX 模型
    validate_onnx_model(onnx_path)

def validate_onnx_model(onnx_path):
    """验证 ONNX 模型格式和推理一致性"""
    import onnx
    import onnxruntime as ort

    # 检查模型格式
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"\n✅ ONNX 模型验证通过: {onnx_path}")

    # 测试推理
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    outputs = sess.run(None, {input_name: dummy_input})
    print(f"输出形状: {[o.shape for o in outputs]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11-Pose 模型转 ONNX")
    parser.add_argument("--pth", type=str, required=True, help="输入 .pth 模型路径")
    parser.add_argument("--onnx", type=str, required=True, help="输出 .onnx 路径")
    parser.add_argument("--opset", type=int, default=11, choices=[10, 11], help="ONNX opset 版本")
    parser.add_argument("--no-simplify", action="store_false", dest="simplify", help="禁用 ONNX 简化")
    args = parser.parse_args()

    convert_yolov11_pose_to_onnx(
        pth_path=args.pth,
        onnx_path=args.onnx,
        opset_version=args.opset,
        simplify=args.simplify
    )