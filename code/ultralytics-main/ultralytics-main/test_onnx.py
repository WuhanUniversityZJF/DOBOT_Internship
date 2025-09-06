import onnxruntime
import numpy as np

# 加载 ONNX 模型
sess = onnxruntime.InferenceSession("yolo11n-pose.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# 生成随机输入
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

# 推理
output = sess.run([output_name], {input_name: input_data})
print("ONNX 推理成功！输出形状:", output[0].shape)