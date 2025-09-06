# 加载D-Robotics 依赖库
from horizon_tc_ui import HB_ONNXRuntime

# 准备模型运行的feed_dict
def prepare_input_dict(input_names):
  feed_dict = dict()
  for input_name in input_names:
      # your_custom_data_prepare代表您的自定义数据
      # 根据输入节点的类型和layout要求准备数据即可
      feed_dict[input_name] = your_custom_data_prepare(input_name)
  return feed_dict

if __name__ == '__main__':
  # 创建推理Session
  sess = HB_ONNXRuntime(model_file='***_quantized_model.onnx')

  # 获取输入节点名称
  input_names = [input.name for input in sess.get_inputs()]
  # 或
  input_names = sess.input_names

  # 获取输出节点名称
  output_names = [output.name for output in sess.get_outputs()]
  # 或
  output_names = sess.output_names

  # 准备模型输入数据
  feed_dict = prepare_input_dict(input_names)
  # 开始模型推理，推理的返回值是一个list，依次与output_names指定名称一一对应
  # 输入图像的类型范围为（RGB/BGR/NV12/YUV444/GRAY）
  outputs = sess.run(output_names, feed_dict, input_offset=128)
  # 输入数据的类型范围为（FEATURE）
  outputs = sess.run_feature(output_names, feed_dict, input_offset=0)