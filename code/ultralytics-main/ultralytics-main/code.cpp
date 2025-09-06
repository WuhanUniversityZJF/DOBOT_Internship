#include <iostream>
#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"

float quanti_shift(int32_t data, uint32_t shift) {
  return static_cast<float>(data) / static_cast<float>(1 << shift);
}

float quanti_scale(int32_t data, float scale) { return data * scale; }

int main(int argc, char **argv) {
  // 第一步加载模型
  hbPackedDNNHandle_t packed_dnn_handle;
  const char* model_file_name= "./mobilenetv1/mobilenetv1_224x224_nv12.bin";
  hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1);

  // 第二步获取模型名称
  const char **model_name_list;
  int model_count = 0;
  hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);

  // 第三步获取dnn_handle
  hbDNNHandle_t dnn_handle;
  hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

  // 第四步准备输入数据
  hbDNNTensor input;
  hbDNNTensorProperties input_properties;
  hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0);
  input.properties = input_properties;
  auto &mem = input.sysMem[0];

  int yuv_length = 224 * 224 * 3 / 2;
  hbSysAllocCachedMem(&mem, yuv_length);
  //memcpy(mem.virAddr, yuv_data, yuv_length);
  //hbSysFlushMem(&mem, HB_SYS_MEM_CACHE_CLEAN);

  // 第五步准备模型输出数据的空间
  int output_count;
  hbDNNGetOutputCount(&output_count, dnn_handle);
  hbDNNTensor *output = new hbDNNTensor[output_count];
  for (int i = 0; i < output_count; i++) {
    hbDNNTensorProperties &output_properties = output[i].properties;
    hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
    int out_aligned_size = output_properties.alignedByteSize;
    hbSysMem &mem = output[i].sysMem[0];
    hbSysAllocCachedMem(&mem, out_aligned_size);
  }

  // 第六步推理模型
  hbDNNTaskHandle_t task_handle = nullptr;
  hbDNNInferCtrlParam infer_ctrl_param;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
  hbDNNInfer(&task_handle,
              &output,
              &input,
              dnn_handle,
              &infer_ctrl_param);

  // 第七步等待任务结束
  hbDNNWaitTaskDone(task_handle, 0);
  //第八步解析模型输出，例子就获取mobilenetv1的top1分类
  float max_prob = -1.0;
  int max_prob_type_id = 0;
  hbSysFlushMem(&(output->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  float *data = reinterpret_cast< float *>(output->sysMem[0].virAddr);
  int *shape = output->properties.validShape.dimensionSize;
  int * aligned_shape = output->properties.alignedShape.dimensionSize;
  auto properties = output->properties;
  int offset = 1;
  if (properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    offset = aligned_shape[2] * aligned_shape[3];
  }
  for (auto i = 0; i < shape[1] * shape[2] * shape[3]; i++) {
    float score;
    if (properties.quantiType == SHIFT) {
      score = quanti_shift(data[i * offset], properties.shift.shiftData[i]);
    } else if (properties.quantiType == SCALE) {
      score = quanti_scale(data[i * offset], properties.scale.scaleData[i]);
    } else if (properties.quantiType == NONE){
      score = data[i * offset];
    } else {
      std::cout << "quanti type error!";
      return -1;
    }
    if(score < max_prob)
      continue;
    max_prob = score;
    max_prob_type_id = i;
  }

  std::cout << "max id: " << max_prob_type_id << std::endl;
  // 释放任务
  hbDNNReleaseTask(task_handle);

  // 释放内存
  hbSysFreeMem(&(input.sysMem[0]));
  hbSysFreeMem(&(output->sysMem[0]));

  // 释放模型
  hbDNNRelease(packed_dnn_handle);

  return 0;
}