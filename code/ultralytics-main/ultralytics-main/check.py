import os

def validate_annotation_files(label_dir):
    for filename in os.listdir(label_dir):
        with open(os.path.join(label_dir, filename), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 29:  # 1(class) + 4(box) + 24(keypoints)
                    print(f"格式错误: {filename} - 找到 {len(parts)} 个值")
                    print(line)
                    return False
    return True

# 使用示例
label_dir = "datasets/tiger-pose/labels/train"
if validate_annotation_files(label_dir):
    print("所有标注文件格式正确")
else:
    print("发现格式不正确的标注文件")