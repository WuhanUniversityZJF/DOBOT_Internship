
class YOLOPredictor:
    def __init__(self):
        """
        初始化YOLO预测器
        """
        self.model_weights = "runs/train/exp6/weights/epoch298.pt"

        self.model = YOLO(self.model_weights)  # 加载模型权重

    def predict_and_save(
            self,
            image_path: Union[str, Path],
            output_dir: Union[str, Path],
            output_filename: Optional[str] = None,
            save: bool = True
    ) -> Results:
        """
        对单张图片进行预测并保存结果

        参数:
            image_path: 输入图片路径
            output_dir: 输出目录路径
            output_filename: 自定义输出文件名 (默认使用输入文件名)
            save: 是否保存预测结果图片

        返回:
            ultralytics的预测结果对象
        """
        # 转换路径类型并创建输出目录
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 读取图片
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")

        # 执行预测
        results = self.model(img)[0]
        print(f"处理完成: {image_path.name}")

        # 保存结果
        if save:
            output_name = output_filename or f"pred_{image_path.name}"
            output_path = output_dir / output_name
            results.save(str(output_path))
            print(f"结果已保存至: {output_path}")

        return results

    def batch_predict(
            self,
            input_dir: Union[str, Path],
            output_dir: Union[str, Path],
            extensions: List[str] = ['.jpg', '.jpeg', '.png']
    ) -> None:
        """
        批量处理文件夹中的所有图片

        参数:
            input_dir: 输入图片文件夹路径
            output_dir: 输出目录路径
            extensions: 支持的图片扩展名列表
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")

        # 获取所有符合条件的图片文件
        image_files = []
        for ext in extensions:
            image_files.extend(list(input_dir.glob(f'*{ext}')))

        print(f"发现 {len(image_files)} 张待处理图片")

        for img_path in image_files:
            try:
                self.predict_and_save(
                    image_path=img_path,
                    output_dir=output_dir,
                    output_filename=f"pred_{img_path.name}",  # 保持原文件名，前面加pred_
                    save=True
                )
            except Exception as e:
                print(f"处理图片 {img_path.name} 时出错: {str(e)}")
