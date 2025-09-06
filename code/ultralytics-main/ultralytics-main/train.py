# -*- coding: utf-8 -*-

import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO(model=r'C:\Users\Dobot\Desktop\ultralytics-main\ultralytics-main\ultralytics\cfg\models\11\yolo11-pose.yaml')
    #model.load('yolo11n.pt')  # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(data=r'C:\Users\Dobot\Desktop\ultralytics-main\ultralytics-main\ultralytics\cfg\datasets\dog-pose.yaml',
                imgsz=640,
                epochs=200,
                batch=4,
                workers=0,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                save=True,#保存训练结果
                save_period=1,#每一个epoch保存一次
                val=True,#开启验证，用于计算mAP并保存最佳模型
                plots=True,#生成训练曲线
                task='pose'#明确指定任务类型
                )



