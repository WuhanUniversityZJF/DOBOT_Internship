import cv2
from ultralytics import YOLO

model = YOLO('C:\\Users\\Dobot\\Desktop\\ultralytics-main\\ultralytics-main\\runs\\train\\exp5\\weights\\best.pt')
#model = YOLO('C:\\Users\\Dobot\\Desktop\\ultralytics-main\\ultralytics-main\\yolo11n.pt')
results = model('C:\\Users\\Dobot\\Desktop\\ultralytics-main\\ultralytics-main\\datasets\\tiger-pose\\images\\val\\Frame_54.jpg')

for result in results:
    img = result.plot()  # 绘制检测框
    kpts = result.keypoints  # 获取关键点

    # 手动绘制关键点
    for kpt in kpts:
        for i, (x, y, conf) in enumerate(kpt):
            if conf > 0.5:  # 只绘制置信度高的关键点
                cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(img, str(i), (int(x), int(y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('Result', img)
    cv2.waitKey(0)