吸烟打电话监测数据集-吸烟、电话检测数据集，3304张，yolo、voc、coco标注方式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0e336531496147c1a6303dd224776687.png)

数据来源于多段办公室监控视频，从视频中间隔取样后标注得到
图像尺寸:998*1776
类别数量:2类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b5f1d8b6fdeb4da6af89ef19a59aa11a.png)

训练集图像数量:2752; 验证集图像数量:280； 测试集图像数量:272
类别名称: 每一类图像数 ，每一类标注数
phone: 1469,2237
smoke: 860,862
image num: 3304
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/03a0a23423ed43539a8bed63d31a1e35.png)


2.模型代码：模型训练使用yolov11n训练，30个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8c12126eccda4a808af49e10b3ee1c35.png)

3.qt界面：运行界面采用pyqt编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a4a8899cda2f40118a0c6d283ac9ac60.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f47af7886af845ebb92108c41342dfae.png)
# 基于YOLOv11 吸烟、打电话检测系统(PyQt5界面)完整代码
## 1. data.yaml配置文件
```yaml
path: ./dataset
train: images/train
val: images/val
test: images/test

names:
  0: smoke
  1: phone
```

## 2.训练代码 train.py
```python
from ultralytics import YOLO

# 加载YOLOv11模型
model = YOLO("yolo11n.pt")

#训练参数
results = model.train(
    data="data.yaml",
    epochs=80,
    imgsz=640,
    batch=8,
    device=0,
    workers=0
)
```

## 3. PyQt5可视化界面主程序 main.py
```python
import sys
import cv2
from PyQt5.QtWidgets import (QApplication,QMainWindow,QPushButton,QFileDialog,
                             QLabel,QTextEdit,QComboBox,QWidget,QVBoxLayout,QHBoxLayout)
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import Qt,QTimer
from ultralytics import YOLO

class DetectWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的吸烟、电话检测系统")
        self.resize(1250,900)
        #加载训练权重
        self.model = YOLO("./runs/detect/train/weights/best.pt")
        self.timer = QTimer()
        self.cap = None
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        #左侧图像展示区
        self.img_label = QLabel("图像预览区")
        self.img_label.setFixedSize(750,550)
        self.img_label.setStyleSheet("border:1px solid #888;")
        layout.addWidget(self.img_label)

        #右侧控制面板
        right_layout = QVBoxLayout()
        layout.addLayout(right_layout)

        #导入文件按钮
        self.btn_img = QPushButton("选择图片")
        self.btn_img.clicked.connect(self.detect_image)
        self.btn_video = QPushButton("选择视频文件")
        self.btn_video.clicked.connect(self.detect_video)
        self.btn_cam = QPushButton("打开摄像头")
        self.btn_cam.clicked.connect(self.open_camera)
        right_layout.addWidget(self.btn_img)
        right_layout.addWidget(self.btn_video)
        right_layout.addWidget(self.btn_cam)

        #检测结果
        self.result_label = QLabel("检测结果")
        self.info_text = QTextEdit()
        self.info_text.setFixedHeight(300)
        self.combo_target = QComboBox()
        self.combo_target.addItems(["全部","smoke","phone"])
        right_layout.addWidget(self.result_label)
        right_layout.addWidget(self.combo_target)
        right_layout.addWidget(self.info_text)

        #操作按钮
        self.btn_save = QPushButton("保存结果")
        self.btn_exit = QPushButton("退出")
        self.btn_exit.clicked.connect(self.close)
        right_layout.addWidget(self.btn_save)
        right_layout.addWidget(self.btn_exit)

    def detect_image(self):
        file_path,_ = QFileDialog.getOpenFileName(self,"打开图片","","Image(*.jpg *.png *.jpeg)")
        if not file_path:
            return
        res = self.model(file_path)[0]
        img = res.plot()
        self.show_img(img)
        self.parse_result(res)

    def detect_video(self):
        vid_path,_ = QFileDialog.getOpenFileName(self,"打开视频","","Video(*.mp4 *.avi)")
        if not vid_path:
            return
        self.cap = cv2.VideoCapture(vid_path)
        self.timer.timeout.connect(self.read_frame)
        self.timer.start(30)

    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.timeout.connect(self.read_frame)
        self.timer.start(30)

    def read_frame(self):
        ret,frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            return
        res = self.model(frame)[0]
        frame = res.plot()
        self.show_img(frame)
        self.parse_result(res)

    def show_img(self,img):
        rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape
        bytes_per = ch*w
        qimg = QImage(rgb.data,w,h,bytes_per,QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.img_label.size(),Qt.KeepAspectRatio))

    def parse_result(self,result):
        self.info_text.clear()
        boxes = result.boxes
        num = len(boxes)
        self.info_text.append(f"用时:{result.speed['inference']:.3f} s  目标数目:{num}")
        for box in boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = self.model.names[cls]
            self.info_text.append(f"类别:{cls_name}  置信度:{conf:.2f}  坐标:[{x1},{y1},{x2},{y2}]")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DetectWindow()
    win.show()
    sys.exit(app.exec_())
```

## 4.环境依赖 requirements.txt
```
ultralytics==8.3.0
opencv‑python
PyQt5
torch
torchvision
numpy
```
>安装命令：`pip install -r requirements.txt`


