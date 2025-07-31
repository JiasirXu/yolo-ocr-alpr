# ALPR 车牌识别完整系统

基于YOLOv10检测和PaddleOCR/FastPlateOCR识别的完整车牌识别系统，支持训练和推理双模式，提供PyQt5图形界面。

## 🚀 主要特性

- **双模式OCR**: 支持PaddleOCR和FastPlateOCR两种识别后端，可动态切换
- **完整训练流程**: 包含YOLOv10检测训练和OCR识别训练
- **多种输入源**: 支持摄像头、图像文件、视频文件
- **图形化界面**: 基于PyQt5的用户友好界面
- **命令行工具**: 提供CLI演示程序
- **数据集支持**: 内置CCPD数据集转换工具
- **性能评估**: 完整的评估指标和可视化

## 📁 项目结构

```
alpr_project/
├── environment.yml            # conda环境配置
├── requirements.txt           # pip依赖列表
├── README.md                  # 项目说明
├── datasets/                  # 数据集目录
│   ├── CCPD/                  # CCPD数据集
│   └── YOLOTest/              # 自定义检测数据集
├── detector/                  # YOLOv10检测模块
│   ├── detect_plate.py        # 车牌检测推理
│   ├── train_yolo.py          # YOLOv10训练
│   └── weights/               # 模型权重目录
├── ocr/                       # OCR识别模块
│   ├── paddle/                # PaddleOCR模块
│   │   ├── run_paddleocr.py   # PaddleOCR推理
│   │   └── train_paddle.sh    # PaddleOCR训练脚本
│   └── fastplate/             # FastPlateOCR模块
│       ├── run_fastplate.py   # FastPlateOCR推理
│       └── train_fastplate.sh # FastPlateOCR训练脚本
├── pipelines/                 # 核心管道
│   ├── runtime.py             # ALPR引擎
│   └── cli_demo.py            # 命令行演示
├── gui/                       # 图形界面
│   ├── main_window.py         # 主窗口
│   └── alpr_thread.py         # 后台线程
├── utils/                     # 工具函数
│   ├── convert_ccpd_to_yolo.py # CCPD转YOLO格式
│   ├── extract_rec_gt.py      # 提取OCR训练数据
│   ├── perspective.py         # 透视变换工具
│   └── eval.py                # 评估工具
└── config/                    # 配置文件
    ├── data_ccpd.yaml         # CCPD数据集配置
    ├── data_yolotest.yaml     # YOLOTest数据集配置
    ├── yolo_v10s.yaml         # YOLOv10模型配置
    └── ocr_finetune/          # OCR微调配置
        ├── paddle_rec.yml     # PaddleOCR配置
        └── fastplate.yaml     # FastPlateOCR配置
```

## 🛠️ 环境安装

### 方法1: 使用Conda（推荐）

```bash
# 创建conda环境
conda env create -f environment.yml

# 激活环境
conda activate alpr_env
```

### 方法2: 使用pip

```bash
# 创建虚拟环境
python -m venv alpr_env

# 激活环境
# Windows:
alpr_env\Scripts\activate
# Linux/Mac:
source alpr_env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### GPU支持配置

如果使用GPU，请根据您的CUDA版本安装对应的PyTorch和PaddlePaddle：

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install paddlepaddle-gpu==2.5.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install paddlepaddle-gpu==2.5.0.post121 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

## 🚀 快速开始

### 1. 图形界面模式

```bash
python gui/main_window.py
```

### 2. 命令行模式

```bash
# 摄像头实时识别
python pipelines/cli_demo.py --source 0 --ocr paddle

# 图片识别
python pipelines/cli_demo.py --source image.jpg --ocr fastplate

# 视频文件识别
python pipelines/cli_demo.py --source video.mp4 --output result.mp4
```

### 3. 使用不同OCR后端

```bash
# 使用PaddleOCR
python pipelines/cli_demo.py --source 0 --ocr paddle

# 使用FastPlateOCR
python pipelines/cli_demo.py --source 0 --ocr fastplate
```

## 📊 数据集准备

### CCPD数据集

1. 下载CCPD数据集到 `datasets/CCPD/` 目录
2. 转换为YOLO格式：

```bash
# 转换检测数据
python utils/convert_ccpd_to_yolo.py --ccpd-dir datasets/CCPD --output-dir datasets/CCPD_YOLO --mode yolo

# 提取OCR训练数据
python utils/extract_rec_gt.py --ccpd-dir datasets/CCPD --output-dir datasets/CCPD_OCR
```

### 自定义数据集

将您的数据按以下格式组织：

```
datasets/YOLOTest/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## 🎯 模型训练

### 1. YOLOv10检测模型训练

```bash
# 使用CCPD数据集训练
python detector/train_yolo.py --data config/data_ccpd.yaml --epochs 150

# 使用自定义数据集训练
python detector/train_yolo.py --data config/data_yolotest.yaml --epochs 200
```

### 2. OCR识别模型训练

#### PaddleOCR训练

```bash
# 运行训练脚本
bash ocr/paddle/train_paddle.sh
```

#### FastPlateOCR训练

```bash
# 运行训练脚本
bash ocr/fastplate/train_fastplate.sh
```

## 📈 模型评估

```bash
# 评估整个系统
python utils/eval.py --dataset datasets/test --gt-file test_gt.json --ocr-backend paddle

# 生成评估报告和可视化图表
python utils/eval.py --dataset datasets/test --gt-file test_gt.json --output-dir evaluation_results
```

## ⚙️ 配置说明

### 检测配置

- `config/data_ccpd.yaml`: CCPD数据集配置
- `config/data_yolotest.yaml`: 自定义数据集配置
- `config/yolo_v10s.yaml`: YOLOv10模型配置

### OCR配置

- `config/ocr_finetune/paddle_rec.yml`: PaddleOCR微调配置
- `config/ocr_finetune/fastplate.yaml`: FastPlateOCR微调配置

## 🔧 高级用法

### 1. 自定义检测器权重

```python
from pipelines.runtime import AlprEngine

# 使用自定义权重
engine = AlprEngine(
    ocr_backend='paddle',
    detector_weights='path/to/your/weights.pt'
)
```

### 2. 调整识别参数

```python
# 设置置信度阈值
results = engine.process_frame(frame, min_conf=0.6)

# 启用图像增强
results = engine.process_frame(frame, enhance_plates=True)
```

### 3. 批量处理

```python
# 处理视频文件
results = engine.process_video(
    'input_video.mp4',
    output_path='output_video.mp4',
    save_results=True
)
```

## 📋 API参考

### AlprEngine类

主要的车牌识别引擎类，提供以下方法：

- `process_frame(frame, min_conf=0.5, enhance_plates=True)`: 处理单帧图像
- `process_video(video_source, output_path=None, display=True)`: 处理视频流
- `set_ocr_backend(backend)`: 切换OCR后端
- `get_stats()`: 获取性能统计信息

### 检测模块

- `detect_plates(img, conf=0.35, imgsz=960)`: 检测车牌
- `detect_plates_with_conf(img, conf=0.35, imgsz=960)`: 检测车牌并返回置信度

### OCR模块

- `recognize_plate(img)`: 识别车牌文字
- `recognize_plate_detailed(img)`: 返回详细识别结果

## 🐛 常见问题

### 1. CUDA内存不足

```bash
# 减小批次大小
python detector/train_yolo.py --batch-size 16

# 或使用CPU训练
python detector/train_yolo.py --device cpu
```

### 2. PaddleOCR安装问题

```bash
# 如果GPU版本安装失败，使用CPU版本
pip uninstall paddlepaddle-gpu
pip install paddlepaddle
```

### 3. PyQt5界面问题

```bash
# Linux系统可能需要安装额外依赖
sudo apt-get install python3-pyqt5
```

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交GitHub Issue
- 发送邮件至项目维护者

## 🙏 致谢

感谢以下开源项目的支持：

- [Ultralytics YOLOv10](https://github.com/ultralytics/ultralytics)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Fast Plate OCR](https://github.com/ankandrew/fast-plate-ocr)
- [CCPD Dataset](https://github.com/detectRecog/CCPD)
