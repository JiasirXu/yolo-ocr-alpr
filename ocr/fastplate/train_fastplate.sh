#!/bin/bash

# Fast Plate OCR 车牌识别模型微调训练脚本
# 使用CCPD数据集训练车牌识别模型

set -e

# 配置参数
WORK_DIR="ocr/fastplate"
CONFIG_FILE="config/ocr_finetune/fastplate.yaml"
BASE_MODEL="cct-xs-v1-global-model"
OUTPUT_DIR="output/fastplate_rec"
DATASET_DIR="datasets/CCPD"
TRAIN_DATA="$DATASET_DIR/train_rec_gt.txt"
VAL_DATA="$DATASET_DIR/val_rec_gt.txt"

echo "=== Fast Plate OCR 车牌识别模型训练 ==="
echo "工作目录: $WORK_DIR"
echo "配置文件: $CONFIG_FILE"
echo "基础模型: $BASE_MODEL"
echo "输出目录: $OUTPUT_DIR"
echo "数据集目录: $DATASET_DIR"

# 检查必要的目录和文件
if [ ! -d "$DATASET_DIR" ]; then
    echo "错误: 数据集目录不存在: $DATASET_DIR"
    echo "请先准备CCPD数据集"
    exit 1
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "错误: 训练数据文件不存在: $TRAIN_DATA"
    echo "请先运行数据预处理脚本生成训练数据"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    echo "请先创建配置文件"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查是否安装了fast-plate-ocr
python -c "import fast_plate_ocr" 2>/dev/null || {
    echo "错误: fast-plate-ocr未安装"
    echo "请运行: pip install fast-plate-ocr"
    exit 1
}

# 检查GPU可用性
if python -c "import torch; print('GPU可用' if torch.cuda.is_available() else 'GPU不可用')"; then
    DEVICE="cuda"
    echo "使用GPU训练"
else
    DEVICE="cpu"
    echo "警告: 未检测到GPU，将使用CPU训练（速度较慢）"
fi

echo "开始训练..."

# 使用fast-plate-ocr的训练命令
# 注意: 这里的命令可能需要根据实际的fast-plate-ocr API进行调整
python -c "
import os
import sys
from fast_plate_ocr import train_model
from fast_plate_ocr.config import load_config

# 加载配置
config = load_config('$CONFIG_FILE')

# 设置训练参数
config.update({
    'base_model': '$BASE_MODEL',
    'train_data': '$TRAIN_DATA',
    'val_data': '$VAL_DATA',
    'output_dir': '$OUTPUT_DIR',
    'device': '$DEVICE',
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'save_interval': 10,
    'eval_interval': 5,
    'early_stopping_patience': 15,
    'warmup_epochs': 5,
    'weight_decay': 0.0001,
    'gradient_clip_norm': 1.0,
    'mixed_precision': True if '$DEVICE' == 'cuda' else False,
})

print('训练配置:')
for key, value in config.items():
    print(f'  {key}: {value}')

# 开始训练
try:
    train_model(config)
    print('训练完成!')
except Exception as e:
    print(f'训练失败: {e}')
    sys.exit(1)
"

# 检查训练结果
if [ -d "$OUTPUT_DIR" ]; then
    echo "训练结果保存在: $OUTPUT_DIR"
    
    # 查找最佳模型
    BEST_MODEL=$(find "$OUTPUT_DIR" -name "best_model*" -type f | head -1)
    if [ -n "$BEST_MODEL" ]; then
        echo "最佳模型: $BEST_MODEL"
        
        # 复制最佳模型到推理目录
        INFERENCE_DIR="$WORK_DIR/models"
        mkdir -p "$INFERENCE_DIR"
        
        cp "$BEST_MODEL" "$INFERENCE_DIR/custom_model.pth"
        echo "推理模型已复制到: $INFERENCE_DIR/custom_model.pth"
    fi
    
    # 查找最终模型
    FINAL_MODEL=$(find "$OUTPUT_DIR" -name "final_model*" -type f | head -1)
    if [ -n "$FINAL_MODEL" ]; then
        echo "最终模型: $FINAL_MODEL"
    fi
else
    echo "警告: 未找到训练结果目录"
fi

# 模型评估
if [ -f "$VAL_DATA" ] && [ -n "$BEST_MODEL" ]; then
    echo "开始模型评估..."
    
    python -c "
import sys
from fast_plate_ocr import LicensePlateRecognizer
from fast_plate_ocr.evaluation import evaluate_model

# 加载训练好的模型
try:
    model = LicensePlateRecognizer('$BEST_MODEL')
    
    # 评估模型
    results = evaluate_model(model, '$VAL_DATA')
    
    print('评估结果:')
    print(f'  准确率: {results.get(\"accuracy\", 0):.4f}')
    print(f'  字符准确率: {results.get(\"char_accuracy\", 0):.4f}')
    print(f'  平均置信度: {results.get(\"avg_confidence\", 0):.4f}')
    
except Exception as e:
    print(f'模型评估失败: {e}')
    sys.exit(1)
"
fi

echo "=== 训练完成 ==="
echo "模型文件位置:"
echo "  训练输出: $OUTPUT_DIR"
echo "  推理模型: $INFERENCE_DIR"
echo ""
echo "使用方法:"
echo "  在 run_fastplate.py 中使用自定义模型路径来加载训练好的模型"

# 创建使用说明文件
cat > "$OUTPUT_DIR/README.md" << EOF
# Fast Plate OCR 训练结果

## 模型文件
- 最佳模型: \`best_model.pth\`
- 最终模型: \`final_model.pth\`
- 训练日志: \`training.log\`

## 使用方法

### Python代码中使用
\`\`\`python
from fast_plate_ocr import LicensePlateRecognizer

# 加载自定义训练的模型
ocr = LicensePlateRecognizer('$OUTPUT_DIR/best_model.pth')

# 识别车牌
result = ocr.run(image)
print(f"识别结果: {result}")
\`\`\`

### 在项目中使用
将模型文件复制到 \`ocr/fastplate/models/\` 目录，然后在 \`run_fastplate.py\` 中指定模型路径。

## 训练配置
- 基础模型: $BASE_MODEL
- 训练数据: $TRAIN_DATA
- 验证数据: $VAL_DATA
- 设备: $DEVICE
EOF

echo "使用说明已保存到: $OUTPUT_DIR/README.md"
