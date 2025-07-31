#!/bin/bash

# PaddleOCR 车牌识别模型微调训练脚本
# 使用CCPD数据集训练车牌识别模型

set -e

# 配置参数
WORK_DIR="ocr/paddle"
CONFIG_FILE="config/ocr_finetune/paddle_rec.yml"
PRETRAINED_MODEL="ch_PP-OCRv4_rec_train"
OUTPUT_DIR="output/paddle_plate_rec"
DATASET_DIR="datasets/CCPD"

echo "=== PaddleOCR 车牌识别模型训练 ==="
echo "工作目录: $WORK_DIR"
echo "配置文件: $CONFIG_FILE"
echo "预训练模型: $PRETRAINED_MODEL"
echo "输出目录: $OUTPUT_DIR"
echo "数据集目录: $DATASET_DIR"

# 检查必要的目录和文件
if [ ! -d "$DATASET_DIR" ]; then
    echo "错误: 数据集目录不存在: $DATASET_DIR"
    echo "请先准备CCPD数据集"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    echo "请先创建配置文件"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查是否安装了PaddleOCR
python -c "import paddleocr" 2>/dev/null || {
    echo "错误: PaddleOCR未安装"
    echo "请运行: pip install paddleocr"
    exit 1
}

# 检查是否有GPU
if python -c "import paddle; print('GPU可用' if paddle.is_compiled_with_cuda() else 'GPU不可用')"; then
    USE_GPU="True"
else
    USE_GPU="False"
    echo "警告: 未检测到GPU，将使用CPU训练（速度较慢）"
fi

echo "开始训练..."

# 执行训练
python -m paddle.distributed.launch \
    --gpus="0" \
    tools/train.py \
    -c "$CONFIG_FILE" \
    -o Global.pretrained_model="$PRETRAINED_MODEL" \
    -o Global.save_model_dir="$OUTPUT_DIR" \
    -o Global.use_gpu="$USE_GPU" \
    -o Train.dataset.data_dir="$DATASET_DIR" \
    -o Eval.dataset.data_dir="$DATASET_DIR"

echo "训练完成!"

# 检查训练结果
if [ -d "$OUTPUT_DIR/best_accuracy" ]; then
    echo "最佳模型保存在: $OUTPUT_DIR/best_accuracy"
    
    # 复制最佳模型到推理目录
    INFERENCE_DIR="$WORK_DIR/models"
    mkdir -p "$INFERENCE_DIR"
    
    if [ -f "$OUTPUT_DIR/best_accuracy/inference.pdmodel" ]; then
        cp "$OUTPUT_DIR/best_accuracy/inference.pdmodel" "$INFERENCE_DIR/"
        cp "$OUTPUT_DIR/best_accuracy/inference.pdiparams" "$INFERENCE_DIR/"
        cp "$OUTPUT_DIR/best_accuracy/inference.pdiparams.info" "$INFERENCE_DIR/"
        echo "推理模型已复制到: $INFERENCE_DIR"
    fi
else
    echo "警告: 未找到训练结果"
fi

# 模型转换为推理格式（如果需要）
if [ -f "$OUTPUT_DIR/latest/model.pdparams" ]; then
    echo "转换模型为推理格式..."
    
    python tools/export_model.py \
        -c "$CONFIG_FILE" \
        -o Global.pretrained_model="$OUTPUT_DIR/latest/model" \
        -o Global.save_inference_dir="$OUTPUT_DIR/inference"
    
    echo "推理模型保存在: $OUTPUT_DIR/inference"
fi

echo "=== 训练完成 ==="
echo "模型文件位置:"
echo "  训练模型: $OUTPUT_DIR"
echo "  推理模型: $INFERENCE_DIR"
echo ""
echo "使用方法:"
echo "  在 run_paddleocr.py 中设置 rec_model_dir='$INFERENCE_DIR' 来使用训练好的模型"
