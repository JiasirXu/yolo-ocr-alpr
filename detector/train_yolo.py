"""
YOLOv10 车牌检测模型训练脚本
支持使用CCPD和YOLOTest数据集训练车牌检测模型
"""

import os
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

def train_yolo(data_config, model_size='s', epochs=150, batch_size=32, imgsz=960, 
               device='auto', workers=8, project='runs/detect', name='yolo_plate'):
    """
    训练YOLOv10车牌检测模型
    
    Args:
        data_config: 数据配置文件路径
        model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
        epochs: 训练轮数
        batch_size: 批次大小
        imgsz: 输入图像尺寸
        device: 训练设备
        workers: 数据加载器工作进程数
        project: 项目保存路径
        name: 实验名称
    """
    
    # 检查数据配置文件是否存在
    if not os.path.exists(data_config):
        raise FileNotFoundError(f"数据配置文件不存在: {data_config}")
    
    # 加载预训练模型
    model_name = f'yolov10{model_size}.pt'
    print(f"加载预训练模型: {model_name}")
    
    try:
        model = YOLO(model_name)
        print(f"成功加载预训练模型: {model_name}")
    except Exception as e:
        print(f"加载预训练模型失败: {e}")
        return None
    
    # 开始训练
    print(f"开始训练，配置参数:")
    print(f"  数据配置: {data_config}")
    print(f"  模型大小: {model_size}")
    print(f"  训练轮数: {epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  图像尺寸: {imgsz}")
    print(f"  设备: {device}")
    print(f"  工作进程: {workers}")
    
    try:
        # 训练模型
        results = model.train(
            data=data_config,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            workers=workers,
            project=project,
            name=name,
            save=True,
            save_period=10,  # 每10个epoch保存一次
            val=True,
            plots=True,
            verbose=True,
            patience=50,  # 早停耐心值
            close_mosaic=10,  # 最后10个epoch关闭mosaic增强
            amp=True,  # 自动混合精度
            fraction=1.0,  # 使用全部数据
            profile=False,
            freeze=None,  # 不冻结任何层
            lr0=0.01,  # 初始学习率
            lrf=0.01,  # 最终学习率因子
            momentum=0.937,  # SGD动量
            weight_decay=0.0005,  # 权重衰减
            warmup_epochs=3.0,  # 预热轮数
            warmup_momentum=0.8,  # 预热动量
            warmup_bias_lr=0.1,  # 预热偏置学习率
            box=7.5,  # 边界框损失权重
            cls=0.5,  # 分类损失权重
            dfl=1.5,  # DFL损失权重
            pose=12.0,  # 姿态损失权重
            kobj=2.0,  # 关键点目标损失权重
            label_smoothing=0.0,  # 标签平滑
            nbs=64,  # 标准批次大小
            hsv_h=0.015,  # HSV色调增强
            hsv_s=0.7,  # HSV饱和度增强
            hsv_v=0.4,  # HSV明度增强
            degrees=0.0,  # 旋转增强角度
            translate=0.1,  # 平移增强
            scale=0.5,  # 缩放增强
            shear=0.0,  # 剪切增强
            perspective=0.0,  # 透视增强
            flipud=0.0,  # 上下翻转概率
            fliplr=0.5,  # 左右翻转概率
            mosaic=1.0,  # mosaic增强概率
            mixup=0.0,  # mixup增强概率
            copy_paste=0.0,  # 复制粘贴增强概率
        )
        
        print("训练完成!")
        
        # 获取最佳权重路径
        best_weights = results.save_dir / 'weights' / 'best.pt'
        last_weights = results.save_dir / 'weights' / 'last.pt'
        
        print(f"最佳权重保存在: {best_weights}")
        print(f"最后权重保存在: {last_weights}")
        
        # 复制最佳权重到detector/weights目录
        target_path = Path('detector/weights/yolov10_plate.pt')
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy2(best_weights, target_path)
        print(f"最佳权重已复制到: {target_path}")
        
        return results
        
    except Exception as e:
        print(f"训练过程出错: {e}")
        return None

def export_model(weights_path, formats=['onnx'], imgsz=960):
    """
    导出训练好的模型到其他格式
    
    Args:
        weights_path: 权重文件路径
        formats: 导出格式列表
        imgsz: 输入图像尺寸
    """
    try:
        model = YOLO(weights_path)
        
        for fmt in formats:
            print(f"导出模型到 {fmt} 格式...")
            model.export(format=fmt, imgsz=imgsz)
            print(f"成功导出到 {fmt} 格式")
            
    except Exception as e:
        print(f"模型导出失败: {e}")

def validate_model(weights_path, data_config, imgsz=960):
    """
    验证训练好的模型
    
    Args:
        weights_path: 权重文件路径
        data_config: 数据配置文件路径
        imgsz: 输入图像尺寸
    """
    try:
        model = YOLO(weights_path)
        results = model.val(data=data_config, imgsz=imgsz)
        
        print("验证结果:")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        
        return results
        
    except Exception as e:
        print(f"模型验证失败: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='YOLOv10车牌检测模型训练')
    parser.add_argument('--data', type=str, default='config/data_ccpd.yaml', 
                       help='数据配置文件路径')
    parser.add_argument('--model-size', type=str, default='s', 
                       choices=['n', 's', 'm', 'l', 'x'], help='模型大小')
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=960, help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='auto', help='训练设备')
    parser.add_argument('--workers', type=int, default=8, help='数据加载器工作进程数')
    parser.add_argument('--project', type=str, default='runs/detect', help='项目保存路径')
    parser.add_argument('--name', type=str, default='yolo_plate', help='实验名称')
    parser.add_argument('--export', action='store_true', help='训练后导出模型')
    parser.add_argument('--export-formats', nargs='+', default=['onnx'], 
                       help='导出格式')
    parser.add_argument('--validate', action='store_true', help='训练后验证模型')
    
    args = parser.parse_args()
    
    # 训练模型
    results = train_yolo(
        data_config=args.data,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name
    )
    
    if results is None:
        print("训练失败!")
        return
    
    # 获取训练后的权重路径
    best_weights = results.save_dir / 'weights' / 'best.pt'
    
    # 导出模型
    if args.export:
        print("\n开始导出模型...")
        export_model(str(best_weights), args.export_formats, args.imgsz)
    
    # 验证模型
    if args.validate:
        print("\n开始验证模型...")
        validate_model(str(best_weights), args.data, args.imgsz)

if __name__ == "__main__":
    main()
