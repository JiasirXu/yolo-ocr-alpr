"""
CCPD数据集转换为YOLO格式的工具
将CCPD数据集的标注信息转换为YOLO检测训练所需的格式
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil
import random

def parse_ccpd_filename(filename):
    """
    解析CCPD文件名获取标注信息
    
    CCPD文件名格式: 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
    
    Args:
        filename: CCPD文件名
        
    Returns:
        dict: 包含车牌信息的字典
    """
    try:
        # 移除文件扩展名
        basename = os.path.splitext(filename)[0]
        
        # 按'-'分割
        parts = basename.split('-')
        
        if len(parts) < 7:
            return None
        
        # 解析各部分
        area_code = parts[0]  # 地区代码
        tilt_degree = parts[1].split('_')  # 倾斜角度和水平角度
        
        # 边界框坐标 (4个点)
        bbox_coords = parts[2].split('_')
        if len(bbox_coords) != 8:
            return None
        
        # 转换为4个点的坐标
        points = []
        for i in range(0, 8, 2):
            x = int(bbox_coords[i].split('&')[0])
            y = int(bbox_coords[i + 1].split('&')[0])
            points.append([x, y])
        
        # 车牌字符
        plate_chars = parts[4].split('_')
        
        # 亮度
        brightness = parts[5]
        
        # 模糊度
        blurriness = parts[6]
        
        return {
            'area_code': area_code,
            'tilt_degree': tilt_degree,
            'bbox_points': points,
            'plate_chars': plate_chars,
            'brightness': brightness,
            'blurriness': blurriness
        }
        
    except Exception as e:
        print(f"解析文件名失败 {filename}: {e}")
        return None

def points_to_yolo_bbox(points, img_width, img_height):
    """
    将4个角点转换为YOLO格式的边界框
    
    Args:
        points: 4个角点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        img_width: 图像宽度
        img_height: 图像高度
        
    Returns:
        tuple: (center_x, center_y, width, height) 归一化坐标
    """
    # 转换为numpy数组
    points = np.array(points)
    
    # 计算边界框
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    
    # 计算中心点和尺寸
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # 归一化
    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height
    
    return center_x, center_y, width, height

def convert_ccpd_to_yolo(ccpd_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    将CCPD数据集转换为YOLO格式
    
    Args:
        ccpd_dir: CCPD数据集目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    
    # 检查比例是否正确
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("训练、验证、测试集比例之和必须为1")
    
    # 创建输出目录结构
    output_path = Path(output_dir)
    
    # 创建数据集目录
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 获取所有CCPD图像文件
    ccpd_path = Path(ccpd_dir)
    image_files = []
    
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(ccpd_path.glob(ext)))
    
    if not image_files:
        print(f"在 {ccpd_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 随机打乱文件列表
    random.shuffle(image_files)
    
    # 计算分割点
    total_files = len(image_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # 分割数据集
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }
    
    print(f"数据集分割: 训练集 {len(splits['train'])}, 验证集 {len(splits['val'])}, 测试集 {len(splits['test'])}")
    
    # 统计信息
    total_converted = 0
    total_failed = 0
    
    # 处理每个分割
    for split_name, files in splits.items():
        print(f"\n处理 {split_name} 集...")
        
        split_converted = 0
        split_failed = 0
        
        for img_file in tqdm(files, desc=f"转换{split_name}集"):
            try:
                # 解析文件名
                annotation = parse_ccpd_filename(img_file.name)
                if annotation is None:
                    split_failed += 1
                    continue
                
                # 读取图像获取尺寸
                img = cv2.imread(str(img_file))
                if img is None:
                    split_failed += 1
                    continue
                
                img_height, img_width = img.shape[:2]
                
                # 转换边界框
                center_x, center_y, width, height = points_to_yolo_bbox(
                    annotation['bbox_points'], img_width, img_height
                )
                
                # 复制图像文件
                dst_img_path = output_path / split_name / 'images' / img_file.name
                shutil.copy2(img_file, dst_img_path)
                
                # 创建标注文件
                label_filename = img_file.stem + '.txt'
                dst_label_path = output_path / split_name / 'labels' / label_filename
                
                with open(dst_label_path, 'w') as f:
                    # YOLO格式: class_id center_x center_y width height
                    # 车牌类别ID为0
                    f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                
                split_converted += 1
                
            except Exception as e:
                print(f"处理文件失败 {img_file.name}: {e}")
                split_failed += 1
        
        print(f"{split_name}集转换完成: 成功 {split_converted}, 失败 {split_failed}")
        total_converted += split_converted
        total_failed += split_failed
    
    # 创建数据配置文件
    create_yolo_config(output_path, splits)
    
    print(f"\n总计转换完成: 成功 {total_converted}, 失败 {total_failed}")
    print(f"输出目录: {output_dir}")

def create_yolo_config(output_path, splits):
    """
    创建YOLO训练配置文件
    
    Args:
        output_path: 输出路径
        splits: 数据集分割信息
    """
    
    # 创建数据配置文件
    config_content = f"""# CCPD车牌检测数据集配置

# 数据集路径
path: {output_path.absolute()}

# 训练、验证、测试集路径
train: train/images
val: val/images
test: test/images

# 类别数量
nc: 1

# 类别名称
names:
  0: license_plate

# 数据集统计
# 训练集: {len(splits['train'])} 张图像
# 验证集: {len(splits['val'])} 张图像  
# 测试集: {len(splits['test'])} 张图像
"""
    
    config_path = output_path / 'data.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"数据配置文件已创建: {config_path}")

def extract_plate_images(ccpd_dir, output_dir, target_size=(200, 64)):
    """
    从CCPD数据集中提取车牌图像用于OCR训练
    
    Args:
        ccpd_dir: CCPD数据集目录
        output_dir: 输出目录
        target_size: 目标尺寸 (width, height)
    """
    
    ccpd_path = Path(ccpd_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(ccpd_path.glob(ext)))
    
    if not image_files:
        print(f"在 {ccpd_dir} 中未找到图像文件")
        return
    
    print(f"开始提取车牌图像，共 {len(image_files)} 个文件")
    
    # 创建标注文件
    gt_file = output_path / 'rec_gt.txt'
    
    extracted_count = 0
    failed_count = 0
    
    with open(gt_file, 'w', encoding='utf-8') as f:
        for img_file in tqdm(image_files, desc="提取车牌图像"):
            try:
                # 解析文件名
                annotation = parse_ccpd_filename(img_file.name)
                if annotation is None:
                    failed_count += 1
                    continue
                
                # 读取图像
                img = cv2.imread(str(img_file))
                if img is None:
                    failed_count += 1
                    continue
                
                # 提取车牌区域
                points = np.array(annotation['bbox_points'], dtype=np.float32)
                
                # 计算边界框
                x_min = int(np.min(points[:, 0]))
                x_max = int(np.max(points[:, 0]))
                y_min = int(np.min(points[:, 1]))
                y_max = int(np.max(points[:, 1]))
                
                # 裁剪车牌区域
                plate_img = img[y_min:y_max, x_min:x_max]
                
                if plate_img.size == 0:
                    failed_count += 1
                    continue
                
                # 调整尺寸
                if target_size:
                    plate_img = cv2.resize(plate_img, target_size)
                
                # 保存车牌图像
                plate_filename = f"plate_{extracted_count:06d}.jpg"
                plate_path = output_path / plate_filename
                cv2.imwrite(str(plate_path), plate_img)
                
                # 构建车牌文字（这里需要根据CCPD的字符编码规则）
                # 简化处理，实际使用时需要完整的字符映射
                plate_text = "车牌文字"  # 这里应该根据annotation['plate_chars']解码
                
                # 写入标注文件
                f.write(f"{plate_filename}\t{plate_text}\n")
                
                extracted_count += 1
                
            except Exception as e:
                print(f"提取车牌失败 {img_file.name}: {e}")
                failed_count += 1
    
    print(f"车牌图像提取完成: 成功 {extracted_count}, 失败 {failed_count}")
    print(f"输出目录: {output_dir}")
    print(f"标注文件: {gt_file}")

def main():
    parser = argparse.ArgumentParser(description='CCPD数据集转换工具')
    parser.add_argument('--ccpd-dir', type=str, required=True, help='CCPD数据集目录')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    parser.add_argument('--mode', type=str, choices=['yolo', 'ocr'], default='yolo',
                       help='转换模式: yolo(检测) 或 ocr(识别)')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--plate-width', type=int, default=200, help='车牌图像宽度')
    parser.add_argument('--plate-height', type=int, default=64, help='车牌图像高度')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ccpd_dir):
        print(f"CCPD数据集目录不存在: {args.ccpd_dir}")
        return
    
    if args.mode == 'yolo':
        # 转换为YOLO检测格式
        convert_ccpd_to_yolo(
            args.ccpd_dir, 
            args.output_dir,
            args.train_ratio,
            args.val_ratio, 
            args.test_ratio
        )
    
    elif args.mode == 'ocr':
        # 提取车牌图像用于OCR训练
        extract_plate_images(
            args.ccpd_dir,
            args.output_dir,
            (args.plate_width, args.plate_height)
        )

if __name__ == "__main__":
    main()
