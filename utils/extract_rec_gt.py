"""
从CCPD数据集提取车牌识别训练数据
生成OCR训练所需的图像和标注文件
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json

# CCPD字符映射表
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def parse_ccpd_filename(filename):
    """
    解析CCPD文件名获取标注信息
    
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
        
        # 解析边界框坐标 (4个点)
        bbox_coords = parts[2].split('_')
        if len(bbox_coords) != 8:
            return None
        
        # 转换为4个点的坐标
        points = []
        for i in range(0, 8, 2):
            x = int(bbox_coords[i].split('&')[0])
            y = int(bbox_coords[i + 1].split('&')[0])
            points.append([x, y])
        
        # 车牌字符编码
        plate_chars_encoded = parts[4].split('_')
        
        return {
            'bbox_points': points,
            'plate_chars_encoded': plate_chars_encoded
        }
        
    except Exception as e:
        print(f"解析文件名失败 {filename}: {e}")
        return None

def decode_plate_chars(encoded_chars):
    """
    解码车牌字符
    
    Args:
        encoded_chars: 编码的字符列表
        
    Returns:
        str: 解码后的车牌号
    """
    try:
        if len(encoded_chars) < 7:
            return None
        
        # 第一个字符是省份
        province_idx = int(encoded_chars[0])
        if province_idx >= len(PROVINCES):
            return None
        province = PROVINCES[province_idx]
        
        # 第二个字符是字母
        letter_idx = int(encoded_chars[1])
        if letter_idx >= len(ALPHABETS):
            return None
        letter = ALPHABETS[letter_idx]
        
        # 后面5个字符可能是字母或数字
        remaining_chars = []
        for i in range(2, 7):
            char_idx = int(encoded_chars[i])
            
            # 根据位置判断是字母还是数字
            if i == 2:  # 第三位可能是字母或数字
                if char_idx < len(ALPHABETS):
                    remaining_chars.append(ALPHABETS[char_idx])
                else:
                    remaining_chars.append(DIGITS[char_idx - len(ALPHABETS)])
            else:  # 其他位置主要是数字
                if char_idx < len(DIGITS):
                    remaining_chars.append(DIGITS[char_idx])
                else:
                    remaining_chars.append(ALPHABETS[char_idx - len(DIGITS)])
        
        plate_text = province + letter + ''.join(remaining_chars)
        return plate_text
        
    except Exception as e:
        print(f"解码字符失败: {e}")
        return None

def extract_plate_region(image, points):
    """
    从图像中提取车牌区域
    
    Args:
        image: 原始图像
        points: 车牌四个角点
        
    Returns:
        numpy.ndarray: 车牌区域图像
    """
    try:
        # 转换为numpy数组
        points = np.array(points, dtype=np.float32)
        
        # 计算边界框
        x_min = int(np.min(points[:, 0]))
        x_max = int(np.max(points[:, 0]))
        y_min = int(np.min(points[:, 1]))
        y_max = int(np.max(points[:, 1]))
        
        # 添加一些边距
        margin = 5
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(image.shape[1], x_max + margin)
        y_max = min(image.shape[0], y_max + margin)
        
        # 裁剪车牌区域
        plate_img = image[y_min:y_max, x_min:x_max]
        
        return plate_img
        
    except Exception as e:
        print(f"提取车牌区域失败: {e}")
        return None

def enhance_plate_image(plate_img):
    """
    增强车牌图像质量
    
    Args:
        plate_img: 车牌图像
        
    Returns:
        numpy.ndarray: 增强后的图像
    """
    try:
        # 转换为灰度图
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
        
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 高斯滤波去噪
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 转回彩色图像
        result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        return result
        
    except Exception as e:
        print(f"图像增强失败: {e}")
        return plate_img

def extract_recognition_data(ccpd_dir, output_dir, target_size=(200, 64), 
                           train_ratio=0.8, val_ratio=0.1, enhance=True):
    """
    从CCPD数据集提取车牌识别训练数据
    
    Args:
        ccpd_dir: CCPD数据集目录
        output_dir: 输出目录
        target_size: 目标尺寸 (width, height)
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        enhance: 是否进行图像增强
    """
    
    ccpd_path = Path(ccpd_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    test_dir = output_path / 'test'
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(ccpd_path.glob(ext)))
    
    if not image_files:
        print(f"在 {ccpd_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 随机打乱文件列表
    import random
    random.shuffle(image_files)
    
    # 计算分割点
    total_files = len(image_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # 分割数据集
    splits = {
        'train': (image_files[:train_end], train_dir),
        'val': (image_files[train_end:val_end], val_dir),
        'test': (image_files[val_end:], test_dir)
    }
    
    print(f"数据集分割: 训练集 {len(splits['train'][0])}, 验证集 {len(splits['val'][0])}, 测试集 {len(splits['test'][0])}")
    
    # 统计信息
    total_extracted = 0
    total_failed = 0
    char_stats = {}
    
    # 处理每个分割
    for split_name, (files, split_dir) in splits.items():
        print(f"\n处理 {split_name} 集...")
        
        # 创建标注文件
        gt_file = split_dir / 'rec_gt.txt'
        
        split_extracted = 0
        split_failed = 0
        
        with open(gt_file, 'w', encoding='utf-8') as f:
            for img_file in tqdm(files, desc=f"提取{split_name}集"):
                try:
                    # 解析文件名
                    annotation = parse_ccpd_filename(img_file.name)
                    if annotation is None:
                        split_failed += 1
                        continue
                    
                    # 解码车牌字符
                    plate_text = decode_plate_chars(annotation['plate_chars_encoded'])
                    if plate_text is None:
                        split_failed += 1
                        continue
                    
                    # 读取图像
                    img = cv2.imread(str(img_file))
                    if img is None:
                        split_failed += 1
                        continue
                    
                    # 提取车牌区域
                    plate_img = extract_plate_region(img, annotation['bbox_points'])
                    if plate_img is None or plate_img.size == 0:
                        split_failed += 1
                        continue
                    
                    # 图像增强
                    if enhance:
                        plate_img = enhance_plate_image(plate_img)
                    
                    # 调整尺寸
                    if target_size:
                        plate_img = cv2.resize(plate_img, target_size, interpolation=cv2.INTER_CUBIC)
                    
                    # 保存车牌图像
                    plate_filename = f"plate_{split_extracted:06d}.jpg"
                    plate_path = split_dir / plate_filename
                    cv2.imwrite(str(plate_path), plate_img)
                    
                    # 写入标注文件
                    f.write(f"{plate_filename}\t{plate_text}\n")
                    
                    # 统计字符
                    for char in plate_text:
                        char_stats[char] = char_stats.get(char, 0) + 1
                    
                    split_extracted += 1
                    
                except Exception as e:
                    print(f"处理文件失败 {img_file.name}: {e}")
                    split_failed += 1
        
        print(f"{split_name}集提取完成: 成功 {split_extracted}, 失败 {split_failed}")
        total_extracted += split_extracted
        total_failed += split_failed
    
    # 保存字符统计信息
    stats_file = output_path / 'char_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(char_stats, f, ensure_ascii=False, indent=2)
    
    # 创建字符字典文件
    create_char_dict(output_path, char_stats)
    
    # 创建数据集配置文件
    create_dataset_config(output_path, splits, char_stats)
    
    print(f"\n总计提取完成: 成功 {total_extracted}, 失败 {total_failed}")
    print(f"输出目录: {output_dir}")
    print(f"字符统计: {stats_file}")

def create_char_dict(output_path, char_stats):
    """
    创建字符字典文件
    
    Args:
        output_path: 输出路径
        char_stats: 字符统计信息
    """
    
    # 按频率排序字符
    sorted_chars = sorted(char_stats.items(), key=lambda x: x[1], reverse=True)
    
    # 创建字符字典
    dict_file = output_path / 'char_dict.txt'
    with open(dict_file, 'w', encoding='utf-8') as f:
        for char, count in sorted_chars:
            f.write(f"{char}\n")
    
    print(f"字符字典已创建: {dict_file}")
    print(f"总字符数: {len(sorted_chars)}")

def create_dataset_config(output_path, splits, char_stats):
    """
    创建数据集配置文件
    
    Args:
        output_path: 输出路径
        splits: 数据集分割信息
        char_stats: 字符统计信息
    """
    
    config_content = f"""# CCPD车牌识别数据集配置

# 数据集路径
dataset_root: {output_path.absolute()}

# 训练、验证、测试集路径
train_data: train/rec_gt.txt
val_data: val/rec_gt.txt
test_data: test/rec_gt.txt

# 字符字典路径
char_dict: char_dict.txt

# 数据集统计
train_samples: {len(splits['train'][0])}
val_samples: {len(splits['val'][0])}
test_samples: {len(splits['test'][0])}
total_chars: {len(char_stats)}

# 图像配置
image_width: 200
image_height: 64
channels: 3

# 字符统计（前10个最常见字符）
top_chars:
"""
    
    # 添加最常见的字符
    sorted_chars = sorted(char_stats.items(), key=lambda x: x[1], reverse=True)
    for char, count in sorted_chars[:10]:
        config_content += f"  {char}: {count}\n"
    
    config_path = output_path / 'dataset_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"数据集配置文件已创建: {config_path}")

def main():
    parser = argparse.ArgumentParser(description='CCPD车牌识别数据提取工具')
    parser.add_argument('--ccpd-dir', type=str, required=True, help='CCPD数据集目录')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    parser.add_argument('--width', type=int, default=200, help='车牌图像宽度')
    parser.add_argument('--height', type=int, default=64, help='车牌图像高度')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--no-enhance', action='store_true', help='不进行图像增强')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ccpd_dir):
        print(f"CCPD数据集目录不存在: {args.ccpd_dir}")
        return
    
    # 提取识别数据
    extract_recognition_data(
        args.ccpd_dir,
        args.output_dir,
        target_size=(args.width, args.height),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        enhance=not args.no_enhance
    )

if __name__ == "__main__":
    main()
