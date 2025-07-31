"""
PaddleOCR 车牌识别模块
支持中文车牌识别，可使用官方预训练模型或自定义微调模型
"""

import os
import cv2
import numpy as np
from paddleocr import PaddleOCR

# 车牌字符字典路径
PLATE_DICT_PATH = 'ocr/paddle/license_plate_dict.txt'

# 全局OCR实例
ocr_instance = None

def create_plate_dict():
    """
    创建车牌字符字典文件
    """
    # 中国车牌字符集
    provinces = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', 
                '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', 
                '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新']
    
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 
               'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    # 特殊字符
    special = ['警', '学', '领', '港', '澳', '挂', '试', '超', '使']
    
    # 组合所有字符
    all_chars = provinces + letters + numbers + special
    
    # 确保目录存在
    os.makedirs(os.path.dirname(PLATE_DICT_PATH), exist_ok=True)
    
    # 写入字典文件
    with open(PLATE_DICT_PATH, 'w', encoding='utf-8') as f:
        for char in all_chars:
            f.write(char + '\n')
    
    print(f"车牌字典文件已创建: {PLATE_DICT_PATH}")

def load_ocr(use_angle_cls=True, use_gpu=True, lang='ch', 
             det_model_dir=None, rec_model_dir=None, cls_model_dir=None):
    """
    加载PaddleOCR模型
    
    Args:
        use_angle_cls: 是否使用角度分类器
        use_gpu: 是否使用GPU
        lang: 语言，'ch'表示中文
        det_model_dir: 检测模型路径
        rec_model_dir: 识别模型路径
        cls_model_dir: 分类模型路径
    """
    global ocr_instance
    
    # 检查字典文件是否存在，不存在则创建
    if not os.path.exists(PLATE_DICT_PATH):
        create_plate_dict()
    
    try:
        # 初始化PaddleOCR
        ocr_instance = PaddleOCR(
            use_angle_cls=use_angle_cls,
            use_gpu=use_gpu,
            lang=lang,
            det=False,  # 只使用识别模块，不使用检测模块
            rec_char_dict_path=PLATE_DICT_PATH,
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            cls_model_dir=cls_model_dir,
            show_log=False
        )
        
        print("PaddleOCR模型加载成功")
        return True
        
    except Exception as e:
        print(f"PaddleOCR模型加载失败: {e}")
        return False

def recognize_plate(img, use_cls=False):
    """
    识别车牌图像中的文字
    
    Args:
        img: 输入图像 (numpy array)
        use_cls: 是否使用角度分类
        
    Returns:
        tuple: (识别文字, 平均置信度)
    """
    global ocr_instance
    
    # 如果OCR实例未加载，尝试加载
    if ocr_instance is None:
        if not load_ocr():
            return '', 0.0
    
    try:
        # 图像预处理
        if len(img.shape) == 3 and img.shape[2] == 3:
            # BGR转RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        # 执行OCR识别
        result = ocr_instance.ocr(img_rgb, cls=use_cls)
        
        if not result or not result[0]:
            return '', 0.0
        
        # 提取识别结果
        text_info = result[0]
        if not text_info:
            return '', 0.0
        
        # 解析识别结果
        chars = []
        confidences = []
        
        for line in text_info:
            if len(line) >= 2:
                text, conf = line[1]
                chars.append(text)
                confidences.append(conf)
        
        # 合并文字和计算平均置信度
        final_text = ''.join(chars)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return final_text, avg_confidence
        
    except Exception as e:
        print(f"车牌识别过程出错: {e}")
        return '', 0.0

def recognize_plate_detailed(img, use_cls=False):
    """
    识别车牌图像中的文字，返回详细信息
    
    Args:
        img: 输入图像 (numpy array)
        use_cls: 是否使用角度分类
        
    Returns:
        dict: 详细识别结果
    """
    global ocr_instance
    
    # 如果OCR实例未加载，尝试加载
    if ocr_instance is None:
        if not load_ocr():
            return {'text': '', 'confidence': 0.0, 'chars': [], 'char_confidences': []}
    
    try:
        # 图像预处理
        if len(img.shape) == 3 and img.shape[2] == 3:
            # BGR转RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        # 执行OCR识别
        result = ocr_instance.ocr(img_rgb, cls=use_cls)
        
        if not result or not result[0]:
            return {'text': '', 'confidence': 0.0, 'chars': [], 'char_confidences': []}
        
        # 提取识别结果
        text_info = result[0]
        if not text_info:
            return {'text': '', 'confidence': 0.0, 'chars': [], 'char_confidences': []}
        
        # 解析识别结果
        chars = []
        confidences = []
        
        for line in text_info:
            if len(line) >= 2:
                text, conf = line[1]
                chars.append(text)
                confidences.append(conf)
        
        # 合并文字和计算平均置信度
        final_text = ''.join(chars)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'text': final_text,
            'confidence': avg_confidence,
            'chars': chars,
            'char_confidences': confidences
        }
        
    except Exception as e:
        print(f"车牌识别过程出错: {e}")
        return {'text': '', 'confidence': 0.0, 'chars': [], 'char_confidences': []}

def preprocess_plate_image(img):
    """
    车牌图像预处理
    
    Args:
        img: 输入图像
        
    Returns:
        numpy.ndarray: 预处理后的图像
    """
    # 转换为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 高斯滤波去噪
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # 转回BGR格式
    result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    return result

# 兼容性函数，保持与原框架一致
def recognize(img):
    """
    兼容性函数，返回简单的识别结果
    """
    return recognize_plate(img)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PaddleOCR车牌识别测试')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--use-gpu', action='store_true', help='使用GPU')
    parser.add_argument('--use-cls', action='store_true', help='使用角度分类器')
    parser.add_argument('--preprocess', action='store_true', help='启用图像预处理')
    args = parser.parse_args()
    
    # 加载OCR模型
    load_ocr(use_gpu=args.use_gpu)
    
    # 读取图像
    img = cv2.imread(args.image)
    if img is None:
        print(f"无法读取图像: {args.image}")
        exit(1)
    
    # 图像预处理
    if args.preprocess:
        img = preprocess_plate_image(img)
    
    # 执行识别
    result = recognize_plate_detailed(img, args.use_cls)
    
    print(f"识别结果: {result['text']}")
    print(f"置信度: {result['confidence']:.4f}")
    print(f"字符详情: {result['chars']}")
    print(f"字符置信度: {[f'{c:.4f}' for c in result['char_confidences']]}")
    
    # 显示图像
    cv2.imshow('车牌图像', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
