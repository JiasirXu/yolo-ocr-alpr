"""
Fast Plate OCR 车牌识别模块
基于fast-plate-ocr库的车牌识别，支持多种预训练模型
"""

import os
import cv2
import numpy as np

# 全局OCR实例
ocr_instance = None

def load_ocr(model_name='cct-xs-v1-global-model', device='auto'):
    """
    加载Fast Plate OCR模型
    
    Args:
        model_name: 模型名称，可选:
                   - 'cct-xs-v1-global-model' (全球模型，小尺寸)
                   - 'cct-small-v1-global-model' (全球模型，小尺寸)
                   - 'cct-base-v1-global-model' (全球模型，基础尺寸)
                   - 'cct-large-v1-global-model' (全球模型，大尺寸)
        device: 设备选择 ('auto', 'cpu', 'cuda')
    """
    global ocr_instance
    
    try:
        from fast_plate_ocr import LicensePlateRecognizer
        
        # 初始化识别器
        ocr_instance = LicensePlateRecognizer(
            model_name=model_name,
            device=device
        )
        
        print(f"Fast Plate OCR模型加载成功: {model_name}")
        return True
        
    except ImportError:
        print("错误: fast-plate-ocr库未安装")
        print("请运行: pip install fast-plate-ocr")
        return False
    except Exception as e:
        print(f"Fast Plate OCR模型加载失败: {e}")
        return False

def recognize_plate(img):
    """
    识别车牌图像中的文字
    
    Args:
        img: 输入图像 (numpy array)
        
    Returns:
        tuple: (识别文字, 置信度)
    """
    global ocr_instance
    
    # 如果OCR实例未加载，尝试加载默认模型
    if ocr_instance is None:
        if not load_ocr():
            return '', 0.0
    
    try:
        # 执行识别
        result = ocr_instance.run(img)
        
        if isinstance(result, tuple) and len(result) >= 2:
            text, confidence = result[0], result[1]
            return text, float(confidence)
        elif isinstance(result, str):
            # 某些版本可能只返回文字
            return result, 1.0
        else:
            return '', 0.0
            
    except Exception as e:
        print(f"车牌识别过程出错: {e}")
        return '', 0.0

def recognize_plate_batch(imgs):
    """
    批量识别车牌图像
    
    Args:
        imgs: 图像列表
        
    Returns:
        list: 识别结果列表 [(text, confidence), ...]
    """
    global ocr_instance
    
    # 如果OCR实例未加载，尝试加载默认模型
    if ocr_instance is None:
        if not load_ocr():
            return [('', 0.0)] * len(imgs)
    
    results = []
    for img in imgs:
        try:
            result = ocr_instance.run(img)
            
            if isinstance(result, tuple) and len(result) >= 2:
                text, confidence = result[0], result[1]
                results.append((text, float(confidence)))
            elif isinstance(result, str):
                results.append((result, 1.0))
            else:
                results.append(('', 0.0))
                
        except Exception as e:
            print(f"批量识别过程出错: {e}")
            results.append(('', 0.0))
    
    return results

def preprocess_plate_image(img, target_height=64):
    """
    车牌图像预处理，适配Fast Plate OCR输入要求
    
    Args:
        img: 输入图像
        target_height: 目标高度
        
    Returns:
        numpy.ndarray: 预处理后的图像
    """
    # 获取原始尺寸
    h, w = img.shape[:2]
    
    # 计算新的宽度，保持宽高比
    aspect_ratio = w / h
    target_width = int(target_height * aspect_ratio)
    
    # 调整图像尺寸
    resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    
    # 转换为RGB格式（如果是BGR）
    if len(resized.shape) == 3 and resized.shape[2] == 3:
        # 假设输入是BGR，转换为RGB
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    else:
        rgb_img = resized
    
    return rgb_img

def enhance_plate_image(img):
    """
    增强车牌图像质量
    
    Args:
        img: 输入图像
        
    Returns:
        numpy.ndarray: 增强后的图像
    """
    # 转换为灰度图进行处理
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 高斯滤波去噪
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # 锐化
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # 转回彩色图像
    if len(img.shape) == 3:
        result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    else:
        result = sharpened
    
    return result

def get_available_models():
    """
    获取可用的模型列表
    
    Returns:
        list: 可用模型名称列表
    """
    models = [
        'cct-xs-v1-global-model',
        'cct-small-v1-global-model', 
        'cct-base-v1-global-model',
        'cct-large-v1-global-model'
    ]
    return models

def benchmark_models(img, models=None):
    """
    对比不同模型的识别效果
    
    Args:
        img: 测试图像
        models: 要测试的模型列表，None表示测试所有模型
        
    Returns:
        dict: 各模型的识别结果
    """
    if models is None:
        models = get_available_models()
    
    results = {}
    
    for model_name in models:
        print(f"测试模型: {model_name}")
        
        try:
            # 加载模型
            if load_ocr(model_name):
                # 执行识别
                text, conf = recognize_plate(img)
                results[model_name] = {
                    'text': text,
                    'confidence': conf,
                    'success': True
                }
                print(f"  结果: {text} (置信度: {conf:.4f})")
            else:
                results[model_name] = {
                    'text': '',
                    'confidence': 0.0,
                    'success': False
                }
                print(f"  加载失败")
                
        except Exception as e:
            results[model_name] = {
                'text': '',
                'confidence': 0.0,
                'success': False,
                'error': str(e)
            }
            print(f"  错误: {e}")
    
    return results

# 兼容性函数，保持与原框架一致
def recognize(img):
    """
    兼容性函数，返回简单的识别结果
    """
    return recognize_plate(img)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast Plate OCR车牌识别测试')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--model', type=str, default='cct-xs-v1-global-model', 
                       choices=get_available_models(), help='使用的模型')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda'], help='运行设备')
    parser.add_argument('--enhance', action='store_true', help='启用图像增强')
    parser.add_argument('--benchmark', action='store_true', help='对比所有模型')
    args = parser.parse_args()
    
    # 读取图像
    img = cv2.imread(args.image)
    if img is None:
        print(f"无法读取图像: {args.image}")
        exit(1)
    
    # 图像增强
    if args.enhance:
        img = enhance_plate_image(img)
        print("已启用图像增强")
    
    # 图像预处理
    img = preprocess_plate_image(img)
    
    if args.benchmark:
        # 对比所有模型
        print("=== 模型对比测试 ===")
        results = benchmark_models(img)
        
        print("\n=== 对比结果 ===")
        for model, result in results.items():
            if result['success']:
                print(f"{model}: {result['text']} (置信度: {result['confidence']:.4f})")
            else:
                print(f"{model}: 识别失败")
    
    else:
        # 单模型测试
        print(f"使用模型: {args.model}")
        
        # 加载模型
        if load_ocr(args.model, args.device):
            # 执行识别
            text, confidence = recognize_plate(img)
            
            print(f"识别结果: {text}")
            print(f"置信度: {confidence:.4f}")
        else:
            print("模型加载失败")
    
    # 显示图像
    display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
    cv2.imshow('车牌图像', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
