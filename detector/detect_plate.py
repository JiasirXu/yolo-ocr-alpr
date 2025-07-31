"""
YOLOv10 车牌检测模块
支持热插拔权重文件，可通过环境变量或直接指定权重路径
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

# 默认权重路径，可通过环境变量覆盖
DEFAULT_WEIGHT = 'detector/weights/yolov10s.pt'
WEIGHT_PATH = os.getenv('YOLO_WEIGHTS', DEFAULT_WEIGHT)

# 全局模型实例
model = None

def load_model(weight_path=None):
    """
    加载YOLO模型
    Args:
        weight_path: 权重文件路径，如果为None则使用默认路径
    """
    global model
    if weight_path is None:
        weight_path = WEIGHT_PATH
    
    try:
        model = YOLO(weight_path)
        print(f"成功加载模型权重: {weight_path}")
        return True
    except Exception as e:
        print(f"加载模型失败: {e}")
        return False

def detect_plates(img, conf=0.35, imgsz=960, iou=0.45):
    """
    检测图像中的车牌
    Args:
        img: 输入图像 (numpy array 或 路径字符串)
        conf: 置信度阈值
        imgsz: 输入图像尺寸
        iou: NMS IoU阈值
    Returns:
        list: 检测到的车牌边界框列表 [(x1,y1,x2,y2), ...]
    """
    global model
    
    # 如果模型未加载，尝试加载默认模型
    if model is None:
        if not load_model():
            return []
    
    try:
        # 执行检测
        results = model(img, imgsz=imgsz, conf=conf, iou=iou)
        
        # 提取边界框
        boxes = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    # 获取边界框坐标并转换为整数
                    xyxy = box.xyxy.cpu().numpy().astype(int)
                    if len(xyxy) > 0:
                        x1, y1, x2, y2 = xyxy[0]
                        boxes.append((x1, y1, x2, y2))
        
        return boxes
    
    except Exception as e:
        print(f"检测过程出错: {e}")
        return []

def detect_plates_with_conf(img, conf=0.35, imgsz=960, iou=0.45):
    """
    检测图像中的车牌，返回边界框和置信度
    Args:
        img: 输入图像 (numpy array 或 路径字符串)
        conf: 置信度阈值
        imgsz: 输入图像尺寸
        iou: NMS IoU阈值
    Returns:
        list: 检测结果列表 [((x1,y1,x2,y2), confidence), ...]
    """
    global model
    
    # 如果模型未加载，尝试加载默认模型
    if model is None:
        if not load_model():
            return []
    
    try:
        # 执行检测
        results = model(img, imgsz=imgsz, conf=conf, iou=iou)
        
        # 提取边界框和置信度
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    # 获取边界框坐标和置信度
                    xyxy = box.xyxy.cpu().numpy().astype(int)
                    confidence = float(box.conf.cpu().numpy())
                    
                    if len(xyxy) > 0:
                        x1, y1, x2, y2 = xyxy[0]
                        detections.append(((x1, y1, x2, y2), confidence))
        
        return detections
    
    except Exception as e:
        print(f"检测过程出错: {e}")
        return []

def visualize_detections(img, detections, show_conf=True):
    """
    在图像上可视化检测结果
    Args:
        img: 输入图像
        detections: 检测结果，格式为 [((x1,y1,x2,y2), conf), ...] 或 [(x1,y1,x2,y2), ...]
        show_conf: 是否显示置信度
    Returns:
        numpy.ndarray: 标注后的图像
    """
    img_vis = img.copy()
    
    for detection in detections:
        if isinstance(detection, tuple) and len(detection) == 2:
            # 格式: ((x1,y1,x2,y2), conf)
            (x1, y1, x2, y2), conf = detection
            label = f"Plate {conf:.2f}" if show_conf else "Plate"
        else:
            # 格式: (x1,y1,x2,y2)
            x1, y1, x2, y2 = detection
            label = "Plate"
        
        # 绘制边界框
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img_vis, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(img_vis, label, (x1, y1 - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return img_vis

# 兼容性函数，保持与原框架一致
def detect(img, conf=0.35, imgsz=960):
    """
    兼容性函数，返回简单的边界框列表
    """
    return detect_plates(img, conf, imgsz)

if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description='车牌检测测试')
    parser.add_argument('--source', type=str, default='0', help='输入源 (图片路径、视频路径或摄像头ID)')
    parser.add_argument('--weights', type=str, default=None, help='模型权重路径')
    parser.add_argument('--conf', type=float, default=0.35, help='置信度阈值')
    parser.add_argument('--imgsz', type=int, default=960, help='输入图像尺寸')
    args = parser.parse_args()
    
    # 加载模型
    if args.weights:
        load_model(args.weights)
    else:
        load_model()
    
    # 处理输入源
    if args.source.isdigit():
        # 摄像头
        cap = cv2.VideoCapture(int(args.source))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = detect_plates_with_conf(frame, args.conf, args.imgsz)
            frame_vis = visualize_detections(frame, detections)
            
            cv2.imshow('车牌检测', frame_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        # 图片或视频文件
        if args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # 图片
            img = cv2.imread(args.source)
            if img is not None:
                detections = detect_plates_with_conf(img, args.conf, args.imgsz)
                img_vis = visualize_detections(img, detections)
                
                cv2.imshow('车牌检测', img_vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"无法读取图片: {args.source}")
        
        else:
            # 视频文件
            cap = cv2.VideoCapture(args.source)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                detections = detect_plates_with_conf(frame, args.conf, args.imgsz)
                frame_vis = visualize_detections(frame, detections)
                
                cv2.imshow('车牌检测', frame_vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
