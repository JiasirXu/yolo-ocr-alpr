"""
ALPR 核心运行时引擎
整合车牌检测和OCR识别，提供统一的车牌识别接口
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Optional, Union

# 导入检测模块
from detector.detect_plate import detect_plates_with_conf, load_model as load_detector

# 导入OCR模块
from ocr.paddle.run_paddleocr import recognize_plate as paddle_recognize, load_ocr as load_paddle_ocr
from ocr.fastplate.run_fastplate import recognize_plate as fastplate_recognize, load_ocr as load_fastplate_ocr

# 导入工具函数
from utils.perspective import four_points_transform, order_points

class AlprEngine:
    """
    车牌识别引擎
    整合检测和OCR功能，提供完整的车牌识别流水线
    """
    
    def __init__(self, ocr_backend='paddle', detector_weights=None, 
                 detector_conf=0.35, detector_imgsz=960):
        """
        初始化ALPR引擎
        
        Args:
            ocr_backend: OCR后端 ('paddle' 或 'fastplate')
            detector_weights: 检测器权重路径
            detector_conf: 检测置信度阈值
            detector_imgsz: 检测器输入图像尺寸
        """
        self.ocr_backend = ocr_backend
        self.detector_conf = detector_conf
        self.detector_imgsz = detector_imgsz
        
        # 初始化检测器
        self._init_detector(detector_weights)
        
        # 初始化OCR
        self._init_ocr(ocr_backend)
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_recognitions': 0,
            'avg_detection_time': 0.0,
            'avg_recognition_time': 0.0,
            'avg_total_time': 0.0
        }
    
    def _init_detector(self, weights_path=None):
        """初始化检测器"""
        try:
            if weights_path:
                success = load_detector(weights_path)
            else:
                success = load_detector()
            
            if success:
                print(f"检测器初始化成功")
            else:
                print("检测器初始化失败")
                
        except Exception as e:
            print(f"检测器初始化错误: {e}")
    
    def _init_ocr(self, backend):
        """初始化OCR"""
        try:
            if backend == 'paddle':
                success = load_paddle_ocr()
                self.recognize_func = paddle_recognize
            elif backend == 'fastplate':
                success = load_fastplate_ocr()
                self.recognize_func = fastplate_recognize
            else:
                raise ValueError(f"不支持的OCR后端: {backend}")
            
            if success:
                print(f"OCR后端 {backend} 初始化成功")
            else:
                print(f"OCR后端 {backend} 初始化失败")
                
        except Exception as e:
            print(f"OCR初始化错误: {e}")
            # 使用默认函数作为后备
            self.recognize_func = lambda img: ('', 0.0)
    
    def set_ocr_backend(self, backend):
        """
        切换OCR后端
        
        Args:
            backend: 新的OCR后端 ('paddle' 或 'fastplate')
        """
        if backend != self.ocr_backend:
            self.ocr_backend = backend
            self._init_ocr(backend)
            print(f"OCR后端已切换到: {backend}")
    
    def detect_plates(self, frame):
        """
        检测车牌
        
        Args:
            frame: 输入图像
            
        Returns:
            list: 检测结果 [((x1,y1,x2,y2), confidence), ...]
        """
        start_time = time.time()
        
        try:
            detections = detect_plates_with_conf(
                frame, 
                conf=self.detector_conf, 
                imgsz=self.detector_imgsz
            )
            
            detection_time = time.time() - start_time
            self._update_detection_stats(detection_time, len(detections))
            
            return detections
            
        except Exception as e:
            print(f"车牌检测错误: {e}")
            return []
    
    def recognize_plate_text(self, plate_img):
        """
        识别车牌文字
        
        Args:
            plate_img: 车牌图像
            
        Returns:
            tuple: (识别文字, 置信度)
        """
        start_time = time.time()
        
        try:
            text, confidence = self.recognize_func(plate_img)
            
            recognition_time = time.time() - start_time
            self._update_recognition_stats(recognition_time)
            
            return text, confidence
            
        except Exception as e:
            print(f"车牌识别错误: {e}")
            return '', 0.0
    
    def process_frame(self, frame, min_conf=0.5, enhance_plates=True):
        """
        处理单帧图像，执行完整的车牌识别流程
        
        Args:
            frame: 输入图像
            min_conf: 最小置信度阈值
            enhance_plates: 是否增强车牌图像
            
        Returns:
            list: 识别结果 [((x1,y1,x2,y2), text, confidence, det_conf), ...]
        """
        start_time = time.time()
        results = []
        
        try:
            # 1. 检测车牌
            detections = self.detect_plates(frame)
            
            # 2. 对每个检测到的车牌进行OCR识别
            for (x1, y1, x2, y2), det_conf in detections:
                # 裁剪车牌区域
                plate_img = frame[y1:y2, x1:x2]
                
                if plate_img.size == 0:
                    continue
                
                # 图像增强（可选）
                if enhance_plates:
                    plate_img = self._enhance_plate_image(plate_img)
                
                # OCR识别
                text, ocr_conf = self.recognize_plate_text(plate_img)
                
                # 过滤低置信度结果
                if ocr_conf >= min_conf and text.strip():
                    results.append(((x1, y1, x2, y2), text, ocr_conf, det_conf))
            
            # 更新统计信息
            total_time = time.time() - start_time
            self._update_total_stats(total_time)
            
            return results
            
        except Exception as e:
            print(f"帧处理错误: {e}")
            return []
    
    def process_video(self, video_source, output_path=None, display=True, 
                     save_results=True, min_conf=0.5):
        """
        处理视频流
        
        Args:
            video_source: 视频源 (文件路径、摄像头ID或URL)
            output_path: 输出视频路径
            display: 是否显示实时结果
            save_results: 是否保存识别结果
            min_conf: 最小置信度阈值
        """
        # 打开视频源
        if isinstance(video_source, str) and video_source.isdigit():
            cap = cv2.VideoCapture(int(video_source))
        else:
            cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"无法打开视频源: {video_source}")
            return
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 初始化视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 结果记录
        all_results = []
        frame_count = 0
        
        print(f"开始处理视频: {video_source}")
        print(f"分辨率: {width}x{height}, FPS: {fps}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 处理当前帧
                results = self.process_frame(frame, min_conf)
                
                # 可视化结果
                vis_frame = self.visualize_results(frame, results)
                
                # 保存结果
                if save_results:
                    frame_results = {
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'detections': results
                    }
                    all_results.append(frame_results)
                
                # 写入输出视频
                if writer:
                    writer.write(vis_frame)
                
                # 显示实时结果
                if display:
                    cv2.imshow('ALPR', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 打印进度
                if frame_count % 30 == 0:
                    print(f"已处理 {frame_count} 帧")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        print(f"视频处理完成，共处理 {frame_count} 帧")
        
        # 保存识别结果
        if save_results and all_results:
            self._save_results(all_results, output_path)
        
        return all_results
    
    def visualize_results(self, frame, results):
        """
        可视化识别结果
        
        Args:
            frame: 原始图像
            results: 识别结果
            
        Returns:
            numpy.ndarray: 标注后的图像
        """
        vis_frame = frame.copy()
        
        for (x1, y1, x2, y2), text, ocr_conf, det_conf in results:
            # 绘制边界框
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 准备标签文本
            label = f"{text} ({ocr_conf:.2f})"
            
            # 计算文本尺寸
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 绘制文本背景
            cv2.rectangle(vis_frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), (0, 255, 0), -1)
            
            # 绘制文本
            cv2.putText(vis_frame, label, (x1, y1 - 8), 
                       font, font_scale, (0, 0, 0), thickness)
        
        # 添加统计信息
        self._draw_stats(vis_frame)
        
        return vis_frame
    
    def _enhance_plate_image(self, plate_img):
        """增强车牌图像"""
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
            if len(plate_img.shape) == 3:
                result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            else:
                result = denoised
            
            return result
            
        except Exception as e:
            print(f"图像增强错误: {e}")
            return plate_img
    
    def _update_detection_stats(self, detection_time, num_detections):
        """更新检测统计信息"""
        self.stats['total_detections'] += num_detections
        
        # 更新平均检测时间
        total_frames = self.stats['total_frames']
        if total_frames > 0:
            self.stats['avg_detection_time'] = (
                (self.stats['avg_detection_time'] * total_frames + detection_time) / 
                (total_frames + 1)
            )
        else:
            self.stats['avg_detection_time'] = detection_time
    
    def _update_recognition_stats(self, recognition_time):
        """更新识别统计信息"""
        self.stats['total_recognitions'] += 1
        
        # 更新平均识别时间
        total_recs = self.stats['total_recognitions']
        if total_recs > 1:
            self.stats['avg_recognition_time'] = (
                (self.stats['avg_recognition_time'] * (total_recs - 1) + recognition_time) / 
                total_recs
            )
        else:
            self.stats['avg_recognition_time'] = recognition_time
    
    def _update_total_stats(self, total_time):
        """更新总体统计信息"""
        self.stats['total_frames'] += 1
        
        # 更新平均总处理时间
        total_frames = self.stats['total_frames']
        if total_frames > 1:
            self.stats['avg_total_time'] = (
                (self.stats['avg_total_time'] * (total_frames - 1) + total_time) / 
                total_frames
            )
        else:
            self.stats['avg_total_time'] = total_time
    
    def _draw_stats(self, frame):
        """在图像上绘制统计信息"""
        stats_text = [
            f"OCR: {self.ocr_backend}",
            f"Frames: {self.stats['total_frames']}",
            f"Detections: {self.stats['total_detections']}",
            f"Avg Time: {self.stats['avg_total_time']*1000:.1f}ms"
        ]
        
        y_offset = 30
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _save_results(self, results, output_path):
        """保存识别结果到文件"""
        import json
        
        if output_path:
            result_path = output_path.replace('.mp4', '_results.json')
        else:
            result_path = 'alpr_results.json'
        
        try:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"识别结果已保存到: {result_path}")
            
        except Exception as e:
            print(f"保存结果失败: {e}")
    
    def get_stats(self):
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_recognitions': 0,
            'avg_detection_time': 0.0,
            'avg_recognition_time': 0.0,
            'avg_total_time': 0.0
        }
