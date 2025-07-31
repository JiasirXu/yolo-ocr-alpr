"""
ALPR PyQt线程模块
在独立线程中运行车牌识别，避免阻塞GUI界面
"""

import cv2
import sys
import time
import numpy as np
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from PyQt5.QtGui import QImage, QPixmap

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.runtime import AlprEngine

class AlprThread(QThread):
    """
    ALPR识别线程
    在后台线程中执行车牌识别，通过信号与GUI通信
    """
    
    # 信号定义
    frame_updated = pyqtSignal(QImage)  # 更新帧图像
    results_updated = pyqtSignal(list)  # 更新识别结果
    stats_updated = pyqtSignal(dict)    # 更新统计信息
    error_occurred = pyqtSignal(str)    # 发生错误
    status_changed = pyqtSignal(str)    # 状态变化
    
    def __init__(self, source=0, ocr_backend='paddle', parent=None):
        """
        初始化ALPR线程
        
        Args:
            source: 视频源 (摄像头ID、文件路径或URL)
            ocr_backend: OCR后端 ('paddle' 或 'fastplate')
            parent: 父对象
        """
        super().__init__(parent)
        
        self.source = source
        self.ocr_backend = ocr_backend
        self.running = False
        self.paused = False
        
        # 线程同步
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        
        # 参数设置
        self.detector_conf = 0.35
        self.detector_imgsz = 960
        self.min_ocr_conf = 0.5
        self.enhance_plates = True
        self.fps_limit = None
        
        # ALPR引擎
        self.engine = None
        
        # 统计信息
        self.frame_count = 0
        self.start_time = None
        
    def set_source(self, source):
        """设置视频源"""
        self.source = source
    
    def set_ocr_backend(self, backend):
        """设置OCR后端"""
        if backend != self.ocr_backend:
            self.ocr_backend = backend
            if self.engine:
                self.engine.set_ocr_backend(backend)
    
    def set_detector_params(self, conf=None, imgsz=None):
        """设置检测器参数"""
        if conf is not None:
            self.detector_conf = conf
        if imgsz is not None:
            self.detector_imgsz = imgsz
    
    def set_ocr_params(self, min_conf=None, enhance=None):
        """设置OCR参数"""
        if min_conf is not None:
            self.min_ocr_conf = min_conf
        if enhance is not None:
            self.enhance_plates = enhance
    
    def set_fps_limit(self, fps):
        """设置FPS限制"""
        self.fps_limit = fps
    
    def pause(self):
        """暂停处理"""
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()
        self.status_changed.emit("已暂停")
    
    def resume(self):
        """恢复处理"""
        self.mutex.lock()
        self.paused = False
        self.condition.wakeAll()
        self.mutex.unlock()
        self.status_changed.emit("运行中")
    
    def stop(self):
        """停止线程"""
        self.mutex.lock()
        self.running = False
        self.paused = False
        self.condition.wakeAll()
        self.mutex.unlock()
        self.status_changed.emit("已停止")
    
    def run(self):
        """线程主循环"""
        try:
            self._initialize_engine()
            self._process_video_stream()
        except Exception as e:
            self.error_occurred.emit(f"线程运行错误: {str(e)}")
        finally:
            self.status_changed.emit("已停止")
    
    def _initialize_engine(self):
        """初始化ALPR引擎"""
        try:
            self.engine = AlprEngine(
                ocr_backend=self.ocr_backend,
                detector_conf=self.detector_conf,
                detector_imgsz=self.detector_imgsz
            )
            self.status_changed.emit("引擎初始化成功")
        except Exception as e:
            raise Exception(f"ALPR引擎初始化失败: {str(e)}")
    
    def _process_video_stream(self):
        """处理视频流"""
        # 打开视频源
        cap = self._open_video_source()
        if cap is None:
            return
        
        self.running = True
        self.frame_count = 0
        self.start_time = time.time()
        last_frame_time = 0
        frame_interval = 1.0 / self.fps_limit if self.fps_limit else 0
        
        self.status_changed.emit("开始处理")
        
        try:
            while self.running:
                # 检查暂停状态
                self.mutex.lock()
                if self.paused:
                    self.condition.wait(self.mutex)
                self.mutex.unlock()
                
                if not self.running:
                    break
                
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    if self._is_camera_source():
                        self.error_occurred.emit("摄像头连接断开")
                    else:
                        self.status_changed.emit("视频处理完成")
                    break
                
                current_time = time.time()
                
                # FPS限制
                if self.fps_limit and (current_time - last_frame_time) < frame_interval:
                    continue
                last_frame_time = current_time
                
                # 处理帧
                self._process_frame(frame)
                
                self.frame_count += 1
                
                # 定期更新统计信息
                if self.frame_count % 30 == 0:
                    self._update_stats()
        
        finally:
            cap.release()
    
    def _open_video_source(self):
        """打开视频源"""
        try:
            if isinstance(self.source, str) and self.source.isdigit():
                cap = cv2.VideoCapture(int(self.source))
            else:
                cap = cv2.VideoCapture(self.source)
            
            if not cap.isOpened():
                self.error_occurred.emit(f"无法打开视频源: {self.source}")
                return None
            
            # 设置缓冲区大小（减少延迟）
            if self._is_camera_source():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            return cap
            
        except Exception as e:
            self.error_occurred.emit(f"打开视频源失败: {str(e)}")
            return None
    
    def _is_camera_source(self):
        """判断是否为摄像头源"""
        return (isinstance(self.source, str) and self.source.isdigit()) or isinstance(self.source, int)
    
    def _process_frame(self, frame):
        """处理单帧"""
        try:
            # 执行车牌识别
            results = self.engine.process_frame(
                frame, 
                self.min_ocr_conf, 
                self.enhance_plates
            )
            
            # 可视化结果
            vis_frame = self.engine.visualize_results(frame, results)
            
            # 转换为QImage并发送信号
            qimage = self._cv2_to_qimage(vis_frame)
            self.frame_updated.emit(qimage)
            
            # 发送识别结果
            self.results_updated.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(f"帧处理错误: {str(e)}")
    
    def _cv2_to_qimage(self, cv_img):
        """将OpenCV图像转换为QImage"""
        try:
            # 确保图像是BGR格式
            if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
                # BGR转RGB
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv_img
            
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            return qimage
            
        except Exception as e:
            print(f"图像转换错误: {e}")
            # 返回空白图像
            return QImage(640, 480, QImage.Format_RGB888)
    
    def _update_stats(self):
        """更新统计信息"""
        if self.engine and self.start_time:
            engine_stats = self.engine.get_stats()
            elapsed_time = time.time() - self.start_time
            
            stats = {
                'frame_count': self.frame_count,
                'elapsed_time': elapsed_time,
                'avg_fps': self.frame_count / elapsed_time if elapsed_time > 0 else 0,
                'total_detections': engine_stats.get('total_detections', 0),
                'avg_detection_time': engine_stats.get('avg_detection_time', 0) * 1000,  # ms
                'avg_recognition_time': engine_stats.get('avg_recognition_time', 0) * 1000,  # ms
                'avg_total_time': engine_stats.get('avg_total_time', 0) * 1000,  # ms
                'ocr_backend': self.ocr_backend
            }
            
            self.stats_updated.emit(stats)

class ImageProcessThread(QThread):
    """
    图像处理线程
    用于处理单张图像的车牌识别
    """
    
    # 信号定义
    processing_finished = pyqtSignal(QImage, list)  # 处理完成
    error_occurred = pyqtSignal(str)  # 发生错误
    
    def __init__(self, image_path, ocr_backend='paddle', parent=None):
        """
        初始化图像处理线程
        
        Args:
            image_path: 图像文件路径
            ocr_backend: OCR后端
            parent: 父对象
        """
        super().__init__(parent)
        
        self.image_path = image_path
        self.ocr_backend = ocr_backend
        
        # 参数设置
        self.detector_conf = 0.35
        self.detector_imgsz = 960
        self.min_ocr_conf = 0.5
        self.enhance_plates = True
    
    def set_params(self, detector_conf=None, detector_imgsz=None, 
                   min_ocr_conf=None, enhance_plates=None):
        """设置处理参数"""
        if detector_conf is not None:
            self.detector_conf = detector_conf
        if detector_imgsz is not None:
            self.detector_imgsz = detector_imgsz
        if min_ocr_conf is not None:
            self.min_ocr_conf = min_ocr_conf
        if enhance_plates is not None:
            self.enhance_plates = enhance_plates
    
    def run(self):
        """线程主函数"""
        try:
            # 初始化ALPR引擎
            engine = AlprEngine(
                ocr_backend=self.ocr_backend,
                detector_conf=self.detector_conf,
                detector_imgsz=self.detector_imgsz
            )
            
            # 读取图像
            image = cv2.imread(self.image_path)
            if image is None:
                self.error_occurred.emit(f"无法读取图像: {self.image_path}")
                return
            
            # 执行识别
            results = engine.process_frame(
                image, 
                self.min_ocr_conf, 
                self.enhance_plates
            )
            
            # 可视化结果
            vis_image = engine.visualize_results(image, results)
            
            # 转换为QImage
            qimage = self._cv2_to_qimage(vis_image)
            
            # 发送结果
            self.processing_finished.emit(qimage, results)
            
        except Exception as e:
            self.error_occurred.emit(f"图像处理错误: {str(e)}")
    
    def _cv2_to_qimage(self, cv_img):
        """将OpenCV图像转换为QImage"""
        try:
            if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv_img
            
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            return qimage
            
        except Exception as e:
            print(f"图像转换错误: {e}")
            return QImage(640, 480, QImage.Format_RGB888)
