"""
ALPR PyQt5 主界面
提供图形化的车牌识别界面，支持实时摄像头、图像和视频文件处理
"""

import sys
import os
import time
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QComboBox, 
                            QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit,
                            QFileDialog, QGroupBox, QGridLayout, QSplitter,
                            QTableWidget, QTableWidgetItem, QHeaderView,
                            QProgressBar, QStatusBar, QMenuBar, QAction,
                            QMessageBox, QSlider, QFrame)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QPixmap, QFont, QIcon, QKeySequence

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gui.alpr_thread import AlprThread, ImageProcessThread

class MainWindow(QMainWindow):
    """
    ALPR主窗口类
    提供完整的图形化车牌识别界面
    """
    
    def __init__(self):
        super().__init__()
        
        # 窗口属性
        self.setWindowTitle('ALPR 车牌识别系统')
        self.setGeometry(100, 100, 1400, 900)
        
        # 线程对象
        self.alpr_thread = None
        self.image_thread = None
        
        # 当前识别结果
        self.current_results = []
        
        # 统计信息
        self.total_detections = 0
        self.session_start_time = None
        
        # 初始化界面
        self._init_ui()
        self._init_menu()
        self._init_status_bar()
        
        # 启动定时器更新界面
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(100)  # 100ms更新一次
        
        print("ALPR主界面初始化完成")
    
    def _init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧控制面板
        control_panel = self._create_control_panel()
        splitter.addWidget(control_panel)
        
        # 右侧显示区域
        display_area = self._create_display_area()
        splitter.addWidget(display_area)
        
        # 设置分割器比例
        splitter.setSizes([400, 1000])
    
    def _create_control_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 输入源设置
        input_group = self._create_input_group()
        layout.addWidget(input_group)
        
        # OCR设置
        ocr_group = self._create_ocr_group()
        layout.addWidget(ocr_group)
        
        # 检测设置
        detection_group = self._create_detection_group()
        layout.addWidget(detection_group)
        
        # 控制按钮
        control_group = self._create_control_group()
        layout.addWidget(control_group)
        
        # 统计信息
        stats_group = self._create_stats_group()
        layout.addWidget(stats_group)
        
        # 添加弹性空间
        layout.addStretch()
        
        return panel
    
    def _create_input_group(self):
        """创建输入源设置组"""
        group = QGroupBox("输入源设置")
        layout = QGridLayout(group)
        
        # 输入类型选择
        layout.addWidget(QLabel("输入类型:"), 0, 0)
        self.input_type_combo = QComboBox()
        self.input_type_combo.addItems(["摄像头", "图像文件", "视频文件"])
        self.input_type_combo.currentTextChanged.connect(self._on_input_type_changed)
        layout.addWidget(self.input_type_combo, 0, 1)
        
        # 摄像头ID
        layout.addWidget(QLabel("摄像头ID:"), 1, 0)
        self.camera_id_spin = QSpinBox()
        self.camera_id_spin.setRange(0, 10)
        self.camera_id_spin.setValue(0)
        layout.addWidget(self.camera_id_spin, 1, 1)
        
        # 文件路径
        layout.addWidget(QLabel("文件路径:"), 2, 0)
        self.file_path_label = QLabel("未选择文件")
        self.file_path_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        layout.addWidget(self.file_path_label, 2, 1)
        
        # 浏览文件按钮
        self.browse_button = QPushButton("浏览文件")
        self.browse_button.clicked.connect(self._browse_file)
        self.browse_button.setEnabled(False)
        layout.addWidget(self.browse_button, 3, 0, 1, 2)
        
        return group
    
    def _create_ocr_group(self):
        """创建OCR设置组"""
        group = QGroupBox("OCR设置")
        layout = QGridLayout(group)
        
        # OCR后端选择
        layout.addWidget(QLabel("OCR后端:"), 0, 0)
        self.ocr_backend_combo = QComboBox()
        self.ocr_backend_combo.addItems(["paddle", "fastplate"])
        layout.addWidget(self.ocr_backend_combo, 0, 1)
        
        # OCR置信度阈值
        layout.addWidget(QLabel("置信度阈值:"), 1, 0)
        self.ocr_conf_spin = QDoubleSpinBox()
        self.ocr_conf_spin.setRange(0.0, 1.0)
        self.ocr_conf_spin.setSingleStep(0.05)
        self.ocr_conf_spin.setValue(0.5)
        self.ocr_conf_spin.setDecimals(2)
        layout.addWidget(self.ocr_conf_spin, 1, 1)
        
        # 图像增强
        self.enhance_checkbox = QCheckBox("启用图像增强")
        self.enhance_checkbox.setChecked(True)
        layout.addWidget(self.enhance_checkbox, 2, 0, 1, 2)
        
        return group
    
    def _create_detection_group(self):
        """创建检测设置组"""
        group = QGroupBox("检测设置")
        layout = QGridLayout(group)
        
        # 检测置信度阈值
        layout.addWidget(QLabel("检测置信度:"), 0, 0)
        self.det_conf_spin = QDoubleSpinBox()
        self.det_conf_spin.setRange(0.0, 1.0)
        self.det_conf_spin.setSingleStep(0.05)
        self.det_conf_spin.setValue(0.35)
        self.det_conf_spin.setDecimals(2)
        layout.addWidget(self.det_conf_spin, 0, 1)
        
        # 输入图像尺寸
        layout.addWidget(QLabel("图像尺寸:"), 1, 0)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(960)
        layout.addWidget(self.imgsz_spin, 1, 1)
        
        # FPS限制
        layout.addWidget(QLabel("FPS限制:"), 2, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(30)
        self.fps_spin.setSpecialValueText("无限制")
        layout.addWidget(self.fps_spin, 2, 1)
        
        return group
    
    def _create_control_group(self):
        """创建控制按钮组"""
        group = QGroupBox("控制")
        layout = QVBoxLayout(group)
        
        # 开始/停止按钮
        self.start_button = QPushButton("开始识别")
        self.start_button.clicked.connect(self._start_recognition)
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        layout.addWidget(self.start_button)
        
        # 暂停/恢复按钮
        self.pause_button = QPushButton("暂停")
        self.pause_button.clicked.connect(self._pause_resume)
        self.pause_button.setEnabled(False)
        layout.addWidget(self.pause_button)
        
        # 停止按钮
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self._stop_recognition)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        layout.addWidget(self.stop_button)
        
        # 截图按钮
        self.screenshot_button = QPushButton("截图")
        self.screenshot_button.clicked.connect(self._take_screenshot)
        self.screenshot_button.setEnabled(False)
        layout.addWidget(self.screenshot_button)
        
        return group
    
    def _create_stats_group(self):
        """创建统计信息组"""
        group = QGroupBox("统计信息")
        layout = QGridLayout(group)
        
        # 创建标签
        self.stats_labels = {
            'frame_count': QLabel("0"),
            'fps': QLabel("0.0"),
            'detections': QLabel("0"),
            'avg_time': QLabel("0.0ms"),
            'ocr_backend': QLabel("paddle")
        }
        
        # 添加到布局
        layout.addWidget(QLabel("处理帧数:"), 0, 0)
        layout.addWidget(self.stats_labels['frame_count'], 0, 1)
        
        layout.addWidget(QLabel("平均FPS:"), 1, 0)
        layout.addWidget(self.stats_labels['fps'], 1, 1)
        
        layout.addWidget(QLabel("检测总数:"), 2, 0)
        layout.addWidget(self.stats_labels['detections'], 2, 1)
        
        layout.addWidget(QLabel("平均耗时:"), 3, 0)
        layout.addWidget(self.stats_labels['avg_time'], 3, 1)
        
        layout.addWidget(QLabel("OCR后端:"), 4, 0)
        layout.addWidget(self.stats_labels['ocr_backend'], 4, 1)

        return group

    def _create_display_area(self):
        """创建右侧显示区域"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 创建分割器
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        # 图像显示区域
        image_frame = QFrame()
        image_frame.setFrameStyle(QFrame.StyledPanel)
        image_layout = QVBoxLayout(image_frame)

        self.image_label = QLabel("等待输入...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("border: 2px dashed #ccc; background-color: #f9f9f9;")
        image_layout.addWidget(self.image_label)

        splitter.addWidget(image_frame)

        # 结果显示区域
        results_frame = QFrame()
        results_frame.setFrameStyle(QFrame.StyledPanel)
        results_layout = QVBoxLayout(results_frame)

        results_layout.addWidget(QLabel("识别结果:"))

        # 结果表格
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(["序号", "车牌号", "OCR置信度", "检测置信度", "位置"])

        # 设置表格属性
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)

        results_layout.addWidget(self.results_table)

        splitter.addWidget(results_frame)

        # 设置分割器比例
        splitter.setSizes([600, 300])

        return widget

    def _init_menu(self):
        """初始化菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu('文件')

        # 打开图像
        open_image_action = QAction('打开图像', self)
        open_image_action.setShortcut(QKeySequence.Open)
        open_image_action.triggered.connect(self._open_image_file)
        file_menu.addAction(open_image_action)

        # 打开视频
        open_video_action = QAction('打开视频', self)
        open_video_action.triggered.connect(self._open_video_file)
        file_menu.addAction(open_video_action)

        file_menu.addSeparator()

        # 保存结果
        save_results_action = QAction('保存结果', self)
        save_results_action.setShortcut(QKeySequence.Save)
        save_results_action.triggered.connect(self._save_results)
        file_menu.addAction(save_results_action)

        file_menu.addSeparator()

        # 退出
        exit_action = QAction('退出', self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 设置菜单
        settings_menu = menubar.addMenu('设置')

        # 重置统计
        reset_stats_action = QAction('重置统计', self)
        reset_stats_action.triggered.connect(self._reset_stats)
        settings_menu.addAction(reset_stats_action)

        # 帮助菜单
        help_menu = menubar.addMenu('帮助')

        # 关于
        about_action = QAction('关于', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _init_status_bar(self):
        """初始化状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    # 事件处理方法
    @pyqtSlot(str)
    def _on_input_type_changed(self, input_type):
        """输入类型改变事件"""
        if input_type == "摄像头":
            self.camera_id_spin.setEnabled(True)
            self.browse_button.setEnabled(False)
            self.file_path_label.setText("使用摄像头")
        else:
            self.camera_id_spin.setEnabled(False)
            self.browse_button.setEnabled(True)
            self.file_path_label.setText("未选择文件")

    def _browse_file(self):
        """浏览文件"""
        input_type = self.input_type_combo.currentText()

        if input_type == "图像文件":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图像文件", "",
                "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*)"
            )
        elif input_type == "视频文件":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频文件", "",
                "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*)"
            )
        else:
            return

        if file_path:
            self.file_path_label.setText(os.path.basename(file_path))
            self.file_path_label.setToolTip(file_path)

    def _start_recognition(self):
        """开始识别"""
        try:
            # 获取输入源
            source = self._get_input_source()
            if source is None:
                return

            # 获取参数
            ocr_backend = self.ocr_backend_combo.currentText()

            # 根据输入类型选择处理方式
            input_type = self.input_type_combo.currentText()

            if input_type == "图像文件":
                self._process_image(source, ocr_backend)
            else:
                self._start_video_processing(source, ocr_backend)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动识别失败: {str(e)}")

    def _get_input_source(self):
        """获取输入源"""
        input_type = self.input_type_combo.currentText()

        if input_type == "摄像头":
            return self.camera_id_spin.value()
        else:
            file_path = self.file_path_label.toolTip()
            if not file_path or file_path == "未选择文件":
                QMessageBox.warning(self, "警告", "请先选择文件")
                return None

            if not os.path.exists(file_path):
                QMessageBox.warning(self, "警告", "文件不存在")
                return None

            return file_path

    def _process_image(self, image_path, ocr_backend):
        """处理单张图像"""
        # 停止当前线程
        self._stop_current_threads()

        # 创建图像处理线程
        self.image_thread = ImageProcessThread(image_path, ocr_backend)

        # 设置参数
        self.image_thread.set_params(
            detector_conf=self.det_conf_spin.value(),
            detector_imgsz=self.imgsz_spin.value(),
            min_ocr_conf=self.ocr_conf_spin.value(),
            enhance_plates=self.enhance_checkbox.isChecked()
        )

        # 连接信号
        self.image_thread.processing_finished.connect(self._on_image_processed)
        self.image_thread.error_occurred.connect(self._on_error)

        # 启动线程
        self.image_thread.start()

        # 更新界面状态
        self._set_processing_state(True)
        self.status_label.setText("正在处理图像...")

    def _start_video_processing(self, source, ocr_backend):
        """开始视频处理"""
        # 停止当前线程
        self._stop_current_threads()

        # 创建ALPR线程
        self.alpr_thread = AlprThread(source, ocr_backend)

        # 设置参数
        self.alpr_thread.set_detector_params(
            conf=self.det_conf_spin.value(),
            imgsz=self.imgsz_spin.value()
        )
        self.alpr_thread.set_ocr_params(
            min_conf=self.ocr_conf_spin.value(),
            enhance=self.enhance_checkbox.isChecked()
        )
        self.alpr_thread.set_fps_limit(self.fps_spin.value() if self.fps_spin.value() > 0 else None)

        # 连接信号
        self.alpr_thread.frame_updated.connect(self._on_frame_updated)
        self.alpr_thread.results_updated.connect(self._on_results_updated)
        self.alpr_thread.stats_updated.connect(self._on_stats_updated)
        self.alpr_thread.error_occurred.connect(self._on_error)
        self.alpr_thread.status_changed.connect(self._on_status_changed)

        # 启动线程
        self.alpr_thread.start()

        # 更新界面状态
        self._set_processing_state(True)
        self.session_start_time = time.time()

    def _pause_resume(self):
        """暂停/恢复处理"""
        if self.alpr_thread and self.alpr_thread.isRunning():
            if self.pause_button.text() == "暂停":
                self.alpr_thread.pause()
                self.pause_button.setText("恢复")
            else:
                self.alpr_thread.resume()
                self.pause_button.setText("暂停")

    def _stop_recognition(self):
        """停止识别"""
        self._stop_current_threads()
        self._set_processing_state(False)
        self.status_label.setText("已停止")

    def _stop_current_threads(self):
        """停止当前运行的线程"""
        if self.alpr_thread and self.alpr_thread.isRunning():
            self.alpr_thread.stop()
            self.alpr_thread.wait(3000)  # 等待3秒
            if self.alpr_thread.isRunning():
                self.alpr_thread.terminate()

        if self.image_thread and self.image_thread.isRunning():
            self.image_thread.terminate()
            self.image_thread.wait(1000)

    def _set_processing_state(self, processing):
        """设置处理状态"""
        self.start_button.setEnabled(not processing)
        self.pause_button.setEnabled(processing and self.alpr_thread is not None)
        self.stop_button.setEnabled(processing)
        self.screenshot_button.setEnabled(processing and self.alpr_thread is not None)

    def _take_screenshot(self):
        """截图"""
        if self.image_label.pixmap():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"

            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存截图", filename,
                "PNG文件 (*.png);;JPEG文件 (*.jpg);;所有文件 (*)"
            )

            if file_path:
                self.image_label.pixmap().save(file_path)
                QMessageBox.information(self, "成功", f"截图已保存到: {file_path}")

    # 信号槽方法
    @pyqtSlot(QImage)
    def _on_frame_updated(self, qimage):
        """帧更新事件"""
        # 缩放图像以适应显示区域
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    @pyqtSlot(list)
    def _on_results_updated(self, results):
        """识别结果更新事件"""
        self.current_results = results
        self._update_results_table()

    @pyqtSlot(dict)
    def _on_stats_updated(self, stats):
        """统计信息更新事件"""
        self.stats_labels['frame_count'].setText(str(stats.get('frame_count', 0)))
        self.stats_labels['fps'].setText(f"{stats.get('avg_fps', 0):.1f}")
        self.stats_labels['detections'].setText(str(stats.get('total_detections', 0)))
        self.stats_labels['avg_time'].setText(f"{stats.get('avg_total_time', 0):.1f}ms")
        self.stats_labels['ocr_backend'].setText(stats.get('ocr_backend', 'unknown'))

    @pyqtSlot(str)
    def _on_error(self, error_msg):
        """错误事件"""
        QMessageBox.critical(self, "错误", error_msg)
        self._set_processing_state(False)
        self.status_label.setText("发生错误")

    @pyqtSlot(str)
    def _on_status_changed(self, status):
        """状态变化事件"""
        self.status_label.setText(status)

    @pyqtSlot(QImage, list)
    def _on_image_processed(self, qimage, results):
        """图像处理完成事件"""
        # 显示结果图像
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

        # 更新结果
        self.current_results = results
        self._update_results_table()

        # 更新状态
        self._set_processing_state(False)
        self.status_label.setText(f"图像处理完成，检测到 {len(results)} 个车牌")

    def _update_results_table(self):
        """更新结果表格"""
        self.results_table.setRowCount(len(self.current_results))

        for i, ((x1, y1, x2, y2), text, ocr_conf, det_conf) in enumerate(self.current_results):
            # 序号
            self.results_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))

            # 车牌号
            self.results_table.setItem(i, 1, QTableWidgetItem(text))

            # OCR置信度
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{ocr_conf:.3f}"))

            # 检测置信度
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{det_conf:.3f}"))

            # 位置
            position = f"({x1},{y1})-({x2},{y2})"
            self.results_table.setItem(i, 4, QTableWidgetItem(position))

    def _update_display(self):
        """定时更新显示"""
        # 这里可以添加需要定时更新的内容
        pass

    # 菜单事件处理
    def _open_image_file(self):
        """打开图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*)"
        )

        if file_path:
            self.input_type_combo.setCurrentText("图像文件")
            self.file_path_label.setText(os.path.basename(file_path))
            self.file_path_label.setToolTip(file_path)

    def _open_video_file(self):
        """打开视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*)"
        )

        if file_path:
            self.input_type_combo.setCurrentText("视频文件")
            self.file_path_label.setText(os.path.basename(file_path))
            self.file_path_label.setToolTip(file_path)

    def _save_results(self):
        """保存结果"""
        if not self.current_results:
            QMessageBox.information(self, "提示", "没有可保存的结果")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"alpr_results_{timestamp}.txt"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", filename,
            "文本文件 (*.txt);;CSV文件 (*.csv);;所有文件 (*)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("ALPR识别结果\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"总检测数: {len(self.current_results)}\n\n")

                    for i, ((x1, y1, x2, y2), text, ocr_conf, det_conf) in enumerate(self.current_results, 1):
                        f.write(f"结果 {i}:\n")
                        f.write(f"  车牌号: {text}\n")
                        f.write(f"  OCR置信度: {ocr_conf:.3f}\n")
                        f.write(f"  检测置信度: {det_conf:.3f}\n")
                        f.write(f"  位置: ({x1},{y1})-({x2},{y2})\n\n")

                QMessageBox.information(self, "成功", f"结果已保存到: {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def _reset_stats(self):
        """重置统计"""
        if self.alpr_thread and hasattr(self.alpr_thread, 'engine') and self.alpr_thread.engine:
            self.alpr_thread.engine.reset_stats()

        # 重置界面显示
        for label in self.stats_labels.values():
            if label != self.stats_labels['ocr_backend']:
                label.setText("0" if "count" in label.objectName() else "0.0")

        self.total_detections = 0
        self.session_start_time = time.time()

        QMessageBox.information(self, "提示", "统计信息已重置")

    def _show_about(self):
        """显示关于对话框"""
        about_text = """
        <h3>ALPR 车牌识别系统</h3>
        <p>版本: 1.0.0</p>
        <p>基于YOLOv10检测和PaddleOCR/FastPlateOCR识别的车牌识别系统</p>
        <p>支持实时摄像头、图像文件和视频文件处理</p>
        <br>
        <p><b>主要功能:</b></p>
        <ul>
        <li>实时车牌检测和识别</li>
        <li>支持多种输入源</li>
        <li>可切换OCR后端</li>
        <li>参数可调节</li>
        <li>结果导出</li>
        </ul>
        """

        QMessageBox.about(self, "关于", about_text)

    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止所有线程
        self._stop_current_threads()

        # 停止定时器
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()

        event.accept()

def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用程序属性
    app.setApplicationName("ALPR车牌识别系统")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("ALPR Team")

    # 创建主窗口
    window = MainWindow()
    window.show()

    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
