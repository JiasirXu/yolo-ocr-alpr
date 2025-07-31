"""
ALPR系统评估工具
提供车牌检测和识别的性能评估功能
"""

import os
import cv2
import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        float: IoU值
    """
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # 交集面积
    intersection = (x2 - x1) * (y2 - y1)
    
    # 并集面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_detection_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    计算检测指标
    
    Args:
        pred_boxes: 预测边界框列表 [(x1,y1,x2,y2,conf), ...]
        gt_boxes: 真实边界框列表 [(x1,y1,x2,y2), ...]
        iou_threshold: IoU阈值
        
    Returns:
        dict: 包含TP, FP, FN的字典
    """
    if not pred_boxes and not gt_boxes:
        return {'tp': 0, 'fp': 0, 'fn': 0}
    
    if not pred_boxes:
        return {'tp': 0, 'fp': 0, 'fn': len(gt_boxes)}
    
    if not gt_boxes:
        return {'tp': 0, 'fp': len(pred_boxes), 'fn': 0}
    
    # 按置信度排序预测框
    pred_boxes = sorted(pred_boxes, key=lambda x: x[4] if len(x) > 4 else 1.0, reverse=True)
    
    tp = 0
    fp = 0
    matched_gt = set()
    
    for pred_box in pred_boxes:
        pred_coords = pred_box[:4]
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            
            iou = calculate_iou(pred_coords, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    fn = len(gt_boxes) - len(matched_gt)
    
    return {'tp': tp, 'fp': fp, 'fn': fn}

def calculate_recognition_metrics(pred_texts, gt_texts):
    """
    计算识别指标
    
    Args:
        pred_texts: 预测文本列表
        gt_texts: 真实文本列表
        
    Returns:
        dict: 包含各种识别指标的字典
    """
    if len(pred_texts) != len(gt_texts):
        print(f"警告: 预测文本数量({len(pred_texts)})与真实文本数量({len(gt_texts)})不匹配")
        min_len = min(len(pred_texts), len(gt_texts))
        pred_texts = pred_texts[:min_len]
        gt_texts = gt_texts[:min_len]
    
    if not pred_texts:
        return {
            'exact_match': 0.0,
            'char_accuracy': 0.0,
            'edit_distance': 0.0,
            'total_samples': 0
        }
    
    exact_matches = 0
    total_chars = 0
    correct_chars = 0
    total_edit_distance = 0
    
    for pred, gt in zip(pred_texts, gt_texts):
        # 精确匹配
        if pred == gt:
            exact_matches += 1
        
        # 字符级准确率
        total_chars += len(gt)
        for i, char in enumerate(gt):
            if i < len(pred) and pred[i] == char:
                correct_chars += 1
        
        # 编辑距离
        edit_dist = levenshtein_distance(pred, gt)
        total_edit_distance += edit_dist
    
    return {
        'exact_match': exact_matches / len(pred_texts),
        'char_accuracy': correct_chars / total_chars if total_chars > 0 else 0.0,
        'avg_edit_distance': total_edit_distance / len(pred_texts),
        'total_samples': len(pred_texts)
    }

def levenshtein_distance(s1, s2):
    """
    计算两个字符串的编辑距离
    
    Args:
        s1: 字符串1
        s2: 字符串2
        
    Returns:
        int: 编辑距离
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

class ALPREvaluator:
    """
    ALPR系统评估器
    """
    
    def __init__(self, engine):
        """
        初始化评估器
        
        Args:
            engine: ALPR引擎实例
        """
        self.engine = engine
        self.results = {
            'detection': defaultdict(list),
            'recognition': defaultdict(list),
            'timing': defaultdict(list)
        }
    
    def evaluate_dataset(self, dataset_path, gt_file, iou_threshold=0.5):
        """
        评估整个数据集
        
        Args:
            dataset_path: 数据集路径
            gt_file: 真实标注文件路径
            iou_threshold: IoU阈值
            
        Returns:
            dict: 评估结果
        """
        # 加载真实标注
        gt_data = self._load_ground_truth(gt_file)
        
        if not gt_data:
            print("无法加载真实标注数据")
            return None
        
        print(f"开始评估数据集，共 {len(gt_data)} 个样本")
        
        all_detection_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
        all_pred_texts = []
        all_gt_texts = []
        all_times = []
        
        for item in tqdm(gt_data, desc="评估进度"):
            image_path = Path(dataset_path) / item['image']
            
            if not image_path.exists():
                continue
            
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            # 执行识别
            start_time = time.time()
            results = self.engine.process_frame(image)
            process_time = time.time() - start_time
            
            all_times.append(process_time)
            
            # 提取预测结果
            pred_boxes = []
            pred_texts = []
            
            for (x1, y1, x2, y2), text, ocr_conf, det_conf in results:
                pred_boxes.append((x1, y1, x2, y2, det_conf))
                pred_texts.append(text)
            
            # 计算检测指标
            gt_boxes = item.get('boxes', [])
            detection_metrics = calculate_detection_metrics(pred_boxes, gt_boxes, iou_threshold)
            
            all_detection_metrics['tp'] += detection_metrics['tp']
            all_detection_metrics['fp'] += detection_metrics['fp']
            all_detection_metrics['fn'] += detection_metrics['fn']
            
            # 收集识别文本
            gt_texts = item.get('texts', [])
            all_pred_texts.extend(pred_texts)
            all_gt_texts.extend(gt_texts)
        
        # 计算最终指标
        detection_results = self._calculate_final_detection_metrics(all_detection_metrics)
        recognition_results = calculate_recognition_metrics(all_pred_texts, all_gt_texts)
        timing_results = {
            'avg_time': np.mean(all_times),
            'std_time': np.std(all_times),
            'min_time': np.min(all_times),
            'max_time': np.max(all_times),
            'avg_fps': 1.0 / np.mean(all_times) if np.mean(all_times) > 0 else 0
        }
        
        return {
            'detection': detection_results,
            'recognition': recognition_results,
            'timing': timing_results,
            'total_samples': len(gt_data)
        }
    
    def _load_ground_truth(self, gt_file):
        """
        加载真实标注数据
        
        Args:
            gt_file: 标注文件路径
            
        Returns:
            list: 标注数据列表
        """
        try:
            if gt_file.endswith('.json'):
                with open(gt_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 假设是简单的文本格式
                gt_data = []
                with open(gt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            gt_data.append({
                                'image': parts[0],
                                'texts': [parts[1]],
                                'boxes': []  # 需要根据实际格式解析
                            })
                return gt_data
        except Exception as e:
            print(f"加载标注文件失败: {e}")
            return []
    
    def _calculate_final_detection_metrics(self, metrics):
        """
        计算最终检测指标
        
        Args:
            metrics: 包含TP, FP, FN的字典
            
        Returns:
            dict: 最终指标
        """
        tp = metrics['tp']
        fp = metrics['fp']
        fn = metrics['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def generate_report(self, results, output_path):
        """
        生成评估报告
        
        Args:
            results: 评估结果
            output_path: 输出路径
        """
        report_content = f"""
# ALPR系统评估报告

## 数据集信息
- 总样本数: {results['total_samples']}

## 检测性能
- 精确率 (Precision): {results['detection']['precision']:.4f}
- 召回率 (Recall): {results['detection']['recall']:.4f}
- F1分数: {results['detection']['f1_score']:.4f}
- 真正例 (TP): {results['detection']['tp']}
- 假正例 (FP): {results['detection']['fp']}
- 假负例 (FN): {results['detection']['fn']}

## 识别性能
- 精确匹配率: {results['recognition']['exact_match']:.4f}
- 字符准确率: {results['recognition']['char_accuracy']:.4f}
- 平均编辑距离: {results['recognition']['avg_edit_distance']:.2f}
- 识别样本数: {results['recognition']['total_samples']}

## 性能指标
- 平均处理时间: {results['timing']['avg_time']:.4f}秒
- 标准差: {results['timing']['std_time']:.4f}秒
- 最小处理时间: {results['timing']['min_time']:.4f}秒
- 最大处理时间: {results['timing']['max_time']:.4f}秒
- 平均FPS: {results['timing']['avg_fps']:.2f}

## 评估时间
- 评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"评估报告已保存到: {output_path}")
    
    def plot_metrics(self, results, output_dir):
        """
        绘制评估指标图表
        
        Args:
            results: 评估结果
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 检测指标柱状图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 检测指标
        detection_metrics = ['Precision', 'Recall', 'F1-Score']
        detection_values = [
            results['detection']['precision'],
            results['detection']['recall'],
            results['detection']['f1_score']
        ]
        
        ax1.bar(detection_metrics, detection_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('检测性能指标')
        ax1.set_ylabel('分数')
        ax1.set_ylim(0, 1)
        
        for i, v in enumerate(detection_values):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 识别指标
        recognition_metrics = ['精确匹配率', '字符准确率']
        recognition_values = [
            results['recognition']['exact_match'],
            results['recognition']['char_accuracy']
        ]
        
        ax2.bar(recognition_metrics, recognition_values, color=['#96CEB4', '#FFEAA7'])
        ax2.set_title('识别性能指标')
        ax2.set_ylabel('分数')
        ax2.set_ylim(0, 1)
        
        for i, v in enumerate(recognition_values):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"指标图表已保存到: {output_path / 'metrics.png'}")

def main():
    parser = argparse.ArgumentParser(description='ALPR系统评估工具')
    parser.add_argument('--dataset', type=str, required=True, help='数据集路径')
    parser.add_argument('--gt-file', type=str, required=True, help='真实标注文件')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='输出目录')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IoU阈值')
    parser.add_argument('--ocr-backend', type=str, choices=['paddle', 'fastplate'], 
                       default='paddle', help='OCR后端')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 初始化ALPR引擎
    try:
        from pipelines.runtime import AlprEngine
        engine = AlprEngine(ocr_backend=args.ocr_backend)
        print(f"ALPR引擎初始化成功，OCR后端: {args.ocr_backend}")
    except Exception as e:
        print(f"ALPR引擎初始化失败: {e}")
        return
    
    # 创建评估器
    evaluator = ALPREvaluator(engine)
    
    # 执行评估
    print("开始评估...")
    results = evaluator.evaluate_dataset(args.dataset, args.gt_file, args.iou_threshold)
    
    if results is None:
        print("评估失败")
        return
    
    # 生成报告
    report_path = output_path / 'evaluation_report.md'
    evaluator.generate_report(results, report_path)
    
    # 绘制图表
    evaluator.plot_metrics(results, args.output_dir)
    
    # 保存详细结果
    results_path = output_path / 'detailed_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"评估完成，结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
