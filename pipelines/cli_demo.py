"""
ALPR 命令行演示程序
支持图像、视频文件和实时摄像头输入
"""

import cv2
import argparse
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.runtime import AlprEngine

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='ALPR 车牌识别命令行演示',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 摄像头实时识别
  python pipelines/cli_demo.py --source 0
  
  # 图片识别
  python pipelines/cli_demo.py --source image.jpg
  
  # 视频文件识别
  python pipelines/cli_demo.py --source video.mp4 --output result.mp4
  
  # 使用不同OCR后端
  python pipelines/cli_demo.py --source 0 --ocr fastplate
  
  # 调整参数
  python pipelines/cli_demo.py --source 0 --conf 0.5 --min-ocr-conf 0.6
        """
    )
    
    # 输入源
    parser.add_argument('--source', type=str, default='0',
                       help='输入源: 摄像头ID(0,1,2...), 图片路径, 视频路径, 或RTSP流URL')
    
    # OCR后端
    parser.add_argument('--ocr', type=str, choices=['paddle', 'fastplate'], 
                       default='paddle', help='OCR后端选择')
    
    # 检测参数
    parser.add_argument('--detector-weights', type=str, default=None,
                       help='检测器权重文件路径')
    parser.add_argument('--conf', type=float, default=0.35,
                       help='检测置信度阈值')
    parser.add_argument('--imgsz', type=int, default=960,
                       help='检测器输入图像尺寸')
    
    # OCR参数
    parser.add_argument('--min-ocr-conf', type=float, default=0.5,
                       help='OCR最小置信度阈值')
    parser.add_argument('--enhance', action='store_true',
                       help='启用车牌图像增强')
    
    # 输出参数
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频文件路径')
    parser.add_argument('--save-results', action='store_true',
                       help='保存识别结果到JSON文件')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示实时画面')
    
    # 其他参数
    parser.add_argument('--fps-limit', type=int, default=None,
                       help='限制处理FPS（用于降低CPU使用率）')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细信息')
    
    return parser.parse_args()

def is_image_file(path):
    """判断是否为图像文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return Path(path).suffix.lower() in image_extensions

def is_video_file(path):
    """判断是否为视频文件"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'}
    return Path(path).suffix.lower() in video_extensions

def process_image(engine, image_path, args):
    """处理单张图像"""
    print(f"处理图像: {image_path}")
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        return
    
    # 执行识别
    start_time = time.time()
    results = engine.process_frame(image, args.min_ocr_conf, args.enhance)
    process_time = time.time() - start_time
    
    # 打印结果
    print(f"处理时间: {process_time:.3f}秒")
    print(f"检测到 {len(results)} 个车牌:")
    
    for i, ((x1, y1, x2, y2), text, ocr_conf, det_conf) in enumerate(results, 1):
        print(f"  {i}. 位置: ({x1},{y1})-({x2},{y2})")
        print(f"     车牌号: {text}")
        print(f"     OCR置信度: {ocr_conf:.3f}")
        print(f"     检测置信度: {det_conf:.3f}")
    
    # 可视化结果
    if not args.no_display:
        vis_image = engine.visualize_results(image, results)
        
        # 调整显示尺寸
        height, width = vis_image.shape[:2]
        if height > 800:
            scale = 800 / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            vis_image = cv2.resize(vis_image, (new_width, new_height))
        
        cv2.imshow('ALPR 结果', vis_image)
        print("按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 保存结果图像
    if args.output:
        output_path = args.output
        if not output_path.lower().endswith(('.jpg', '.png')):
            output_path += '.jpg'
        
        vis_image = engine.visualize_results(image, results)
        cv2.imwrite(output_path, vis_image)
        print(f"结果图像已保存到: {output_path}")

def process_video_or_camera(engine, source, args):
    """处理视频文件或摄像头"""
    # 确定输入类型
    if source.isdigit():
        source_type = "摄像头"
        cap_source = int(source)
    elif source.startswith(('rtsp://', 'http://', 'https://')):
        source_type = "网络流"
        cap_source = source
    else:
        source_type = "视频文件"
        cap_source = source
        if not os.path.exists(source):
            print(f"错误: 文件不存在 {source}")
            return
    
    print(f"处理{source_type}: {source}")
    
    # 打开视频源
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"错误: 无法打开{source_type}")
        return
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source_type == "视频文件" else -1
    
    print(f"视频属性: {width}x{height}, FPS: {fps:.1f}")
    if total_frames > 0:
        print(f"总帧数: {total_frames}")
    
    # 初始化视频写入器
    writer = None
    if args.output and source_type != "摄像头":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"输出视频: {args.output}")
    
    # 处理参数
    display = not args.no_display
    fps_limit = args.fps_limit
    frame_interval = 1.0 / fps_limit if fps_limit else 0
    
    # 统计变量
    frame_count = 0
    total_detections = 0
    start_time = time.time()
    last_frame_time = 0
    
    print("开始处理... (按 'q' 退出)")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if source_type == "视频文件":
                    print("视频处理完成")
                else:
                    print("摄像头连接断开")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # FPS限制
            if fps_limit and (current_time - last_frame_time) < frame_interval:
                continue
            last_frame_time = current_time
            
            # 执行识别
            results = engine.process_frame(frame, args.min_ocr_conf, args.enhance)
            total_detections += len(results)
            
            # 可视化结果
            vis_frame = engine.visualize_results(frame, results)
            
            # 添加帧信息
            info_text = f"Frame: {frame_count}"
            if total_frames > 0:
                info_text += f"/{total_frames} ({frame_count/total_frames*100:.1f}%)"
            
            cv2.putText(vis_frame, info_text, (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 保存到输出视频
            if writer:
                writer.write(vis_frame)
            
            # 显示画面
            if display:
                cv2.imshow('ALPR 实时识别', vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户退出")
                    break
                elif key == ord('s') and source_type == "摄像头":
                    # 保存当前帧
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    save_path = f"capture_{timestamp}.jpg"
                    cv2.imwrite(save_path, vis_frame)
                    print(f"截图已保存: {save_path}")
            
            # 打印进度信息
            if args.verbose and frame_count % 30 == 0:
                elapsed_time = current_time - start_time
                avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f"已处理 {frame_count} 帧, 平均FPS: {avg_fps:.1f}, "
                      f"总检测数: {total_detections}")
    
    except KeyboardInterrupt:
        print("\n用户中断处理")
    
    finally:
        # 清理资源
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    
    # 打印最终统计
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n=== 处理完成 ===")
    print(f"总帧数: {frame_count}")
    print(f"总检测数: {total_detections}")
    print(f"处理时间: {elapsed_time:.1f}秒")
    print(f"平均FPS: {avg_fps:.1f}")
    
    # 显示引擎统计
    stats = engine.get_stats()
    print(f"\n=== 引擎统计 ===")
    print(f"平均检测时间: {stats['avg_detection_time']*1000:.1f}ms")
    print(f"平均识别时间: {stats['avg_recognition_time']*1000:.1f}ms")
    print(f"平均总处理时间: {stats['avg_total_time']*1000:.1f}ms")

def main():
    """主函数"""
    args = parse_arguments()
    
    print("=== ALPR 车牌识别演示 ===")
    print(f"OCR后端: {args.ocr}")
    print(f"检测置信度: {args.conf}")
    print(f"OCR置信度: {args.min_ocr_conf}")
    print(f"图像增强: {'启用' if args.enhance else '禁用'}")
    
    # 初始化ALPR引擎
    try:
        engine = AlprEngine(
            ocr_backend=args.ocr,
            detector_weights=args.detector_weights,
            detector_conf=args.conf,
            detector_imgsz=args.imgsz
        )
        print("ALPR引擎初始化成功")
    except Exception as e:
        print(f"ALPR引擎初始化失败: {e}")
        return 1
    
    # 根据输入类型选择处理方式
    source = args.source
    
    if os.path.isfile(source) and is_image_file(source):
        # 处理图像文件
        process_image(engine, source, args)
    
    elif os.path.isfile(source) and is_video_file(source):
        # 处理视频文件
        process_video_or_camera(engine, source, args)
    
    elif source.isdigit() or source.startswith(('rtsp://', 'http://', 'https://')):
        # 处理摄像头或网络流
        process_video_or_camera(engine, source, args)
    
    else:
        print(f"错误: 不支持的输入源类型或文件不存在: {source}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
