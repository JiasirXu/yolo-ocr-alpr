"""
透视变换工具函数
用于车牌图像的透视校正和几何变换
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

def order_points(pts):
    """
    对四个点进行排序，按照左上、右上、右下、左下的顺序
    
    Args:
        pts: 四个点的坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
    Returns:
        numpy.ndarray: 排序后的点坐标
    """
    # 初始化坐标数组
    rect = np.zeros((4, 2), dtype="float32")
    
    # 计算点的和，左上角的点和最小，右下角的点和最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上角
    rect[2] = pts[np.argmax(s)]  # 右下角
    
    # 计算点的差，右上角的点差最小，左下角的点差最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上角
    rect[3] = pts[np.argmax(diff)]  # 左下角
    
    return rect

def four_points_transform(image, pts):
    """
    对图像进行四点透视变换
    
    Args:
        image: 输入图像
        pts: 四个角点坐标
        
    Returns:
        numpy.ndarray: 透视变换后的图像
    """
    # 获取有序的坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # 计算新图像的宽度
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # 计算新图像的高度
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # 构建目标点集合
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # 计算透视变换矩阵并应用
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def auto_perspective_correction(image, contour):
    """
    自动透视校正
    
    Args:
        image: 输入图像
        contour: 轮廓点
        
    Returns:
        numpy.ndarray: 校正后的图像，如果失败返回原图像
    """
    try:
        # 近似轮廓为多边形
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 如果近似后有4个点，进行透视变换
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            return four_points_transform(image, pts)
        else:
            # 如果不是四边形，使用最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            return four_points_transform(image, box)
            
    except Exception as e:
        print(f"透视校正失败: {e}")
        return image

def correct_plate_perspective(plate_image, debug=False):
    """
    车牌透视校正
    
    Args:
        plate_image: 车牌图像
        debug: 是否显示调试信息
        
    Returns:
        numpy.ndarray: 校正后的车牌图像
    """
    try:
        # 转换为灰度图
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return plate_image
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 检查轮廓面积是否足够大
        contour_area = cv2.contourArea(largest_contour)
        image_area = gray.shape[0] * gray.shape[1]
        
        if contour_area < image_area * 0.1:  # 轮廓面积小于图像面积的10%
            return plate_image
        
        # 进行透视校正
        corrected = auto_perspective_correction(plate_image, largest_contour)
        
        if debug:
            print(f"轮廓面积: {contour_area}, 图像面积: {image_area}")
            cv2.imshow("Original", plate_image)
            cv2.imshow("Threshold", thresh)
            cv2.imshow("Corrected", corrected)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return corrected
        
    except Exception as e:
        print(f"车牌透视校正失败: {e}")
        return plate_image

def resize_with_aspect_ratio(image, target_width=None, target_height=None, inter=cv2.INTER_AREA):
    """
    保持宽高比的图像缩放
    
    Args:
        image: 输入图像
        target_width: 目标宽度
        target_height: 目标高度
        inter: 插值方法
        
    Returns:
        numpy.ndarray: 缩放后的图像
    """
    # 获取原始尺寸
    (h, w) = image.shape[:2]
    
    # 如果目标宽度和高度都为None，返回原图像
    if target_width is None and target_height is None:
        return image
    
    # 如果只指定了宽度
    if target_width is not None and target_height is None:
        r = target_width / float(w)
        dim = (target_width, int(h * r))
    
    # 如果只指定了高度
    elif target_height is not None and target_width is None:
        r = target_height / float(h)
        dim = (int(w * r), target_height)
    
    # 如果同时指定了宽度和高度，选择较小的缩放比例
    else:
        r_w = target_width / float(w)
        r_h = target_height / float(h)
        r = min(r_w, r_h)
        dim = (int(w * r), int(h * r))
    
    # 执行缩放
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def rotate_image(image, angle):
    """
    旋转图像
    
    Args:
        image: 输入图像
        angle: 旋转角度（度）
        
    Returns:
        numpy.ndarray: 旋转后的图像
    """
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    
    # 计算旋转中心
    center = (w // 2, h // 2)
    
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算新的边界尺寸
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # 调整旋转矩阵的平移部分
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # 执行旋转
    rotated = cv2.warpAffine(image, M, (new_w, new_h))
    return rotated

def detect_skew_angle(image):
    """
    检测图像的倾斜角度
    
    Args:
        image: 输入图像
        
    Returns:
        float: 倾斜角度（度）
    """
    try:
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 霍夫直线检测
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return 0.0
        
        # 计算角度
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if -45 <= angle <= 45:
                angles.append(angle)
        
        if not angles:
            return 0.0
        
        # 返回角度的中位数
        return np.median(angles)
        
    except Exception as e:
        print(f"倾斜角度检测失败: {e}")
        return 0.0

def auto_rotate_image(image):
    """
    自动旋转图像以校正倾斜
    
    Args:
        image: 输入图像
        
    Returns:
        numpy.ndarray: 校正后的图像
    """
    try:
        # 检测倾斜角度
        angle = detect_skew_angle(image)
        
        # 如果角度很小，不需要旋转
        if abs(angle) < 1.0:
            return image
        
        # 旋转图像
        rotated = rotate_image(image, -angle)
        return rotated
        
    except Exception as e:
        print(f"自动旋转失败: {e}")
        return image

if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description='透视变换工具测试')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--mode', type=str, choices=['perspective', 'rotate', 'resize'], 
                       default='perspective', help='处理模式')
    args = parser.parse_args()
    
    # 读取图像
    image = cv2.imread(args.image)
    if image is None:
        print(f"无法读取图像: {args.image}")
        exit(1)
    
    if args.mode == 'perspective':
        # 透视校正
        result = correct_plate_perspective(image, debug=True)
    elif args.mode == 'rotate':
        # 自动旋转
        result = auto_rotate_image(image)
        print(f"检测到的倾斜角度: {detect_skew_angle(image):.2f}度")
    elif args.mode == 'resize':
        # 缩放
        result = resize_with_aspect_ratio(image, target_width=400)
    
    # 显示结果
    cv2.imshow('原图像', image)
    cv2.imshow('处理结果', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
