#!/usr/bin/env python3
"""
清理自动驾驶训练数据中的格式错误或低质量图片。

Usage:
    clean_image.py --tub=<tub_path> [--remove] [--max_brightness=<max>] 
                   [--min_brightness=<min>] [--max_zero_angle_count=<count>]

选项:
    --tub=<tub_path>                  数据目录的路径
    --remove                          是否直接移除问题图片（默认只打印报告）
    --max_brightness=<max>            最大平均亮度 [默认: 250]
    --min_brightness=<min>            最小平均亮度 [默认: 20]
    --max_zero_angle_count=<count>    最大连续零转向角数 [默认: 10]
"""

import os
import json
import glob
from docopt import docopt
import numpy as np
import cv2
from PIL import Image
import re

def get_image_path(tub_path, img_filename):
    """查找图像的正确路径，检查根目录和images子目录"""
    # 首先检查tub根目录
    img_path = os.path.join(tub_path, img_filename)
    if os.path.exists(img_path):
        return img_path
    
    # 检查images子目录
    img_path = os.path.join(tub_path, 'images', img_filename)
    if os.path.exists(img_path):
        return img_path
    
    return None

def is_valid_image(img_path):
    """检查图像是否可以正常打开且有效"""
    if img_path is None:
        return False
    
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"图片验证错误 {img_path}: {e}")
        return False

def check_brightness(img_path, min_brightness=20, max_brightness=250):
    """检查图像亮度是否在合理范围内"""
    if img_path is None:
        return False, 0
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False, 0
        
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 计算平均亮度
        brightness = np.mean(gray)
        
        if brightness < min_brightness or brightness > max_brightness:
            return False, brightness
        return True, brightness
    except Exception as e:
        print(f"亮度检查错误 {img_path}: {e}")
        return False, 0

# [函数保持不变]
def get_zero_angle_sequences(records):
    """检测连续零转向角的序列"""
    zero_sequences = []
    current_seq = []
    
    for index, record in enumerate(records):
        angle = record.get('user/angle', None)
        if angle is not None and abs(angle) < 0.05:  # 判断角度是否接近0
            current_seq.append(index)
        else:
            if len(current_seq) > 0:
                zero_sequences.append(current_seq)
                current_seq = []
    
    # 添加最后一个序列（如果存在）
    if len(current_seq) > 0:
        zero_sequences.append(current_seq)
    
    return zero_sequences

def clean_tub_data(tub_path, remove=False, max_brightness=250, min_brightness=20, max_zero_angle_count=10):
    """清理tub目录中的问题图像"""
    if not os.path.exists(tub_path):
        print(f"目录不存在: {tub_path}")
        return
    
    # 查找所有catalog文件
    catalog_files = glob.glob(os.path.join(tub_path, "catalog_*.catalog"))
    if not catalog_files:
        print(f"未找到catalog文件在: {tub_path}")
        return
    
    # 查找manifest.json文件
    manifest_path = os.path.join(tub_path, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"未找到manifest.json文件在: {tub_path}")
        return
    
    # 读取manifest文件
    deleted_indexes = set()
    with open(manifest_path, 'r') as f:
        manifest_content = f.read()
        for line in manifest_content.split('\n'):
            if '"deleted_indexes":' in line:
                # 提取删除的索引列表
                match = re.search(r'"deleted_indexes": \[(.*?)\]', line)
                if match:
                    indexes = match.group(1).strip()
                    if indexes:  # 确保不是空列表
                        deleted_indexes = set(int(idx.strip()) for idx in indexes.split(','))
                break
    
    print(f"已经标记为删除的索引数: {len(deleted_indexes)}")
    
    # 读取所有记录
    all_records = []
    for catalog_file in catalog_files:
        with open(catalog_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    all_records.append(record)
                except json.JSONDecodeError:
                    print(f"无法解析记录: {line}")
    
    print(f"总记录数: {len(all_records)}")
    
    problem_images = []
    
    # 检查每个记录的图像
    for record in all_records:
        index = record.get('_index')
        
        # 如果索引已经在删除列表中，跳过
        if index in deleted_indexes:
            continue
        
        # 获取图像路径
        img_filename = record.get('cam/image_array')
        if not img_filename:
            continue
        
        # 查找图像的正确路径
        img_path = get_image_path(tub_path, img_filename)
        
        # 检查图像是否存在
        if img_path is None:
            problem_images.append((index, os.path.join(tub_path, img_filename), "文件不存在"))
            continue
        
        # 检查图像是否格式有效
        if not is_valid_image(img_path):
            problem_images.append((index, img_path, "图片格式无效"))
            continue
        
        # 检查亮度
        valid_brightness, brightness = check_brightness(img_path, min_brightness, max_brightness)
        if not valid_brightness:
            problem_images.append((index, img_path, f"亮度异常: {brightness:.1f}"))
            continue
    
    # 检查连续零转向角
    zero_sequences = get_zero_angle_sequences(all_records)
    for sequence in zero_sequences:
        if len(sequence) > max_zero_angle_count:
            # 计算序列前后的转向角平均值
            pre_seq_angles = []
            post_seq_angles = []
            
            # 获取序列前5个记录的转向角
            for i in range(max(0, sequence[0]-5), sequence[0]):
                if i < len(all_records):
                    angle = all_records[i].get('user/angle', None)
                    if angle is not None:
                        pre_seq_angles.append(abs(angle))
            
            # 获取序列后5个记录的转向角
            for i in range(sequence[-1]+1, min(len(all_records), sequence[-1]+6)):
                angle = all_records[i].get('user/angle', None)
                if angle is not None:
                    post_seq_angles.append(abs(angle))
            
            # 如果序列前后有明显转向（可能是转弯区域），则不标记为问题
            pre_avg = sum(pre_seq_angles) / len(pre_seq_angles) if pre_seq_angles else 0
            post_avg = sum(post_seq_angles) / len(post_seq_angles) if post_seq_angles else 0
            
            # 如果前后有一个位置是转弯，则保留该序列
            if pre_avg > 0.1 or post_avg > 0.1:
                continue  # 跳过此序列，不标记为问题
                
            # 标记超出允许长度的零转向角序列
            for idx in sequence:
                record = all_records[idx]
                rec_idx = record.get('_index')
                if rec_idx not in deleted_indexes and rec_idx not in [p[0] for p in problem_images]:
                    img_filename = record.get('cam/image_array')
                    if img_filename:
                        img_path = get_image_path(tub_path, img_filename)
                        if img_path:
                            problem_images.append((rec_idx, img_path, "连续零转向角序列"))
    
    # 输出结果
    if problem_images:
        print(f"发现 {len(problem_images)} 个问题图像:")
        for index, path, reason in problem_images:
            print(f"索引 {index}: {path} - {reason}")
        
        if remove:
            # 更新manifest.json文件以标记需要删除的索引
            new_deleted = [index for index, _, _ in problem_images]
            deleted_indexes.update(new_deleted)
            
            # 更新deleted_indexes
            pattern = r'"deleted_indexes": \[.*?\]'
            replacement = f'"deleted_indexes": {json.dumps(sorted(list(deleted_indexes)))}'
            new_content = re.sub(pattern, replacement, manifest_content)
            
            # 写回manifest文件
            with open(manifest_path, 'w') as f:
                f.write(new_content)
            
            print(f"已将 {len(new_deleted)} 个问题图像添加到删除列表")
    else:
        print("未发现问题图像")

def main():
    args = docopt(__doc__)
    tub_path = args['--tub']
    remove = args['--remove']
    max_brightness = int(args.get('--max_brightness') or 250)
    min_brightness = int(args.get('--min_brightness') or 20)
    max_zero_angle_count = int(args.get('--max_zero_angle_count') or 25)
    
    clean_tub_data(tub_path, remove, max_brightness, min_brightness, max_zero_angle_count)
    
if __name__ == "__main__":
    main()