#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MP4视频文件合并工具
使用OpenCV将多个MP4文件合并为一个
"""

import os
import sys
from pathlib import Path
import cv2
import argparse


def get_video_info(video_path):
    """获取视频的基本信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'fps': fps,
        'width': width,
        'height': height,
        'frame_count': frame_count,
        'duration': duration
    }


def merge_videos(input_files, output_file, target_fps=None, target_size=None):
    """
    合并多个视频文件
    
    Args:
        input_files (list): 输入视频文件路径列表
        output_file (str): 输出视频文件路径
        target_fps (float): 目标帧率，None表示使用第一个视频的帧率
        target_size (tuple): 目标分辨率(width, height)，None表示使用第一个视频的分辨率
    """
    print(f"开始合并 {len(input_files)} 个视频文件...")
    
    # 验证输入文件并获取视频信息
    video_infos = []
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在 - {file_path}")
            continue
            
        if not file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"警告: 跳过不支持的视频格式 - {file_path}")
            continue
            
        info = get_video_info(file_path)
        if info is None:
            print(f"错误: 无法读取视频信息 - {file_path}")
            continue
            
        video_infos.append((file_path, info))
        print(f"已加载: {file_path} (时长: {info['duration']:.2f}秒, {info['width']}x{info['height']}, {info['fps']:.2f}fps)")
    
    if not video_infos:
        print("错误: 没有有效的视频文件可以合并")
        return False
    
    # 确定输出视频的参数
    first_info = video_infos[0][1]
    output_fps = target_fps if target_fps else first_info['fps']
    output_width = target_size[0] if target_size else first_info['width']
    output_height = target_size[1] if target_size else first_info['height']
    
    print(f"输出视频参数: {output_width}x{output_height}, {output_fps:.2f}fps")
    
    # 检查所有视频的参数是否一致
    inconsistent_videos = []
    for file_path, info in video_infos:
        if info['width'] != output_width or info['height'] != output_height or abs(info['fps'] - output_fps) > 0.1:
            inconsistent_videos.append(file_path)
    
    if inconsistent_videos:
        print("警告: 以下视频的参数与输出参数不一致，将进行缩放/帧率转换:")
        for video in inconsistent_videos:
            print(f"  - {video}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, output_fps, (output_width, output_height))
    
    if not out.isOpened():
        print(f"错误: 无法创建输出视频文件 - {output_file}")
        return False
    
    try:
        total_frames_processed = 0
        total_frames = sum(info['frame_count'] for _, info in video_infos)
        
        for i, (file_path, info) in enumerate(video_infos):
            print(f"处理第 {i+1}/{len(video_infos)} 个视频: {os.path.basename(file_path)}")
            
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"错误: 无法打开视频文件 - {file_path}")
                continue
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 调整帧大小
                if frame.shape[1] != output_width or frame.shape[0] != output_height:
                    frame = cv2.resize(frame, (output_width, output_height))
                
                # 写入帧
                out.write(frame)
                frame_count += 1
                total_frames_processed += 1
                
                # 显示进度
                if frame_count % 30 == 0:  # 每30帧显示一次进度
                    progress = (total_frames_processed / total_frames) * 100
                    print(f"进度: {progress:.1f}% ({total_frames_processed}/{total_frames} 帧)")
            
            cap.release()
            print(f"完成: {os.path.basename(file_path)} ({frame_count} 帧)")
        
        out.release()
        print(f"✅ 视频合并完成: {output_file}")
        print(f"总共处理了 {total_frames_processed} 帧")
        return True
        
    except Exception as e:
        print(f"❌ 合并失败: {e}")
        out.release()
        # 删除不完整的输出文件
        if os.path.exists(output_file):
            os.remove(output_file)
        return False


def get_video_files_in_directory(directory, extensions=('.mp4', '.avi', '.mov', '.mkv')):
    """获取目录中的所有视频文件"""
    video_files = []
    directory_path = Path(directory)
    
    for ext in extensions:
        video_files.extend(directory_path.glob(f"*{ext}"))
        video_files.extend(directory_path.glob(f"*{ext.upper()}"))
    
    return sorted([str(f) for f in video_files])


def main():
    parser = argparse.ArgumentParser(description='合并多个视频文件')
    parser.add_argument('--output', default='merged_video.mp4', help='输出文件名 (默认: merged_video.mp4)')
    parser.add_argument('--directory', default='video_path', help='指定目录，合并该目录下所有视频文件')
    parser.add_argument('--fps', type=float, help='目标帧率 (默认使用第一个视频的帧率)')
    parser.add_argument('--resolution', help='目标分辨率，格式: WIDTHxHEIGHT (例如: 1920x1080)')
    
    args = parser.parse_args()
    
    # 解析分辨率参数
    target_size = None
    if args.resolution:
        try:
            width, height = map(int, args.resolution.split('x'))
            target_size = (width, height)
        except ValueError:
            print("错误: 分辨率格式不正确，应为 WIDTHxHEIGHT (例如: 1920x1080)")
            return
    
    # 获取输入文件
    input_files = []
    
    if args.directory and args.directory != 'directory':
        # 从目录获取视频文件
        input_files = get_video_files_in_directory(args.directory)
        if not input_files:
            print(f"在目录 {args.directory} 中未找到视频文件")
            return
        print(f"在目录 {args.directory} 中找到 {len(input_files)} 个视频文件")
    
    # 检查输入文件
    if len(input_files) < 2:
        print("错误: 至少需要2个视频文件才能合并")
        return
    
    # 显示将要合并的文件
    print("\n将要合并的文件:")
    for i, file_path in enumerate(input_files, 1):
        print(f"{i}. {file_path}")
    
    # 确认输出文件名
    if args.directory == 'directory':
        output_file = input(f"\n输出文件名 (默认: {args.output}): ").strip()
        if output_file:
            args.output = output_file
    
    print(f"\n输出文件: {args.output}")
    if args.fps:
        print(f"目标帧率: {args.fps} fps")
    if target_size:
        print(f"目标分辨率: {target_size[0]}x{target_size[1]}")
    
    # 执行合并
    success = merge_videos(input_files, args.output, args.fps, target_size)
    
    if success:
        # 显示文件信息
        if os.path.exists(args.output):
            file_size = os.path.getsize(args.output) / (1024 * 1024)  # MB
            print(f"输出文件大小: {file_size:.2f} MB")
            
            # 显示输出视频信息
            output_info = get_video_info(args.output)
            if output_info:
                print(f"输出视频信息: {output_info['width']}x{output_info['height']}, "
                      f"{output_info['duration']:.2f}秒, {output_info['fps']:.2f}fps")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序出错: {e}")
