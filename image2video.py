#!/usr/bin/env python3
import cv2
import os
import argparse
import glob
from natsort import natsorted

def images_to_video(image_folder, output_video, fps=30, img_formats=None):
    """
    将文件夹中的图片序列转换为视频
    
    参数:
        image_folder: 包含图片序列的文件夹路径
        output_video: 输出视频文件路径
        fps: 帧率
        img_formats: 图片格式列表，例如['*.jpg', '*.png']
    """
    if img_formats is None:
        img_formats = ['*.jpg']
    
    # 获取所有支持格式的图片并合并
    images = []
    for fmt in img_formats:
        format_images = glob.glob(os.path.join(image_folder, fmt))
        images.extend(format_images)
    
    # 按自然顺序排序所有图片
    images = natsorted(images)
    
    if not images:
        print(f"在文件夹 {image_folder} 中未找到任何支持格式的图片")
        return
    
    # 读取第一张图片获取图像尺寸
    frame = cv2.imread(images[0])
    height, width, channels = frame.shape
    
    # 定义视频编码器和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4V编码
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # 遍历所有图片并添加到视频
    total_images = len(images)
    for i, image_path in enumerate(images):
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
            continue
        
        video_writer.write(img)
        print(f"处理中... {i+1}/{total_images}: {os.path.basename(image_path)}", end='\r')
    
    # 释放视频写入器
    video_writer.release()
    print(f"\n视频已成功生成: {output_video}")
    print(f"总共处理了 {total_images} 张图片")

def main():
    parser = argparse.ArgumentParser(description='将图片序列转换为视频')
    parser.add_argument('-i', '--input', required=True, help='图片文件夹路径')
    parser.add_argument('-o', '--output', required=True, help='输出视频文件路径')
    parser.add_argument('-f', '--fps', type=int, default=12, help='帧率 (默认: 30)')
    parser.add_argument('--formats', nargs='+', default=['*.jpg'], 
                        help='图片格式列表 (默认: *.jpg，可指定多个格式，如: --formats *.jpg *.png)')
    
    args = parser.parse_args()
    
    images_to_video(args.input, args.output, args.fps, args.formats)

if __name__ == "__main__":
    main()
