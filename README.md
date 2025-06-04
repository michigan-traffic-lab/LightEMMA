# 视频合并工具 (OpenCV版)

这是一个用Python编写的视频文件合并工具，使用OpenCV库将多个视频文件合并为一个。

## 支持格式

- MP4
- AVI
- MOV
- MKV

## 安装依赖

首先安装所需的Python包：

```bash
pip install -r requirements.txt
```

或者直接安装OpenCV：

```bash
pip install opencv-python
```

## 使用方法

### 1. 交互式模式

直接运行脚本，按提示操作：

```bash
python merge_videos.py
```

### 2. 目录模式

#### 合并目录中所有视频文件
```bash
python merge_videos.py --directory /path/to/video/directory --output merged_output.mp4
```

#### 指定输出帧率
```bash
python merge_videos.py --directory /path/to/videos --output output.mp4 --fps 30
```

#### 指定输出分辨率
```bash
python merge_videos.py --directory /path/to/videos --output output.mp4 --resolution 1920x1080
```

## 参数说明

- `--output`: 输出文件名（默认：merged_video.mp4）
- `--directory`: 指定目录，合并该目录下所有视频文件
- `--fps`: 目标帧率（默认使用第一个视频的帧率）
- `--resolution`: 目标分辨率，格式为WIDTHxHEIGHT（默认使用第一个视频的分辨率）

## 使用示例

### 示例1：合并当前目录的所有视频文件
```bash
python merge_videos.py --directory . --output all_videos_merged.mp4
```

### 示例2：合并指定目录的视频文件
```bash
python merge_videos.py --directory /home/user/videos --output complete_movie.mp4
```

### 示例3：指定输出参数
```bash
python merge_videos.py --directory ./clips --output final.mp4 --fps 25 --resolution 1280x720
```

### 示例4：交互式模式示例
```bash
python merge_videos.py
# 然后按提示选择模式1（手动输入文件）或模式2（当前目录所有视频）
```

## 特性

1. **多格式支持**: 支持MP4, AVI, MOV, MKV等常见视频格式  
2. **自动格式统一**: 自动将所有视频转换为统一的帧率和分辨率
3. **实时进度显示**: 显示合并进度和处理信息
4. **参数自定义**: 可以指定输出的帧率和分辨率
5. **错误处理**: 自动处理文件错误和格式不兼容问题
6. **交互式操作**: 支持交互式选择文件或目录

## 注意事项

1. **文件格式**: 支持MP4, AVI, MOV, MKV格式的视频文件
2. **文件顺序**: 视频会按照文件名字母顺序进行合并（目录模式）或输入顺序（交互模式）
3. **音频处理**: **注意：OpenCV版本不包含音频处理，输出视频将没有音频**
4. **处理时间**: 大文件的合并可能需要较长时间，请耐心等待
5. **参数统一**: 
   - 如果输入视频的分辨率不同，会自动缩放到第一个视频的分辨率（或指定分辨率）
   - 如果输入视频的帧率不同，会统一使用第一个视频的帧率（或指定帧率）

## 音频处理说明

**重要提醒**: 由于使用了OpenCV而不是moviepy，此版本的工具**不会保留音频**。如果您需要保留音频，有以下选择：

1. 使用moviepy版本（需要修改代码使用moviepy库）
2. 使用FFmpeg等工具后续添加音频
3. 考虑使用其他支持音频的视频处理库

## 依赖说明

- **opencv-python**: 用于视频读取、处理和写入
- **pathlib**: 用于文件路径操作（Python 3.4+自带）

## 错误处理

脚本会自动处理以下情况：
- 文件不存在
- 不支持的视频格式
- 损坏的视频文件
- 无法读取的视频文件

如遇到问题，请检查：
1. 文件路径是否正确
2. 文件是否为支持的视频格式
3. 是否有足够的磁盘空间
4. 视频文件是否损坏 