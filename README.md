# PyNvVideoPipe

![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg?style=for-the-badge)
![CUDA](https://img.shields.io/badge/NVIDIA_CUDA-Optimized-76B900?logo=nvidia&logoColor=white&style=for-the-badge)
![Platform](https://img.shields.io/badge/platform-Linux-77216F?logo=linux&logoColor=white&style=for-the-badge)

High-Performance Video Processing Pipeline in Python, Powered by NVIDIA CUDA

Supports multi-stream, multi-GPU, and multi-model inference

Minimizes memory copies and CPU–GPU data transfers for maximum efficiency

基于 NVIDIA CUDA 的 Python 高性能视频处理流水线实现

支持多路视频流、多 GPU 与多模型推理

最大限度减少内存拷贝和 CPU–GPU 数据传输，提升整体效率

|            | Open开源 | Learning Curve学习成本           | Developer-Friendliness二次开发友好 | Performance性能 |
| ---------- | -------- | -------------------------------- | ---------------------------------- | --------------- |
| DeepStream | NO       | High                             | Low                                | High            |
| VideoPipe  | YES      | medium（requires C++ knowledge） | High                               | Medium          |
| our        | YES      | ≈ 0                              | High +++++++++++                   | Medium ---      |

### Quick Start

1. 环境准备

   Docker >= 24.0.0

   NVIDIA Driver >= 590

   NVIDIA Container Toolkit >= 1.13.0

   服务器满足以上三个条件，推荐 docker 容器运行（不推荐自己本地装环境）

   ```bash
   cd docker
   docker build -t PyNvVideoPipe:cuda12.6 .
   ```

   镜像生成后，进入容器，不报错即成功，后面示例默认容器内运行

   ```bash
   docker run -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
     -v {your_path}/PyNvVideoPipe:/workspace \
     PyNvVideoPipe:cuda12.6 \
     bash
   ```

   编译硬件加速库实现

   ```bash
   python setup.py build --inplace
   ```

2. 视觉模型导入

   将通过 [ultralytics](https://github.com/ultralytics/ultralytics) 训练的模型导入到`yolo26`目录下

   ```bash
   cd yolo26
   python pt2trt.py  --w yolo26n.pt --fp16
   ```

   🚀 推理尺寸固定为`(576,1024)`，跳过`letterbox`降低计算开销

3. 运行

   修改并理解`main.py`

   ```bash
   python main.py
   ```

### Notes

- 更多细节和技巧请阅读 `main.py` 注释
- 大简之道是最美的艺术，没有之一
- 工程不是追求完美的数学解，而是在资源受限、时间紧迫、需求模糊的情况下，寻找一个可用的最优解

### License

[Apache 2.0](https://github.com/lmk123568/PyNvVideoPipe/blob/main/LICENSE)