<p style="" align="center">
  <img src="./assets/logo.png" alt="Logo" width="20%">
</p>
<h1 align="center">PyVideoProc</h1>
<p style="margin:0px" align="center">
    <img src="https://img.shields.io/badge/license-BSD--2-blue.svg?&logo=c&logoColor=white&style=for-the-badge">
    <img src="https://img.shields.io/badge/CUDA-12.8-76B900?&logo=nvidia&logoColor=white&style=for-the-badge">
    <img src="https://img.shields.io/badge/OS-Linux-FCC624?&logo=linux&logoColor=white&style=for-the-badge">
    <img src="https://img.shields.io/badge/ENV-Docker-2496ED?logo=docker&logoColor=white&style=for-the-badge">
    <img src="https://img.shields.io/badge/Video-YouTube-FF0000?logo=youtube&logoColor=white&style=for-the-badge">
    <img src="https://img.shields.io/badge/Video-Bilibili-FB7299?logo=bilibili&logoColor=white&style=for-the-badge">
</p>

---

PyVideoProc 提供基于 CUDA 加速优化的高性能 Python SDK，可高效实现多路、多卡、多模型的视频解码、AI 推理与编码，显著降低开发复杂度并提升吞吐性能

PyVideoProc provides a high-performance Python SDK optimized with CUDA acceleration, enabling efficient multi-stream, multi-GPU, and multi-model video decoding, AI inference, and encoding—significantly reducing development complexity and boosting throughput performance.

[https://github.com/user-attachments/assets/402f9080-004e-457e-8d36-3fefdb56f21d](https://github.com/user-attachments/assets/bb12981c-ac43-40c8-a802-983671356646)

⭐ 多进程绕过 GIL 限制，提升 Python 并发性能

⭐ 减少 Host-Device 数据传输，降低 GPU 显存冗余拷贝，提升推理速度

⭐ 尽可能在 GPU 上计算，以降低 CPU 计算负担

⭐ 开箱即用，简单易懂，扩展性强，适合中小型项目快速部署

|                                                           | Open Source |          Learning Curve          |      Developer Friendliness      | Performance |      Architecture Design       |
| :-------------------------------------------------------: | :---------: | :------------------------------: | :------------------------------: | :---------: | :----------------------------: |
| [DeepStream](https://developer.nvidia.com/deepstream-sdk) |      ❌      |               High               |               Low                |    High     | Single-process, multi-threaded |
| [VideoPipe](https://github.com/sherlockchou86/VideoPipe)  |      ✅      | medium（requires cpp knowledge） | Medium（requires cpp knowledge） |   Medium    | Single-process, multi-threaded |
|                            Our                            |      ✅      |               ≈ 0                |         High       |   Medium    | Multi-process, single-threaded |

🔗 Bilibili: [https://www.bilibili.com/video/BV12TcvzbEcZ](https://www.bilibili.com/video/BV12TcvzbEcZ) 

🔗 YouTube: [https://youtu.be/-soVRjH1rb4](https://youtu.be/-soVRjH1rb4)

## Quick Start

本项目推荐 Docker 容器运行，首先确保本地环境满足以下三个条件：

- Docker >= 24.0.0
- NVIDIA Driver >= 590
- NVIDIA Container Toolkit >= 1.13.0

### 1. 生成镜像

clone 本项目，生成包含完整开发环境的镜像

```bash
git clone https://github.com/lmk123568/PyVideoProc.git
cd PyVideoProc/docker
docker build -t pyvideoproc:cuda12.8 .
```

镜像生成后，进入容器，不报错即成功

```bash
docker run -it \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v {your_path}/PyVideoProc:/workspace \
  pyvideoproc:cuda12.8 \
  bash
```

后续示例代码默认在容器内`/workspace`运行

> ⚠️ 不推荐自己本地装环境，如果一定要自己装，请参考 Dockerfile

### 2. 编译加速包

```bash
python scripts/setup.py install
```

这里面包含了硬件编解码、YOLO26 推理优化的 C++ 实现，并通过 Pybind11 给 Python 调用

### 3. 训练模型权重转换

将通过 [ultralytics](https://github.com/ultralytics/ultralytics) 训练的`pt`模型导入到当前目录（`/workspace`）下（示例模型为 [yolo26n.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt)）

```bash
python scripts/pt2trt.py  --w ./yolo26n.pt --fp16
```

 转换过程中会与 ultralytics 官方结果进行推理对齐

> 💡 TensorRT 编译生成 .engine 过程中，推理尺寸默认设置为`(576,1024)`，可以跳过`letterbox`降低计算开销

> 💡 遇到警告 `requirements: Ultralytics requirement ['onnxruntime-gpu'] not found ...` 可以 `Ctrl + C` 跳过

### 4. 运行

开启 MPS（Multi-Process Service）

```bash
nvidia-cuda-mps-control -d
# echo quit | nvidia-cuda-mps-control  关闭 MPS
```

阅读理解其代码并运行

```bash
python main.py
```

## Benchmark

测试日期: 2026-01-25

测试硬件: AMD Ryzen 9 5950 X + NVIDIA GeForce RTX 3090

测试任务: 4 × RTSP Decoders → YOLO26 (TensorRT) → 4 × RTMP Encoders

|                           | CPU     | RAM     | GPU VRAM | **GPU-Util** |
| ------------------------- | ------- | ------- | -------- | ------------ |
| VideoPipe（ffmpeg codec） | 511.6 % | 1.5 GiB | 2677 MiB | 16 %         |
| Our                       | 40 %    | 1.2GiB  | 3932 MiB | 12 %         |

## Pipeline

<img src="./assets/pipeline.png" alt="pipe" style="zoom:70%;" />

## License

[BSD-2-Clause](https://github.com/lmk123568/PyVideoProc/blob/main/LICENSE)
