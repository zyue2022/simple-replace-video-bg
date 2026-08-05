# 🚀 极速 AI 视频抠像与背景替换引擎

本项目是一个基于 **Robust Video Matting (RVM - ResNet50)** 的简易视频抠像与背景替换处理流水线。

通过在 Python 端实现**底层锁页内存池 (Ring Buffer)**、**IPC 管线短读防护**与**严格的 CUDA 异步 DMA 同步机制**，本引擎彻底消除了传统逐帧读取带来的 CPU/GPU 内存搬运瓶颈。在单机环境下即可压榨出物理硬件的极致吞吐量，完美胜任 1080P/4K 高帧率视频的批量、无损、自动化处理。

## ✨ 核心特性

- ⚡️ **极致性能 (Zero-Copy)**：利用 Ring Buffer 与 `readinto()` 内存视图，消除 Python 解释器内昂贵的帧对象创建与销毁开销。
- 🛡️ **画面防撕裂**：硬核 `torch.cuda.synchronize()` 杜绝异步 DMA 与 CPU 读写的内存竞态，杜绝画面花屏与静默崩溃。
- 🎵 **无损音频秒合并**：采用 FFmpeg 流复制 (`-c:a copy`)，避免音频二次有损重编码。
- 🧠 **AI 算力全下放**：将 BGR2RGB、色彩归一化、Channels-last 通道重排等前处理全部推入 CUDA 核心并行加速。
- 📦 **完美应对高负载**：内置 memoryview 循环补齐逻辑，彻底解决 4K 巨型帧引发的 OS 管道短读 (Short Read) 截断问题。

---

## 🛠️ 环境配置

### 1. 硬件要求
- **GPU**: 必须配备 NVIDIA 独立显卡（推荐显存 8GB 及以上以支持高并发与 4K 处理）。
- **OS**: Windows / Linux。

### 2. 系统依赖 (FFmpeg)
引擎极其依赖底层 FFmpeg 进程进行硬件编解码与多路复用。
- 请前往 [FFmpeg 官网](https://ffmpeg.org/download.html) 下载，并解压。 Windows上也可使用winget命令。
- **关键配置**：确保 `ffmpeg` 命令已加入系统的环境变量（PATH），或者在脚本配置区手动指定 `ffmpeg.exe` 的绝对路径。

### 3. Python 依赖包
推荐使用 Python 3.8 或以上版本。请在终端执行以下命令安装依赖：

```bash
# 1. 安装 PyTorch (必须带有 CUDA 支持，请根据你的显卡驱动版本选择对应的安装命令)
# 参考 PyTorch 官网: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 2. 安装其他必需组件
pip install opencv-python numpy tqdm
```

---

## 📂 项目结构与使用方法

### 1. 准备目录与文件
请在脚本同级目录下，按照以下结构准备文件：

```text
📁 你的项目文件夹/
├── 📄 main.py              # 本项目的 Python 主脚本
├── 🖼️ bg.jpg               # 你想要替换的背景图片（必填，尺寸会自动适应视频）
├── 📁 input_videos/        # 放入所有需要处理的原视频 (.mp4, .mov, .mkv 等)
└── 📁 output_videos/       # 脚本运行时会自动创建，用于存放最终结果
```

> **💡 背景图制作提示（Clean Plate 法）**：
> 如果你的原视频是固定机位拍摄，建议截取原视频一帧进入 Photoshop，使用 AI 创成式填充把人物抹掉，仅替换你需要修改的背景物品，另存为 `bg.jpg`。这样合成后空间透视最自然！

### 2. 执行脚本
打开终端，运行脚本。脚本会自动扫描 `input_videos` 文件夹并开启批量处理：

```bash
python main.py
```

支持安全的中断机制：运行中途随时可按 `Ctrl + C`，程序会优雅截断并清理内存资源，不会留下僵尸进程。

---

## ⚙️ 核心参数配置说明

你可以用文本编辑器打开 `main.py`，在头部的 `# ================= 配置区 =================` 修改核心参数：

```python
INPUT_DIR = "input_videos"        # 待处理视频的存放目录
OUTPUT_DIR = "output_videos"      # 输出成品的存放目录
BG_IMAGE_PATH = "bg.jpg"          # 统一替换的背景图路径
FFMPEG_PATH = "ffmpeg"            # 若未配置环境变量，这里可填入绝对路径如 "C:/ffmpeg/bin/ffmpeg.exe"

USE_NVENC = True                  # 是否使用 N 卡硬件加速编码。若编码报错可改为 False(使用 CPU x264 编码)
QUEUE_SIZE = 32                   # 环形内存池深度。显存够大可调高(如 64)增加吞吐，显存不足可降至 16
```

---

## 常见问题 (FAQ)

**Q: 运行提示 `❌ 编码失败，FFmpeg 日志...` 怎么办？**
**A**: 通常是由于显卡的 NVENC 编码器并发限制，或者分辨率不支持导致的。尝试在配置区将 `USE_NVENC` 改为 `False` 强制使用通用 CPU 编码重试。

**Q: 模型下载太慢 / 提示网络连接失败？**
**A**: 脚本默认通过 `torch.hub.load` 从 GitHub 拉取 RVM 模型权重。可自行提前下载预训练模型并改为本地加载，或开启终端代理。

**Q: 程序可以用来做实时直播抠像吗？**
**A**: 本代码优化方向为“最高吞吐量的离线批量压制”。若需改为实时直播链路（如接入 OBS 虚拟摄像头），需将输入源从 `FFmpeg subprocess` 改为流媒体读取或 `cv2.VideoCapture`，并将输出接入 `pyvirtualcam`。