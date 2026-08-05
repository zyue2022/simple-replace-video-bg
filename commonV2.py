import os

# 针对 CPU 架构优化：限制内部线程，防止多线程抢占导致的操作系统调度卡顿
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import time
import sys
import subprocess
import threading
from queue import Queue, Full, Empty
import cv2
import numpy as np
import torch
from tqdm import tqdm

# ================= 配置区 =================
INPUT_DIR = "input_videos"
OUTPUT_DIR = "output_videos"
BG_IMAGE_PATH = "bg.jpg"
FFMPEG_PATH = "ffmpeg"

USE_NVENC = True
QUEUE_SIZE = 32  # 环形内存池深度 (越高性能要求越高)


# =========================================

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}小时{m}分{s}秒" if h > 0 else f"{m}分{s}秒"


def get_bg_tensor(bg_path, target_w, target_h):
    bg_img = cv2.imread(bg_path)
    if bg_img is None:
        raise FileNotFoundError(f"无法读取背景图: {bg_path}")
    bg_img = cv2.resize(bg_img, (target_w, target_h))
    bg_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    # 推入 GPU 并做 channels_last 优化
    bg_tensor = torch.from_numpy(bg_rgb).permute(2, 0, 1).float().cuda() / 255.0
    return bg_tensor.unsqueeze(0).to(memory_format=torch.channels_last)


def get_video_info(ffmpeg_path, video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        output = subprocess.check_output(cmd).decode("utf-8").strip().splitlines()
        w = int(output[0])
        h = int(output[1])
        fps_parts = output[2].split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(output[2])
        total_frames = int(output[3])
        return w, h, fps, total_frames
    except Exception:
        return None, None, None, None


def reader_thread_zero_copy(ffmpeg_path, video_path, width, height, in_free_q, in_ready_q, stop_event):
    cmd = [
        ffmpeg_path, "-hwaccel", "cuda",
        "-i", video_path,
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8)
    frame_size = width * height * 3

    try:
        while not stop_event.is_set():
            try:
                buffer_tensor = in_free_q.get(timeout=0.5)
            except Empty:
                continue

            buf_1d = buffer_tensor.view(-1).numpy()

            # 【核心修复】：使用 memoryview 循环补齐读取，完美解决 4K 下 IPC Pipe 短读 (Short Read) 隐患
            mem_view = memoryview(buf_1d)
            total_read = 0
            while total_read < frame_size and not stop_event.is_set():
                bytes_read = proc.stdout.readinto(mem_view[total_read:])
                if not bytes_read:
                    break  # 遭遇真正的 EOF
                total_read += bytes_read

            # 校验是否因中断或视频结束导致未能读满一帧
            if total_read != frame_size:
                in_free_q.put(buffer_tensor)  # 归还内存块
                break

            while not stop_event.is_set():
                try:
                    in_ready_q.put(buffer_tensor, timeout=0.5)
                    break
                except Full:
                    continue
    finally:
        proc.terminate()
        proc.wait()
        # 优雅注入 EOF 标记，通知主线程结束
        while not stop_event.is_set():
            try:
                in_ready_q.put(None, timeout=0.5)
                break
            except Full:
                pass


def writer_thread_zero_copy(ffmpeg_proc, out_ready_q, out_free_q, stop_event):
    # stop_event 置位后，依然要把 out_ready_q 里的遗留成品帧清空，保证片尾不掉帧
    while not stop_event.is_set() or not out_ready_q.empty():
        try:
            out_tensor = out_ready_q.get(timeout=0.5)
        except Empty:
            continue

        if out_tensor is None:
            break

        try:
            ffmpeg_proc.stdin.write(out_tensor.numpy())
        except (OSError, IOError):
            break

        out_free_q.put(out_tensor)


def create_encoder_process(ffmpeg_path, temp_output_path, width, height, fps, use_nvenc=True):
    vcodec_args = ["-c:v", "h264_nvenc", "-preset", "p6", "-profile:v", "high", "-rc", "vbr", "-cq", "21", "-pix_fmt",
                   "yuv420p"] if use_nvenc else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt",
                                                 "yuv420p"]
    cmd = [ffmpeg_path, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{width}x{height}", "-pix_fmt", "bgr24",
           "-r", str(fps), "-i", "-", *vcodec_args, temp_output_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=False)


def process_video(video_path, output_path, bg_path, model):
    start_time = time.time()
    video_name = os.path.basename(video_path)

    w, h, fps, total_frames = get_video_info(FFMPEG_PATH, video_path)
    if not w or total_frames == 0:
        print(f"❌ 视频损坏或无法读取元数据: {video_name}", flush=True)
        return

    bg_tensor = get_bg_tensor(bg_path, w, h)
    base_name, ext = os.path.splitext(output_path)
    temp_output_path = f"{base_name}_temp{ext}"

    # ================= 建立双向锁页内存池 =================
    in_free_q = Queue(maxsize=QUEUE_SIZE)
    in_ready_q = Queue(maxsize=QUEUE_SIZE)
    out_free_q = Queue(maxsize=QUEUE_SIZE)
    out_ready_q = Queue(maxsize=QUEUE_SIZE)

    for _ in range(QUEUE_SIZE):
        in_free_q.put(torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True))
        out_free_q.put(torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True))

    stop_event = threading.Event()
    ffmpeg_proc = create_encoder_process(FFMPEG_PATH, temp_output_path, w, h, fps, use_nvenc=USE_NVENC)

    reader = threading.Thread(target=reader_thread_zero_copy,
                              args=(FFMPEG_PATH, video_path, w, h, in_free_q, in_ready_q, stop_event))
    writer = threading.Thread(target=writer_thread_zero_copy, args=(ffmpeg_proc, out_ready_q, out_free_q, stop_event))
    reader.start()
    writer.start()

    rec = [None] * 4
    downsample_ratio = 0.5 if (w * h <= 1920 * 1080) else 0.25
    processed_frames = 0

    print(f"\n🎬 开始处理: {video_name} | 原画 [{w}x{h}] | {fps:.2f} FPS", flush=True)

    try:
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            pbar = tqdm(total=total_frames, desc="   AI 抠像渲染中", unit="帧", leave=True, file=sys.stdout)

            while True:
                try:
                    in_tensor = in_ready_q.get(timeout=1.0)
                except Empty:
                    if not reader.is_alive(): break
                    continue

                if in_tensor is None:
                    break

                # 1. 异步 DMA 移入 GPU
                src_uint8 = in_tensor.cuda(non_blocking=True)

                # 2. GPU 并行完成重排、归一化与 channels_last
                src = src_uint8.permute(2, 0, 1).unsqueeze(0)[:, [2, 1, 0], :, :]
                src = src.to(memory_format=torch.channels_last).float().div_(255.0)

                # 3. AI 推理合成
                fgr, pha, *rec = model(src, *rec, downsample_ratio=downsample_ratio)
                com = fgr * pha + bg_tensor * (1.0 - pha)

                # 还原至 uint8 HWC 格式
                com_bgr = (com[0, [2, 1, 0]] * 255).clamp(0, 255).to(torch.uint8)
                com_bgr_hwc = com_bgr.permute(1, 2, 0).contiguous()

                # 4. 获取空闲输出内存并发起异步 D2H 写回
                out_tensor = out_free_q.get(timeout=1.0)
                out_tensor.copy_(com_bgr_hwc)

                # 5. 必须进行 CUDA 硬件级同步，防止 CPU 提前收回内存导致数据撕裂
                torch.cuda.synchronize()

                # 安全收回/提交内存
                in_free_q.put(in_tensor)
                out_ready_q.put(out_tensor)

                processed_frames += 1
                pbar.update(1)

            pbar.close()

    except (Exception, KeyboardInterrupt) as e:
        if isinstance(e, KeyboardInterrupt):
            print("\n🛑 用户手动强制中断！正在清理内存...", flush=True)
        else:
            print(f"\n❌ 处理过程中发生异常: {e}", flush=True)
    finally:
        # 【核心修复】：精简终止流，只需通知 stop_event，后台线程凭借 timeout 会安全自尽
        stop_event.set()

        reader.join()
        writer.join()

        if ffmpeg_proc.stdin:
            try:
                ffmpeg_proc.stdin.close()
            except:
                pass

        stderr_output = ffmpeg_proc.stderr.read().decode("utf-8", errors="ignore")
        ffmpeg_proc.wait()

        del rec, bg_tensor
        torch.cuda.empty_cache()

    if ffmpeg_proc.returncode != 0:
        print(f"❌ 编码失败，FFmpeg 日志:\n{stderr_output}", flush=True)
        return

    # 若未发生异常且生成了临时文件，则压制音频
    if os.path.exists(temp_output_path):
        print("   🎵 正在流式合并原视频音频...", flush=True)
        command = [
            FFMPEG_PATH, "-y",
            "-i", temp_output_path,
            "-i", video_path,
            "-c:v", "copy",
            "-c:a", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            output_path
        ]
        result = subprocess.run(command, capture_output=True)

        os.remove(temp_output_path)
        elapsed_time = time.time() - start_time
        avg_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0

        if result.returncode != 0:
            print(f"⚠️ 音频合并失败:\n{result.stderr.decode('utf-8', errors='ignore')}", flush=True)
        else:
            print(f"✅ 完成: {video_name} | 耗时: {format_time(elapsed_time)} | 均速: {avg_fps:.2f} FPS", flush=True)
    else:
        print("⚠️ 未找到输出视频，可能是由于任务中断导致。", flush=True)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("🚀 正在加载 AI 旗舰抠像模型 (Robust Video Matting - ResNet50)...", flush=True)
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50").cuda()
    model.eval()
    model = model.to(memory_format=torch.channels_last)

    videos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
    if not videos:
        print(f"❌ 未在 '{INPUT_DIR}' 目录下找到任何支持的视频文件！", flush=True)
        return

    print(f"📂 检测到 {len(videos)} 个视频，Zero-Copy 并发流水线就绪...", flush=True)
    print("=" * 60, flush=True)

    batch_start_time = time.time()
    for idx, video_name in enumerate(videos, 1):
        print(f"\n[ 任务进度: {idx} / {len(videos)} ]", flush=True)
        process_video(os.path.join(INPUT_DIR, video_name), os.path.join(OUTPUT_DIR, video_name), BG_IMAGE_PATH, model)

    print("\n" + "=" * 60, flush=True)
    print(f"🎉 全部处理完毕！总耗时: {format_time(time.time() - batch_start_time)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 用户强制结束了总任务。", flush=True)
        sys.exit(0)