import os
import time
import sys
import subprocess
import threading
from queue import Queue
import cv2
import numpy as np
import torch
from tqdm import tqdm

# ================= 配置区 =================
INPUT_DIR = "input_videos"  # 输入视频文件夹
OUTPUT_DIR = "output_videos"  # 输出视频文件夹
BG_IMAGE_PATH = "bg.jpg"  # 背景图片路径
FFMPEG_PATH = "ffmpeg"  # FFmpeg 路径

USE_NVENC = True  # 开启 NVENC 硬件编码
QUEUE_SIZE = 32  # 缓存队列深度


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
    bg_tensor = torch.from_numpy(bg_rgb).permute(2, 0, 1).float().cuda() / 255.0
    return bg_tensor.unsqueeze(0)


def frame_reader_thread(cap, in_queue, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            in_queue.put(None)
            break
        in_queue.put(frame)


def frame_writer_thread(ffmpeg_proc, out_queue, stop_event):
    while not stop_event.is_set():
        out_bgr_bytes = out_queue.get()
        if out_bgr_bytes is None:
            break
        try:
            ffmpeg_proc.stdin.write(out_bgr_bytes)
        except (OSError, IOError):
            break


def create_encoder_process(ffmpeg_path, temp_output_path, width, height, fps, use_nvenc=True):
    if use_nvenc:
        vcodec_args = [
            "-c:v", "h264_nvenc",
            "-preset", "p6",
            "-profile:v", "high",
            "-rc", "vbr",
            "-cq", "21",
            "-pix_fmt", "yuv420p"
        ]
    else:
        vcodec_args = [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "21",
            "-pix_fmt", "yuv420p"
        ]

    cmd = [
        ffmpeg_path,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        *vcodec_args,
        temp_output_path
    ]

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=False
    )


def process_video(video_path, output_path, bg_path, model):
    start_time = time.time()
    video_name = os.path.basename(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_name}", flush=True)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0 or total_frames == 0:
        print(f"❌ 视频元数据读取异常: {video_name}", flush=True)
        cap.release()
        return

    bg_tensor = get_bg_tensor(bg_path, w, h)
    base_name, ext = os.path.splitext(output_path)
    temp_output_path = f"{base_name}_temp{ext}"

    ffmpeg_proc = create_encoder_process(FFMPEG_PATH, temp_output_path, w, h, fps, use_nvenc=USE_NVENC)

    in_queue = Queue(maxsize=QUEUE_SIZE)
    out_queue = Queue(maxsize=QUEUE_SIZE)
    stop_event = threading.Event()

    reader = threading.Thread(target=frame_reader_thread, args=(cap, in_queue, stop_event))
    writer = threading.Thread(target=frame_writer_thread, args=(ffmpeg_proc, out_queue, stop_event))
    reader.start()
    writer.start()

    rec = [None] * 4

    print(f"\n🎬 开始处理: {video_name}", flush=True)
    print(f"   规格: 原画 [{w}x{h}] | 帧率: {fps:.2f} FPS", flush=True)
    print(f"   架构: 3线程流水线 + 混合精度 Autocast + NVENC p6 (极速峰值版)", flush=True)

    downsample_ratio = 0.125 if (w * h >= 3840 * 2160) else 0.25
    processed_frames = 0

    try:
        with torch.inference_mode():
            with torch.amp.autocast('cuda'):
                # 绑定 file=sys.stdout 彻底解决日志错位
                pbar = tqdm(total=total_frames, desc="   原画级 AI渲染中", unit="帧", leave=True, file=sys.stdout)

                while True:
                    frame = in_queue.get()
                    if frame is None:
                        break

                    src = torch.from_numpy(frame).cuda(non_blocking=True).permute(2, 0, 1).float() / 255.0
                    src = src[[2, 1, 0], :, :].unsqueeze(0)

                    fgr, pha, *rec = model(src, *rec, downsample_ratio=downsample_ratio)
                    com = fgr * pha + bg_tensor * (1.0 - pha)

                    com_bgr = com[0, [2, 1, 0], :, :]
                    out_bgr = (com_bgr.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    out_bgr_bytes = np.ascontiguousarray(out_bgr).tobytes()

                    out_queue.put(out_bgr_bytes)

                    processed_frames += 1
                    pbar.update(1)

                pbar.close()

    finally:
        out_queue.put(None)
        stop_event.set()
        reader.join()
        writer.join()
        cap.release()

        if ffmpeg_proc.stdin:
            ffmpeg_proc.stdin.close()
        stderr_output = ffmpeg_proc.stderr.read().decode("utf-8", errors="ignore")
        ffmpeg_proc.wait()

        del rec, bg_tensor
        torch.cuda.empty_cache()

    if ffmpeg_proc.returncode != 0:
        print(f"❌ 编码失败，日志:\n{stderr_output}", flush=True)
        return

    print("   🎵 正在合并原视频音频...", flush=True)
    command = [
        FFMPEG_PATH, "-y",
        "-i", temp_output_path,
        "-i", video_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        output_path
    ]

    result = subprocess.run(command, capture_output=True)
    stderr_msg = result.stderr.decode("utf-8", errors="ignore")

    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    elapsed_time = time.time() - start_time
    avg_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0

    if result.returncode != 0:
        print(f"⚠️ 音频合并报错:\n{stderr_msg}", flush=True)
    else:
        print(f"✅ 完成: {video_name} | 耗时: {format_time(elapsed_time)} | 渲染速度: {avg_fps:.2f} FPS", flush=True)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    torch.backends.cudnn.benchmark = True

    print("🚀 正在加载 AI 抠像模型 (MobilenetV3)...", flush=True)
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").cuda()
    model.eval()

    videos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
    total_videos = len(videos)

    if total_videos == 0:
        print(f"❌ 未在 '{INPUT_DIR}' 目录下找到任何视频！", flush=True)
        return

    print(f"📂 共检测到 {total_videos} 个待处理视频，准备开启极速流水线处理...", flush=True)
    print("=" * 60, flush=True)

    batch_start_time = time.time()

    for idx, video_name in enumerate(videos, 1):
        print(f"\n[ 任务进度: {idx} / {total_videos} ]", flush=True)
        in_path = os.path.join(INPUT_DIR, video_name)
        out_path = os.path.join(OUTPUT_DIR, video_name)

        process_video(in_path, out_path, BG_IMAGE_PATH, model)

    total_batch_time = time.time() - batch_start_time
    print("\n" + "=" * 60, flush=True)
    print(f"🎉 所有视频全部处理完毕！", flush=True)
    print(f"📊 任务汇总: 成功处理 {total_videos} 个视频 | 总耗时: {format_time(total_batch_time)}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()