# MOSS-Transcribe-Diarize vLLM Integration

将下面命令复制到目标项目即可。服务提供 OpenAI-compatible 的音频转写接口：`POST /v1/audio/transcriptions`。

## 环境

- Linux、NVIDIA GPU、可工作的 NVIDIA 驱动。
- Python 3.12、`uv`、`ffmpeg`、`curl`。
- CUDA 12 使用项目指定的 vLLM `cu129` nightly wheel。

```bash
uv venv .venv --python 3.12
source .venv/bin/activate

uv pip install -U vllm \
  --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/68b4a1d582818e67adc903bf1b8fc5a5447da2fa/cu129
uv pip install 'vllm[audio]'
```

`vllm[audio]` 是读取 MP3、WAV 等音频文件所必需的依赖。

## 启动

首次运行会下载模型。不能直连 Hugging Face 时使用镜像：

```bash
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

vllm serve OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90
```

检查服务：

```bash
curl http://127.0.0.1:8000/health
```

## 调用

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize \
  -F file=@/path/to/audio.mp3 \
  -F response_format=json \
  -F temperature=0
```

返回示例：

```json
{
  "text": "[0.48][S01]Welcome everyone[1.66][12.26][S02]The meeting starts now[13.81]"
}
```

`text` 的格式为 `[起始秒数][说话人]正文[结束秒数]`。目标项目可用时间戳生成 SRT、ASS 或其他字幕格式。

## 长音频

默认编码缓存下，单个音频建议不超过 150 秒。长音频先切片，再按顺序调用接口；合并结果时将每个片段内的时间戳加上此前片段的总时长。

```bash
mkdir -p audio_parts
ffmpeg -i input.mp3 -map 0:a:0 -c:a copy \
  -f segment -segment_time 150 -reset_timestamps 1 \
  audio_parts/part_%03d.mp3
```

服务使用一张 RTX 4090 时，模型权重约占 1.72 GiB，默认 131072 上下文会分配约 19 GiB KV Cache，总显存约 22 GiB。
