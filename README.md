# Audio Power

音频处理工具库，集成语音识别（ASR）、语音合成（TTS）、声纹/特征提取、语音增强、语音翻译与变声等能力。

## 功能模块

| 模块 | 说明 |
|------|------|
| **audio_preprocess** | 音频预处理：HuBERT/ContentVec/DPHuBERT 特征、Mel、F0 预测、Wenet、降噪(FRCRN)、VAD(Silero) |
| **asr** | 语音识别：Azure、Faster Whisper、FunASR、SenseVoice |
| **speak_recognition** | 语音识别：Damo(FunASR)、WhisperX |
| **audio_translation** | 语音翻译：OpenAI、Azure、Cloudflare |
| **tts** | 语音合成：Azure TTS、Bark、Coqui AI、CosyVoice、Index TTS |
| **vc** / **voice_conversion** | 声音克隆/变声：RVC、SVC(So-VITS) |
| **audio_dirazation** | 音视频转写/说话人分离：阿里云通义听悟、FunASR/pyannote 实验归档、MOSS-Transcribe-Diarize vLLM 服务 |

## 环境要求

- Python 3.8+
- 部分模块需 CUDA（GPU）

## 安装

```bash
git clone <repo_url>
cd audio_power
pip install -r requirements.txt
```

依赖包括：WhisperX、OpenAI、python-dotenv、scipy、librosa、FunASR 等，FCPE 从 GitHub 安装。
FunASR API 与 pyannote 服务依赖较重，可按需单独安装：

- `audio_dirazation/funasr/requirements.txt`
- `audio_dirazation/pyannote/requirements.txt`
- `audio_dirazation/moss_transcribe_diarize/README.md`

## 配置

复制环境示例并填写密钥：

```bash
cp .env.example .env
```

`.env` 中可配置：

- `OPENAI_API_KEY` / `OPENAI_API_BASE` / `MODEL_NAME`：OpenAI 翻译等
- `HF_TOKEN`：Hugging Face 模型下载（可选）
- `BYTEDANCE_APPID` / `BYTEDANCE_ACCESS_TOKEN`：火山引擎
- `DASHSCOPE_API_KEY`：阿里 DashScope
- `HF_ENDPOINT`：国内可设为 `https://hf-mirror.com` 加速下载
- `ALIBABA_CLOUD_ACCESS_KEY_ID` / `ALIBABA_CLOUD_ACCESS_KEY_SECRET`：阿里云 AK
- `TINGWU_APP_KEY`：通义听悟 AppKey
- `TINGWU_REGION_ID` / `TINGWU_ENDPOINT`：听悟地域与接入点
- `FUNASR_DEVICE` / `FUNASR_DEFAULT_MODEL`：本地 FunASR 实验 API 推理设备与默认模型
- `PYANNOTE_MODEL_PATH` / `PYANNOTE_DEVICE`：pyannote 离线分离模型路径与推理设备

**密钥与配置请勿写入代码，仅通过环境变量使用。**

## 使用示例

### 音频特征提取（HuBERT 等）

```python
import librosa
from audio_preprocess.audio_feature.hubert import HubertAilia

audio_path = 'resources/guangdian.wav'
input_audio = librosa.load(audio_path, sr=16000, mono=True)[0][None, ...]

hubert = HubertAilia('hubert_base_ailia', 'gpu', device=0)
embedding_out = hubert.forward(input_audio)
print(embedding_out.shape)
```

更多示例见 `examples/audio_feature_extract.py`，支持 Hubert、ContentVec、DPHuBERT、FairseqHubert 等。

### VAD + Whisper 语音识别

```python
from speak_detection import SileroVAD
import whisperx

svad = SileroVAD()
tss = svad.forward('audio_vocals.wav')  # 获取说话时间段

whisper_model = whisperx.load_model('large', language='zh', device='cuda')
audio_array = whisperx.load_audio('audio_vocals.wav')
for ts in tss:
    audio_partial = audio_array[ts['start']:ts['end']]
    rec_result = whisper_model.transcribe(audio_partial, batch_size=32, language='zh')
    print(rec_result)
```

参考 `examples/silent_whisper_speak_recognize.py`。

### 通义听悟离线转写

```python
import os
from dotenv import load_dotenv
from audio_dirazation.tingwu import TingwuClient

load_dotenv()
client = TingwuClient(
    access_key_id=os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID"),
    access_key_secret=os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),
    app_key=os.getenv("TINGWU_APP_KEY"),
)
payload = client.create_offline_task(file_url="https://example.com/demo.mp4", source_language="cn", speaker_count=2)
task_id = client.extract_task_id(payload)
task_info = client.wait_for_task(task_id)
result_urls = client.extract_result_urls(task_info)
client.download_result_files(result_urls, output_dir="work/tingwu/downloads")
```

CLI: `python -m audio_dirazation.tingwu.cli --file-url 'https://...' --source-language cn --speaker-count 2 --wait --download-results --export-srt`

### FunASR diarization 实验归档

结论：不推荐将 FunASR 作为短剧 speaker diarization 主方案，主要原因是对白场景下 turn 切分和 speaker 归属不够稳定。

保留 API 入口用于复现实验或临时对照：

```bash
cd audio_power
python3.10 -m venv .venv-funasr --system-site-packages
.venv-funasr/bin/pip install -r audio_dirazation/funasr/requirements.txt
FUNASR_DEVICE=cuda:0 .venv-funasr/bin/uvicorn audio_dirazation.funasr.api:app --host 0.0.0.0 --port 8001
```

```bash
curl http://127.0.0.1:8001/v1/audio/transcriptions \
  -F file=@episode.mp3 \
  -F model=sensevoice \
  -F response_format=verbose_json \
  -F spk=true \
  -F preset_speaker_count=2
```

详情见 `audio_dirazation/funasr/README.md`。

### pyannote 本地说话人分离（不做 ASR）

`speaker-diarization-community-1` 只能做说话人分离，不做 ASR。它可以输出 speaker turn，但不会产生转写文本。

原 `pyannote-diarization-eval` 的 `community-1` 模型快照已迁移到本机忽略目录：

```bash
pretrain_models/audio_dirazation/pyannote/community-1
```

CLI:

```bash
python -m audio_dirazation.pyannote.cli input.wav --num-speakers 2 --output-json work/pyannote/result.json
```

API:

```bash
PYANNOTE_DEVICE=auto python -m uvicorn audio_dirazation.pyannote.server:app --host 0.0.0.0 --port 8002
```

详情见 `audio_dirazation/pyannote/README.md`。

### MOSS-Transcribe-Diarize vLLM 服务

基于 `OpenMOSS-Team/MOSS-Transcribe-Diarize` 的 vLLM 部署说明，提供 OpenAI-compatible 音频转写接口 `POST /v1/audio/transcriptions`。

```bash
source .venv/bin/activate
uv pip install -U vllm   --torch-backend=auto   --extra-index-url https://wheels.vllm.ai/68b4a1d582818e67adc903bf1b8fc5a5447da2fa/cu129
uv pip install 'vllm[audio]'

vllm serve OpenMOSS-Team/MOSS-Transcribe-Diarize   --host 0.0.0.0   --port 8000   --trust-remote-code   --dtype bfloat16   --max-model-len 131072   --max-num-seqs 1   --gpu-memory-utilization 0.90
```

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions   -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize   -F file=@/path/to/audio.mp3   -F response_format=json   -F temperature=0
```

返回的 `text` 包含 `[起始秒数][说话人]正文[结束秒数]` 时间戳，适合进一步生成 SRT/ASS。

长音频建议先切成不超过 150 秒的片段，顺序调用接口，再把时间戳按累计时长做偏移。

详情见 `audio_dirazation/moss_transcribe_diarize/README.md`。

### 翻译（OpenAI）

使用 `audio_translation/openai_translate.py`，需在 `.env` 中配置 `OPENAI_API_KEY`。

### 语音识别（Damo / FunASR）

使用 `speak_recognition/damo_api.py`，需将模型下载到 `pretrain_models/audio_lib/speak_recognition/damo/` 对应路径。

## 目录结构概览

```
audio_power/
├── audio_preprocess/     # 预处理与特征
│   ├── audio_feature/    # HuBERT、Mel、F0、Wenet
│   ├── noise_reducer/    # 降噪
│   └── speak_detection/ # VAD
├── asr/                  # 多引擎 ASR
├── speak_recognition/    # Damo、WhisperX
├── audio_translation/    # 翻译接口
├── tts/                  # 多引擎 TTS
├── vc/                   # RVC、SVC 变声
├── voice_conversion/     # 同上（兼容）
├── audio_dirazation/     # 音视频转写/说话人分离
│   ├── tingwu/           # 阿里云通义听悟
│   ├── funasr/           # FunASR diarization 实验归档
│   ├── pyannote/         # pyannote 分离 CLI/API（不做 ASR）
│   ├── nemo/             # NeMo 说话人分离实验说明
│   └── moss_transcribe_diarize/ # MOSS-Transcribe-Diarize vLLM 服务文档
├── examples/             # 示例脚本
├── resources/            # 示例音频
├── pretrain_models/      # 预训练模型（需自行下载）
└── output/               # 输出目录
```

## 预训练模型

部分功能需自行下载模型到 `pretrain_models/` 下对应子目录，例如：

- Damo ASR：见 `speak_recognition/damo_api.py` 内路径与 ModelScope 链接
- Whisper / WhisperX：首次运行时会按 `download_root` 自动下载
- pyannote community-1：默认路径 `pretrain_models/audio_dirazation/pyannote/community-1`
- MOSS-Transcribe-Diarize：部署见 `audio_dirazation/moss_transcribe_diarize/README.md`
- RVC / SVC：按各模块 README 或脚本内说明放置权重

## License

见项目仓库说明。
