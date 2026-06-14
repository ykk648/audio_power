# Audio Power

音频处理工具库，集成语音识别（ASR）、语音合成（TTS）、声纹/特征提取、语音增强、语音翻译与变声等能力。

## 功能模块

| 模块 | 说明 |
|------|------|
| **audio_preprocess** | 音频预处理：HuBERT/ContentVec/DPHuBERT 特征、Mel、F0 预测、Wenet、降噪(FRCRN)、VAD(Silero)、说话人分割 |
| **asr** | 语音识别：Azure、Faster Whisper、FunASR、SenseVoice |
| **speak_recognition** | 语音识别：Damo(FunASR)、WhisperX |
| **audio_translation** | 语音翻译：OpenAI、Azure、Cloudflare |
| **tts** | 语音合成：Azure TTS、Bark、Coqui AI、CosyVoice、Index TTS |
| **vc** / **voice_conversion** | 声音克隆/变声：RVC、SVC(So-VITS) |
| **audio_dirazation** | 音视频转写/说话人分离：阿里云通义听悟（离线任务） |

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
│   ├── speak_detection/ # VAD
│   └── speak_dirazation/# 说话人分割
├── asr/                  # 多引擎 ASR
├── speak_recognition/    # Damo、WhisperX
├── audio_translation/    # 翻译接口
├── tts/                  # 多引擎 TTS
├── vc/                   # RVC、SVC 变声
├── voice_conversion/     # 同上（兼容）
├── audio_dirazation/     # 音视频转写/说话人分离（听悟）
│   └── tingwu/           # 阿里云通义听悟
├── examples/             # 示例脚本
├── resources/            # 示例音频
├── pretrain_models/      # 预训练模型（需自行下载）
└── output/               # 输出目录
```

## 预训练模型

部分功能需自行下载模型到 `pretrain_models/` 下对应子目录，例如：

- Damo ASR：见 `speak_recognition/damo_api.py` 内路径与 ModelScope 链接
- Whisper / WhisperX：首次运行时会按 `download_root` 自动下载
- RVC / SVC：按各模块 README 或脚本内说明放置权重

## License

见项目仓库说明。
