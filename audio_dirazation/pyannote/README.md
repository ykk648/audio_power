# pyannote 本地说话人分离（不做 ASR）

从原 `pyannote-diarization-eval` 整理而来，保留 `community-1` 离线模型路径约定，并补成 Python API、CLI 和可选 FastAPI 服务。

## 结论

`speaker-diarization-community-1` 只能做说话人分离，不做 ASR。它可以输出谁在什么时间段说话，但不会输出转写文本。

## 离线模型

离线模型默认读取：

```bash
pretrain_models/audio_dirazation/pyannote/community-1
```

`pretrain_models/` 在 `audio_power` 中被 `.gitignore` 忽略，不会提交到 Git。新机器部署时可重新下载：

```bash
git lfs install
git clone https://hf.co/pyannote/speaker-diarization-community-1 \
  pretrain_models/audio_dirazation/pyannote/community-1
```

## 环境

```bash
cd audio_power
python3.10 -m venv .venv-pyannote --system-site-packages
.venv-pyannote/bin/pip install -r audio_dirazation/pyannote/requirements.txt
```

依赖系统命令：`ffmpeg`。代码会先转成 16k mono WAV，再用 `torchaudio.load` 送入 pyannote，避免 `torchcodec` 缺失导致内置解码失败。

## CLI

```bash
cd audio_power
.venv-pyannote/bin/python -m audio_dirazation.pyannote.cli input.wav \
  --num-speakers 2 \
  --output-json work/pyannote/result.json \
  --output-rttm work/pyannote/result.rttm
```

默认模型选择顺序：

1. `PYANNOTE_MODEL_PATH` / `PYANNOTE_MODEL`
2. `pretrain_models/audio_dirazation/pyannote/community-1`
3. `pyannote/speaker-diarization-community-1`（需要 `HF_TOKEN`）

## FastAPI

```bash
cd audio_power
export CUDA_VISIBLE_DEVICES=0
export PYANNOTE_DEVICE=auto
export PYANNOTE_PRELOAD=true
export PYANNOTE_TRUST_MODEL_PICKLE=true
.venv-pyannote/bin/uvicorn audio_dirazation.pyannote.server:app --host 0.0.0.0 --port 8002
```

```bash
curl http://127.0.0.1:8002/healthz

curl http://127.0.0.1:8002/v1/audio/diarizations \
  -F file=@episode.wav \
  -F num_speakers=2
```

返回格式：

```json
{
  "input_duration_seconds": 34.203,
  "elapsed_seconds": 1.003,
  "device": "cuda:0",
  "turns": [
    {"start": 0.098, "end": 1.178, "speaker": "SPEAKER_00"}
  ]
}
```

## PyTorch 2.6+ 权重加载

原评测日志里已经遇到过 `torch.load(weights_only=True)` 导致的 pyannote checkpoint 加载失败。这里默认对本地可信模型启用 `PYANNOTE_TRUST_MODEL_PICKLE=true`，会在加载模型时临时把 `torch.load` 的 `weights_only` 设为 `False`。

只对自己下载并信任的模型使用这个开关。
