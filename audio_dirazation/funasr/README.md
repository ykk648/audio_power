# FunASR diarization 实验归档

从原 `funasr-diarize-api` 合并而来。这里保留 OpenAI 风格 multipart API 入口，主要用于复现实验和做横向对照。

## 结论

不推荐将 FunASR 作为短剧 speaker diarization 主方案，主要原因是对白场景下 turn 切分和 speaker 归属不够稳定。

## 环境

```bash
cd audio_power
python3.10 -m venv .venv-funasr --system-site-packages
.venv-funasr/bin/pip install -r audio_dirazation/funasr/requirements.txt
```

依赖系统命令：`ffmpeg`、`ffprobe`。

## 启动

```bash
cd audio_power
export CUDA_VISIBLE_DEVICES=0
export FUNASR_DEVICE=cuda:0
export FUNASR_DEFAULT_MODEL=sensevoice
export FUNASR_PRELOAD=true
.venv-funasr/bin/uvicorn audio_dirazation.funasr.api:app --host 0.0.0.0 --port 8001
```

## API

```bash
curl http://127.0.0.1:8001/healthz

curl http://127.0.0.1:8001/v1/audio/transcriptions \
  -F file=@episode.mp3 \
  -F model=sensevoice \
  -F response_format=verbose_json \
  -F spk=true \
  -F preset_speaker_count=2
```

`verbose_json` 返回秒级时间戳：

```json
{
  "segments": [
    {"id": 0, "start": 1.2, "end": 2.6, "speaker": "S01", "text": "..."}
  ]
}
```

说话人 ID 只在单次完整请求内有效，不要把多个短切片分别分离后直接拼接 `S01/S02`。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `FUNASR_DEVICE` | `cuda:0` | FunASR 推理设备 |
| `FUNASR_DEFAULT_MODEL` | `sensevoice` | 默认模型别名 |
| `FUNASR_PRELOAD` | `true` | 服务启动时是否预加载默认模型 |
| `FUNASR_MAX_CONCURRENCY` | `1` | 同时推理数量 |
| `FUNASR_BATCH_SIZE_S` | `300` | FunASR batch size |
| `FUNASR_MAX_UPLOAD_MB` | `500` | 单文件上传大小限制 |

可用模型别名：`sensevoice`、`sensevoice-eres2net`、`paraformer`、`fun-asr-nano`。
