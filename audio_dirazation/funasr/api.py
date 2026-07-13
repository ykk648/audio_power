from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse


LOGGER = logging.getLogger("audio_power.funasr_diarization")
TAG_RE = re.compile(r"<\|[^|]*\|>")
ALLOWED_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".mp4", ".ogg", ".opus", ".wav", ".webm"}

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "sensevoice": {
        "model": "iic/SenseVoiceSmall",
        "vad_model": "fsmn-vad",
        "spk_model": "cam++",
        "vad_kwargs": {"max_single_segment_time": 30000},
    },
    "sensevoice-eres2net": {
        "model": "iic/SenseVoiceSmall",
        "vad_model": "fsmn-vad",
        "spk_model": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
        "vad_kwargs": {"max_single_segment_time": 30000},
    },
    "paraformer": {
        "model": "paraformer-zh",
        "vad_model": "fsmn-vad",
        "punc_model": "ct-punc",
        "spk_model": "cam++",
        "spk_mode": "punc_segment",
        "vad_kwargs": {"max_single_segment_time": 30000},
    },
    "fun-asr-nano": {
        "model": "FunAudioLLM/Fun-ASR-Nano-2512",
        "trust_remote_code": True,
        "vad_model": "fsmn-vad",
        "spk_model": "cam++",
        "vad_kwargs": {"max_single_segment_time": 30000},
    },
}


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        LOGGER.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _clean_text(value: Any) -> str:
    return TAG_RE.sub("", str(value or "")).strip()


def _speaker_label(value: Any) -> str | None:
    if value is None or value == "":
        return None
    raw = str(value).strip()
    match = re.fullmatch(r"(?:spk|speaker|s)?\s*(\d+)", raw, flags=re.IGNORECASE)
    if match:
        return f"S{int(match.group(1)) + 1:02d}"
    return raw


def _probe_duration_seconds(audio_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return float(result.stdout.strip())


def _convert_to_16k_mono(source_path: Path, output_path: Path) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "ffmpeg audio conversion failed")


class ModelRegistry:
    def __init__(self, *, device: str) -> None:
        self.device = device
        self.models: dict[str, Any] = {}

    def get(self, alias: str) -> Any:
        if alias not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model '{alias}'. Available: {', '.join(MODEL_CONFIGS)}")
        if alias not in self.models:
            from funasr import AutoModel

            config = dict(MODEL_CONFIGS[alias])
            config.update(device=self.device, disable_update=True, hub="ms")
            LOGGER.info("Loading %s on %s", alias, self.device)
            self.models[alias] = AutoModel(**config)
            LOGGER.info("Loaded %s", alias)
        return self.models[alias]


def _build_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    speaker_map: dict[str, str] = {}
    for sentence in result.get("sentence_info") or []:
        start_ms = int(sentence.get("start") or 0)
        end_ms = int(sentence.get("end") or 0)
        text = _clean_text(sentence.get("sentence") or sentence.get("text"))
        if not text or end_ms <= start_ms:
            continue
        raw_speaker = _speaker_label(sentence.get("spk"))
        speaker = None
        if raw_speaker:
            speaker = speaker_map.setdefault(raw_speaker, f"S{len(speaker_map) + 1:02d}")
        rows.append(
            {
                "start": round(start_ms / 1000, 3),
                "end": round(end_ms / 1000, 3),
                "text": text,
                "speaker": speaker,
            }
        )
    return rows


def _transcribe(
    registry: ModelRegistry,
    *,
    model_alias: str,
    audio_path: Path,
    language: str | None,
    hotwords: str | None,
    diarize: bool,
    preset_speaker_count: int | None,
) -> tuple[dict[str, Any], float]:
    model = registry.get(model_alias)
    kwargs: dict[str, Any] = {
        "input": str(audio_path),
        "batch_size_s": _read_int_env("FUNASR_BATCH_SIZE_S", 300),
        "return_spk_res": diarize,
    }
    if language:
        kwargs["language"] = language
    if hotwords:
        kwargs["hotword"] = hotwords
    if preset_speaker_count:
        kwargs["preset_spk_num"] = preset_speaker_count
    started = time.perf_counter()
    result = model.generate(**kwargs)[0]
    return result, time.perf_counter() - started


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry = ModelRegistry(device=os.getenv("FUNASR_DEVICE", "cuda:0"))
    app.state.inference_semaphore = asyncio.Semaphore(max(1, _read_int_env("FUNASR_MAX_CONCURRENCY", 1)))
    app.state.default_model = os.getenv("FUNASR_DEFAULT_MODEL", "sensevoice")
    if _read_bool_env("FUNASR_PRELOAD", True):
        await run_in_threadpool(app.state.registry.get, app.state.default_model)
    yield


app = FastAPI(
    title="Audio Power FunASR Diarization API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": app.state.registry.device,
        "default_model": app.state.default_model,
        "loaded_models": sorted(app.state.registry.models),
    }


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    loaded = app.state.registry.models
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "owned_by": "funasr", "ready": name in loaded}
            for name in MODEL_CONFIGS
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    language: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    spk: bool = Form(default=True),
    hotwords: str | None = Form(default=None),
    preset_speaker_count: int | None = Form(default=None),
):
    model_alias = model or app.state.default_model
    if model_alias not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_alias}'")
    if response_format not in {"json", "text", "verbose_json"}:
        raise HTTPException(status_code=400, detail="response_format must be json, text, or verbose_json")
    if preset_speaker_count is not None and preset_speaker_count < 1:
        raise HTTPException(status_code=400, detail="preset_speaker_count must be positive")

    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported audio extension: {suffix or '(none)'}")
    content = await file.read()
    max_upload_bytes = _read_int_env("FUNASR_MAX_UPLOAD_MB", 500) * 1024 * 1024
    if len(content) > max_upload_bytes:
        raise HTTPException(status_code=413, detail="Upload exceeds FUNASR_MAX_UPLOAD_MB")
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload")

    with tempfile.TemporaryDirectory(prefix="audio-power-funasr-") as temp_dir:
        temp_path = Path(temp_dir)
        uploaded_path = temp_path / f"upload{suffix}"
        normalized_path = temp_path / "normalized.wav"
        uploaded_path.write_bytes(content)
        try:
            await run_in_threadpool(_convert_to_16k_mono, uploaded_path, normalized_path)
            duration = await run_in_threadpool(_probe_duration_seconds, normalized_path)
            async with app.state.inference_semaphore:
                raw_result, processing_seconds = await run_in_threadpool(
                    _transcribe,
                    app.state.registry,
                    model_alias=model_alias,
                    audio_path=normalized_path,
                    language=language,
                    hotwords=hotwords,
                    diarize=spk,
                    preset_speaker_count=preset_speaker_count,
                )
        except RuntimeError as exc:
            LOGGER.exception("Transcription failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except subprocess.CalledProcessError as exc:
            raise HTTPException(status_code=500, detail=exc.stderr or "Audio probing failed") from exc

    text = _clean_text(raw_result.get("text"))
    segments = _build_segments(raw_result)
    if not segments and text:
        segments = [{"start": 0.0, "end": round(duration, 3), "text": text, "speaker": None}]
    speaker_ids = sorted({segment["speaker"] for segment in segments if segment["speaker"]})
    verbose = {
        "task": "transcribe",
        "language": language or "auto",
        "duration": round(duration, 3),
        "text": text,
        "model": model_alias,
        "processing_seconds": round(processing_seconds, 3),
        "diarization": {
            "enabled": spk,
            "speaker_count": len(speaker_ids),
            "speaker_ids": speaker_ids,
            "scope": "request",
        },
        "segments": [{"id": index, **segment} for index, segment in enumerate(segments)],
    }
    if response_format == "text":
        return JSONResponse(content={"text": text})
    if response_format == "verbose_json":
        return JSONResponse(content=verbose)
    return JSONResponse(content={"text": text})

