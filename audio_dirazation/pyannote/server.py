from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from .pyannote_api import ALLOWED_EXTENSIONS, PyannoteDiarizer, resolve_default_model


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.diarizer = PyannoteDiarizer(
        model=os.getenv("PYANNOTE_MODEL_PATH") or os.getenv("PYANNOTE_MODEL") or resolve_default_model(),
        device=os.getenv("PYANNOTE_DEVICE", "auto"),
        token=os.getenv("HF_TOKEN"),
        cache_dir=os.getenv("HF_HOME"),
        trust_model_pickle=_read_bool_env("PYANNOTE_TRUST_MODEL_PICKLE", True),
        use_exclusive=_read_bool_env("PYANNOTE_USE_EXCLUSIVE", True),
    )
    app.state.inference_semaphore = asyncio.Semaphore(max(1, _read_int_env("PYANNOTE_MAX_CONCURRENCY", 1)))
    if _read_bool_env("PYANNOTE_PRELOAD", True):
        await run_in_threadpool(app.state.diarizer.load)
    yield


app = FastAPI(
    title="Audio Power pyannote Diarization API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": app.state.diarizer.model,
        "device": app.state.diarizer.device,
        "loaded": app.state.diarizer.pipeline is not None,
    }


@app.post("/v1/audio/diarizations")
async def diarize(
    file: UploadFile = File(...),
    num_speakers: int | None = Form(default=None),
    min_speakers: int | None = Form(default=None),
    max_speakers: int | None = Form(default=None),
    preprocess: bool = Form(default=True),
):
    for name, value in {
        "num_speakers": num_speakers,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
    }.items():
        if value is not None and value < 1:
            raise HTTPException(status_code=400, detail=f"{name} must be positive")

    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported audio extension: {suffix or '(none)'}")
    content = await file.read()
    max_upload_bytes = _read_int_env("PYANNOTE_MAX_UPLOAD_MB", 500) * 1024 * 1024
    if len(content) > max_upload_bytes:
        raise HTTPException(status_code=413, detail="Upload exceeds PYANNOTE_MAX_UPLOAD_MB")
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload")

    with tempfile.TemporaryDirectory(prefix="audio-power-pyannote-api-") as temp_dir:
        uploaded_path = Path(temp_dir) / f"upload{suffix}"
        uploaded_path.write_bytes(content)
        try:
            async with app.state.inference_semaphore:
                result = await run_in_threadpool(
                    app.state.diarizer.diarize,
                    uploaded_path,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    preprocess=preprocess,
                )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    result["input"] = file.filename or result["input"]
    return JSONResponse(content=result)
