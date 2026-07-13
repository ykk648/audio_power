from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_MODEL = REPO_ROOT / "pretrain_models" / "audio_dirazation" / "pyannote" / "community-1"
DEFAULT_HF_MODEL = "pyannote/speaker-diarization-community-1"
ALLOWED_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".mp4", ".ogg", ".opus", ".wav", ".webm"}


@dataclass(frozen=True)
class PyannoteTurn:
    start: float
    end: float
    speaker: str


def resolve_default_model() -> str:
    env_model = os.getenv("PYANNOTE_MODEL_PATH") or os.getenv("PYANNOTE_MODEL")
    if env_model:
        return env_model
    if DEFAULT_LOCAL_MODEL.exists():
        return str(DEFAULT_LOCAL_MODEL)
    return DEFAULT_HF_MODEL


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def convert_to_16k_mono(source_path: str | Path, output_path: str | Path) -> None:
    source = Path(source_path)
    output = Path(output_path)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "ffmpeg audio conversion failed")


@contextlib.contextmanager
def trusted_torch_load_pickle(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    import torch

    original_load = torch.load

    def patched_load(*args: Any, **kwargs: Any) -> Any:
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load


def annotation_to_turns(annotation: Any) -> list[PyannoteTurn]:
    turns: list[PyannoteTurn] = []
    if hasattr(annotation, "itertracks"):
        for segment, _track, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                PyannoteTurn(
                    start=round(float(segment.start), 3),
                    end=round(float(segment.end), 3),
                    speaker=str(speaker),
                )
            )
        return turns

    for item in annotation:
        if len(item) == 2:
            segment, speaker = item
        else:
            segment, _track, speaker = item
        turns.append(
            PyannoteTurn(
                start=round(float(segment.start), 3),
                end=round(float(segment.end), 3),
                speaker=str(speaker),
            )
        )
    return turns


def write_rttm(turns: list[PyannoteTurn], output_path: str | Path, *, uri: str = "audio") -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for turn in turns:
        duration = max(0.0, turn.end - turn.start)
        rows.append(
            f"SPEAKER {uri} 1 {turn.start:.3f} {duration:.3f} <NA> <NA> {turn.speaker} <NA> <NA>"
        )
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


class PyannoteDiarizer:
    def __init__(
        self,
        *,
        model: str | Path | None = None,
        device: str = "auto",
        token: str | None = None,
        cache_dir: str | Path | None = None,
        trust_model_pickle: bool | None = None,
        use_exclusive: bool = True,
    ) -> None:
        self.model = str(model or resolve_default_model())
        self.device = device
        self.resolved_device: str | None = None
        self.token = token if token is not None else os.getenv("HF_TOKEN")
        self.cache_dir = str(cache_dir) if cache_dir else os.getenv("HF_HOME")
        self.trust_model_pickle = (
            _read_bool_env("PYANNOTE_TRUST_MODEL_PICKLE", True)
            if trust_model_pickle is None
            else trust_model_pickle
        )
        self.use_exclusive = use_exclusive
        self.pipeline: Any | None = None

    def load(self) -> Any:
        if self.pipeline is not None:
            return self.pipeline

        from pyannote.audio import Pipeline
        import torch

        kwargs: dict[str, Any] = {}
        if self.token:
            kwargs["token"] = self.token
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        with trusted_torch_load_pickle(self.trust_model_pickle):
            pipeline = Pipeline.from_pretrained(self.model, **kwargs)
        if self.device:
            self.resolved_device = self._resolve_device(torch)
            pipeline.to(torch.device(self.resolved_device))
        self.pipeline = pipeline
        return pipeline

    def _resolve_device(self, torch_module: Any) -> str:
        if self.device.lower() == "auto":
            return "cuda:0" if torch_module.cuda.is_available() else "cpu"
        return self.device

    def diarize(
        self,
        input_path: str | Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        preprocess: bool = True,
    ) -> dict[str, Any]:
        source = Path(input_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        suffix = source.suffix.lower()
        if suffix and suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported audio extension: {suffix}")

        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="audio-power-pyannote-") as temp_dir:
            audio_path = source
            if preprocess:
                audio_path = Path(temp_dir) / "normalized.wav"
                convert_to_16k_mono(source, audio_path)

            import torchaudio

            waveform, sample_rate = torchaudio.load(str(audio_path))
            duration = round(float(waveform.shape[-1]) / float(sample_rate), 3)

            pipeline = self.load()
            call_kwargs: dict[str, Any] = {}
            if num_speakers is not None:
                call_kwargs["num_speakers"] = num_speakers
            if min_speakers is not None:
                call_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                call_kwargs["max_speakers"] = max_speakers

            output = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **call_kwargs)
            annotation = output.speaker_diarization
            diarization_kind = "speaker_diarization"
            if self.use_exclusive and hasattr(output, "exclusive_speaker_diarization"):
                annotation = output.exclusive_speaker_diarization
                diarization_kind = "exclusive_speaker_diarization"

        turns = annotation_to_turns(annotation)
        return {
            "input": str(source),
            "input_duration_seconds": duration,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "device": self.resolved_device or self.device,
            "model": self.model,
            "diarization_kind": diarization_kind,
            "num_speakers_requested": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "turns": [asdict(turn) for turn in turns],
        }

    @staticmethod
    def save_json(payload: Mapping[str, Any], output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
