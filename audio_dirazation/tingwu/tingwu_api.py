# -- coding: utf-8 --
# @Time : 2026/6/14
# @Author : ykk648
"""
阿里云通义听悟 离线音视频转写/说话人分离 API
- ref: https://help.aliyun.com/zh/tingwu/api-tingwu-2023-09-30-overview
- pip install alibabacloud-tea-openapi alibabacloud-tingwu20230930 requests python-dotenv
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import urlparse

import requests
from alibabacloud_tea_openapi import utils_models as open_api_models
from alibabacloud_tingwu20230930.client import Client as TingwuOpenApiClient
from alibabacloud_tingwu20230930 import models as tingwu_models

SUCCESS_CODES = {"0", 0}
TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELED"}


@dataclass
class TingwuSubtitleSegment:
    sentence_id: str
    speaker_id: str
    start_ms: int
    end_ms: int
    text: str


class TingwuClient:
    def __init__(
        self,
        *,
        access_key_id: str,
        access_key_secret: str,
        app_key: str,
        region_id: str = "cn-beijing",
        endpoint: str = "tingwu.cn-beijing.aliyuncs.com",
        connect_timeout: int = 10_000,
        read_timeout: int = 300_000,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.app_key = app_key
        self.session = session or requests.Session()
        self.client = TingwuOpenApiClient(
            open_api_models.Config(
                access_key_id=access_key_id,
                access_key_secret=access_key_secret,
                region_id=region_id,
                endpoint=endpoint,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
        )

    def create_offline_task(
        self,
        *,
        file_url: str,
        source_language: str,
        task_key: Optional[str] = None,
        translation_languages: Optional[Iterable[str]] = None,
        speaker_count: Optional[int] = None,
        enable_meeting_assistance: bool = False,
        enable_summarization: bool = False,
    ) -> Dict[str, Any]:
        translation_targets = [lang for lang in (translation_languages or []) if lang]

        transcription = tingwu_models.CreateTaskRequestParametersTranscription()
        if speaker_count is not None:
            transcription.diarization_enabled = True
            transcription.diarization = (
                tingwu_models.CreateTaskRequestParametersTranscriptionDiarization(
                    speaker_count=speaker_count
                )
            )

        parameters = tingwu_models.CreateTaskRequestParameters(
            transcription=transcription,
            meeting_assistance_enabled=enable_meeting_assistance,
            summarization_enabled=enable_summarization,
        )
        if translation_targets:
            parameters.translation_enabled = True
            parameters.translation = tingwu_models.CreateTaskRequestParametersTranslation(
                target_languages=translation_targets
            )

        request = tingwu_models.CreateTaskRequest(
            app_key=self.app_key,
            type="offline",
            input=tingwu_models.CreateTaskRequestInput(
                file_url=file_url,
                source_language=source_language,
                task_key=task_key,
            ),
            parameters=parameters,
        )
        response = self.client.create_task(request)
        payload = response.to_map()
        self._ensure_success(payload)
        return payload

    def get_task_info(self, task_id: str) -> Dict[str, Any]:
        response = self.client.get_task_info(task_id)
        payload = response.to_map()
        self._ensure_success(payload)
        return payload

    def wait_for_task(
        self,
        task_id: str,
        *,
        poll_interval_seconds: float = 10.0,
        timeout_seconds: float = 3600.0,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last_payload: Optional[Dict[str, Any]] = None
        while time.monotonic() <= deadline:
            payload = self.get_task_info(task_id)
            last_payload = payload
            status = self.extract_task_status(payload)
            if status in TERMINAL_STATUSES:
                return payload
            time.sleep(poll_interval_seconds)
        raise TimeoutError(
            f"Timed out waiting for task {task_id} after {timeout_seconds:.1f}s. "
            f"Last status: {self.extract_task_status(last_payload or {}) or 'UNKNOWN'}"
        )

    @staticmethod
    def extract_task_id(payload: Mapping[str, Any]) -> str:
        data = payload.get("body", {}).get("Data", {})
        task_id = data.get("TaskId")
        if not task_id:
            raise RuntimeError(f"TaskId not found in response: {json.dumps(payload, ensure_ascii=False)}")
        return str(task_id)

    @staticmethod
    def extract_task_status(payload: Mapping[str, Any]) -> str:
        data = payload.get("body", {}).get("Data", {})
        return str(data.get("TaskStatus", ""))

    @staticmethod
    def extract_result_urls(payload: Mapping[str, Any]) -> Dict[str, str]:
        data = payload.get("body", {}).get("Data", {})
        result = data.get("Result") or {}
        result_urls = {
            "transcription": str(result.get("Transcription", "") or ""),
            "translation": str(result.get("Translation", "") or ""),
            "summarization": str(result.get("Summarization", "") or ""),
            "meeting_assistance": str(result.get("MeetingAssistance", "") or ""),
            "auto_chapters": str(result.get("AutoChapters", "") or ""),
            "content_extraction": str(result.get("ContentExtraction", "") or ""),
            "identity_recognition": str(result.get("IdentityRecognition", "") or ""),
            "text_polish": str(result.get("TextPolish", "") or ""),
            "custom_prompt": str(result.get("CustomPrompt", "") or ""),
            "service_inspection": str(result.get("ServiceInspection", "") or ""),
            "ppt_extraction": str(result.get("PptExtraction", "") or ""),
            "output_mp3": str(data.get("OutputMp3Path", "") or ""),
            "output_mp4": str(data.get("OutputMp4Path", "") or ""),
            "output_thumbnail": str(data.get("OutputThumbnailPath", "") or ""),
            "output_spectrum": str(data.get("OutputSpectrumPath", "") or ""),
        }
        return {key: value for key, value in result_urls.items() if value}

    def download_result_files(
        self,
        result_urls: Mapping[str, str],
        *,
        output_dir: str | Path,
    ) -> List[Path]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        for key, url in result_urls.items():
            target_path = root / self._build_download_name(key, url)
            response = self.session.get(url, timeout=300)
            response.raise_for_status()
            target_path.write_bytes(response.content)
            written.append(target_path)
        return written

    @staticmethod
    def load_json(path: str | Path) -> Dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def extract_transcription_segments(payload: Mapping[str, Any]) -> List[TingwuSubtitleSegment]:
        transcription = payload.get("Transcription") or {}
        paragraphs = transcription.get("Paragraphs") or []
        rows: List[TingwuSubtitleSegment] = []
        for paragraph in paragraphs:
            if not isinstance(paragraph, Mapping):
                continue
            speaker_id = str(paragraph.get("SpeakerId", "") or "")
            words = paragraph.get("Words") or []
            merged_by_sentence: Dict[str, Dict[str, Any]] = {}
            sentence_order: List[str] = []
            for word in words:
                if not isinstance(word, Mapping):
                    continue
                sentence_id = str(word.get("SentenceId", "") or "")
                text = str(word.get("Text", "") or "")
                if not sentence_id or not text:
                    continue
                if sentence_id not in merged_by_sentence:
                    merged_by_sentence[sentence_id] = {
                        "start_ms": int(word.get("Start", 0) or 0),
                        "end_ms": int(word.get("End", 0) or 0),
                        "parts": [text],
                    }
                    sentence_order.append(sentence_id)
                else:
                    merged = merged_by_sentence[sentence_id]
                    merged["end_ms"] = int(word.get("End", merged["end_ms"]) or merged["end_ms"])
                    merged["parts"].append(text)
            for sentence_id in sentence_order:
                merged = merged_by_sentence[sentence_id]
                rows.append(
                    TingwuSubtitleSegment(
                        sentence_id=sentence_id,
                        speaker_id=speaker_id,
                        start_ms=int(merged["start_ms"]),
                        end_ms=int(merged["end_ms"]),
                        text="".join(merged["parts"]).strip(),
                    )
                )
        rows.sort(key=lambda item: (item.start_ms, item.end_ms, item.sentence_id))
        return rows

    @staticmethod
    def write_srt(segments: Iterable[TingwuSubtitleSegment], output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blocks: List[str] = []
        for idx, seg in enumerate(segments, start=1):
            text = seg.text
            if seg.speaker_id:
                text = f"[S{int(seg.speaker_id):02d}] {text}" if seg.speaker_id.isdigit() else f"[{seg.speaker_id}] {text}"
            blocks.append(
                "\n".join(
                    [
                        str(idx),
                        f"{TingwuClient._ms_to_srt_time(seg.start_ms)} --> {TingwuClient._ms_to_srt_time(seg.end_ms)}",
                        text,
                    ]
                )
            )
        path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")

    @staticmethod
    def save_json(payload: Mapping[str, Any], output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _ensure_success(payload: Mapping[str, Any]) -> None:
        body = payload.get("body", {})
        code = body.get("Code")
        if code not in SUCCESS_CODES:
            raise RuntimeError(
                f"Tingwu API error: code={code!r}, message={body.get('Message')!r}, "
                f"payload={json.dumps(payload, ensure_ascii=False)}"
            )

    @staticmethod
    def _build_download_name(key: str, url: str) -> str:
        parsed = urlparse(url)
        ext = Path(parsed.path).suffix or ".json"
        safe_key = key.replace("/", "_")
        return f"{safe_key}{ext}"

    @staticmethod
    def _ms_to_srt_time(value: int) -> str:
        total_ms = max(0, int(value))
        hours, rem = divmod(total_ms, 3_600_000)
        minutes, rem = divmod(rem, 60_000)
        seconds, millis = divmod(rem, 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
