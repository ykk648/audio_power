from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="阿里云通义听悟离线音视频转写 CLI"
    )
    parser.add_argument("--file-url", required=True, help="音视频 http(s) URL")
    parser.add_argument("--source-language", default="cn", choices=["cn", "en", "ja", "yue", "fspk"])
    parser.add_argument("--task-key", help="可选的业务任务 key")
    parser.add_argument("--translation-language", action="append", dest="translation_languages",
                        help="可重复，目标翻译语言: cn/en/ja")
    parser.add_argument("--speaker-count", type=int,
                        help="说话人分离数量，0 表示自动推断")
    parser.add_argument("--enable-meeting-assistance", action="store_true")
    parser.add_argument("--enable-summarization", action="store_true")
    parser.add_argument("--wait", action="store_true", help="轮询直到任务完成")
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--download-results", action="store_true", help="下载结果文件")
    parser.add_argument("--export-srt", action="store_true", help="从 transcription.json 导出 SRT 字幕")
    parser.add_argument("--output-dir", default="work/tingwu")
    parser.add_argument("--access-key-id", default=os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID"))
    parser.add_argument("--access-key-secret", default=os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET"))
    parser.add_argument("--app-key", default=os.getenv("TINGWU_APP_KEY"))
    parser.add_argument("--region-id", default=os.getenv("TINGWU_REGION_ID", "cn-beijing"))
    parser.add_argument("--endpoint", default=os.getenv("TINGWU_ENDPOINT", "tingwu.cn-beijing.aliyuncs.com"))
    return parser


def require_arg(value: str | None, name: str) -> str:
    if value:
        return value
    raise ValueError(f"Missing required argument or env var: {name}")


def main() -> int:
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'python-dotenv'. Install with: pip install python-dotenv") from exc

    load_dotenv(find_dotenv(filename=".env", usecwd=True))
    args = build_parser().parse_args()

    from .tingwu_api import TingwuClient

    client = TingwuClient(
        access_key_id=require_arg(args.access_key_id, "ALIBABA_CLOUD_ACCESS_KEY_ID / --access-key-id"),
        access_key_secret=require_arg(args.access_key_secret, "ALIBABA_CLOUD_ACCESS_KEY_SECRET / --access-key-secret"),
        app_key=require_arg(args.app_key, "TINGWU_APP_KEY / --app-key"),
        region_id=args.region_id,
        endpoint=args.endpoint,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    create_payload = client.create_offline_task(
        file_url=args.file_url,
        source_language=args.source_language,
        task_key=args.task_key,
        translation_languages=args.translation_languages,
        speaker_count=args.speaker_count,
        enable_meeting_assistance=args.enable_meeting_assistance,
        enable_summarization=args.enable_summarization,
    )
    create_path = output_dir / "create_task_response.json"
    client.save_json(create_payload, create_path)

    task_id = client.extract_task_id(create_payload)
    task_status = client.extract_task_status(create_payload)
    print(f"[INFO] TaskId: {task_id}")
    print(f"[INFO] Initial status: {task_status}")
    print(f"[INFO] Wrote {create_path}")

    if not args.wait:
        return 0

    task_info = client.wait_for_task(
        task_id,
        poll_interval_seconds=args.poll_interval,
        timeout_seconds=args.timeout,
    )
    task_info_path = output_dir / "task_info.json"
    client.save_json(task_info, task_info_path)
    final_status = client.extract_task_status(task_info)
    print(f"[INFO] Final status: {final_status}")
    print(f"[INFO] Wrote {task_info_path}")

    result_urls = client.extract_result_urls(task_info)
    result_urls_path = output_dir / "result_urls.json"
    client.save_json(result_urls, result_urls_path)
    print(f"[INFO] Wrote {result_urls_path}")

    if final_status != "COMPLETED":
        data = task_info.get("body", {}).get("Data", {})
        raise RuntimeError(
            f"Tingwu task failed: status={final_status}, "
            f"error_code={data.get('ErrorCode')!r}, error_message={data.get('ErrorMessage')!r}"
        )

    if args.download_results:
        downloads = client.download_result_files(result_urls, output_dir=output_dir / "downloads")
        for path in downloads:
            print(f"[INFO] Downloaded {path}")
        transcription_path = output_dir / "downloads" / "transcription.json"
        if args.export_srt and transcription_path.is_file():
            transcription_payload = client.load_json(transcription_path)
            segments = client.extract_transcription_segments(transcription_payload)
            srt_path = output_dir / "speaker_subtitles.srt"
            client.write_srt(segments, srt_path)
            print(f"[INFO] Wrote {srt_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
