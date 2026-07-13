from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .pyannote_api import PyannoteDiarizer, PyannoteTurn, resolve_default_model, write_rttm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pyannote speaker diarization CLI")
    parser.add_argument("input", help="Audio or video file")
    parser.add_argument("--model", default=resolve_default_model(), help="Local model path or Hugging Face model id")
    parser.add_argument("--device", default=os.getenv("PYANNOTE_DEVICE", "auto"))
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--cache-dir", default=os.getenv("HF_HOME"))
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--output-json", help="Write JSON result to this path")
    parser.add_argument("--output-rttm", help="Write RTTM result to this path")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip ffmpeg 16k mono normalization")
    parser.add_argument("--regular", action="store_true", help="Use regular diarization instead of exclusive diarization")
    parser.add_argument(
        "--trust-model-pickle",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("PYANNOTE_TRUST_MODEL_PICKLE", "true").lower() in {"1", "true", "yes", "on"},
        help="Allow trusted local pyannote checkpoints to load with torch weights_only=False",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    diarizer = PyannoteDiarizer(
        model=args.model,
        device=args.device,
        token=args.hf_token,
        cache_dir=args.cache_dir,
        trust_model_pickle=args.trust_model_pickle,
        use_exclusive=not args.regular,
    )
    result = diarizer.diarize(
        args.input,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        preprocess=not args.no_preprocess,
    )

    if args.output_json:
        diarizer.save_json(result, args.output_json)
    if args.output_rttm:
        turns = [PyannoteTurn(**row) for row in result["turns"]]
        write_rttm(turns, args.output_rttm, uri=Path(args.input).stem)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
