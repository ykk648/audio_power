# -- coding: utf-8 --
import argparse
import json
import os
import sys

# 将父目录添加到 sys.path 中，以便能够导入 asr 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asr.vibevoice import VibeVoiceASR

def main():
    parser = argparse.ArgumentParser(description="Test VibeVoice ASR API and save JSON result")
    parser.add_argument("--audio_path", default=r"", help="Path to the audio or video file")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="VibeVoice API URL")
    parser.add_argument("--out", default="vibevoice_result.json", help="Output JSON file path (default: vibevoice_result.json)")
    parser.add_argument("--hotwords", default="", help="Optional hotwords (e.g., 'Microsoft,Azure')")
    
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"❌ Error: File not found - {args.audio_path}")
        return

    print(f"Connecting to {args.url}...")
    asr = VibeVoiceASR(base_url=args.url)
    
    print(f"Transcribing '{args.audio_path}'...")
    try:
        # 调用 ASR 接口
        result = asr.forward(audio_path=args.audio_path, context_info=args.hotwords)
        
        # 将结果保存为 JSON 文件
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
            
        print(f"✅ Transcription successful! Result saved to {args.out}")
        
        # 打印部分结果以供预览
        print("\n--- Result Preview ---")
        if isinstance(result, list):
            for item in result[:3]:  # 只打印前三条
                print(item)
            if len(result) > 3:
                print("...")
        else:
            print(result)
            
    except Exception as e:
        print(f"❌ Error during transcription: {e}")

if __name__ == "__main__":
    main()
