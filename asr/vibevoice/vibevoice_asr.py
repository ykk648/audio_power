# -- coding: utf-8 --
import requests
import json
import base64
import os
import subprocess

class VibeVoiceASR:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def _guess_mime_type(self, path: str) -> str:
        """Guess MIME type from file extension."""
        ext = os.path.splitext(path)[1].lower()
        mime_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".mp4": "video/mp4",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".opus": "audio/ogg",
        }
        return mime_map.get(ext, "application/octet-stream")

    def _get_duration_seconds_ffprobe(self, path: str) -> float:
        """Get audio duration using ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8").strip()
        return float(out)

    def _is_video_file(self, path: str) -> bool:
        """Check if the file is a video file that needs audio extraction."""
        ext = os.path.splitext(path)[1].lower()
        return ext in (".mp4", ".m4v", ".mov", ".webm", ".avi", ".mkv")

    def _extract_audio_from_video(self, video_path: str) -> str:
        """
        Extract audio from video file (mp4/mov/webm) to a temporary mp3 file.
        Returns the path to the extracted audio file.
        """
        import tempfile
        fd, audio_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",  # No video
            "-acodec", "libmp3lame",
            "-q:a", "2",  # High quality
            audio_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return audio_path

    def forward(self, audio_path: str, context_info: str = None, language: str = 'zh'):
        """
        :param audio_path: Path to the audio file
        :param context_info: Hotwords string (e.g., "Microsoft,Azure,VibeVoice")
        :param language: Language code
        :return: JSON formatted transcription result
        """
        temp_audio_path = None
        actual_audio_path = audio_path
        
        if self._is_video_file(audio_path):
            temp_audio_path = self._extract_audio_from_video(audio_path)
            actual_audio_path = temp_audio_path
            
        try:
            duration = self._get_duration_seconds_ffprobe(actual_audio_path)
            
            with open(actual_audio_path, "rb") as f:
                audio_bytes = f.read()
            
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            
        except Exception as e:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            raise Exception(f"Error preparing audio: {e}")

        url = f"{self.base_url}/v1/chat/completions"
        show_keys = ["Start time", "End time", "Speaker ID", "Content"]
        
        if context_info and context_info.strip():
            prompt_text = (
                f"This is a {duration:.2f} seconds audio, with extra info: {context_info.strip()}\n\n"
                f"Please transcribe it with these keys: " + ", ".join(show_keys) + ".\n"
                f"Please output the result purely in JSON format as a list of dictionaries."
            )
        else:
            prompt_text = (
                f"This is a {duration:.2f} seconds audio, please transcribe it with these keys: "
                + ", ".join(show_keys) + ".\n"
                f"Please output the result purely in JSON format as a list of dictionaries."
            )

        mime = self._guess_mime_type(actual_audio_path)
        data_url = f"data:{mime};base64,{audio_b64}"
        
        payload = {
            "model": "vibevoice",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that transcribes audio input into text output in JSON format."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": data_url}},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ],
            "max_tokens": 2048,       
            "temperature": 0.0,      
            "stream": False,
            "top_p": 1.0,
        }
        
        try:
            response = requests.post(url, json=payload, timeout=12000)
            if response.status_code == 200:
                result_json = response.json()
                content = result_json['choices'][0]['message']['content']
                
                # 尝试解析返回的文本为JSON
                try:
                    # 去除可能存在的markdown代码块标记
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    
                    parsed_content = json.loads(content.strip())
                    return parsed_content
                except json.JSONDecodeError:
                    # 如果解析失败，返回原始文本和结构
                    return {"raw_content": content}
            else:
                raise Exception(f"API Error {response.status_code}: {response.text}")
                
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
