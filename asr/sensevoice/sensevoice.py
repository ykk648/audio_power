# -- coding: utf-8 --
# @Time : 2024/7/21
# @Author : ykk648
"""
SenseVoice API 接口
- ref: https://huggingface.co/FunAudioLLM/SenseVoiceSmall
- 时间戳：2024.11 起支持基于 CTC 对齐的 output_timestamp，用法见 demo2.py
"""
from dotenv import load_dotenv

load_dotenv()
import os
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from ..utils import split_thres, SegmentDummy

# SenseVoice支持的语言代码映射
LANGUAGE_MAP = {
    'zh': 'zn',  # 中文
    'en': 'en',  # 英文
    'yue': 'yue',  # 粤语
    'ja': 'ja',  # 日语
    'ko': 'ko',  # 韩语
}


class SenseVoiceAPI:
    def __init__(self, model_dir="FunAudioLLM/SenseVoiceSmall", device="cuda:0", use_vad=True):
        """
        初始化SenseVoice模型
        
        :param model_dir: 模型路径或HuggingFace模型名称
        :param device: 设备，如 "cuda:0" 或 "cpu"
        :param use_vad: 是否使用VAD（语音活动检测）
        """
        if use_vad:
            self.model = AutoModel(
                model=model_dir,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=device,
                hub="hf",
            )
        else:
            self.model = AutoModel(
                model=model_dir,
                device=device,
                hub="hf",
            )
        self.use_vad = use_vad

    def forward(self, audio_path, language=None, word_timestamp=True):
        """
        对音频进行语音识别

        :param audio_path: 音频文件路径
        :param language: 语言代码，None表示自动检测。支持: 'zh', 'en', 'yue', 'ja', 'ko'
        :param word_timestamp: 是否返回时间戳（SenseVoice 基于 CTC 对齐，2024.11 起支持 output_timestamp）
        :return: SegmentDummy 对象列表
        """
        # 转换语言代码
        if language:
            sensevoice_lang = LANGUAGE_MAP.get(language, "auto")
        else:
            sensevoice_lang = "auto"

        # 调用模型进行识别（支持 output_timestamp 获取基于 CTC 对齐的时间戳，参考 demo2.py）
        gen_kwargs = dict(
            input=audio_path,
            cache={},
            language=sensevoice_lang,
            output_timestamp=word_timestamp,
        )
        if self.use_vad:
            res = self.model.generate(
                **gen_kwargs,
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
        else:
            res = self.model.generate(
                **gen_kwargs,
                use_itn=False,
                batch_size=64,
                hub="hf",
            )

        # 处理结果
        recon_segments = []

        # SenseVoice 返回格式：res 为列表，每项为字典或列表；开启 output_timestamp 时有 timestamp 字段（CTC 对齐）
        def parse_one(item):
            if isinstance(item, dict):
                if "timestamp" in item and item.get("timestamp") and word_timestamp:
                    # 基于 CTC 对齐的时间戳：可能是 [[s,e],[s,e],...] 或 [s, e]
                    ts = item["timestamp"]
                    text = (item.get("text") or "")
                    if text:
                        text = rich_transcription_postprocess(text)
                    if isinstance(ts, (list, tuple)) and len(ts) > 0:
                        first = ts[0]
                        if isinstance(first, (list, tuple)) and len(first) >= 2:
                            # 多段 [[s,e],[s,e],...]
                            for pair in ts:
                                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                                    s, e = float(pair[0]), float(pair[1])
                                    recon_segments.append(SegmentDummy(start=s, end=e, text=""))
                            if recon_segments and text:
                                recon_segments[0].text = text
                        elif isinstance(first, (int, float)) and len(ts) >= 2:
                            # 单段 [s, e]
                            recon_segments.append(SegmentDummy(start=float(ts[0]), end=float(ts[1]), text=text))
                        elif text:
                            recon_segments.append(SegmentDummy(start=0.0, end=0.0, text=text))
                    elif text:
                        recon_segments.append(SegmentDummy(start=0.0, end=0.0, text=text))
                elif "segments" in item and isinstance(item["segments"], list):
                    for seg in item["segments"]:
                        if isinstance(seg, dict):
                            start = seg.get("start", 0.0)
                            end = seg.get("end", 0.0)
                            text = seg.get("text", "")
                            if text:
                                text = rich_transcription_postprocess(text)
                            recon_segments.append(SegmentDummy(start=start, end=end, text=text))
                elif "text" in item:
                    text = item["text"]
                    if text:
                        text = rich_transcription_postprocess(text)
                    start = item.get("start", 0.0)
                    end = item.get("end", 0.0)
                    if start == 0.0 and end == 0.0 and "timestamp" in item:
                        ts = item["timestamp"]
                        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                            start, end = float(ts[0]), float(ts[1])
                    recon_segments.append(SegmentDummy(start=start, end=end, text=text))
            elif isinstance(item, str) and item:
                recon_segments.append(SegmentDummy(start=0.0, end=0.0, text=rich_transcription_postprocess(item)))

        if isinstance(res, list):
            for result in res:
                if isinstance(result, list):
                    for sub in result:
                        parse_one(sub)
                else:
                    parse_one(result)
        
        # 如果没有获取到任何结果，返回空列表
        if not recon_segments:
            print("Warning: No transcription results returned from SenseVoice")
            return recon_segments
        
        # 若全部 segment 均无时间戳则保持原样
        has_timestamps = any(seg.start > 0 or seg.end > 0 for seg in recon_segments)
        if not has_timestamps and len(recon_segments) == 1:
            pass

        return recon_segments

