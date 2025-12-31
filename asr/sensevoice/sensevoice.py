# -- coding: utf-8 --
# @Time : 2024/7/21
# @Author : ykk648
"""
SenseVoice API接口
ref https://huggingface.co/FunAudioLLM/SenseVoiceSmall
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
        :param word_timestamp: 是否返回词级别时间戳（SenseVoice可能不支持，保留参数以保持接口一致性）
        :return: SegmentDummy对象列表
        """
        # 转换语言代码
        if language:
            sensevoice_lang = LANGUAGE_MAP.get(language, "auto")
        else:
            sensevoice_lang = "auto"

        # 调用模型进行识别
        if self.use_vad:
            res = self.model.generate(
                input=audio_path,
                cache={},
                language=sensevoice_lang,
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
        else:
            res = self.model.generate(
                input=audio_path,
                cache={},
                language=sensevoice_lang,
                use_itn=False,
                batch_size=64,
                hub="hf",
            )

        # 处理结果
        recon_segments = []
        
        # SenseVoice返回的结果格式：res是一个列表，每个元素是一个字典，包含text等字段
        if isinstance(res, list) and len(res) > 0:
            # 遍历所有结果
            for result in res:
                if isinstance(result, dict):
                    # 如果结果包含segments信息（分段结果）
                    if "segments" in result and isinstance(result["segments"], list):
                        for seg in result["segments"]:
                            if isinstance(seg, dict):
                                start = seg.get("start", 0.0)
                                end = seg.get("end", 0.0)
                                text = seg.get("text", "")
                                if text:
                                    text = rich_transcription_postprocess(text)
                                    recon_segments.append(SegmentDummy(start=start, end=end, text=text))
                    # 如果结果只有text字段（单个结果）
                    elif "text" in result:
                        text = result["text"]
                        if text:
                            text = rich_transcription_postprocess(text)
                            # 尝试获取时间戳信息
                            start = result.get("start", 0.0)
                            end = result.get("end", 0.0)
                            # 如果时间戳为0，尝试从timestamp字段获取
                            if start == 0.0 and end == 0.0 and "timestamp" in result:
                                timestamp = result["timestamp"]
                                if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
                                    start, end = timestamp[0], timestamp[1]
                            recon_segments.append(SegmentDummy(start=start, end=end, text=text))
                elif isinstance(result, str):
                    # 如果直接返回字符串
                    text = rich_transcription_postprocess(result)
                    recon_segments.append(SegmentDummy(start=0.0, end=0.0, text=text))
        
        # 如果没有获取到任何结果，返回空列表
        if not recon_segments:
            print("Warning: No transcription results returned from SenseVoice")
            return recon_segments
        
        # 如果所有segments都没有时间戳，尝试根据文本长度进行简单分割
        # 注意：SenseVoice可能不支持时间戳，根据README说明
        has_timestamps = any(seg.start > 0 or seg.end > 0 for seg in recon_segments)
        if not has_timestamps and len(recon_segments) == 1:
            # 如果只有一个没有时间戳的segment，保持原样
            # 可以根据需要添加文本分割逻辑
            pass

        return recon_segments

