# -- coding: utf-8 --
# @Time : 2024/7/21
# @Author : ykk648
"""
ref https://github.com/jianchang512/pyvideotrans/blob/main/videotrans/recognition/all.py
_LANGUAGE_CODES = ( "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh", "yue", )
"""
from dotenv import load_dotenv

load_dotenv()
import os
from faster_whisper import WhisperModel, BatchedInferencePipeline
import math
from tqdm import tqdm
from ..utils import split_thres, SegmentDummy

PROMPTS = {
    'zh': "合理断句，保留标点。",
    'en': 'Break sentences reasonably, keep punctuation.',
    'ja': 'Break sentences reasonably, keep punctuation.',
    'fr': 'Break sentences reasonably, keep punctuation.',
    'ko': 'Break sentences reasonably, keep punctuation.',
}


class FastWhisperAPI:
    def __init__(self):
        model_size = "large-v3"

        # Run on GPU with FP16
        model = WhisperModel(model_size, device="cuda", compute_type="float16", )

        self.model = BatchedInferencePipeline(model=model, chunk_length=30, )

    def forward(self, audio_path, language=None, word_timestamp=True):
        """

        :param audio_path:
        :param language: None means auto detect
        :param word_timestamp:
        :return:
        """
        try:
            initial_prompt = PROMPTS[language]
        except KeyError:
            initial_prompt = PROMPTS['en']

        segments, info = self.model.transcribe(audio_path,
                                               language=language,
                                               beam_size=5,
                                               length_penalty=1,
                                               initial_prompt=initial_prompt,
                                               temperature=0,
                                               word_timestamps=True,
                                               batch_size=16,
                                               # hotwords="",
                                               # vad_filter=True,
                                               # vad_parameters=dict(min_silence_duration_ms=2000, max_speech_duration_s=float('inf'), ),
                                               )

        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        recon_segments = []
        # line_count = 1
        timestamps = 0.0  # for progress bar
        with tqdm(total=info.duration, unit=" audio seconds") as pbar:
            for i, segment in enumerate(segments, start=1):
                # print("[%.3fs -> %.3fs] %s" % (segment.start, segment.end, segment.text))
                if len(segment.text) <= split_thres[language] or not word_timestamp:
                    recon_segments.append(segment)
                else:
                    cur = None
                    for word in segment.words:
                        if not cur:
                            cur = SegmentDummy(start=word.start, end=word.end, text=word.word)
                            continue
                        if word.start - cur.end < 0.5:
                            cur.end = word.end
                            cur.text += word.word
                            continue
                        recon_segments.append(cur)
                        cur = SegmentDummy(start=word.start, end=word.end, text=word.word)
                    if cur:
                        recon_segments.append(cur)

                pbar.update(segment.end - timestamps)
                timestamps = segment.end
            if timestamps < info.duration:  # silence at the end of the audio
                pbar.update(info.duration - timestamps)
        return recon_segments
