# -- coding: utf-8 --
# @Time : 2024/8/1
# @Author : ykk648
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from cv2box.utils import MyTimer
from utils.CVAudio import CVAudio


class CosyVoiceAPI:
    def __init__(self):
        self.model = CosyVoice('pretrained_models/temp/CosyVoice-300M')

    def forward(self, prompt_speech_16k, tts_text, dest_lang, output_path='zero_shot.wav'):
        assert dest_lang in ['en', 'zh', 'yue', 'ko', 'ja']
        with MyTimer('cosyvoice infer') as _:
            output = self.model.inference_cross_lingual(f'<|{dest_lang}|>{tts_text}', prompt_speech_16k)
        torchaudio.save(output_path, output['tts_speech'], 22050)
        return output_path