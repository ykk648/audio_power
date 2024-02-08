# -- coding: utf-8 --
# @Time : 2024/1/2
# @LastEdit : 2024/2/8
# @Author : ykk648

"""
ref https://github.com/coqui-ai/TTS
https://github.com/jianchang512/clone-voice/blob/main/clone/logic.py

pip install TTS

model download to ~/.local/share/tts
"""
from TTS.api import TTS


class TTSFreeVCAPI:
    def __init__(self, device='cuda'):
        self.tts = TTS(model_name='voice_conversion_models/multilingual/vctk/freevc24', progress_bar=False).to(device)

    def forward(self, source_wav, target_wav, output_path):
        self.tts.voice_conversion_to_file(source_wav=source_wav, target_wav=target_wav,
                                          file_path=output_path)


class TTSXTTSAPI:
    def __init__(self, device='cuda'):
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def forward(self, source_txt, target_wav, output_path):
        self.tts.tts_to_file(
            text=source_txt,
            speaker_wav=target_wav,
            language="zh", file_path=output_path)


class TTSBarkAPI:
    def __init__(self, device='cuda'):
        self.tts = TTS("tts_models/multilingual/multi-dataset/bark").to(device)

    def forward(self, source_txt, target_dir, output_path):
        # Cloning a new speaker
        # This expects to find a mp3 or wav file like `bark_voices/new_speaker/speaker.wav`
        # It computes the cloning values and stores in `bark_voices/new_speaker/speaker.npz`
        self.tts.tts_to_file(
            text=source_txt,
            file_path=output_path,
            voice_dir=target_dir,
            speaker="new_speaker")


if __name__ == '__main__':
    source_wav = "resources/daihuo.wav"
    target_wav = "resources/guangdian.wav"

    ttsfvc = TTSFreeVCAPI()
    ttsfvc.forward(source_wav, target_wav, 'output.wav')

    source_txt = '大家好，我是袁方。我祝贺群里的美女们笑容如花般绽放，生活如诗般甜美，愿你的每一天都充满阳光和欢笑，每一刻都洋溢着幸福和快乐。'
    ttsxtts = TTSXTTSAPI()
    ttsxtts.forward(
        source_txt,
        target_wav, 'output.wav')

    target_dir = ''
    ttsbark = TTSBarkAPI()
    ttsbark.forward(source_txt, target_wav, 'output.wav')
