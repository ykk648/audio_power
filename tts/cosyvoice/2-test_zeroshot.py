import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from cv2box.utils import MyTimer

with MyTimer() as mt:
    cosyvoice = CosyVoice('pretrained_models/temp/CosyVoice-300M')

with MyTimer() as mt:
    # zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
    prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
    output = cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k)
    torchaudio.save('zero_shot.wav', output['tts_speech'], 22050)

# # cross_lingual usage
# prompt_speech_16k = load_wav('cross_lingual_prompt.wav', 16000)
# output = cosyvoice.inference_cross_lingual('<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.', prompt_speech_16k)
# torchaudio.save('cross_lingual.wav', output['tts_speech'], 22050)