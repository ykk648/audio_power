# -- coding: utf-8 --
# @LastEdit : 2024/5/13
# @Author : ykk648

import librosa
import torch
import numpy as np
from cv2box import MyFpsCounter
from audio_preprocess.audio_feature.f0_predictor.rmvpe_mel_include_ac import RMVPEMel
import librosa


audio_path = 'resources/guangdian.wav'
audio_all = librosa.load(audio_path, sr=16000, mono=True)[0]

rmvpe = RMVPEMel(model_type='rmvpe_mel_include', provider='gpu')
# fps: 12 GPU: 6G-1.7G
with MyFpsCounter() as mfc:
    for _ in range(10):
        f0 = rmvpe.forward(audio_all, 0.03)
print(rmvpe.get_course())
print(rmvpe.get_course().shape)
print(f0[:100])
print(f0.shape)
