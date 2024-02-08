# -- coding: utf-8 --
# @Time : 2024/2/8
# @Author : ykk648

import librosa
import torch

from audio_preprocess.audio_feature.hubert.dphubert import DPHuBERT
from audio_preprocess.audio_feature.hubert import Hubert, FairseqHubert, ContentVec

audio_path = 'resources/guangdian.wav'
input_audio = librosa.load(audio_path, sr=16000, mono=True)[0][None, ...]
# input_audio = np.random.random((1, 7040)).astype(np.float32)

# DPHuBERT
# hubert = DPHuBERT('dphubert', 'cuda')
# Hubert
# hubert = Hubert('hubert_base_12_768_fp16', 'gpu')
# # ContentVec
# hubert = ContentVec('vec-768-layer-12', 'gpu')
# # FairseqHubert
hubert = FairseqHubert('hubert_base', 'cuda')

from cv2box import MyFpsCounter

with MyFpsCounter() as mfc:
    for _ in range(100):
        embedding_out = hubert.forward(input_audio)
    # print(embedding_out)
    print(embedding_out.shape)
