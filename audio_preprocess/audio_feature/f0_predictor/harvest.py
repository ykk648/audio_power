# -- coding: utf-8 --
# @Time : 2024/1/9
# @Author : ykk648

import pyworld
import numpy as np
import scipy.signal as signal


class Harvest:
    def __init__(self, model_type='rmvpe_mel_include', provider='gpu'):
        self.model_type = model_type

        self.sr = 16000
        self.f0_min = 50
        self.f0_max = 1100

    def forward(self, audio):
        audio = audio.astype(np.double)
        fs = self.sr
        frame_period = 10
        f0, t = pyworld.harvest(
            audio,
            fs=fs,
            f0_ceil=self.f0_max,
            f0_floor=self.f0_min,
            frame_period=frame_period,
        )
        f0 = pyworld.stonemask(audio, f0, t, fs)

        filter_radius = 3
        if filter_radius > 2:
            f0 = signal.medfilt(f0, 3)

        return f0


if __name__ == '__main__':
    import librosa

    audio_all = librosa.load("resources/guangdian.wav", sr=16000, mono=True)[0]

    from cv2box import MyFpsCounter

    harvest = Harvest()
    # fps: 0.4 GPU: 0
    with MyFpsCounter() as mfc:
        for _ in range(100):
            f0 = harvest.forward(audio_all)
        print(f0.shape)
