# -- coding: utf-8 --
# @Time : 2024/1/9
# @Author : ykk648

import parselmouth
import numpy as np
import scipy.signal as signal


class PM:
    def __init__(self):

        self.sr = 16000
        self.f0_min = 50
        self.f0_max = 1100
        self.window = 160

        self.time_step = self.window / self.sr * 1000

    def forward(self, audio, p_len):
        f0 = (
            parselmouth.Sound(audio, self.sr)
            .to_pitch_ac(
                time_step=self.time_step / 1000,
                voicing_threshold=0.6,
                pitch_floor=self.f0_min,
                pitch_ceiling=self.f0_max,
            )
            .selected_array["frequency"]
        )
        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.pad(
                f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
            )

        return f0


if __name__ == '__main__':
    import librosa

    audio_all = librosa.load("resources/guangdian.wav", sr=16000, mono=True)[0]

    from cv2box import MyFpsCounter
    pm = PM()
    # fps: 150 GPU: 0
    with MyFpsCounter() as mfc:
        for _ in range(100):
            f0 = pm.forward(audio_all, 1001)
        print(f0.shape)
