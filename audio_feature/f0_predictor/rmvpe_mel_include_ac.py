# -- coding: utf-8 --
# @Time : 2023/11/10
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
from voice-changer, mel calculate(torch) included in onnx file
partly borrowed from https://github.com/w-okada/voice-changer/blob/master/server/voice_changer/RVC/pitchExtractor/RMVPEOnnxPitchExtractor.py
"""
from apstone import ModelBase
import numpy as np

"""
input_name:['waveform', 'threshold'], 
shape:[[1, 'waveform_dynamic_axes_1'], [1]], 
type:['tensor(float)', 'tensor(float)']
output_name:['pitchf'], 
shape:[['Mulpitchf_dim_0', 'Mulpitchf_dim_1']], 
type:['tensor(float)']
"""

MODEL_ZOO = {
    'rmvpe_mel_include': {
        'model_path': 'pretrain_models/audio_lib/rmvpe/rmvpe_mel_include.onnx'
    },
}


class RMVPEMel(ModelBase):
    def __init__(self, model_type='rmvpe_mel_include', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type
        self.f0 = None

        # for mel calculate
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def forward(self, audio, thres: float):
        """
        :param audio: (samples)
        :param thres:
        :return: (samples//160+1)
        """
        audio = np.expand_dims(audio, axis=0)
        output = self.model.forward([audio.astype(np.float32), np.array([thres]).astype(np.float32)])
        self.f0 = output[0].squeeze()
        return self.f0

    def get_course(self):
        # got mel
        f0_mel = 1127.0 * np.log(1.0 + self.f0 / 700.0)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(int)
        return f0_coarse


if __name__ == '__main__':
    import librosa

    audio_path = 'resources/guangdian.wav'
    audio_all = librosa.load(audio_path, sr=16000, mono=True)[0]

    from cv2box import MyFpsCounter
    rmvpe = RMVPEMel(model_type='rmvpe_mel_include', provider='gpu')
    # fps: 12 GPU: 6G-1.7G
    with MyFpsCounter() as mfc:
        for _ in range(10):
            f0 = rmvpe.forward(audio_all, 0.03)
    print(rmvpe.get_course())
    print(rmvpe.get_course().shape)
    print(f0[:100])
    print(f0.shape)
