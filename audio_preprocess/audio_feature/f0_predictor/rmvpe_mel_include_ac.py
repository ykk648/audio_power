# -- coding: utf-8 --
# @LastEdit : 2025/5/15
# @Author : ykk648

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
    def __init__(self, model_type='rmvpe_mel_include', provider='gpu', device=0):
        super().__init__(MODEL_ZOO[model_type], provider, device)
        self.model_type = model_type
        self.f0 = None

        # for mel calculate
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        self.max_sample = 2000000

    def forward_ori(self, audio, thres: float):
        """
        :param audio: (samples)
        :param thres:
        :return: (samples//160+1)
        """
        audio = np.expand_dims(audio, axis=0)
        output = self.model.forward([audio.astype(np.float32), np.array([thres]).astype(np.float32)])
        self.f0 = output[0].squeeze()
        return self.f0

    def forward(self, audio, thres: float):
        """
        chunked infer, avoid a very long audio cause memory leak
        :param audio: (samples)
        :param thres:
        :return: (samples//160+1)
        """
        audio = np.expand_dims(audio, axis=0)

        f0_results = []
        start = 0
        while start <= audio.shape[1]:
            # print(start, start + self.max_sample)
            output = self.model.forward([audio[..., start:start + self.max_sample].astype(np.float32), np.array([thres]).astype(np.float32)])
            f0_results.append(output[0].squeeze())
            start += self.max_sample

        self.f0 = np.concatenate(f0_results, axis=0)
        return self.f0

    def get_course(self):
        # got mel
        f0_mel = 1127.0 * np.log(1.0 + self.f0 / 700.0)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(int)
        return f0_coarse
