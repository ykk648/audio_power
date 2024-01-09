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

    def forward(self, audio, thres: float):
        """
        :param audio: (samples)
        :param thres:
        :return: (samples//160+1)
        """
        audio = np.expand_dims(audio, axis=0)
        output = self.model.forward([audio.astype(np.float32), np.array([thres]).astype(np.float32)])
        f0 = output[0].squeeze()
        return f0


if __name__ == '__main__':
    import librosa

    audio_all = librosa.load("resources/guangdian.wav", sr=16000, mono=True)[0]

    from cv2box import MyFpsCounter
    rmvpe = RMVPEMel(model_type='rmvpe_mel_include', provider='gpu')
    # fps: 12 GPU: 6G-1.7G
    with MyFpsCounter() as mfc:
        for _ in range(100):
            f0 = rmvpe.forward(audio_all, 0.3)
        print(f0.shape)
