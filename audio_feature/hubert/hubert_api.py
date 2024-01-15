# -- coding: utf-8 --
# @Time : 2023/11/7
# @Author : ykk648

"""
partly borrowed from https://github.com/w-okada/voice-changer/blob/master/server/voice_changer/RVC/embedder/FairseqHubert.py
"""
from apstone import ModelBase
import numpy as np

"""
input_name:['source', 'padding_mask'], 
shape:[[1, 'source_dynamic_axes_1'], [1, 'padding_mask_dynamic_axes_1']], 
type:['tensor(float)', 'tensor(bool)']
output_name:['output_x'], 
shape:[['Sliceoutput_x_dim_0', 'output_x_dynamic_axes_1', 'output_x_dynamic_axes_2']], 
type:['tensor(float)']
"""
MODEL_ZOO = {
    'hubert_base_12_768': {
        'model_path': 'pretrain_models/audio_lib/hubert/hubert_base_12_768.onnx'
    },
}


class Hubert(ModelBase):
    def __init__(self, model_type='hubert_base_12_768', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type

    def forward(self, input_audio):
        """
        :param input_audio: (samples)
        :param padding_mask: (samples//320, 768)
        :return:
        """
        padding_mask = np.zeros_like(input_audio, dtype=bool)
        return self.model.forward([input_audio[None, ...].astype(np.float32), padding_mask])[0][0]


if __name__ == '__main__':
    import librosa

    audio_path = 'resources/guangdian.wav'
    input_audio = librosa.load(audio_path, sr=16000, mono=True)[0]
    # input_audio = np.random.random((1, 7040)).astype(np.float32)

    hubert = Hubert('hubert_base_12_768', 'gpu')
    embedding_out = hubert.forward(input_audio)
    print(embedding_out)
    print(embedding_out.shape)
