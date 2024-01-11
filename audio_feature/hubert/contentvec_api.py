# -- coding: utf-8 --
# @Time : 2024/1/11
# @Author : ykk648

"""
partly borrowed from https://github.com/w-okada/voice-changer/blob/master/server/voice_changer/RVC/embedder/FairseqHubert.py
"""
from apstone import ModelBase
import numpy as np

MODEL_ZOO = {
    """
    input_name:['audio'], 
    shape:[[1, 'audio_dynamic_axes_1']], 
    type:['tensor(float)']
    output_name:['units9', 'unit12', 'unit12s'], 
    shape:[[1, 'Addunits9_dim_1', 256], [1, 'Addunit12_dim_1', 768], [1, 'Mulunit12s_dim_1', 768]], 
    type:['tensor(float)', 'tensor(float)', 'tensor(float)']
    """
    'content_vec_500': {
        'model_path': 'pretrain_models/audio_lib/hubert/content_vec_500.onnx'
    },

    """
    input_name:['source'], 
    shape:[[1, 1, 'sample_length']], 
    type:['tensor(float)']
    output_name:['embed'], 
    shape:[['Transposeembed_dim_0', 'Transposeembed_dim_1', 768]], 
    type:['tensor(float)']
    """
    'vec-768-layer-12': {
        'model_path': 'pretrain_models/audio_lib/hubert/vec-768-layer-12.onnx'
    },
}


class Hubert(ModelBase):
    def __init__(self, model_type='content_vec_500', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type

    def forward(self, input_audio, padding_mask):
        if self.model_type == 'content_vec_500':
            return self.model.forward([input_audio, padding_mask])[1]
        elif self.model_type == 'vec-768-layer-12':
            input_audio = input_audio[None, ...]
            return self.model.forward([input_audio, padding_mask])[0]


if __name__ == '__main__':
    input_audio = np.random.random((1, 7040)).astype(np.float32)
    padding_mask = np.zeros_like(input_audio, dtype=bool)
    hubert = Hubert('vec-768-layer-12', 'gpu')
    embedding_out = hubert.forward(input_audio, padding_mask)
    print(embedding_out.shape)
