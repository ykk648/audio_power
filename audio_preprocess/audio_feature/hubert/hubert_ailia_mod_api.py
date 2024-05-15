# -- coding: utf-8 --
# @Time : 2024/5/11
# @Author : ykk648

"""
partly borrowed from https://github.com/axinc-ai/ailia-models/blob/master/audio_processing/rvc/rvc.py
"""
from apstone import ModelBase
import numpy as np

"""

"""
MODEL_ZOO = {
    'hubert_base_ailia': {
        'model_path': 'pretrain_models/audio_lib/hubert/hubert_base_ailia_2_output.onnx'
    },
    'hubert_base_ailia_fp16': {
        'model_path': 'pretrain_models/audio_lib/hubert/hubert_base_ailia_2_output_fp16.onnx'
    },
}


class HubertAilia(ModelBase):
    def __init__(self, model_type='hubert_base_ailia', provider='gpu', device=0):
        super().__init__(MODEL_ZOO[model_type], provider, device)

    def forward(self, input_audio, version='v2'):
        padding_mask = np.zeros_like(input_audio, dtype=bool)

        if version == 'v1':
            return self.model.forward([input_audio.astype(self.model.input_type[0]), padding_mask])[0].astype(np.float32)
        elif version == 'v2':
            return self.model.forward([input_audio.astype(self.model.input_type[0]), padding_mask])[1].astype(np.float32)
        else:
            raise RuntimeError('Unsupported')