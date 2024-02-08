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
    'hubert_base_onnx': {
        # 'model_path': 'test.onnx'
        'model_path': 'pretrain_models/audio_lib/hubert/hubert_base.onnx'
    },
    'hubert_base_12_768': {
        'model_path': 'pretrain_models/audio_lib/hubert/hubert_base_12_768.onnx'
    },
    'hubert_base_12_768_fp16': {
        'model_path': 'pretrain_models/audio_lib/hubert/hubert_base_12_768_fp16.onnx'
    },
    'hubert_base_9_256_proj_fp16': {
        'model_path': 'pretrain_models/audio_lib/hubert/hubert_base_9_256_proj_fp16.onnx'
    },
}


class Hubert(ModelBase):
    def __init__(self, model_type='hubert_base_12_768_fp16', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type

    def forward(self, input_audio):
        """
        :param input_audio: (samples)
        :param padding_mask: (samples//320, 768)
        :return:
        """
        # https://github.com/microsoft/onnxruntime/issues/7563
        # 输入维度不能被40整除，onnx不支持
        if input_audio.shape[1] % 40 == 0:
            input_audio = np.pad(input_audio, ((0, 0), (0, 1)))
            # padding_mask = np.pad(padding_mask, ((0, 0), (0, False)))
        padding_mask = np.zeros_like(input_audio, dtype=bool)
        if self.model_type in ['hubert_base_12_768_fp16']:
            input_audio = input_audio.astype(np.float16)
            return self.model.forward([input_audio, padding_mask])[0].astype(np.float32)
        if self.model_type in ['hubert_base_9_256_proj_fp16']:
            input_audio = input_audio.astype(np.float16)
            return self.model.forward([input_audio, padding_mask])[0].astype(np.float32)[None, ...]
        return self.model.forward([input_audio, padding_mask])[0]

