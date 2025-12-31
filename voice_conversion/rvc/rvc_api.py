# -- coding: utf-8 --
# @Time : 2023/11/6
# @Author : ykk648

"""
partly borrowed from https://github.com/w-okada/voice-changer/blob/master/server/voice_changer/RVC/inferencer/OnnxRVCInferencer2.py
"""
from apstone import ModelBase
import numpy as np

# input_name:['feats', 'p_len', 'pitch', 'pitchf', 'sid'],
# shape:[[1, 'feats_dynamic_axes_1', 768], [1], [1, 'pitch_dynamic_axes_1'], [1, 'pitchf_dynamic_axes_1'], [1]],
# type:# ['tensor(float16)', 'tensor(int64)', 'tensor(int64)', 'tensor(float)', 'tensor(int64)']
# output_name:['audio', 'onnx::Mul_595', 'onnx::Mul_4255', 'onnx::Slice_3718', 'onnx::RandomNormalLike_3710', 'onnx::Exp_3711'],
# shape:[['Tanhaudio_dim_0', 1, 'Tanhaudio_dim_2'], [1, 1, 'Castonnx::Mul_595_dim_2'], [1, 192, 'Concatonnx::Mul_4255_dim_2'], [1, 192, 'Mulonnx::Slice_3718_dim_2'], [1, 192, 'Splitonnx::RandomNormalLike_3710_dim_2'], [1, 192, 'Splitonnx::RandomNormalLike_3710_dim_2']],
# type:# ['tensor(float16)', 'tensor(float16)', 'tensor(float16)', 'tensor(float16)', 'tensor(float16)', 'tensor(float16)']


class RVC(ModelBase):
    def __init__(self, model_path=None, provider='gpu'):
        super().__init__({'model_path': model_path}, provider)
        # self.model_type = model_type

    def forward(self, feats, pitch_length, pitch, pitchf, sid):
        audio = self.model.forward([feats.astype(np.float16),
                                   pitch_length.astype(np.int64),
                                   pitch.astype(np.int64),
                                   pitchf.astype(np.float32),
                                   sid.astype(np.int64)]
                                   )
        if len(audio[0].shape) >= 2:
            # for RVC v2
            res = np.array(audio[0][0, 0])
            res = np.clip(res, -1.0, 1.0)
        else:
            # for RVC v2.1 v1.1
            res = audio[0]
        return res


if __name__ == '__main__':
    pass
