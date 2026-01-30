# -- coding: utf-8 --
# @Time : 2024/6/10
# @Author : ykk648

from apstone import ModelBase
import numpy as np

"""
ref https://github.com/audeering/w2v2-age-gender-how-to
"""

MODEL_ZOO = {
    'w2v2-L-robust-6-age-gender': {
        'model_path': r'pretrain_models\audio_lib\age_gender\w2v2-L-robust-6-age-gender.onnx'
    },
}
"""
input_name:['signal'], 
shape:[[1, 'time']], 
type:[<class 'numpy.float32'>]
output_name:['hidden_states', 'logits_age', 'logits_gender'], 
shape:[[1, 1024], [1, 1], [1, 3]], 
type:['tensor(float)', 'tensor(float)', 'tensor(float)']
"""


class AgeGender(ModelBase):
    def __init__(self, model_type='w2v2-L-robust-6-age-gender', provider='gpu', device=0):
        super().__init__(MODEL_ZOO[model_type], provider, device)
        self.labels = ['female', 'male', 'child']

    def forward(self, input_audio):
        return self.labels[np.argmax(self.model.forward(input_audio)[2][0])]
