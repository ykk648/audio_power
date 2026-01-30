# -- coding: utf-8 --
# @Time : 2024/1/31
# @Author : ykk648
"""
ref https://modelscope.cn/models/iic/speech_frcrn_ans_cirm_16k/summary
"""
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='/your_local_path/speech_frcrn_ans_cirm_16k')
result = ans(
    'https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/speech_with_noise.wav',
    output_path='output.wav')

