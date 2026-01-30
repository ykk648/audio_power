# -- coding: utf-8 --
# @Time : 2024/1/31
# @Author : ykk648
"""
ref https://modelscope.cn/models/iic/speech_frcrn_ans_cirm_16k/summary
"""
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


ans = pipeline(Tasks.acoustic_noise_suppression, model='damo/speech_frcrn_ans_cirm_16k')
result = ans(r"Anne-Hathaway.mp3", output_path='output.wav')
