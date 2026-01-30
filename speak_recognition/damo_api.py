# -- coding: utf-8 --
# @Time : 2024/1/30
# @Author : ykk648
"""
mod from https://github.com/RVC-Boss/GPT-SoVITS/blob/main/tools/damo_asr/cmd-asr.py
https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files
https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files
https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files
"""

import os
import sys
import traceback
from funasr import AutoModel


dir = ''
# opt_name=dir.split("\\")[-1].split("/")[-1]
opt_name = os.path.basename(dir)


path_asr = 'pretrain_models/audio_lib/speak_recognition/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
path_vad = 'pretrain_models/audio_lib/speak_recognition/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch'
path_punc = 'pretrain_models/audio_lib/speak_recognition/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'

model = AutoModel(model=path_asr, model_revision="v2.0.4",
                  vad_model=path_vad,
                  vad_model_revision="v2.0.4",
                  punc_model=path_punc,
                  punc_model_revision="v2.0.4",
                  )

opt = []
for name in os.listdir(dir):
    try:
        text = model.generate(input="%s/%s" % (dir, name))[0]["text"]
        opt.append("%s/%s|%s|ZH|%s" % (dir, name, opt_name, text))
    except:
        print(traceback.format_exc())

opt_dir = "output/asr_opt"
os.makedirs(opt_dir, exist_ok=True)
with open("%s/%s.list" % (opt_dir, opt_name), "w", encoding="utf-8") as f: f.write("\n".join(opt))
