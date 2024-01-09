# -- coding: utf-8 --
# @Time : 2024/1/3
# @Author : ykk648
# @Project : https://github.com/ykk648/RVC_realtime
"""
partly borrowed from https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/0de65053247fb616fa696e4c40d32411174b6310/tools/rvc_for_realtime.py#L321
"""
from apstone import ModelBase
import numpy as np
from torchfcpe import spawn_bundled_infer_model
import torch

# """
# input_name:['input'], shape:[[1, 128, 'time']]
# output_name:['output'], shape:[['Sigmoidoutput_dim_0', 'Sigmoidoutput_dim_1', 360]]
# """
# MODEL_ZOO = {
#     'rmvpe_onnx': {
#         'model_path': 'pretrain_models/audio_lib/rmvpe/rmvpe.onnx'
#     },
# }


class FCPE:
    def __init__(self, model_type='rmvpe_onnx', provider='gpu'):
        self.model_type = model_type
        self.input_std = self.input_mean = 127.5
        self.input_size = (112, 112)

        self.device = 'cuda'

    def forward(self, audio_mel, thred=0.3):
        """
        Args:
            audio_mel: (128, samples)
        Returns: torch.Size([samples])
        """
        print("Loading fcpe model")
        self.model_fcpe = spawn_bundled_infer_model(self.device)
        f0 = self.model_fcpe.infer(
            audio_mel.to(self.device).unsqueeze(0).float(),
            sr=16000,
            decoder_mode="local_argmax",
            threshold=0.006,
        )
        return f0[0].squeeze(1)


if __name__ == '__main__':
    import librosa

    audio_all = librosa.load("resources/guangdian.wav", sr=16000, mono=True)[0]

    from cv2box import MyFpsCounter

    fcpe = FCPE(model_type='rmvpe_mel_include', provider='gpu')
    # fps: 11.1 GPU: 0.6G
    with MyFpsCounter() as mfc:
        for _ in range(100):
            f0 = fcpe.forward(torch.tensor(audio_all), 0.3)
        # print(f0)
        print(f0.shape)