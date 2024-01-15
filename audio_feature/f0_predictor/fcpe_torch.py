# -- coding: utf-8 --
# @Time : 2024/1/3
# @Author : ykk648

"""
partly borrowed from https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/0de65053247fb616fa696e4c40d32411174b6310/tools/rvc_for_realtime.py#L321
"""

from torchfcpe import spawn_bundled_infer_model
import torch

"""
pip install git+https://github.com/CNChTu/FCPE
"""


class FCPE:
    def __init__(self, model_type='rmvpe_onnx', device='cuda'):
        self.model_type = model_type
        self.device = device
        self.model_fcpe = spawn_bundled_infer_model(self.device)

    def forward(self, audio_mel, threshold=0.006):
        """
        Args:
            audio_mel: (128, samples)
            threshold: (128, samples)
        Returns: torch.Size([samples])
        """
        f0 = self.model_fcpe.infer(audio_mel.to(self.device).unsqueeze(0).float(),
                                   sr=16000,
                                   decoder_mode="local_argmax",
                                   threshold=threshold,
                                   )
        return f0[0].squeeze(1)


if __name__ == '__main__':
    import librosa

    audio_all = librosa.load("resources/guangdian.wav", sr=16000, mono=True)[0]

    from cv2box import MyFpsCounter

    fcpe = FCPE(model_type='rmvpe_mel_include', device='cuda')
    # fps: 128 GPU: 0.6G
    with MyFpsCounter() as mfc:
        for _ in range(100):
            f0 = fcpe.forward(torch.tensor(audio_all), 0.3)
        # print(f0)
        print(f0.shape)
