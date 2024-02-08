# -- coding: utf-8 --
# @Time : 2024/1/31
# @Author : ykk648

"""
partly borrowed from https://github.com/pyf98/DPHuBERT
"""
import torch
import numpy as np
from .wav2vec2.model import wav2vec2_model

MODEL_ZOO = {
    'dphubert': {
        'model_path': 'pretrain_models/audio_lib/hubert/DPHuBERT-sp0.75.pth'
    },
}


class DPHuBERT:
    def __init__(self, model_type='dphubert', device='cuda'):

        ckpt = torch.load(MODEL_ZOO[model_type]['model_path'])
        self.model = wav2vec2_model(**ckpt["config"])
        self.model.eval()
        self.model.load_state_dict(ckpt["state_dict"], strict=False)

        self.device = device
        self.model = self.model.to(device)
        # if isHalf:
        #     model = model.half()

    def forward(self, feats):
        """
        :param feats: (batch_size, seq_len)
        :return: (batch_size, seq_len//320, 768)
        """
        if isinstance(feats, np.ndarray):
            feats = torch.tensor(feats).to(self.device)
        with torch.no_grad():
            return self.model.extract_features(feats.to(self.device), num_layers=12)[0][0].cpu().detach().numpy()
