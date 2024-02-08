# -- coding: utf-8 --
# @Time : 2023/11/7
# @Author : ykk648

"""
partly borrowed from https://github.com/w-okada/voice-changer/blob/master/server/voice_changer/RVC/embedder/FairseqHubert.py
"""
import torch
from fairseq import checkpoint_utils
import numpy as np

MODEL_ZOO = {
    'hubert_base': {
        'model_path': 'pretrain_models/audio_lib/hubert/hubert_base.pt'
    },
}


class FairseqHubert:
    def __init__(self, model_type='hubert_base', device='cuda'):
        self.models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [MODEL_ZOO[model_type]['model_path']], suffix="", )
        self.model = self.models[0]
        self.model.eval()

        self.device = device
        self.model = self.model.to(device)
        # if isHalf:
        #     model = model.half()

    def forward(self, feats, embOutputLayer=12, useFinalProj=False):
        """
        Args:
            feats: torch.Size([1, 4960])
            embOutputLayer: 9/12
            useFinalProj:
        Returns: torch.Size([1, 15, 768])
        """
        if isinstance(feats, np.ndarray):
            feats = torch.tensor(feats).to(self.device)

        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": embOutputLayer,  # 9 or 12
        }

        with torch.no_grad():
            logits = self.model.extract_features(**inputs)
            if useFinalProj:
                feats = self.model.final_proj(logits[0])
            else:
                feats = logits[0]
        return feats.cpu().detach().numpy()
