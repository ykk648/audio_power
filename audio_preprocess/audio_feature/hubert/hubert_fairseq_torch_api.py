# -- coding: utf-8 --
# @Time : 2023/11/7
# @Author : ykk648

"""
partly borrowed from https://github.com/w-okada/voice-changer/blob/master/server/voice_changer/RVC/embedder/FairseqHubert.py
"""
import torch
from fairseq import checkpoint_utils

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

    def forward(self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True) -> torch.Tensor:
        """
        Args:
            feats: torch.Size([1, 4960])
            embOutputLayer: 9/12
            useFinalProj:
        Returns: torch.Size([1, 15, 768])
        """
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
        return feats


if __name__ == '__main__':
    import librosa

    audio_path = 'resources/guangdian.wav'
    input_audio = torch.tensor(librosa.load(audio_path, sr=16000, mono=True)[0][None, ...])
    # input_audio = np.random.random((1, 7040)).astype(np.float32)

    fh = FairseqHubert('hubert_base', 'cuda')

    from cv2box import MyFpsCounter
    with MyFpsCounter() as mfc:
        for _ in range(100):
            embedding_out = fh.forward(input_audio, 9, True)
        print(embedding_out)
        print(embedding_out.shape)