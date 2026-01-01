# -- coding: utf-8 --
# @Time : 2023/11/6
# @Author : ykk648

"""
partly borrowed from
"""
from audio_lib.rvc.torch_utils import SynthesizerTrnMs768NSFsid_ONNX, SynthesizerTrnMs256NSFsid_ONNX
import numpy as np
import torch


class RVCTorch:
    def __init__(self, model_path=None, device='cuda'):
        self.device = device
        # self.model_type = model_type
        model_tmp = torch.load(model_path, map_location="cpu")
        # print(*model_tmp["config"])
        if model_tmp['weight']['enc_p.emb_phone.weight'].shape == torch.Size([192, 256]):
            # print(model_tmp['weight']['enc_p.emb_phone.weight'].shape)
            # 1025 32 192 192 768 2 6 3 0 1 [3, 7, 11] [[1, 3, 5], [1, 3, 5], [1, 3, 5]] [10, 10, 2, 2] 512 [16, 16, 4, 4] 109 256 40000
            self.model = SynthesizerTrnMs256NSFsid_ONNX(*model_tmp["config"], is_half=False)
            print('detect RVCv1 model !!!')
            # raise Exception('detect RVCv1 model , can not be support now!!!')
        elif model_tmp['weight']['enc_p.emb_phone.weight'].shape == torch.Size([192, 768]):
            # 1025 32 192 192 768 2 6 3 0 1 [3, 7, 11] [[1, 3, 5], [1, 3, 5], [1, 3, 5]] [10, 10, 2, 2] 512 [16, 16, 4, 4] 109 256 40000
            print('detect RVCv2 model !!!')
            self.model = SynthesizerTrnMs768NSFsid_ONNX(*model_tmp["config"], is_half=False)
        else:
            raise 'can not support enc_p emb layer weight besides 256/768 !!!'
        self.model.eval().to(device)
        self.model.load_state_dict(model_tmp["weight"], strict=False)

    def forward(self, feats, pitch_length, pitch, pitchf, sid):
        with torch.no_grad():
            feats = torch.tensor(feats, dtype=torch.float).to(self.device)
            pitch_length = torch.tensor(pitch_length, dtype=torch.long).to(self.device)
            sid = torch.tensor(sid, dtype=torch.long).to(self.device)
            pitch = torch.tensor(pitch, dtype=torch.int64).to(self.device)
            pitchf = torch.tensor(pitchf, dtype=torch.float).to(self.device)

            res = self.model.forward(feats, pitch_length.unsqueeze(0), pitch, pitchf, sid)
            # if len(audio[0].shape) >= 2:
            #     # for RVC v2
            #     res = np.array(audio[0][0, 0])
            #     res = np.clip(res, -1.0, 1.0)
            # else:
            #     # for RVC v2.1 v1.1
            #     res = audio[0]
            return res.cpu().detach().numpy()


if __name__ == '__main__':
    device = 'cuda'
    featsLength = 64
    feats = torch.FloatTensor(1, featsLength, 768).to(device)
    p_len = torch.LongTensor([featsLength]).to(device)
    sid = torch.LongTensor([0]).to(device)
    pitch = torch.zeros(1, featsLength, dtype=torch.int64).to(device)
    pitchf = torch.FloatTensor(1, featsLength).to(device)

    rvct = RVCTorch()

    print(rvct.forward(feats, p_len, pitch, pitchf, sid).shape)
