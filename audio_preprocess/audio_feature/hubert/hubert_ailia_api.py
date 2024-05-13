# -- coding: utf-8 --
# @Time : 2024/5/11
# @Author : ykk648

"""
partly borrowed from https://github.com/axinc-ai/ailia-models/blob/master/audio_processing/rvc/rvc.py
"""
import numpy as np
import onnxruntime
# import ailia # needs purchase
import onnx
from onnx import shape_inference

"""

"""
MODEL_ZOO = {
    'hubert_base_ailia': {
        'model_path': 'pretrain_models/audio_lib/hubert/hubert_base_ailia.onnx'
        # 'model_path': 'pretrain_models/audio_lib/hubert/hubert_base_ailia_2_output.onnx'
        # 'model_path': 'pretrain_models/audio_lib/hubert/hubert_base_ailia_mod.onnx' # need IR 10, means build newest onnx repo
    },
}


# import onnxruntime as ort
# ort.set_default_logger_severity(1)

class HubertAilia:
    def __init__(self, model_type='hubert_base_ailia', version='v2', provider='gpu', device=0):
        self.model_type = model_type
        self.model_plain = onnx.load(MODEL_ZOO[model_type]['model_path'])
        self.provider = provider
        self.version = version

        if self.provider in ['gpu', 'cuda']:
            self.providers = (
                "CUDAExecutionProvider",
                {'device_id': device, }
            )
        elif self.provider == 'dml':
            self.providers = (
                "DmlExecutionProvider",
                {'device_id': device, }
            )
        else:
            self.providers = "CPUExecutionProvider"

        value_info_protos = []
        shape_info = shape_inference.infer_shapes(self.model_plain)
        for idx, node in enumerate(shape_info.graph.value_info):
            if node.name in ['/encoder/Slice_5_output_0']:
                # print(idx, node)
                value_info_protos.append(node)
        self.model_plain.graph.output.extend(value_info_protos)
        onnx.checker.check_model(self.model_plain)
        # onnx.save(self.model_plain, './test.onnx')

        self.onnx_session = onnxruntime.InferenceSession(self.model_plain.SerializeToString(), providers=[self.providers])

    def forward(self, input_audio):
        padding_mask = np.zeros_like(input_audio, dtype=bool)

        # self.onnx_session.disable_fallback()
        # ort_outs = self.onnx_session.run(['feats'], {'source': input_audio, 'padding_mask': padding_mask})
        ort_outs = self.onnx_session.run(['feats', '/encoder/Slice_5_output_0'], {'source': input_audio, 'padding_mask': padding_mask})
        if self.version == 'v1':
            return ort_outs[0]
        elif self.version == 'v2':
            return ort_outs[1]
        else:
            raise RuntimeError('Unsupported')
