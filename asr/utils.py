# -- coding: utf-8 --
# @Time : 2024/8/1
# @Author : ykk648
from dataclasses import dataclass

@dataclass
class SegmentDummy:
    start: float
    end: float
    text: str


split_thres = {
    'zh': 20,
    'en': 50,
    'ja': 20,
    'fr': 50,
    'ko': 20,
}

language_dict = {
    'zh': 'zh-CN',
    'en': 'en-US',
    'ja': 'ja-JP',
    'fr': 'fr-FR',
    'ko': 'ko-KR',
}