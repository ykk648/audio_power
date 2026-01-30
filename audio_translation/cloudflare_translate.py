# -- coding: utf-8 --
# @Time : 2024/7/22
# @Author : ykk648
"""
ref https://github.com/jianchang512/pyvideotrans/blob/main/videotrans/translator/transapi.py
https://pyvideotrans.com/06.html  在cloudflare上创建m2m100翻译API
language support:
阿非利卡语（af）、阿姆哈拉语（am）、阿拉伯语（ar）、阿斯图里亚斯语（ast）、阿塞拜疆语（az）、巴什基尔语（ba）、白俄罗斯语（be）、 保加利亚语（bg）、孟加拉语（bn）、布列塔尼语（br）、波斯尼亚语（bs）、加泰罗尼亚语（ca）、宿务语（ceb）、捷克语（cs）、威尔士语（cy）、 丹麦语（da）、德语（de）、希腊语（el）、英语（en）、西班牙语（es）、爱沙尼亚语（et）、波斯语（fa）、富拉语（ff）、芬兰语（fi）、 法语（fr）、西弗里西亚语（fy）、爱尔兰语（ga）、苏格兰盖尔语（gd）、加利西亚语（gl）、古吉拉特语（gu）、豪萨语（ha）、希伯来语（he）、 印地语（hi）、克罗地亚语（hr）、海地克里奥尔语（ht）、匈牙利语（hu）、亚美尼亚语（hy）、印度尼西亚语（id）、伊博语（ig）、 伊洛卡诺语（ilo）、冰岛语（is）、意大利语（it）、日语（ja）、爪哇语（jv）、格鲁吉亚语（ka）、哈萨克语（kk）、中央高棉语（km）、 卡纳达语（kn）、韩语（ko）、卢森堡语（lb）、甘达语（lg）、林加拉语（ln）、老挝语（lo）、立陶宛语（lt）、拉脱维亚语（lv）、 马达加斯加语（mg）、马其顿语（mk）、马拉雅拉姆语（ml）、蒙古语（mn）、马拉地语（mr）、马来语（ms）、缅甸语（my）、尼泊尔语（ne）、 荷兰语（nl）、挪威语（no）、北索托语（ns）、奥克西塔尼语（oc），奥里亚语（or）、旁遮普语（pa）、波兰语（pl）、普什图语（ps）、 葡萄牙语（pt）、罗马尼亚语（ro）、俄语（ru）、信德语（sd）、僧伽罗语（si）、斯洛伐克语（sk）、斯洛文尼亚语（sl）、索马里语（so）、 阿尔巴尼亚语（sq）、塞尔维亚语（sr）、斯瓦蒂语（ss）、巽他语（su）、瑞典语（sv）、斯瓦希里语（sw）、泰米尔语（ta）、泰语（th）、 他加禄语（tl）、茨瓦纳语（tn）、土耳其语（tr）、乌克兰语（uk）、乌尔都语（ur）、乌兹别克语（uz）、越南语（vi）、沃洛夫语（wo）、 科萨语（xh）、依地语（yi）、约鲁巴语（yo）、中文（zh）、祖鲁语（zu）
"""
import requests
import os


class CloudFlareTranslator:
    def __init__(self):
        self.key = os.getenv('CF_TRANSLATE')
        self.base_url = os.getenv('CF_TRANSLATE_URL', '').rstrip('/')  # 如 https://xxx.workers.dev
        self.retry_num = int(os.getenv('CF_TRANSLATE_RETRY_NUM', '3'))

    def translate(self, text, src_lang='', dest_lang='en'):
        if not self.base_url:
            raise ValueError('请设置环境变量 CF_TRANSLATE_URL（你的 Cloudflare Workers 翻译 API 地址）')
        while self.retry_num > 0:
            # do not support \n
            text = text.replace('\n', '')
            # zh en
            response = requests.get(url=f"{self.base_url}/?text={text}&source_language={src_lang}&target_language={dest_lang}&secret={self.key}")
            result = response.json()
            if result['msg'] != 'ok':
                self.retry_num -= 1
                err = result['text']
                print(f'CloudFlareTranslator fail: {err}, retry: {self.retry_num}')
                continue
            split = result['text'].split('. ')
            result = [x + '.' for x in split[:-1]] + [split[-1]]
            for i, res in enumerate(result):
                # remove empty res
                if len(res) == 1:
                    result.pop(i)
            return result
