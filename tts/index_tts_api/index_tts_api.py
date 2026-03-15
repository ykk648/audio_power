"""
Index TTS API - 支持音频路径和角色名两种方式的TTS服务
ref https://github.com/Ksuriuri/index-tts-vllm/blob/master/api_example.py
"""
from typing import List, Union
import requests


class IndexTTSAPI:
    """Index TTS API 客户端"""
    
    def __init__(self, server_host: str = "0.0.0.0", server_port: int = 6006):
        """
        初始化Index TTS API客户端
        
        :param server_host: TTS服务器地址，默认 "0.0.0.0"
        :param server_port: TTS服务器端口，默认 6006
        """
        self.server_host = server_host
        self.server_port = server_port
        self.base_url = f"http://{server_host}:{server_port}"
    
    def tts_with_audio_paths(self, 
                              text: str, 
                              audio_paths: Union[str, List[str]], 
                              output_path: str) -> bool:
        """
        使用本地音频文件作为参考音频进行TTS合成
        
        :param text: 要合成的文本
        :param audio_paths: 参考音频文件路径，可以是单个路径字符串或路径列表（支持多参考音频）
        :param output_path: 输出音频文件路径
        :return: 成功返回True，失败返回False
        """
        url = f"{self.base_url}/tts_url"
        
        # 确保audio_paths是列表格式
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
        
        data = {
            "text": text,
            "audio_paths": audio_paths
        }
        
        try:
            response = requests.post(url, json=data, timeout=60)
            response.raise_for_status()
            
            # 保存音频文件
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            print(f"[成功] TTS合成完成，已保存到: {output_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"[错误] TTS请求失败: {e}")
            return False
        except Exception as e:
            print(f"[错误] 保存音频文件失败: {e}")
            return False
    
    def tts_with_character(self, 
                           text: str, 
                           character: str, 
                           output_path: str) -> bool:
        """
        使用角色名进行TTS合成
        
        :param text: 要合成的文本
        :param character: 角色名称（需要在服务器端注册，参考 assets/speaker.json）
        :param output_path: 输出音频文件路径
        :return: 成功返回True，失败返回False
        """
        url = f"{self.base_url}/tts"
        
        data = {
            "text": text,
            "character": character
        }
        
        try:
            response = requests.post(url, json=data, timeout=60)
            response.raise_for_status()
            
            # 保存音频文件
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            print(f"[成功] TTS合成完成，已保存到: {output_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"[错误] TTS请求失败: {e}")
            return False
        except Exception as e:
            print(f"[错误] 保存音频文件失败: {e}")
            return False


if __name__ == '__main__':
    # 使用示例
    
    # 创建API客户端
    api = IndexTTSAPI(server_host="127.0.0.1", server_port=11996)
    
    # 方式1: 使用本地音频文件作为参考（单个参考音频）
    print("=" * 60)
    print("示例1: 使用单个参考音频")
    print("=" * 60)
    api.tts_with_audio_paths(
        text="还是会想你，还是想登你",
        audio_paths="assets/jay_promptvn.wav",
        output_path="output_single.wav"
    )

    # 方式2: 使用多个参考音频
    print("\n" + "=" * 60)
    print("示例2: 使用多个参考音频")
    print("=" * 60)
    api.tts_with_audio_paths(
        text="还是会想你，还是想登你",
        audio_paths=[
            "assets/jay_promptvn.wav",
            "assets/vo_card_klee_endOfGame_fail_01.wav"
        ],
        output_path="output_multi.wav"
    )
    
    # 方式3: 使用角色名
    print("\n" + "=" * 60)
    print("示例3: 使用角色名")
    print("=" * 60)
    api.tts_with_character(
        text="还是会想你，还是想登你",
        character="jay_klee",
        output_path="output_character.wav"
    )

