# -- coding: utf-8 --
# @Time : 2024/1/5
# @Author : ykk648
# @Project : https://github.com/ykk648/audio_lib
"""
ref https://github.com/liuzhao1225/YouDub-webui/blob/master/youdub/step030_translation.py
"""
import json
import os
import re
import openai
from dotenv import load_dotenv
import time
from loguru import logger

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
model_name = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')


def translation_postprocess(result):
    result = re.sub(r'\（[^)]*\）', '', result)
    result = result.replace('...', '，')
    result = re.sub(r'(?<=\d),(?=\d)', '', result)
    result = result.replace('²', '的平方').replace(
        '————', '：').replace('——', '：').replace('°', '度')
    result = result.replace("AI", '人工智能')
    result = result.replace('变压器', "Transformer")
    return result


def valid_translation(text, translation):
    if len(text) <= 10:
        if len(translation) > 15:
            return False, f'Only translate the following sentence and give me the result.'
    elif len(translation) > len(text) * 0.75:
        return False, f'The translation is too long. Only translate the following sentence and give me the result.'

    if (translation.startswith('“') and translation.endswith('”')) or (
            translation.startswith('"') and translation.endswith('"')):
        translation = translation[1:-1]
        return True, translation_postprocess(translation)

    forbidden = ['翻译', '这句', '\n', '简体中文', '中文']
    translation = translation.strip()
    for word in forbidden:
        if word in translation:
            return False, f"Don't include {word} in the translation. Only translate the following sentence and give me the result."

    return True, translation_postprocess(translation)


def _translate(transcript, target_language='简体中文'):
    full_translation = []
    fixed_message = [
        {'role': 'system', 'content': f'Translate the sentence into {target_language}.'},
        {'role': 'user', 'content': 'What language do you need to translate the title into?'},
        {'role': 'assistant', 'content': target_language}]

    for line in transcript:
        text = line['text']
        history = ''.join(full_translation[:-30])

        retry_message = 'Only translate the quoted sentence and give me the final translation.'
        retry = 0
        while retry < 30 and retry != -1:
            messages = fixed_message + \
                       [{'role': 'user', 'content': '\n'.join(
                           [history, retry_message,
                            f'Translate the following sentence into {target_language}: \n"{text}"'])}]
            print(messages)
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                    timeout=240
                )
                translation = response.choices[0].message.content.replace('\n', '')
                logger.info(f'原文：{text}')
                logger.info(f'译文：{translation}')
                success, translation = valid_translation(text, translation)
                if not success:
                    retry_message += 'Only translate the following sentence and give me the final translation.'
                    raise Exception('Invalid translation')
                full_translation.append(translation)
                retry = -1
            except Exception as e:
                print(e)
                retry += 1
                logger.warning('翻译失败')
                if retry == 30:
                    full_translation.append(translation)
                time.sleep(1)
    return full_translation


def translate(folder, target_language='简体中文'):
    if os.path.exists(os.path.join(folder, 'translation.json')):
        logger.info(f'Translation already exists in {folder}')
        return True

    transcript_path = os.path.join(folder, 'transcript.json')
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    translation_path = os.path.join(folder, 'translation.json')
    translation = _translate(transcript, target_language)
    for i, line in enumerate(transcript):
        line['translation'] = translation[i]
    with open(translation_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    return True


if __name__ == '__main__':
    # audio_vocals.wav in it
    json_folder = './'

    translate(json_folder, 'English')
