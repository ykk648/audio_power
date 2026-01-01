# -- coding: utf-8 --
# @Time : 2024/7/21
# @Author : ykk648
"""
ref https://learn.microsoft.com/zh-cn/azure/ai-services/openai/assistants-quickstart?tabs=command-line%2Ctypescript&pivots=programming-language-python
https://github.com/openai/openai-cookbook/blob/main/examples/Assistants_API_overview_python.ipynb
"""
import requests
import uuid
import json
import os
import re
import time
from pathlib import Path
from .config import language_prompt_name
from openai import AzureOpenAI, APIError


class AzureOpenaiTranslator:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version="2024-05-01-preview",
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            # http_client=httpx.Client()
        )

        self.thread = self.client.beta.threads.create()
        print(self.thread.id)
        self.first = True

    def make_assistant_if_not_exist(self, name, prompt):
        all_assistants = self.client.beta.assistants.list()

        for assistant in all_assistants:
            if assistant.name == name:
                return assistant.id

        assistant = self.client.beta.assistants.create(
            name=name,
            model="gpt-4o-mini",  # replace with model deployment name.
            instructions=prompt,
            temperature=0.5,
            top_p=1,
        )
        return assistant.id

    def translate(self, text, src_lang='', dest_lang='en'):

        assert dest_lang not in ["ar"]

        prompt_path = Path('utils/azure.txt') if dest_lang == 'zh' else Path('utils/azure-en.txt')
        prompt = prompt_path.read_text(encoding='utf-8').strip() + "\n"
        target_language = language_prompt_name[dest_lang]
        prompt = prompt.replace('{lang}', target_language)

        assistant_id = self.make_assistant_if_not_exist(dest_lang, prompt)

        message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=text
        )

        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=assistant_id
        )

        # Looping until the run completes or fails
        while run.status in ['queued', 'in_progress', 'cancelling']:
            time.sleep(1)
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=run.id
            )

        results = ''
        if run.status == 'completed':
            # if self.first:
            messages = self.client.beta.threads.messages.list(
                thread_id=self.thread.id
            )
            #     self.first = False
            # else:
            #     messages = self.client.beta.threads.messages.list(
            #         thread_id=self.thread.id, order="asc", after=message.id
            #     )
            results = messages.data[0].content[0].text.value
        elif run.status == 'requires_action':
            # the assistant requires calling some functions
            # and submit the tool outputs back to the run
            pass
        else:
            if run.last_error.code == 'rate_limit_exceeded':
                print('Chatgpt rate limit exceeded...')
            else:
                print(run.status)
                print(run.last_error)

        # self.client.beta.threads.delete(self.thread.id)

        # segment
        results = results.replace('##', '').strip().replace('&#39;', '"').replace('&quot;', "'")
        results = re.sub(r'\n{2,}', "\n", results)
        results = results.strip()
        results = results.split('\n')
        results = [result.lstrip() for result in results]
        return results

    def __del__(self):
        self.client.beta.threads.delete(self.thread.id)
        print(f"Thread {self.thread.id} had been deleted.")
