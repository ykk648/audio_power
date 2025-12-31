# -- coding: utf-8 --
# @Time : 2024/7/21
# @Author : ykk648

import requests, uuid, json
import os

# Add your key and endpoint
key = os.getenv('AZURE_TRANS_KEY')
endpoint = os.getenv('AZURE_TRANS_ENDPOINT')
location = os.getenv('AZURE_TRANS_LOCATION')

path = '/translate'
constructed_url = endpoint + path

params = {
    'api-version': '3.0',
    'from': 'zh',
    'to': ['en']
}

headers = {
    'Ocp-Apim-Subscription-Key': key,
    # location required if you're using a multi-service or regional (not global) resource.
    'Ocp-Apim-Subscription-Region': location,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

# You can pass more than one object in body.
body = [{
    'text': '合理断句，保留标点。'
}]

request = requests.post(constructed_url, params=params, headers=headers, json=body)
response = request.json()

print(json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': ')))