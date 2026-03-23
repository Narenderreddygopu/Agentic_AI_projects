import requests
import json

url = "https://openrouter.ai/api/v1/chat/completions"

API_KEY = "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"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model" : "openai/gpt-oss-120b:free",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What can AI agents do?"}
    ]
}

response = requests.post(url, headers=headers, json=data)

result = response.json()
print(json.dumps(result, indent=4))