# api_client/openai_client.py
import requests
import json
from .base_client import BaseClient


class OpenAIClient(BaseClient):
    def __init__(self, api_key, model):
        if not api_key:
            raise ValueError("OpenAI platform requires a valid api_key.")
        self.api_key = api_key
        self.model = model

    def send_request(self, messages, **kwargs):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        text = response.json()
        choices = text.get("choices")
        if not choices:
            raise RuntimeError(f"Unexpected OpenAI response: {text}")
        return choices[0].get("message", {}).get("content", "")
