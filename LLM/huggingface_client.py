# LLM/huggingface_client.py
import logging
from typing import List, Dict

import requests

from .base_client import BaseClient


logger = logging.getLogger(__name__)


class HuggingFaceClient(BaseClient):
    """Client for the Hugging Face Inference API."""

    def __init__(self, api_key: str, model: str, api_url: str = None):
        if not api_key:
            raise ValueError(
                "Hugging Face Inference API requires a valid token."
            )
        self.api_key = api_key
        self.model = model
        if api_url:
            self.api_url = api_url.rstrip("/")
        else:
            self.api_url = (
                f"https://api-inference.huggingface.co/models/{self.model}"
            )

    def send_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        **kwargs,
    ) -> str:
        prompt = self._build_prompt(messages)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "return_full_text": False,
            },
        }

        response = requests.post(
            self.api_url, headers=headers, json=payload, timeout=60
        )
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(
                f"Hugging Face API error: {data.get('error')}"
            )

        if isinstance(data, list) and data:
            generated = data[0].get("generated_text")
            if generated is not None:
                return generated.strip()

        raise RuntimeError(
            f"Unexpected response from Hugging Face API: {data}"
        )

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        system_prompt = []
        conversation = []

        for message in messages:
            role = message.get("role")
            content = (message.get("content") or "").strip()
            if not content:
                continue

            if role == "system":
                system_prompt.append(content)
            elif role == "assistant":
                conversation.append(f"Assistant:\n{content}")
            else:
                conversation.append(f"User:\n{content}")

        prompt_sections = []
        if system_prompt:
            prompt_sections.append("\n\n".join(system_prompt))
        prompt_sections.extend(conversation)
        prompt_sections.append("Assistant:\n")
        return "\n\n".join(prompt_sections)

