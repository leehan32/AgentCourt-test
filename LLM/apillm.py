from .llm import LLM
from .openai_client import OpenAIClient
from .wenxin_client import WenxinClient
from .zhipuai_client import ZhipuAIClient
from .hf_client import HuggingFaceClient


class APILLM(LLM):
    def __init__(self, api_key, api_secret=None, platform="wenxin", model="gpt-4"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.platform = platform
        self.model = model

        normalized_key = (self.api_key or "").strip().lower()
        placeholder_tokens = {"put your api_key here", "your_api_key", ""}
        if normalized_key in placeholder_tokens:
            raise ValueError("A valid API key must be provided for API-based LLM usage.")

        if self.platform == "wenxin":
            normalized_secret = (self.api_secret or "").strip().lower()
            placeholder_secrets = {"put your api_secret here", "your_api_secret", ""}
            if normalized_secret in placeholder_secrets:
                raise ValueError(
                    "A valid API secret must be provided when using the Wenxin platform."
                )

        self.client = self._initialize_client()

    def _initialize_client(self):
        if self.platform == "openai":
            return OpenAIClient(self.api_key, self.model)
        elif self.platform == "wenxin":
            return WenxinClient(self.api_key, self.api_secret, self.model)
        elif self.platform == "zhipuai":
            return ZhipuAIClient(self.api_key, self.model)
        elif self.platform == "huggingface":
            return HuggingFaceClient(self.api_key, self.model)
        else:
            raise ValueError(f"Unsupported platform: {self.platform}")

    def generate(self, instruction, prompt, *args, **kwargs):
        if instruction is None:
            instruction = "You are a helpful assistant."

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ]
        return self.client.send_request(messages, *args, **kwargs)
