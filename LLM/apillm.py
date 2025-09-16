from .llm import LLM
from .openai_client import OpenAIClient
from .wenxin_client import WenxinClient
from .zhipuai_client import ZhipuAIClient
from .huggingface_client import HuggingFaceClient


PLACEHOLDER_TOKEN = "put your"


class APILLM(LLM):
    def __init__(
        self,
        api_key,
        api_secret=None,
        platform="wenxin",
        model="gpt-4",
        api_base=None,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.platform = platform
        self.model = model
        self.api_base = api_base
        self._validate_credentials()
        self.client = self._initialize_client()

    def _initialize_client(self):
        if self.platform == "openai":
            return OpenAIClient(self.api_key, self.model)
        elif self.platform == "wenxin":
            return WenxinClient(self.api_key, self.api_secret, self.model)
        elif self.platform == "zhipuai":
            return ZhipuAIClient(self.api_key, self.model)
        elif self.platform == "huggingface":
            return HuggingFaceClient(
                self.api_key, self.model, api_url=self.api_base
            )
        else:
            raise ValueError(f"Unsupported platform: {self.platform}")

    def _validate_credentials(self):
        if self.platform == "wenxin":
            if not self.api_key or PLACEHOLDER_TOKEN in self.api_key.lower():
                raise ValueError(
                    "Wenxin platform requires a valid api_key. Update role_config.json"
                )
            if not self.api_secret or PLACEHOLDER_TOKEN in self.api_secret.lower():
                raise ValueError(
                    "Wenxin platform requires a valid api_secret. Update role_config.json"
                )
        elif self.platform in {"openai", "zhipuai"}:
            if not self.api_key or PLACEHOLDER_TOKEN in self.api_key.lower():
                raise ValueError(
                    f"{self.platform} platform requires a valid api_key. Update role_config.json"
                )
        elif self.platform == "huggingface":
            if not self.api_key or PLACEHOLDER_TOKEN in self.api_key.lower():
                raise ValueError(
                    "Hugging Face platform requires a valid token. Update role_config.json"
                )
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
