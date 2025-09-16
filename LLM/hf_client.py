import logging
from typing import Dict, List

from huggingface_hub import InferenceClient

from .base_client import BaseClient


class HuggingFaceClient(BaseClient):
    """Client wrapper for Hugging Face Inference Endpoints."""

    def __init__(self, api_key: str, model: str):
        self.client = InferenceClient(model=model, token=api_key)
        self.logger = logging.getLogger(__name__)

    def send_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.8,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        **kwargs,
    ) -> str:
        system_prompt = None
        conversation: List[Dict[str, str]] = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                system_prompt = content
                continue

            normalized_role = role if role in {"user", "assistant"} else "user"
            conversation.append({"role": normalized_role, "content": content})

        chat_messages: List[Dict[str, str]] = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.extend(conversation)

        if not chat_messages:
            chat_messages = [{"role": "user", "content": ""}]

        try:
            response = self.client.chat_completion(
                messages=chat_messages,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )

            if response and getattr(response, "choices", None):
                choice = response.choices[0]
                message = getattr(choice, "message", None)

                if isinstance(message, dict):
                    return message.get("content", "")

                if message is not None:
                    content = getattr(message, "content", None)
                    if content is not None:
                        return content

            self.logger.warning(
                "Hugging Face chat completion returned no content. Falling back to text generation."
            )
        except Exception as chat_error:  # pylint: disable=broad-except
            self.logger.warning(
                "Hugging Face chat completion failed: %s. Falling back to text generation.",
                chat_error,
            )

        prompt_sections: List[str] = []
        if system_prompt:
            prompt_sections.append(system_prompt)
        for item in conversation:
            label = "Assistant" if item["role"] == "assistant" else "User"
            prompt_sections.append(f"{label}: {item['content']}")

        prompt = "\n\n".join(prompt_sections).strip()

        if not prompt:
            return ""

        try:
            text_response = self.client.text_generation(
                prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                return_full_text=False,
            )

            if isinstance(text_response, str):
                return text_response

            generated_text = getattr(text_response, "generated_text", None)
            if generated_text is not None:
                return generated_text

        except Exception as generation_error:  # pylint: disable=broad-except
            self.logger.error(
                "Hugging Face text generation failed: %s. Returning empty string.",
                generation_error,
            )

        return ""
