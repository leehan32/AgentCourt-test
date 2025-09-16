# api_client/wenxin_client.py
import json
import logging
import time

import requests

from .base_client import BaseClient


class WenxinClient(BaseClient):
    def __init__(self, api_key, api_secret, model):
        if not api_key or not api_secret:
            raise ValueError(
                "Wenxin platform requires both api_key and api_secret."
            )
        self.api_key = api_key
        self.api_secret = api_secret
        self.model = model

    def get_access_token(self):
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self.api_key}&client_secret={self.api_secret}"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        response = requests.post(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        token = data.get("access_token")
        if not token:
            error_code = data.get("error") or data.get("error_code")
            error_msg = data.get("error_description") or data.get("error_msg")
            raise RuntimeError(
                f"Failed to obtain Wenxin access token ({error_code}): {error_msg}"
            )
        return token

    def send_request(
        self,
        messages,
        temperature=0.8,
        top_p=0.8,
        penalty_score=1.0,
        stream=False,
        enable_system_memory=False,
        system_memory_id=None,
        stop=None,
        disable_search=False,
        enable_citation=False,
        enable_trace=False,
        max_output_tokens: int = None,
        response_format=None,
        user_id=None,
        tool_choice=None,
        *args,
        **kwargs,
    ):

        access_token = self.get_access_token()
        if self.model == "ERNIE-4.0-8K":
            endpoint = "completions_pro"
        elif self.model == "ERNIE-Speed-128K":
            endpoint = "ernie-speed-128k"
        elif self.model == "ERNIE-3.5-8K":
            endpoint = "completions"
        else:
            raise ValueError("Invalid model name")

        base_url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{endpoint}?access_token={access_token}"
        headers = {"Content-Type": "application/json"}

        system_messages = [msg for msg in messages if msg["role"] == "system"]
        if system_messages:
            system = system_messages[0]["content"]
            messages = [msg for msg in messages if msg["role"] != "system"]
        else:
            system = None

        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "penalty_score": penalty_score,
            "stream": stream,
            "enable_system_memory": enable_system_memory,
            "disable_search": disable_search,
            "enable_citation": enable_citation,
            "enable_trace": enable_trace,
            "response_format": response_format,
        }

        if system:
            payload["system"] = system
        if system_memory_id:
            payload["system_memory_id"] = system_memory_id
        if stop:
            payload["stop"] = stop
        if max_output_tokens:
            payload["max_output_tokens"] = max_output_tokens
        if user_id:
            payload["user_id"] = user_id
        if tool_choice:
            payload["tool_choice"] = tool_choice

        response = requests.post(
            base_url, headers=headers, data=json.dumps(payload), timeout=60
        )

        # 속도 제한 처리
        if response.status_code == 429:
            print("경고: 요청 속도가 제한을 초과했습니다!")
            remaining_requests = int(
                response.headers.get("X-Ratelimit-Remaining-Requests", 0)
            )
            remaining_tokens = int(
                response.headers.get("X-Ratelimit-Remaining-Tokens", 0)
            )
            if remaining_requests == 0 or remaining_tokens == 0:
                sleep_time = 60  # 60초 동안 대기한 뒤 재시도
                print(f"할당량이 모두 소진되었습니다. {sleep_time}초 후에 다시 시도합니다...")
                time.sleep(sleep_time)
                return self.send_request(
                    messages,
                    temperature,
                    top_p,
                    penalty_score,
                    stream,
                    enable_system_memory,
                    system_memory_id,
                    stop,
                    disable_search,
                    enable_citation,
                    enable_trace,
                    max_output_tokens,
                    response_format,
                    user_id,
                    tool_choice,
                )

        text = json.loads(response.text)
        logging.debug("Wenxin response: %s", text)

        if text.get("error_code") not in (None, 0):
            raise RuntimeError(
                f"Wenxin API error {text.get('error_code')}: {text.get('error_msg')}"
            )

        result = text.get("result")
        if result is None:
            logging.warning("경고: 응답에서 result 필드를 찾을 수 없습니다! 응답: %s", text)
            return ""

        if text.get("is_truncated"):
            logging.warning("주의: 출력 결과가 잘렸습니다!")

        if text.get("function_call"):
            logging.info("모델이 함수 호출을 생성했습니다: %s", text["function_call"])

        return result
