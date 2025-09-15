import requests
import json


def search_law(query):
    LAW_VECTOR_API_URL = "" # Place your API URL here
    URL = "" # Place your API URL here
    params = {"question": query}
    res = requests.get(URL, params=params)
    res = json.loads(res.text)

    return res


if __name__ == "__main__":
    query = "중화인민공화국 노동법 제43조는 무엇을 규정하고 있나요?"
    print(search_law(query))
