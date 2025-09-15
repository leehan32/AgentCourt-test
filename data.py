from datasets import load_dataset
import json, os

# train split만 존재
dataset = load_dataset("youzi517/AgentCourt", split="train")

# 저장 폴더
os.makedirs("data", exist_ok=True)

# train 데이터를 validation.jsonl 이름으로 저장
with open("data/validation.jsonl", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("변환 완료: data/validation.jsonl (총", len(dataset), "개 사례)")
