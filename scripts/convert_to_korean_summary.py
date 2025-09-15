import json
import re
from pathlib import Path
from typing import List, Tuple

# 간단한 이름 변환을 위한 사전
SURNAME_MAP = {
    "赵": "자오",
    "张": "장",
    "李": "리",
    "刘": "류",
    "陈": "천",
    "王": "왕",
    "黄": "황",
    "莫": "모",
    "潘": "판",
    "沈": "선",
    "雷": "레이",
    "应": "잉",
    "许": "쉬",
    "金": "진",
    "周": "저우",
    "杨": "양",
    "范": "판",
    "谭": "탄",
    "罗": "뤄",
    "张": "장",
    "龚": "궁",
    "廖": "랴오",
    "陈": "천",
    "张": "장",
    "赵": "자오",
    "朱": "주",
    "胡": "후",
    "邓": "덩",
    "沈": "선",
    "刘": "류",
    "蔡": "차이",
    "李": "리",
    "廖": "랴오",
    "范": "판",
    "范": "판",
    "宋": "쑹",
    "吴": "우",
    "郑": "정",
    "韩": "한",
    "冯": "펑",
    "杜": "두",
    "高": "가오",
    "唐": "탕",
    "曹": "차오",
    "顾": "구",
}

COMPANY_TERM_MAP = {
    "有限公司": "유한회사",
    "有限责任公司": "유한책임회사",
    "股份有限公司": "주식회사",
    "公司": "회사",
    "集团": "그룹",
    "建设": "건설",
    "工程": "공정",
    "实业": "실업",
    "科技": "과기",
    "电子": "전자",
    "贸易": "무역",
    "信息": "정보",
    "投资": "투자",
}

CLAIM_KEYWORDS: List[Tuple[str, str]] = [
    ("返还", "피고가 해당 금액을 반환하도록 청구한다"),
    ("归还", "피고가 금원을 돌려주도록 청구한다"),
    ("支付", "피고가 금액을 지급하도록 청구한다"),
    ("赔偿", "피고가 손해를 배상하도록 청구한다"),
    ("补偿", "피고가 손해를 보상하도록 청구한다"),
    ("停止", "피고가 침해 행위를 중단하도록 청구한다"),
    ("消除影响", "피고가 침해로 인한 영향을 제거하도록 청구한다"),
    ("公开道歉", "피고가 공개적으로 사과하도록 청구한다"),
    ("承担连带", "피고가 연대 책임을 부담하도록 청구한다"),
    ("承担", "피고가 관련 책임을 부담하도록 청구한다"),
    ("解除", "계약을 해지하거나 효력을 부인해 달라고 청구한다"),
    ("撤销", "관련 행위를 취소해 달라고 청구한다"),
    ("确认", "법원이 권리 의무 관계를 확인해 달라고 청구한다"),
    ("赔礼", "피고가 사과하도록 청구한다"),
]

FACT_KEYWORDS: List[Tuple[str, str]] = [
    ("借款", "원고는 피고에게 자금을 빌려주었으나 아직 상환받지 못했다고 주장한다."),
    ("欠款", "원고는 피고가 채무를 갚지 않았다고 주장한다."),
    ("合同", "원고는 계약상의 의무가 이행되지 않았다고 주장한다."),
    ("违约", "원고는 피고가 계약을 위반했다고 주장한다."),
    ("租赁", "원고는 임대차 약정이 제대로 이행되지 않았다고 주장한다."),
    ("侵权", "원고는 지식재산권 침해가 있었다고 주장한다."),
    ("商标", "원고는 상표권 침해를 문제 삼고 있다."),
    ("著作权", "원고는 저작권 침해를 문제 삼고 있다."),
    ("专利", "원고는 특허권 침해를 문제 삼고 있다."),
    ("投资", "원고는 투자금 정산이 이루어지지 않았다고 주장한다."),
    ("工资", "원고는 임금 및 보수를 지급받지 못했다고 주장한다."),
    ("劳务", "원고는 용역 대금이 지급되지 않았다고 주장한다."),
    ("车辆", "원고는 차량에 관한 권리 침해를 문제 삼고 있다."),
    ("房", "원고는 부동산 거래와 관련한 분쟁을 제기하였다."),
    ("服务费", "원고는 서비스 대금이 미지급되었다고 주장한다."),
    ("股东", "원고는 지분 투자와 관련한 분쟁을 제기하였다."),
    ("保证", "원고는 보증 의무 이행을 요구하고 있다."),
]

DEFENSE_KEYWORDS: List[Tuple[str, str]] = [
    ("缺乏事实和法律依据", "피고는 원고의 청구에 사실과 법적 근거가 없다고 본다."),
    ("缺乏事实依据", "피고는 원고의 주장이 사실에 근거하지 않는다고 주장한다."),
    ("缺乏法律依据", "피고는 원고의 청구가 법적 근거가 없다고 주장한다."),
    ("不应承担", "피고는 자신이 책임을 질 필요가 없다고 주장한다."),
    ("不应当承担", "피고는 자신이 책임을 질 필요가 없다고 주장한다."),
    ("已经", "피고는 이미 의무를 이행했다고 강조한다."),
    ("金额", "피고는 청구 금액이 부당하다고 다툰다."),
    ("利息", "피고는 이자 청구에 동의하지 않는다."),
    ("合同", "피고는 계약의 효력이나 내용이 다르다고 주장한다."),
    ("合法来源", "피고는 문제가 된 상품이 합법적 출처라고 항변한다."),
    ("重复起诉", "피고는 이번 소송이 중복 제기되었다고 주장한다."),
    ("诉讼时效", "피고는 청구가 소멸시효에 걸렸다고 주장한다."),
]

AMOUNT_PATTERNS = [
    (re.compile(r"([0-9]+(?:[.,][0-9]+)*)万元"), "만 위안"),
    (re.compile(r"([0-9]+(?:[.,][0-9]+)*)万"), "만 위안"),
    (re.compile(r"([0-9]+(?:[.,][0-9]+)*)元"), "위안"),
]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def replace_company_terms(name: str) -> str:
    for src, tgt in COMPANY_TERM_MAP.items():
        name = name.replace(src, tgt)
    return name


def convert_name(raw: str) -> str:
    if not raw:
        return "이름 미상"
    name = raw.strip()
    name = replace_company_terms(name)
    name = name.replace("某某", "모모").replace("某", "모")
    converted = []
    for ch in name:
        if ch in SURNAME_MAP:
            converted.append(SURNAME_MAP[ch])
        elif re.match(r"[A-Za-z0-9]", ch):
            converted.append(ch)
        elif ch in ['·', '-', '_', ' ', '&', '/', '(', ')']:
            converted.append(ch)
        elif ch in [':', '.', ',', '#']:
            converted.append(ch)
        else:
            converted.append('○')
    cleaned = normalize_whitespace(''.join(converted))
    return cleaned if cleaned else '이름 미상'


def extract_plaintiff(lines: List[str]) -> str:
    for line in lines:
        line = line.strip()
        if line.startswith("原告"):
            parts = line.split("：", 1)
            return convert_name(parts[1] if len(parts) > 1 else line.replace("原告", ""))
    return "원고 미상"


def extract_defendants(lines: List[str]) -> List[str]:
    defendants = []
    for line in lines:
        line = line.strip()
        if line.startswith("被告"):
            parts = line.split("：", 1)
            name = convert_name(parts[1] if len(parts) > 1 else line.replace("被告", ""))
            if name:
                defendants.append(name)
    return defendants


def extract_section(text: str, start: str, end_tokens: List[str]) -> str:
    start_idx = text.find(start)
    if start_idx == -1:
        return ""
    section = text[start_idx + len(start):]
    end_idx = len(section)
    for token in end_tokens:
        pos = section.find(token)
        if pos != -1 and pos < end_idx:
            end_idx = pos
    return section[:end_idx]


def format_amounts(text: str) -> List[str]:
    amounts = []
    for pattern, suffix in AMOUNT_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(1).replace(",", "")
            value = value.rstrip("0").rstrip(".") if "." in value else value
            amounts.append(f"{value}{suffix}")
    return amounts


def translate_claim(claim: str) -> str:
    claim_clean = claim.replace("被告", "피고").replace("原告", "원고")
    amount_parts = format_amounts(claim_clean)
    amount_text = ", ".join(sorted(set(amount_parts)))
    summary = None
    for keyword, sentence in CLAIM_KEYWORDS:
        if keyword in claim_clean:
            summary = sentence
            break
    if summary is None:
        summary = "원고는 해당 항목에 대한 법원의 구제를 청구한다"
    if amount_text:
        summary = summary + f"(청구 금액: {amount_text})"
    return summary + "."


def translate_claims(claim_section: str) -> List[str]:
    items = []
    for raw_line in claim_section.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[一二三四五六七八九十0-9\.、]+", "", line)
        if not line:
            continue
        items.append(translate_claim(line))
    if not items:
        items.append("원고는 소송 청구 취지를 따로 기재하였다")
    return items


def summarize_facts(fact_section: str) -> List[str]:
    text = fact_section.replace("原告", "원고").replace("被告", "피고")
    summaries = []
    for keyword, sentence in FACT_KEYWORDS:
        if keyword in text:
            summaries.append(sentence)
    if not summaries:
        summaries.append("원고는 위 청구를 뒷받침하는 사실관계를 근거로 법원의 인용을 구하고 있다.")
    return summaries


def summarize_defense(defense_section: str) -> List[str]:
    text = defense_section.replace("原告", "원고").replace("被告", "피고")
    summaries = []
    for keyword, sentence in DEFENSE_KEYWORDS:
        if keyword in text:
            summaries.append(sentence)
    if "驳回" in text or "驳回" in text:
        summaries.append("피고는 원고의 청구를 기각해 달라고 요청한다.")
    if not summaries:
        summaries.append("피고는 원고의 청구를 받아들일 수 없다고 항변한다.")
    return list(dict.fromkeys(summaries))


def build_plaintiff_statement(raw: str) -> str:
    lines = raw.split("\n")
    plaintiff = extract_plaintiff(lines)
    defendants = extract_defendants(lines)
    claim_section = extract_section(raw, "诉讼请求：", ["事实与理由", "事实与理由：", "事实及理由", "事实和理由", "事实理由", "此致", "综上"]).strip()
    fact_section = extract_section(raw, "事实", ["此致", "综上", "以上", "特此", "附：", "附件", "此致敬礼"]).strip()

    claim_sentences = translate_claims(claim_section)
    fact_sentences = summarize_facts(fact_section)

    defendant_text = ", ".join(defendants) if defendants else "피고 미상"

    parts = [
        "소장",
        "",
        f"원고: {plaintiff}",
        f"피고: {defendant_text}",
        "",
        "소송 청구:",
    ]
    for item in claim_sentences:
        parts.append(f"- {item}")
    parts.append("")
    parts.append("사실과 이유:")
    for sentence in fact_sentences:
        parts.append(f"- {sentence}")
    parts.append("")
    parts.append("이에")
    parts.append("[법원 명칭]")
    parts.append("")
    parts.append(f"소송 제기인: {plaintiff}")
    parts.append("날짜: [소송 제기일]")
    parts.append("")
    parts.append("첨부: 증거 개요")
    parts.append("- 원고는 관련 증거자료를 함께 제출하였다.")

    return "\n".join(parts)


def build_defendant_statement(raw: str, plaintiff: str, defendants: List[str]) -> str:
    defense_section = extract_section(raw, "答辩", ["此致", "综上", "以上", "特此", "附件", "附："]).strip()
    defense_sentences = summarize_defense(defense_section)
    defendant_text = ", ".join(defendants) if defendants else "피고"

    parts = [
        "답변서",
        "",
        f"피고: {defendant_text}",
        f"원고: {plaintiff}",
        "",
        "주요 항변:",
    ]
    for sentence in defense_sentences:
        parts.append(f"- {sentence}")
    parts.append("")
    parts.append("피고는 위 사정을 들어 원고의 청구를 기각해 달라고 요청한다.")
    parts.append("")
    parts.append("이에")
    parts.append("[법원 명칭]")
    parts.append("")
    parts.append(f"답변인: {defendant_text}")
    parts.append("날짜: [답변 제출일]")

    return "\n".join(parts)


def convert_dataset(src: Path, dst: Path) -> None:
    results = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            plaintiff_statement = item.get("plaintiff_statement", "")
            defendant_statement = item.get("defendant_statement", "")
            lines = plaintiff_statement.split("\n")
            plaintiff = extract_plaintiff(lines)
            defendants = extract_defendants(lines)
            new_plaintiff = build_plaintiff_statement(plaintiff_statement)
            new_defendant = build_defendant_statement(defendant_statement, plaintiff, defendants)
            results.append({
                "caseId": item.get("caseId"),
                "plaintiff_statement": new_plaintiff,
                "defendant_statement": new_defendant,
            })
    with dst.open("w", encoding="utf-8") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    src = Path("data/validation_zh.jsonl")
    dst = Path("data/validation.jsonl")
    convert_dataset(src, dst)


if __name__ == "__main__":
    main()
