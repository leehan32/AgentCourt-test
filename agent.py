from typing import List, Dict, Any, Tuple
import re
import json
from LLM.deli_client import search_law
import uuid
import logging


class Agent:
    def __init__(
        self,
        id: int,
        name: str,
        role: str,
        description: str,
        llm: Any,
        db: Any,
        log_think=False,
    ):
        self.id = id
        self.name = name
        self.role = role
        self.description = description
        self.llm = llm
        self.db = db
        self.log_think = log_think

        self.logger = logging.getLogger(__name__)

    def __str__(self):
        return f"{self.name} ({self.role})"

    # --- Plan Phase --- #

    def plan(self, history_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.log_think:
            self.logger.info(f"Agent ({self.role}) starting planning phase")
        history_context = self.prepare_history_context(history_list)
        plans = self._get_plan(history_context)
        if self.log_think:
            self.logger.info(f"Agent ({self.role}) generated plans: {plans}")
        queries = self._prepare_queries(plans, history_context)
        if self.log_think:
            self.logger.info(f"Agent ({self.role}) prepared queries: {queries}")

        return {"plans": plans, "queries": queries}

    def _get_plan(self, history_context: str) -> Dict[str, bool]:
        instruction = f"You are a {self.role}. {self.description}\n\n"
        prompt = "Based on the court history, analyze whether information from the experience, case, or legal database is needed. Return a JSON string with three key-value pairs for experience, case, and legal, with values being true or false."
        response = self.llm.generate(
            instruction=instruction, prompt=prompt + "\n\n" + history_context
        )
        return self._extract_plans(self.extract_response(response))

    def _prepare_queries(
        self, plans: Dict[str, bool], history_context: str
    ) -> Dict[str, str]:
        queries = {}
        if plans["experience"]:
            queries["experience"] = self._prepare_experience_query(history_context)
        if plans["case"]:
            queries["case"] = self._prepare_case_query(history_context)
        if plans["legal"]:
            queries["legal"] = self._prepare_legal_query(history_context)
        return queries

    def _prepare_experience_query(self, history_context: str) -> str:
        instruction = f"You are a {self.role}. {self.description}\n\n"
        prompt = """
        Based on the court history, analyze what kind of experience information is needed.
        Identify the key points and formulate a query to retrieve relevant experiences that can improve logic.
        Provide a JSON string containing query statement.
        like 
        {{
            'query':'노동 분쟁 처리 방법 구체적 단계'
        }}
        """
        response = self.llm.generate(
            instruction=instruction, prompt=prompt + "\n\n" + history_context
        )
        return self.extract_response(response)

    def _prepare_case_query(self, history_context: str) -> str:
        instruction = f"You are a {self.role}. {self.description}\n\n"
        prompt = """
        Based on the court history, analyze what kind of case information is needed.
        Identify the key points and formulate a query to retrieve relevant case precedents that can improve agility.
        Provide a JSON string containing query keywords.
        like 
        {{
            'query':'노동계약 분쟁 판결 분석'
        }}
        """
        response = self.llm.generate(
            instruction=instruction, prompt=prompt + "\n\n" + history_context
        )
        return self.extract_response(response)

    def _prepare_legal_query(self, history_context: str) -> str:
        instruction = f"You are a {self.role}. {self.description}\n\n"
        prompt = """
        Based on the court history, analyze what kind of legal information is needed.
        Identify the relevant laws or regulations, such as Civil Law, Labor Law, Family Law, or Labor Dispute, and formulate a query to retrieve relevant legal references that can improve professionalism.
        Provide a JSON string containing query keywords.
        like 
        {{
            'query':'불법행위자 행동 법률 조항'
        }}
        """
        response = self.llm.generate(
            instruction=instruction, prompt=prompt + "\n\n" + history_context
        )
        return self.extract_response(response)

    # --- Do Phase --- #

    def execute(
        self, plan: Dict[str, Any], history_list: List[Dict[str, str]], prompt: str
    ) -> str:
        if not plan:
            context = self.prepare_history_context(history_list)
        else:
            context = self._prepare_context(plan, history_list)
        return self.speak(context, prompt)

    def speak(self, context: str, prompt: str) -> str:
        instruction = f"You are a {self.role}. {self.description}\n\n"
        full_prompt = f"{context}\n\n{prompt}"
        return self.llm.generate(instruction=instruction, prompt=full_prompt)

    def _prepare_context(
        self, plan: Dict[str, Any], history_list: List[Dict[str, str]]
    ) -> str:
        context = ""
        queries = plan["queries"]

        if "experience" in queries:
            experience_context = self.db.query_experience_metadatas(
                queries["experience"], n_results=3
            )
            context += (
                f"\n다음 경험을 참고하여 응답의 논리적 엄밀성을 강화하십시오:\n{experience_context}\n"
            )

        if "case" in queries:
            case_context = self.db.query_case_metadatas(queries["case"], n_results=3)
            context += f"\nCase Context:\n{case_context}\n"

        if "legal" in queries:
            legal_context = self.db.query_legal(queries["legal"], n_results=3)
            context += f"\nLaw Context:\n{legal_context}\n"

        if self.log_think:
            self.logger.info(f"Agent ({self.role})\n\n{context}")

        history_context = self.prepare_history_context(history_list)
        context += "\nCommunication History:\n" + history_context + "\n"

        return context

    # --- Reflect Phase --- #

    def reflect(self, history_list: List[Dict[str, str]]):

        history_context = self.prepare_history_context(history_list)

        case_content = self.prepare_case_content(history_context)

        # Legal knowledge base reflection
        legal_reflection = self._reflect_on_legal_knowledge(history_context)
        if self.log_think:
            self.logger.info(f"Agent ({self.role})\n\n{legal_reflection}")

        # Experience reflection
        experience_reflection = self._reflect_on_experience(
            case_content, history_context
        )
        if self.log_think:
            self.logger.info(f"Agent ({self.role})\n\n{experience_reflection}")

        # Case reflection
        case_reflection = self._reflect_on_case(case_content, history_context)
        if self.log_think:
            self.logger.info(f"Agent ({self.role})\n\n{case_reflection}")

        return {
            "legal_reflection": legal_reflection,
            "experience_reflection": experience_reflection,
            "case_reflection": case_reflection,
        }

    def _reflect_on_legal_knowledge(self, history_context: str) -> Dict[str, Any]:
        # Determine if legal reference is needed
        need_legal = self._need_legal_reference(history_context)

        if need_legal:
            query = self._prepare_legal_query(history_context)
            laws = search_law(query)

            processed_laws = []
            for law in laws[:3]:  # Limit to 3 laws
                law_id = str(uuid.uuid4())
                processed_law = self._process_law(law)
                self.add_to_legal(
                    law_id, processed_law["content"], processed_law["metadata"]
                )
                processed_laws.append(processed_law)

            return {"needed_reference": True, "query": query, "laws": processed_laws}
        else:
            return {"needed_reference": False}

    def _need_legal_reference(self, history_context: str) -> bool:
        instruction = (
            f"You are a {self.role}. {self.description}\n\n"
            "Review the provided court case history and evaluate its thoroughness and professionalism. "
            "Determine if referencing specific legal statutes or regulations would enhance the quality of the response. "
            "Return 'true' if additional legal references are needed, otherwise return 'false'."
        )
        prompt = (
            "Court Case History:\n\n"
            + history_context
            + "\n\nIs additional legal reference needed? Output true unless it is absolutely unnecessary. Provide only a simple 'true' or 'false' answer."
        )
        response = self.llm.generate(instruction=instruction, prompt=prompt)

        cleaned_response = response.strip().lower()

        # 응답에 'true' 또는 'false'가 포함되어 있는지 확인
        if "true" in cleaned_response:
            return True
        elif "false" in cleaned_response:
            return False
        else:
            return False

    def _process_law(self, law: dict) -> Dict[str, Any]:

        law_content = (
            law["lawsName"] + " " + law["articleTag"] + " " + law["articleContent"]
        )

        return {
            "content": law_content,
            "metadata": {"lawName": law["lawsName"], "articleTag": law["articleTag"]},
        }

    def _reflect_on_experience(
        self, case_content: str, history_context: str
    ) -> Dict[str, Any]:

        experience = self._generate_experience_summary(case_content, history_context)

        experience_entry = {
            "id": str(uuid.uuid4()),
            "content": experience["context"],  # 여기에 사건과 관련된 설명을 저장합니다
            "metadata": {
                "context": experience["content"],  # 여기에 사건과 관련된 지침을 저장합니다
                "focusPoints": experience["focus_points"],
                "guidelines": experience["guidelines"],
            },
        }

        # Add to experience database
        self.add_to_experience(
            experience_entry["id"],
            experience_entry["content"],
            experience_entry["metadata"],
        )

        return experience_entry

    def _generate_experience_summary(
        self, case_content: str, history_context: str
    ) -> Dict[str, Any]:
        instruction = f"당신은 {self.role}입니다. {self.description}\n\n"

        prompt = f"""
        아래의 사례 내용과 대화 기록을 바탕으로 논리적으로 일관된 경험 요약을 생성하십시오. 응답이 논리적으로 치밀하고 유사한 사건을 처리하는 데 효과적인 지침이 되도록 해 주십시오.

        사례 내용: {case_content}
        대화 기록: {history_context}

        다음 내용을 포함해 주십시오:
        1. 사건의 주요 쟁점과 각 측의 입장을 포함한 간단한 사건 배경(실제 인명은 사용하지 마십시오).
        2. 논리적 일관성에 초점을 맞춘 경험 설명(내용)으로, 유사한 사건을 처리할 때 중점적으로 살펴야 할 문제와 전략을 제시하십시오.
        3. 논리적 일관성을 높이는 데 도움이 되는 3~5개의 핵심 포인트를 제시하고, 실제 처리에서 이를 어떻게 적용할지 설명하십시오.
        4. 논리적 일관성을 유지하기 위한 3~5개의 가이드라인을 제시하고, 유사한 사건을 다룰 때 특별히 주의해야 할 사항과 조언을 제공하십시오.

        응답을 다음 구조의 JSON 객체로 작성해 주십시오:
        {{
            "context": "간단한 배경...",
            "content": "논리적 일관성에 초점을 맞춘 경험 설명...",
            "focus_points": "핵심 포인트1, 핵심 포인트2, 핵심 포인트3",
            "guidelines": "가이드라인1, 가이드라인2, 가이드라인3"
        }}
        """

        response = self.llm.generate(instruction, prompt)

        data = self.extract_response(response)

        # 목록을 문자열로 변환합니다
        return self.ensure_ex_string_fields(data)
        # if data and isinstance(data, dict):
        #    for key, value in data.items():
        #        if isinstance(value, list):
        #           data[key] = ", ".join(value)
        # return data

    def _reflect_on_case(
        self, case_content: str, history_context: str
    ) -> Dict[str, Any]:

        case_summary = self._generate_case_summary(case_content, history_context)

        case_entry = {
            "id": str(uuid.uuid4()),
            "content": case_summary["content"],
            "metadata": {
                "caseType": case_summary["case_type"],
                "keywords": case_summary["keywords"],
                "quick_reaction_points": case_summary["quick_reaction_points"],
                "response_directions": case_summary["response_directions"],
            },
        }

        # Add to case database
        self.add_to_case(
            case_entry["id"], case_entry["content"], case_entry["metadata"]
        )

        return case_entry

    def _generate_case_summary(
        self, case_content: str, history_context: str
    ) -> Dict[str, Any]:
        instruction = f"당신은 {self.role}로서 사건을 빠르게 분석하고 민첩하게 대응하는 데 능숙합니다. {self.description}\n\n"

        prompt = f"""
        아래의 사례 내용과 대화 기록을 바탕으로 간결한 사례 요약을 작성하여 유사한 상황에서의 대응 민첩성을 높이십시오. 응답은 사건을 빠르게 이해하고 신속히 대응 전략을 수립하는 데 도움이 되어야 합니다.

        사례 내용: {case_content}
        대화 기록: {history_context}

        다음 내용을 포함해 주십시오:
        1. 사례 이름과 배경: 간결한 사례 이름을 제시하고, 주요 쟁점과 각 측의 입장을 포함한 배경을 간단히 설명하십시오(실제 인명은 사용하지 마십시오).
        2. 사건 유형: 해당 사건이 어떤 유형(예: 노동 분쟁, 계약 분쟁 등)인지 명시하십시오.
        3. 핵심 키워드: 사건의 본질을 빠르게 파악할 수 있는 3~5개의 키워드를 제공하십시오.
        4. 빠른 대응 포인트: 유사한 사건을 신속히 이해하고 처리하는 데 필수적인 3~5개의 핵심 포인트를 제시하십시오.
        5. 대응 방향: 빠르게 대응 전략을 세우기 위한 3~5개의 가능한 방향이나 관점을 제시하십시오.

        응답을 다음 구조의 JSON 객체로 작성해 주십시오:
        {{
            "content": "사례 이름과 배경: ...",
            "case_type": "사건 유형...",
            "keywords": "키워드1, 키워드2, 키워드3",
            "quick_reaction_points": "포인트1, 포인트2, 포인트3",
            "response_directions": "방향1, 방향2, 방향3"
        }}

        주의: 내용은 간결하고 명확해야 하며, 핵심 문제를 빠르게 파악하고 대응 전략을 수립하는 데 도움이 되는 정보에 집중하십시오. 형식은 위에서 설명한 JSON 구조를 따르십시오.
        """

        response = self.llm.generate(instruction, prompt)

        data = self.extract_response(response)

        # 문자열인지 확인합니다
        return self.ensure_case_string_fields(data)
        # if data and isinstance(data, dict):
        #    for key, value in data.items():
        #        if isinstance(value, list):
        #            data[key] = ", ".join(value)
        # return data

    # --- Helper Methods --- #

    def extract_json_from_txt(self, response: str) -> Any:
        pattern = r"\{.*?\}"
        match = re.search(pattern, response, re.DOTALL)
        json_str = match.group()

        data = json.loads(json_str)
        return data

    def extract_response(self, response: str) -> Any:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                cleaned_json_str = re.sub(r"[\x00-\x1F\x7F]", "", json_match.group())
                return json.loads(cleaned_json_str, strict=False)
            except json.JSONDecodeError:
                pass
        return response.strip()

    def _extract_plans(self, plans_str: str) -> Dict[str, bool]:
        try:
            plans = plans_str if isinstance(plans_str, dict) else json.loads(plans_str)
            return {
                "experience": plans.get("experience", False),
                "case": plans.get("case", False),
                "legal": plans.get("legal", False),
            }
        except json.JSONDecodeError:
            return {"experience": False, "case": False, "legal": False}

    def add_to_experience(
        self, id: str, document: str, metadata: Dict[str, Any] = None
    ):
        self.db.add_to_experience(id, document, metadata)

    def add_to_case(self, id: str, document: str, metadata: Dict[str, Any] = None):
        self.db.add_to_case(id, document, metadata)

    def add_to_legal(self, id: str, document: str, metadata: Dict[str, Any] = None):
        self.db.add_to_legal(id, document, metadata)

    def prepare_history_context(self, history_list: List[Dict[str, str]]) -> str:
        formatted_history = []
        for entry in history_list:
            role = entry["role"]
            name = entry["name"]
            content = entry["content"].replace("\n", "\n  ")
            formatted_entry = f"{role} ({name}):\n  {content}"
            formatted_history.append(formatted_entry)
        return "\n\n".join(formatted_history)

    def prepare_case_content(self, history_context: str) -> str:
        instruction = f"당신은 전문적인 판사로서 사건 상황을 요약하는 데 능숙합니다.\n\n"

        prompt = "법정 기록을 바탕으로 세 문장으로 사건 상황을 요약해 주십시오."

        response = self.llm.generate(
            instruction=instruction, prompt=prompt + "\n\n" + history_context
        )

        return response
    
    # 선택 사항: 평점을 활용하여 반성 과정을 진행할 수 있습니다
    def _evaluate_response(self, case_content: str, response: str) -> Dict[str, int]:
        instruction = ""
        prompt = f"""
        아래 사건 내용을 바탕으로 다음 답변을 평가하고, 사고의 민첩성, 지식의 전문성, 논리적 엄밀성 세 가지 관점에서 1~5점으로 점수를 부여하십시오:
        사건 내용:
        {case_content}
        답변 내용:
        {response}
        
        다음 형식으로 점수 결과를 출력해 주십시오:
        {{
            "agility": 점수,
            "professionalism": 점수,
            "logic": 점수
        }}
        """

        evaluation_result = self.llm.generate(instruction, prompt)
        return json.loads(evaluation_result)

    def ensure_ex_string_fields(self, data):
        """
        데이터의 특정 필드가 문자열인지 확인합니다.
        """
        if not isinstance(data, dict):
            fallback_content = "" if not data else str(data)
            self.logger.warning(
                "Experience summary was not returned as JSON. Falling back to a minimal structure."
            )
            return {
                "context": "",
                "content": fallback_content,
                "focus_points": "",
                "guidelines": "",
            }

        fields_to_check = {
            "context": str,
            "content": str,
            "focus_points": lambda x: ", ".join(x) if isinstance(x, list) else x,
            "guidelines": lambda x: ", ".join(x) if isinstance(x, list) else x,
        }

        for field, validator in fields_to_check.items():
            if field in data:
                if callable(validator):
                    data[field] = validator(data[field])
                elif not isinstance(data[field], validator):
                    raise ValueError(f"{field} must be a {validator.__name__}")

        for field in fields_to_check:
            data.setdefault(field, "")

        return data

    def ensure_case_string_fields(self, data):
        """
        데이터의 특정 필드가 문자열인지 확인합니다.
        """
        if not isinstance(data, dict):
            fallback_content = "" if not data else str(data)
            self.logger.warning(
                "Case summary was not returned as JSON. Falling back to a minimal structure."
            )
            return {
                "content": fallback_content,
                "case_type": "",
                "keywords": "",
                "quick_reaction_points": "",
                "response_directions": "",
            }

        fields_to_check = [
            "content",
            "case_type",
            "keywords",
            "quick_reaction_points",
            "response_directions",
        ]

        for field in fields_to_check:
            if field in data:
                if isinstance(data[field], list):
                    data[field] = ", ".join(data[field])
                elif not isinstance(data[field], str):
                    raise ValueError(f"{field} must be a list or a string")

        for field in fields_to_check:
            data.setdefault(field, "")

        return data
