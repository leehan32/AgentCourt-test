import json
import os
import random
import logging
import argparse
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from tqdm import trange

from EMDB.db import db
from LLM.offlinellm import OfflineLLM
from LLM.apillm import APILLM
from agent import Agent

console = Console()


class CourtSimulation:
    def __init__(self, config_path, case_data, log_level, log_think=False):
        """
        법정 시뮬레이션 클래스를 초기화합니다.
        :param config_path: 구성 파일 경로
        :param case_data: 사례 데이터(단일 파일 경로 또는 여러 사례를 포함하는 디렉터리 경로)
        :param log_level: 로그 수준
        """
        self.setup_logging(log_level)
        self.config = self.load_json(config_path)
        self.case_data = self.load_case_data(case_data)
        if self.config["llm_type"] == "offline":
            self.llm = OfflineLLM(self.config["model_path"])
        elif self.config["llm_type"] == "apillm":
            self.llm = APILLM(
                api_key=self.config.get("api_key"),
                api_secret=self.config.get("api_secret", None),
                platform=self.config["model_platform"],
                model=self.config["model_type"],
                api_base=self.config.get("api_base"),
            )
        else:
            raise ValueError(
                f"Unsupported llm_type: {self.config['llm_type']}."
            )

        self.judge = self.create_agent(self.config["judge"], log_think=log_think)
        self.lawyers = [
            self.create_agent(lawyer, log_think=log_think)
            for lawyer in self.config["lawyers"]
        ]
        self.role_colors = {
            "법원 서기": "cyan",
            "재판장": "yellow",
            "원고 변호사": "green",
            "피고 변호사": "red",
        }

    @staticmethod
    def setup_logging(log_level):
        """
        로그 구성을 설정합니다.
        :param log_level: 로그 수준
        """
        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)],
        )

    @staticmethod
    def load_json(file_path):
        """
        JSON 파일을 불러옵니다.
        :param file_path: 파일 경로
        :return: JSON 데이터
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_case_data(case_path):
        """
        사례 데이터를 불러옵니다.
        :param case_path: 사례 파일 경로 또는 디렉터리 경로
        :return: 모든 사례 데이터를 포함한 목록
        """
        cases = []
        with open(case_path, "r", encoding="utf-8") as file:
            for line in file:
                case = json.loads(line)
                cases.append(case)
        return cases

    def create_agent(self, role_config, log_think=False):
        """
        역할 에이전트를 생성합니다.
        :param role_config: 역할 구성
        :return: Agent 인스턴스
        """
        return Agent(
            id=role_config["id"],
            name=role_config["name"],
            role=role_config.get("role", None),
            description=role_config["description"],
            llm=self.llm,
            db=db(role_config["name"]),
            log_think=log_think,
        )

    def add_to_history(self, role, name, content):
        """
        대화를 기록에 추가합니다.
        :param role: 발언 역할
        :param name: 발언자 이름
        :param content: 대화 내용
        """
        self.global_history.append({"role": role, "name": name, "content": content})
        color = self.role_colors.get(role, "white")
        console.print(
            Panel(content, title=f"{role} ({name})", border_style=color, expand=False)
        )

    def initialize_court(self):
        """
        법정을 초기화합니다.
        """
        self.global_history = []
        court_rules = self.config["stenographer"]["court_rules"]
        self.add_to_history("법원 서기", self.config["stenographer"]["name"], court_rules)
        self.add_to_history(
            "재판장",
            self.judge.name,
            "지금부터 개정합니다.",
        )
    # 법정 개정 전에 다양한 사항을 확인하는 예시로, 필요에 따라 대형 모델과 적절한 프롬프트를 활용해 시뮬레이션할 수 있으며, 여기서는 단순화된 예시입니다.
    def confirm_rights_and_obligations(self):
        """
        소송 권리와 의무를 확인합니다.
        """
        self.add_to_history(
            "재판장",
            self.judge.name,
            "각 측은 상대방 출석 인원에 대해 이의가 있습니까?",
        )
        self.add_to_history(
            "원고 변호사",
            self.plaintiff.name,
            "이의 없습니다.",
        )
        self.add_to_history(
            "피고 변호사",
            self.defendant.name,
            "이의 없습니다.",
        )
        self.add_to_history(
            "재판장",
            self.judge.name,
            "확인 결과, 출석한 당사자와 소송대리인의 신분은 모두 법적 요건을 충족하여 본 사건의 공판에 참여할 수 있습니다. 당사자의 소송상 권리와 의무에 관한 사항은 이미 공판 전에 서면으로 양측에게 통지되었습니다. 당사자들은 해당 권리와 의무의 내용을 명확히 이해하고 있습니까?",
        )
        self.add_to_history(
            "원고 변호사",
            self.plaintiff.name,
            "잘 알고 있습니다.",
        )
        self.add_to_history(
            "피고 변호사",
            self.defendant.name,
            "잘 알고 있습니다.",
        )
        self.add_to_history(
            "재판장",
            self.judge.name,
            "민사소송법에 따라, 양측 당사자가 재판부 또는 서기가 본 사건 당사자나 소송대리인의 가까운 친족이거나 사건과 직접적인 이해관계 등으로 공정한 재판에 영향을 줄 수 있다고 판단하는 경우, 사실과 사유를 제시하여 기피를 신청할 수 있습니다. 당사자들은 기피 신청이 필요합니까?",
        )
        self.add_to_history(
            "원고 변호사",
            self.plaintiff.name,
            "신청하지 않습니다.",
        )
        self.add_to_history(
            "피고 변호사",
            self.defendant.name,
            "신청하지 않습니다.",
        )

    def initial_statements(self, case):
        """
        초기 진술
        :param case: 현재 사례 데이터
        """
        self.add_to_history(
            "재판장",
            self.judge.name,
            "먼저 원고 측이 소송 청구, 사실관계 및 이유를 진술해 주십시오.",
        )
        self.add_to_history(
            "원고 변호사", self.plaintiff.name, case["plaintiff_statement"]
        )
        self.add_to_history(
            "재판장",
            self.judge.name,
            "피고 측은 답변해 주십시오.",
        )
        self.add_to_history(
            "피고 변호사", self.defendant.name, case["defendant_statement"]
        )

    def judge_initial_question(self):
        """
        판사의 초기 질문
        """
        content = self.judge.execute(
            None,
            history_list=self.global_history,
            prompt="원고 변호사와 피고 변호사의 진술을 바탕으로 양측 변호사가 어떤 쟁점을 중심으로 변론해야 하는지 요약하십시오. 현실에 부합하는 범위에서 최대한 간결하고 효과적으로 정리해 주십시오.",
        )
        self.add_to_history("재판장", self.judge.name, content)

    def debate_rounds(self, rounds):
        """
        변론 단계
        :param rounds: 변론 라운드 수
        """
        for i in trange(rounds, desc="Debate Rounds"):
            logging.info(f"Starting debate round {i+1}")
            for role, agent in [
                ("원고 변호사", self.plaintiff),
                ("피고 변호사", self.defendant),
            ]:
                p_q = agent.plan(self.global_history)
                content = agent.execute(
                    p_q,
                    self.global_history,
                    prompt=f"경험, 법조문, 판례 및 법정 대화 기록을 바탕으로 변론을 시작하십시오. context에 포함된 법조문을 인용했다면 해당 부분을 명시해 주십시오. 주의: 1. 지금은 법정 변론 단계이며 법정 조사 단계가 아닙니다. 2. 당신은 {role}입니다.",
                )
                self.add_to_history(role, agent.name, content)

    def final_judgment(self):
        """
        최종 판결
        """
        content = self.judge.speak(
            self.global_history, prompt="판사님, 판결을 내려 주십시오. (판결은 현실에 부합해야 합니다.)"
        )
        self.add_to_history("재판장", self.judge.name, content)

    def reflect_and_summary(self):
        """
        반성과 요약
        """
        self.plaintiff.reflect(self.global_history)
        self.defendant.reflect(self.global_history)

    def assign_roles(self):
        """
        역할을 무작위로 배정합니다.
        """
        roles = ["plaintiff", "defendant"]
        # random.shuffle(self.lawyers)
        self.plaintiff = self.lawyers[0]
        self.defendant = self.lawyers[1]
        self.plaintiff.role = roles[0]
        self.defendant.role = roles[1]

    def save_progress(self, index):
        """
        실행 상태를 기록합니다.
        :param index: 현재 사례 인덱스
        """
        progress = {"current_case_index": index}
        with open("progress.json", "w") as f:
            json.dump(progress, f)

    def load_progress(self):
        """
        실행 상태를 불러옵니다.
        :return: 실행 상태 딕셔너리 또는 None
        """
        if os.path.exists("progress.json"):
            with open("progress.json", "r") as f:
                return json.load(f)
        return None

    def run_simulation(self):
        """
        전체 법정 시뮬레이션 과정을 실행합니다.
        """
        progress = self.load_progress()
        start_index = progress["current_case_index"] if progress else 0

        case_data_to_run = self.case_data[:62]
        for index in range(start_index, len(case_data_to_run)):
            case = case_data_to_run[index]
            console.print(f"\n사례 {index + 1} 시뮬레이션을 시작합니다", style="bold")
            console.print("재판장을 제외한 다른 인원이 입장합니다", style="bold")
            self.assign_roles()  # 역할을 무작위로 배정합니다.
            self.initialize_court()
            self.confirm_rights_and_obligations()
            self.initial_statements(case)
            self.judge_initial_question()

            rounds = random.randint(3, 5)
            self.debate_rounds(rounds)
            self.save_progress(index)  # 현재 진행 상황을 기록합니다

            self.final_judgment()
            self.reflect_and_summary()
            console.print(f"사례 {index + 1} 공판이 종료되었습니다", style="bold")
            self.save_court_log(
                f"test_result/ours/1/court_session_test_case_{index + 1}.json"
            )

    def save_court_log(self, file_path):
        """
        법정 기록을 저장합니다.
        :param file_path: 저장할 파일 경로
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.global_history, f, ensure_ascii=False, indent=2)
        logging.info(f"Court session log saved to {file_path}")


def parse_arguments():
    """
    명령줄 인수를 해석합니다.
    :return: 해석된 인수
    """
    parser = argparse.ArgumentParser(description="Run a simulated court session.")
    parser.add_argument(
        "--config",
        default="example_role_config.json",
        help="Path to the role configuration file",
    )
    parser.add_argument(
        "--case",
        default="data/validation.jsonl",
        help="Path to the case data file in JSONL format",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--log_think", action="store_true", help="Log the agent think step"
    )
    return parser.parse_args()


def main():
    """
    메인 함수
    """
    args = parse_arguments()
    simulation = CourtSimulation(args.config, args.case, args.log_level, args.log_think)
    simulation.run_simulation()


if __name__ == "__main__":
    main()
