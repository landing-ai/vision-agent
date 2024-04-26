import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

from vision_agent.agent import Agent
from vision_agent.llm import LLM, OpenAILLM
from vision_agent.lmm import LMM, OpenAILMM
from vision_agent.tools.tools_v2 import TOOLS_DOCSTRING, UTILITIES_DOCSTRING

from .agent_coder_prompts import DEBUG, FIX_BUG, PROGRAM, TEST, VISUAL_TEST
from .execution import IMPORT_HELPER, check_correctness

logging.basicConfig(stream=sys.stdout)
_LOGGER = logging.getLogger(__name__)


def write_tests(question: str, code: str, model: LLM) -> str:
    prompt = TEST.format(
        question=question,
        code=code,
    )
    completion = model(prompt)
    return preprocess_data(completion)


def preprocess_data(code: str) -> str:
    if f"```python" in code:
        code = code[code.find(f"```python") + len(f"```python") :]
        code = code[: code.find("```")]
    return code


def parse_file_name(s: str) -> str:
    # We only output png files
    return "".join([p for p in s.split(" ") if p.endswith(".png")])


def write_program(question: str, feedback: str, model: LLM) -> str:
    prompt = PROGRAM.format(
        docstring=TOOLS_DOCSTRING, question=question, feedback=feedback
    )
    completion = model(prompt)
    return preprocess_data(completion)


def write_debug(question: str, code: str, feedback: str, model: LLM) -> str:
    prompt = DEBUG.format(
        docstring=UTILITIES_DOCSTRING,
        code=code,
        question=question,
        feedback=feedback,
    )
    completion = model(prompt)
    return preprocess_data(completion)


def execute_tests(code: str, tests: str) -> Dict[str, Union[str, bool]]:
    full_code = f"{IMPORT_HELPER}\n{code}\n{tests}"
    return check_correctness(full_code, 20.0)


def run_visual_tests(
    question: str, code: str, viz_file: str, feedback: str, model: LMM
) -> Dict[str, Union[str, bool]]:
    prompt = VISUAL_TEST.format(
        docstring=TOOLS_DOCSTRING,
        code=code,
        question=question,
        feedback=feedback,
    )
    completion = model(prompt, images=[viz_file])
    return json.loads(completion)


def fix_bugs(code: str, tests: str, result: str, feedback: str, model: LLM) -> str:
    prompt = FIX_BUG.format(completion=code, test_case=tests, result=result)
    completion = model(prompt)
    return preprocess_data(completion)


class AgentCoder(Agent):
    def __init__(
        self,
        coder_agent: Optional[LLM] = None,
        tester_agent: Optional[LLM] = None,
        visual_tester_agent: Optional[LMM] = None,
        verbose: bool = False,
    ) -> None:
        self.coder_agent = (
            OpenAILLM(temperature=0.1) if coder_agent is None else coder_agent
        )
        self.tester_agent = (
            OpenAILLM(temperature=0.1) if tester_agent is None else tester_agent
        )
        self.visual_tester_agent = (
            OpenAILMM(temperature=0.1, json_mode=True)
            if visual_tester_agent is None
            else visual_tester_agent
        )
        self.max_turns = 3
        if verbose:
            _LOGGER.setLevel(logging.INFO)

    def __call__(
        self,
        input: Union[List[Dict[str, str]], str],
        image: Optional[Union[str, Path]] = None,
    ) -> str:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        return self.chat(input, image)

    def chat(
        self,
        input: List[Dict[str, str]],
        image: Optional[Union[str, Path]] = None,
    ) -> str:
        question = input[0]["content"]
        if image:
            question += f" Input file path: {os.path.abspath(image)}"

        code = ""
        feedback = ""
        for _ in range(self.max_turns):
            code = write_program(question, feedback, self.coder_agent)
            _LOGGER.info(f"code:\n{code}")
            debug = write_debug(question, code, feedback, self.coder_agent)
            _LOGGER.info(f"debug:\n{debug}")
            results = execute_tests(code, debug)
            _LOGGER.info(
                f"execution results: passed: {results['passed']}\n{results['result']}"
            )

            if not results["passed"]:
                code = fix_bugs(
                    code, debug, results["result"].strip(), feedback, self.coder_agent
                )
                _LOGGER.info(f"fixed code:\n{code}")
            else:
                # TODO: Sometimes it prints nothing, so we need to handle that case
                # TODO: The visual agent reflection does not work very well, needs more testing
                # viz_test_results = run_visual_tests(
                #     question, code, parse_file_name(results["result"].strip()), feedback, self.visual_tester_agent
                # )
                # _LOGGER.info(f"visual test results:\n{viz_test_results}")
                # if viz_test_results["finished"]:
                #     return f"{IMPORT_HELPER}\n{code}"
                # feedback += f"\n{viz_test_results['feedback']}"

                return f"{IMPORT_HELPER}\n{code}"

        return f"{IMPORT_HELPER}\n{code}"

    def log_progress(self, description: str) -> None:
        _LOGGER.info(description)
