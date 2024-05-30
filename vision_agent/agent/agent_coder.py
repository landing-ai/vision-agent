import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.syntax import Syntax

from vision_agent.agent import Agent
from vision_agent.agent.agent_coder_prompts import (
    DEBUG,
    FIX_BUG,
    PROGRAM,
    TEST,
    VISUAL_TEST,
)
from vision_agent.llm import LLM, OpenAILLM
from vision_agent.lmm import LMM, OpenAILMM
from vision_agent.tools import TOOL_DOCSTRING, UTILITIES_DOCSTRING
from vision_agent.utils import CodeInterpreterFactory

IMPORT_HELPER = """
import math
import re
import sys
import copy
import datetime
import itertools
import collections
import heapq
import statistics
import functools
import hashlib
import numpy
import numpy as np
import string
from typing import *
from collections import *
from vision_agent.tools import *
"""
logging.basicConfig(stream=sys.stdout)
_LOGGER = logging.getLogger(__name__)
_EXECUTE = CodeInterpreterFactory.get_default_instance()
_CONSOLE = Console()


def write_tests(question: str, code: str, model: LLM) -> str:
    prompt = TEST.format(
        question=question,
        code=code,
    )
    completion = model(prompt)
    return preprocess_data(completion)


def preprocess_data(code: str) -> str:
    if "```python" in code:
        code = code[code.find("```python") + len("```python") :]
        code = code[: code.find("```")]
    return code


def parse_file_name(s: str) -> str:
    # We only output png files
    return "".join([p for p in s.split(" ") if p.endswith(".png")])


def write_program(
    question: str, feedback: str, model: LLM, media: Optional[Union[str, Path]] = None
) -> str:
    prompt = PROGRAM.format(
        docstring=TOOL_DOCSTRING, question=question, feedback=feedback
    )
    if isinstance(model, OpenAILMM):
        completion = model(prompt, images=[media] if media else None)
    else:
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
    result = _EXECUTE.exec_isolation(full_code)
    return {"code": code, "result": result.text(), "passed": result.success}


def run_visual_tests(
    question: str, code: str, viz_file: str, feedback: str, model: LMM
) -> Dict[str, Union[str, bool]]:
    prompt = VISUAL_TEST.format(
        docstring=TOOL_DOCSTRING,
        code=code,
        question=question,
        feedback=feedback,
    )
    completion = model(prompt, images=[viz_file])
    # type is from the prompt
    return json.loads(completion)  # type: ignore


def fix_bugs(code: str, tests: str, result: str, feedback: str, model: LLM) -> str:
    prompt = FIX_BUG.format(code=code, tests=tests, result=result, feedback=feedback)
    completion = model(prompt)
    return preprocess_data(completion)


class AgentCoder(Agent):
    """AgentCoder is based off of the AgentCoder paper https://arxiv.org/abs/2312.13010
    and it's open source code https://github.com/huangd1999/AgentCoder with some key
    differences. AgentCoder comprises of 3 components: a coder agent, a tester agent,
    and an executor. The tester agents writes code to test the code written by the coder
    agent, but in our case because we are solving a vision task it's difficult to write
    testing code. We instead have the tester agent write code to visualize the output
    of the code written by the coder agent. If the code fails, we pass it back to the
    coder agent to fix the bug, if it succeeds we pass it to a visual tester agent, which
    is an LMM model like GPT4V, to visually inspect the output and make sure it looks
    good."""

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
        self.verbose = verbose
        if self.verbose:
            _LOGGER.setLevel(logging.INFO)

    def __call__(
        self,
        input: Union[List[Dict[str, str]], str],
        media: Optional[Union[str, Path]] = None,
    ) -> str:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        return self.chat(input, media)

    def chat(
        self,
        input: List[Dict[str, str]],
        media: Optional[Union[str, Path]] = None,
    ) -> str:
        question = input[0]["content"]
        if media:
            question += f" Input file path: {os.path.abspath(media)}"

        code = ""
        feedback = ""
        for _ in range(self.max_turns):
            code = write_program(question, feedback, self.coder_agent, media=media)
            if self.verbose:
                _CONSOLE.print(
                    Syntax(code, "python", theme="gruvbox-dark", line_numbers=True)
                )
            debug = write_debug(question, code, feedback, self.tester_agent)
            if self.verbose:
                _CONSOLE.print(
                    Syntax(debug, "python", theme="gruvbox-dark", line_numbers=True)
                )
            results = execute_tests(code, debug)
            _LOGGER.info(
                f"execution results: passed: {results['passed']}\n{results['result']}"
            )

            if not results["passed"]:
                code = fix_bugs(
                    code, debug, results["result"].strip(), feedback, self.coder_agent  # type: ignore
                )
                if self.verbose:
                    _CONSOLE.print(
                        Syntax(code, "python", theme="gruvbox-dark", line_numbers=True)
                    )
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

    def log_progress(self, data: Dict[str, Any]) -> None:
        _LOGGER.info(data)
