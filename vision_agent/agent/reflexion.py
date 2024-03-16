import re
from typing import Dict, List, Tuple, Union

from vision_agent import LLM, OpenAILLM

from .agent import Agent
from .reflexion_prompts import (
    CHECK_FINSH,
    COT_AGENT_REFLECT_INSTRUCTION,
    COT_REFLECT_INSTRUCTION,
    COT_SIMPLE_REFLECTION,
    COTQA_SIMPLE6,
    REFLECTION_HEADER,
)


def format_step(step: str) -> str:
    return step.strip("\n").strip().replace("\n", "")


def parse_action(input: str) -> Tuple[str, str]:
    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, input)

    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument

    raise ValueError(f"Invalid action: {input}")


def format_reflections(reflections: List[str], header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ""
    else:
        return (
            header + "Reflections:\n- " + "\n- ".join([r.strip() for r in reflections])
        )


def format_chat(chat: List[Dict[str, str]]) -> str:
    chat_str = ""
    for c in chat:
        chat_str += c["role"] + ": " + c["content"] + "\n"
    return chat_str.strip()


class Reflexion(Agent):
    def __init__(
        self,
        cot_examples: str = COTQA_SIMPLE6,
        reflect_examples: str = COT_SIMPLE_REFLECTION,
        agent_prompt: str = COT_AGENT_REFLECT_INSTRUCTION,
        reflect_prompt: str = COT_REFLECT_INSTRUCTION,
        finsh_prompt: str = CHECK_FINSH,
        self_reflect_llm: Optional[LLM] = None,
        action_agent: Optional[Union[Agent, LLM]] = None,
    ):
        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt
        self.finsh_prompt = finsh_prompt
        self.cot_examples = cot_examples
        self.refelct_examples = reflect_examples
        self.self_reflect_llm = self_reflect_llm
        self.action_agent = action_agent
        self.reflections: List[str] = []

        if self_reflect_llm is None:
            self.self_reflect_llm = OpenAILLM()
        if action_agent is None:
            self.action_agent = OpenAILLM()

    def __call__(self, input: Union[List[Dict[str, str]], str]) -> str:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        return self.chat(input)

    def chat(self, chat: List[Dict[str, str]]) -> str:
        if len(chat) == 0 or chat[0]["role"] != "user":
            raise ValueError(
                f"Invalid chat. Should start with user and then assistant and contain at least one entry {chat}"
            )
        question = chat[0]["content"]
        if len(chat) == 1:
            results = self._step(question)
            self.last_scratchpad = results["scratchpad"]
            return results["action_arg"]

        # Observe
        chat_str = format_chat(chat)
        is_correct = self.prompt_finish(chat_str)
        self.last_scratchpad += "\nObservation: "
        if is_correct:
            self.last_scratchpad += "Answer is CORRECT"
            return self.self_reflect_llm(chat)
        else:
            self.last_scratchpad += "Answer is INCORRECT"
            chat_context = "The previous conversation was:\n" + chat_str
            reflections = self.reflect(question, chat_context, self.last_scratchpad)
            results = self._step(question, reflections)
            self.last_scratchpad = results["scratchpad"]
            return results["action_arg"]

    def _step(self, question: str, reflections: str = "") -> Dict[str, str]:
        # Think
        scratchpad = "\nThought:"
        scratchpad += " " + self.prompt_agent(question, reflections, scratchpad)

        # Act
        scratchpad += "\nAction:"
        action = self.prompt_agent(question, reflections, scratchpad)
        scratchpad += " " + action
        action_type, argument = parse_action(action)
        return {
            "scratchpad": scratchpad,
            "action_type": action_type,
            "action_arg": argument,
        }

    def reflect(self, question: str, context: str, scratchpad: str) -> str:
        self.reflections += [self.prompt_reflection(question, context, scratchpad)]
        return format_reflections(self.reflections)

    def prompt_agent(self, question: str, reflections: str, scratchpad: str) -> str:
        return format_step(
            self.action_agent(
                self._build_agent_prompt(question, reflections, scratchpad)
            )
        )

    def prompt_reflection(
        self, question: str, context: str = "", scratchpad: str = ""
    ) -> str:
        return format_step(
            self.self_reflect_llm(
                self._build_reflect_prompt(question, context, scratchpad)
            )
        )

    def prompt_finish(self, chat: str) -> bool:
        answer = self.action_agent(self.finsh_prompt.format(chat=chat))
        return "true" in answer.lower()

    def _build_agent_prompt(
        self, question: str, reflections: str, scratchpad: str
    ) -> str:
        return self.agent_prompt.format(
            examples=self.cot_examples,
            reflections=reflections,
            context="",
            question=question,
            scratchpad=scratchpad,
        )

    def _build_reflect_prompt(
        self, question: str, context: str = "", scratchpad: str = ""
    ) -> str:
        return self.reflect_prompt.format(
            examples=self.refelct_examples,
            context=context,
            question=question,
            scratchpad=scratchpad,
        )
