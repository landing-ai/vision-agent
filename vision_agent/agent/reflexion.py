import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from vision_agent.llm import LLM, OpenAILLM
from vision_agent.lmm import LMM

from .agent import Agent
from .reflexion_prompts import (
    CHECK_FINSH,
    COT_AGENT_REFLECT_INSTRUCTION,
    COT_REFLECT_INSTRUCTION,
    COT_SIMPLE_REFLECTION,
    COTQA_SIMPLE6,
    REFLECTION_HEADER,
)

logging.basicConfig(stream=sys.stdout)

_LOGGER = logging.getLogger(__name__)


def format_step(step: str) -> str:
    return step.strip("\n").strip().replace("\n", "")


def parse_action(input: str) -> Tuple[str, str]:
    # Make the pattern slightly less strict, the LMMs are not as good at following
    # instructions so they often would fail on the original regex.
    pattern = r"(\w+)\[(.+)\]"
    match = re.search(pattern, input)

    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument

    _LOGGER.error(f"Invalid action: {input}")
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
    r"""This is an implementation of the Reflexion paper https://arxiv.org/abs/2303.11366
    based on the original implementation https://github.com/noahshinn/reflexion in the
    hotpotqa folder. There are several differences between this implementation and the
    original one. Because we do not have instant feedback on whether or not the agent
    was correct, we use user feedback to determine if the agent was correct. The user
    feedback is evaluated by the self_reflect_model with a new prompt. We also expand
    Reflexion to include the ability to use an image as input to the action_agent and the
    self_reflect_model. Using Reflexion with LMMs may not work well, if it gets it wrong
    the first time, chances are it can't actually see the thing you want it to see.

    Example
    -------
        >>> from vision_agent.agent import Reflexion
        >>> agent = Reflexion()
        >>> question = "How many tires does a truck have?"
        >>> resp = agent(question)
        >>> print(resp)
        "18"
        >>> resp = agent([
        >>>     {"role": "user", "content": question},
        >>>     {"role": "assistant", "content": resp},
        >>>     {"role": "user", "content": "No I mean those regular trucks but where the back tires are double."}
        >>> ])
        >>> print(resp)
        "6"
        >>> agent = Reflexion(
        >>>     self_reflect_model=va.lmm.OpenAILMM(),
        >>>     action_agent=va.lmm.OpenAILMM()
        >>> )
        >>> quesiton = "How many hearts are in this image?"
        >>> resp = agent(question, image="cards.png")
        >>> print(resp)
        "6"
        >>> resp = agent([
        >>>     {"role": "user", "content": question},
        >>>     {"role": "assistant", "content": resp},
        >>>     {"role": "user", "content": "No, please count the hearts on the bottom card."}
        >>> ], image="cards.png")
        >>> print(resp)
        "4"
        )
    """

    def __init__(
        self,
        cot_examples: str = COTQA_SIMPLE6,
        reflect_examples: str = COT_SIMPLE_REFLECTION,
        agent_prompt: str = COT_AGENT_REFLECT_INSTRUCTION,
        reflect_prompt: str = COT_REFLECT_INSTRUCTION,
        finsh_prompt: str = CHECK_FINSH,
        self_reflect_model: Optional[Union[LLM, LMM]] = None,
        action_agent: Optional[Union[Agent, LLM, LMM]] = None,
        verbose: bool = False,
    ):
        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt
        self.finsh_prompt = finsh_prompt
        self.cot_examples = cot_examples
        self.reflect_examples = reflect_examples
        self.reflections: List[str] = []
        if verbose:
            _LOGGER.setLevel(logging.INFO)

        if isinstance(self_reflect_model, LLM) and not isinstance(action_agent, LLM):
            raise ValueError(
                "If self_reflect_model is an LLM, then action_agent must also be an LLM."
            )
        if isinstance(self_reflect_model, LMM) and isinstance(action_agent, LLM):
            raise ValueError(
                "If self_reflect_model is an LMM, then action_agent must also be an agent or LMM."
            )

        self.self_reflect_model = (
            OpenAILLM() if self_reflect_model is None else self_reflect_model
        )
        self.action_agent = OpenAILLM() if action_agent is None else action_agent

    def __call__(
        self,
        input: Union[str, List[Dict[str, str]]],
        image: Optional[Union[str, Path]] = None,
    ) -> str:
        """Invoke the vision agent.

        Parameters:
            input: a prompt that describe the task or a conversation in the format of [{"role": "user", "content": "describe your task here..."}].
            image: the input image referenced in the prompt parameter.

        Returns:
            A text response.
        """
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        return self.chat(input, image)

    def chat(
        self, chat: List[Dict[str, str]], image: Optional[Union[str, Path]] = None
    ) -> str:
        if len(chat) == 0 or chat[0]["role"] != "user":
            raise ValueError(
                f"Invalid chat. Should start with user and alternate between user"
                f"and assistant and contain at least one entry {chat}"
            )
        if image is not None and isinstance(self.action_agent, LLM):
            raise ValueError(
                "If image is provided, then action_agent must be an agent or LMM."
            )

        question = chat[0]["content"]
        if len(chat) == 1:
            results = self._step(question, image=image)
            self.last_scratchpad = results["scratchpad"]
            return results["action_arg"]

        # Observe
        chat_str = format_chat(chat)
        is_correct = self.prompt_finish(chat_str)
        self.last_scratchpad += "\nObservation: "
        if is_correct:
            self.last_scratchpad += "Answer is CORRECT"
            return self.self_reflect_model(chat)
        else:
            self.last_scratchpad += "Answer is INCORRECT"
            chat_context = "The previous conversation was:\n" + chat_str
            reflections = self.reflect(
                question, chat_context, self.last_scratchpad, image
            )
            _LOGGER.info(f" {reflections}")
            results = self._step(question, reflections, image=image)
            self.last_scratchpad = results["scratchpad"]
            return results["action_arg"]

    def _step(
        self,
        question: str,
        reflections: str = "",
        image: Optional[Union[str, Path]] = None,
    ) -> Dict[str, str]:
        # Think
        scratchpad = "\nThought:"
        scratchpad += " " + self.prompt_agent(question, reflections, scratchpad, image)
        _LOGGER.info(f" {scratchpad}")

        # Act
        scratchpad += "\nAction:"
        action = self.prompt_agent(question, reflections, scratchpad, image)
        _LOGGER.info(f" {action}")
        scratchpad += " " + action
        action_type, argument = parse_action(action)
        return {
            "scratchpad": scratchpad,
            "action_type": action_type,
            "action_arg": argument,
        }

    def reflect(
        self,
        question: str,
        context: str,
        scratchpad: str,
        image: Optional[Union[str, Path]],
    ) -> str:
        self.reflections += [
            self.prompt_reflection(question, context, scratchpad, image)
        ]
        return format_reflections(self.reflections)

    def prompt_agent(
        self,
        question: str,
        reflections: str,
        scratchpad: str,
        image: Optional[Union[str, Path]] = None,
    ) -> str:
        if isinstance(self.action_agent, LLM):
            return format_step(
                self.action_agent(
                    self._build_agent_prompt(question, reflections, scratchpad)
                )
            )
        elif isinstance(self.action_agent, LMM):
            return format_step(
                self.action_agent(
                    self._build_agent_prompt(question, reflections, scratchpad),
                    images=[image] if image is not None else None,
                )
            )
        elif isinstance(self.action_agent, Agent):
            return format_step(
                self.action_agent(
                    self._build_agent_prompt(question, reflections, scratchpad),
                    image=image,
                )
            )

    def prompt_reflection(
        self,
        question: str,
        context: str = "",
        scratchpad: str = "",
        image: Optional[Union[str, Path]] = None,
    ) -> str:
        if isinstance(self.self_reflect_model, LLM):
            return format_step(
                self.self_reflect_model(
                    self._build_reflect_prompt(question, context, scratchpad)
                )
            )
        return format_step(
            self.self_reflect_model(
                self._build_reflect_prompt(question, context, scratchpad),
                images=[image] if image is not None else None,
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
            examples=self.reflect_examples,
            context=context,
            question=question,
            scratchpad=scratchpad,
        )
