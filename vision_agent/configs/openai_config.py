from typing import Type

from pydantic import BaseModel, Field

from vision_agent.lmm import LMM, OpenAILMM


class Config(BaseModel):
    # for vision_agent_v2
    agent: Type[LMM] = Field(default=OpenAILMM)
    agent_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": "gpt-4o-2024-08-06",
            "temperature": 0.0,
            "image_size": 768,
            "image_detail": "low",
        }
    )

    # for vision_agent_planner_v2
    planner: Type[LMM] = Field(default=OpenAILMM)
    planner_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": "gpt-4o-2024-08-06",
            "temperature": 0.0,
            "image_size": 768,
            "image_detail": "low",
        }
    )

    # for vision_agent_planner_v2
    summarizer: Type[LMM] = Field(default=OpenAILMM)
    summarizer_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": "o1",
            "temperature": 1.0,
            "image_size": 768,
        }
    )

    # for vision_agent_planner_v2
    critic: Type[LMM] = Field(default=OpenAILMM)
    critic_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": "gpt-4o-2024-08-06",
            "temperature": 0.0,
            "image_size": 768,
            "image_detail": "low",
        }
    )

    # for vision_agent_coder_v2
    coder: Type[LMM] = Field(default=OpenAILMM)
    coder_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": "gpt-4o-2024-08-06",
            "temperature": 0.0,
            "image_size": 768,
            "image_detail": "low",
        }
    )

    # for vision_agent_coder_v2
    tester: Type[LMM] = Field(default=OpenAILMM)
    tester_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": "gpt-4o-2024-08-06",
            "temperature": 0.0,
            "image_size": 768,
            "image_detail": "low",
        }
    )

    # for vision_agent_coder_v2
    debugger: Type[LMM] = Field(default=OpenAILMM)
    debugger_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": "gpt-4o-2024-08-06",
            "temperature": 0.0,
            "image_size": 768,
            "image_detail": "low",
        }
    )

    # for get_tool_for_task
    tool_tester: Type[LMM] = Field(default=OpenAILMM)
    tool_tester_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": "gpt-4o-2024-08-06",
            "temperature": 0.0,
            "image_size": 768,
            "image_detail": "low",
        }
    )

    # for get_tool_for_task
    tool_chooser: Type[LMM] = Field(default=OpenAILMM)
    tool_chooser_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": "gpt-4o-2024-08-06",
            "temperature": 1.0,
            "image_size": 768,
            "image_detail": "low",
        }
    )

    # for suggestions module
    suggester: Type[LMM] = Field(default=OpenAILMM)
    suggester_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": "gpt-4o-2024-08-06",
            "temperature": 1.0,
            "image_size": 768,
            "image_detail": "low",
        }
    )

    # for vqa module
    vqa: Type[LMM] = Field(default=OpenAILMM)
    vqa_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": "gpt-4o-2024-08-06",
            "temperature": 0.0,
            "image_size": 768,
            "image_detail": "low",
        }
    )

    def create_agent(self) -> LMM:
        return self.agent(**self.agent_kwargs)

    def create_planner(self) -> LMM:
        return self.planner(**self.planner_kwargs)

    def create_summarizer(self) -> LMM:
        return self.summarizer(**self.summarizer_kwargs)

    def create_critic(self) -> LMM:
        return self.critic(**self.critic_kwargs)

    def create_coder(self) -> LMM:
        return self.coder(**self.coder_kwargs)

    def create_tester(self) -> LMM:
        return self.tester(**self.tester_kwargs)

    def create_debugger(self) -> LMM:
        return self.debugger(**self.debugger_kwargs)

    def create_tool_tester(self) -> LMM:
        return self.tool_tester(**self.tool_tester_kwargs)

    def create_tool_chooser(self) -> LMM:
        return self.tool_chooser(**self.tool_chooser_kwargs)

    def create_suggester(self) -> LMM:
        return self.suggester(**self.suggester_kwargs)

    def create_vqa(self) -> LMM:
        return self.vqa(**self.vqa_kwargs)
