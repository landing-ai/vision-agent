from typing import Type

from pydantic import BaseModel, Field

from vision_agent.lmm import LMM, AnthropicLMM, OpenAILMM, GoogleLMM

model_name = "gemini-2.5-pro-preview-03-25"
lmm_type = GoogleLMM


class Config(BaseModel):
    # for vision_agent_v2
    agent: Type[LMM] = Field(default=lmm_type)
    agent_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": model_name,
            "temperature": 0.0,
            "image_size": 768,
        }
    )

    # for vision_agent_planner_v2
    planner: Type[LMM] = Field(default=lmm_type)
    planner_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": model_name,
            "temperature": 0.0,
            "image_size": 768,
        }
    )

    summarizer: Type[LMM] = Field(default=lmm_type)
    summarizer_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": model_name,
            "temperature": 0.0,
            "image_size": 768,
        }
    )

    # for vision_agent_planner_v2
    critic: Type[LMM] = Field(default=lmm_type)
    critic_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": model_name,
            "temperature": 0.0,
            "image_size": 768,
        }
    )

    # for vision_agent_coder_v2
    coder: Type[LMM] = Field(default=lmm_type)
    coder_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": model_name,
            "temperature": 0.0,
            "image_size": 768,
        }
    )

    # for vision_agent_coder_v2
    tester: Type[LMM] = Field(default=lmm_type)
    tester_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": model_name,
            "temperature": 0.0,
            "image_size": 768,
        }
    )

    # for vision_agent_coder_v2
    debugger: Type[LMM] = Field(default=lmm_type)
    debugger_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": model_name,
            "temperature": 0.0,
            "image_size": 768,
        }
    )

    # for get_tool_for_task
    tool_tester: Type[LMM] = Field(default=lmm_type)
    tool_tester_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": model_name,
            "temperature": 0.0,
            "image_size": 768,
        }
    )

    # for get_tool_for_task
    tool_chooser: Type[LMM] = Field(default=lmm_type)
    tool_chooser_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": model_name,
            "temperature": 1.0,
            "image_size": 768,
        }
    )

    # for get_tool_for_task
    od_judge: Type[LMM] = Field(default=lmm_type)
    od_judge_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": model_name,
            "temperature": 0.0,
            "image_size": 512,
        }
    )

    # for suggestions module
    suggester: Type[LMM] = Field(default=OpenAILMM)
    suggester_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": "o1",
            "temperature": 1.0,
            "image_detail": "high",
            "image_size": 1024,
        }
    )

    # for vqa module
    vqa: Type[LMM] = Field(default=lmm_type)
    vqa_kwargs: dict = Field(
        default_factory=lambda: {
            "model_name": model_name,
            "temperature": 0.0,
            "image_size": 768,
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

    def create_od_judge(self) -> LMM:
        return self.od_judge(**self.od_judge_kwargs)

    def create_suggester(self) -> LMM:
        return self.suggester(**self.suggester_kwargs)

    def create_vqa(self) -> LMM:
        return self.vqa(**self.vqa_kwargs)
