import os
from uuid import UUID
from typing import List

from vision_agent.clients.http import BaseHTTP
from vision_agent.utils.type_defs import LandingaiAPIKey
from vision_agent.tools.meta_tools_types import BboxInputBase64, PromptTask


class LandingPublicAPI(BaseHTTP):
    def __init__(self) -> None:
        landing_url = os.environ.get("LANDINGAI_URL", "https://api.dev.landing.ai")
        landing_api_key = os.environ.get("LANDINGAI_API_KEY", LandingaiAPIKey().api_key)
        headers = {"Content-Type": "application/json", "apikey": landing_api_key}
        super().__init__(base_endpoint=landing_url, headers=headers)

    def launch_fine_tuning_job(
        self, model_name: str, task: PromptTask, bboxes: List[BboxInputBase64]
    ) -> UUID:
        url = "v1/agent/jobs/fine-tuning"
        data = {
            "model": {"name": model_name, "task": task.value},
            "bboxes": [bbox.model_dump(by_alias=True) for bbox in bboxes],
        }
        response = self.post(url, payload=data)
        return UUID(response["jobId"])
