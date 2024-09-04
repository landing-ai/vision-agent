import os
from typing import List
from uuid import UUID

from requests.exceptions import HTTPError

from vision_agent.clients.http import BaseHTTP
from vision_agent.tools.tools_types import BboxInputBase64, JobStatus, PromptTask
from vision_agent.utils.exceptions import FineTuneModelNotFound
from vision_agent.utils.type_defs import LandingaiAPIKey


class LandingPublicAPI(BaseHTTP):
    def __init__(self) -> None:
        landing_url = os.environ.get("LANDINGAI_URL", "https://api.landing.ai")
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

    def check_fine_tuning_job(self, job_id: UUID) -> JobStatus:
        url = f"v1/agent/jobs/fine-tuning/{job_id}/status"
        try:
            get_job = self.get(url)
        except HTTPError as err:
            if err.response.status_code == 404:
                raise FineTuneModelNotFound()
        return JobStatus(get_job["status"])
