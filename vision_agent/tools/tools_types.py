from enum import Enum
from uuid import UUID
from typing import List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, SerializationInfo, field_serializer


class BboxInput(BaseModel):
    image_path: str
    labels: List[str]
    bboxes: List[Tuple[int, int, int, int]]


class BboxInputBase64(BaseModel):
    image: str
    filename: str
    labels: List[str]
    bboxes: List[Tuple[int, int, int, int]]


class PromptTask(str, Enum):
    """Valid task prompts options for the Florence2 model."""

    PHRASE_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"


class Florence2FtRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image: Optional[str] = None
    video: Optional[bytes] = None
    task: PromptTask
    prompt: Optional[str] = ""
    chunk_length_frames: Optional[int] = None
    postprocessing: Optional[str] = None
    job_id: Optional[UUID] = Field(None, alias="jobId")

    @field_serializer("job_id")
    def serialize_job_id(self, job_id: UUID, _info: SerializationInfo) -> str:
        return str(job_id)


class JobStatus(str, Enum):
    """The status of a fine-tuning job.

    CREATED:
        The job has been created and is waiting to be scheduled to run.
    STARTING:
        The job has started running, but not entering the training phase.
    TRAINING:
        The job is training a model.
    EVALUATING:
        The job is evaluating the model and computing metrics.
    PUBLISHING:
        The job is exporting the artifact(s) to an external directory (s3 or local).
    SUCCEEDED:
        The job has finished, including training, evaluation and publishing the
        artifact(s).
    FAILED:
        The job has failed for some reason internally, it can be due to resources
        issues or the code itself.
    STOPPED:
        The job has been stopped by the use locally or in the cloud.
    """

    CREATED = "CREATED"
    STARTING = "STARTING"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    PUBLISHING = "PUBLISHING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


class ODResponseData(BaseModel):
    label: str
    score: float
    bbox: Union[list[int], list[float]] = Field(alias="bounding_box")

    model_config = ConfigDict(
        populate_by_name=True,
    )


BoundingBoxes = list[ODResponseData]
