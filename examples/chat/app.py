import asyncio
import base64
import tempfile
from typing import Any, Dict, List, Optional

import cv2
import httpx
import numpy as np
from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import vision_agent.tools as T
from vision_agent.agent import VisionAgentV2
from vision_agent.models import AgentMessage
from vision_agent.utils.execute import CodeInterpreterFactory
from vision_agent.utils.video import frames_to_bytes

app = FastAPI()
DEBUG_HIL = True

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app's address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
clients = []


async def _async_update_callback(message: Dict[str, Any]):
    async with httpx.AsyncClient() as client:
        await client.post(
            "http://localhost:8000/send_message",
            json=message,
        )


def update_callback(message: Dict[str, Any]):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_async_update_callback(message))
    loop.close()


if DEBUG_HIL:
    # Use this code for debugging human-in-the-loop mode.
    agent = VisionAgentV2(
        verbose=True,
        update_callback=update_callback,
        hil=True,
    )
    code_interpreter = CodeInterpreterFactory.new_instance(non_exiting=True)
else:
    agent = VisionAgentV2(
        verbose=True,
        update_callback=update_callback,
    )
    code_interpreter = CodeInterpreterFactory.new_instance()


def process_messages_background(messages: List[Dict[str, Any]]):
    for message in messages:
        if "media" in message and message["media"] is None:
            del message["media"]

    response = agent.chat(
        [
            AgentMessage(
                role=message["role"],
                content=message["content"],
                media=message.get("media", None),
            )
            for message in messages
        ],
        code_interpreter=code_interpreter,
    )


class Message(BaseModel):
    role: str
    content: str
    media: Optional[List[str]] = None


class Detection(BaseModel):
    label: str
    bbox: List[int]
    confidence: float
    mask: Optional[List[int]] = None


def b64_video_to_frames(b64_video: str) -> List[np.ndarray]:
    video_frames = []
    # Convert base64 to video file bytes
    video_bytes = base64.b64decode(
        b64_video.split(",")[1] if "," in b64_video else b64_video
    )

    # Create a temporary file to store the video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_video:
        temp_video.write(video_bytes)
        temp_video.flush()

        # Read video frames using OpenCV
        cap = cv2.VideoCapture(temp_video.name)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    return video_frames


@app.post("/chat")
async def chat(
    messages: List[Message], background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    background_tasks.add_task(
        process_messages_background, [elt.model_dump() for elt in messages]
    )
    return {
        "status": "Processing started",
        "message": "Your messages are being processed in the background",
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            await websocket.receive_json()
    except WebSocketDisconnect:
        clients.remove(websocket)


@app.post("/send_message")
async def send_message(message: Message):
    for client in clients:
        await client.send_json(message.model_dump())
