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

from vision_agent.agent import VisionAgentV2
from vision_agent.models import AgentMessage
from vision_agent.lmm import AnthropicLMM
from vision_agent.utils.execute import CodeInterpreterFactory

from dotenv import load_dotenv
import os

PORT_FRONTEND = os.getenv("PORT_FRONTEND")
DEBUG_HIL = os.getenv("DEBUG_HIL")

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://localhost:{PORT_FRONTEND}"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single WebSocket client tracking
active_client: Optional[WebSocket] = None
active_client_lock = asyncio.Lock()

# Add a global flag to track if processing should be canceled
processing_canceled = False
processing_canceled_lock = asyncio.Lock()

async def _async_update_callback(message: Dict[str, Any]):
    global processing_canceled
    
    # Check if processing has been canceled
    async with processing_canceled_lock:
        if processing_canceled:
            # Skip sending updates if processing has been canceled
            return
    
    # Try to send message to active WebSocket client
    async with active_client_lock:
        if active_client:
            try:
                await active_client.send_json(message)
            except Exception:
                print("Client disconnected unexpectedly.")
        else:
            print("No active client to send to.")


def update_callback(message: Dict[str, Any]):
    # Needed for non-async context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_async_update_callback(message))
    loop.close()


# Agent setup
if DEBUG_HIL:
    agent = VisionAgentV2(
        verbose=True,
        update_callback=update_callback,
        hil=True,
    )
    code_interpreter = CodeInterpreterFactory.new_instance(non_exiting=True)
else:
    agent = VisionAgentV2(
        agent=AnthropicLMM(model_name="claude-3-7-sonnet-20250219"),
        verbose=True,
        update_callback=update_callback,
    )
    code_interpreter = CodeInterpreterFactory.new_instance()

async def reset_cancellation_flag():
    global processing_canceled
    async with processing_canceled_lock:
        processing_canceled = False


def process_messages_background(messages: List[Dict[str, Any]]):
    global processing_canceled
    if processing_canceled:
        return
        
    for message in messages:
        if "media" in message and message["media"] is None:
            del message["media"]

    # Process messages normally (since cancellation is checked in the callback)

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
    video_bytes = base64.b64decode(
        b64_video.split(",")[1] if "," in b64_video else b64_video
    )
    video_frames = []
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_video:
        temp_video.write(video_bytes)
        temp_video.flush()

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
    # Reset cancellation flag before starting new processing
    await reset_cancellation_flag()
    
    background_tasks.add_task(
        process_messages_background, [m.model_dump() for m in messages]
    )
    return {"status": "Processing started"}


@app.post("/cancel")
async def cancel_processing():
    """Cancel any ongoing message processing."""
    global processing_canceled
    async with processing_canceled_lock:
        processing_canceled = True
    
    # Also clear the active websocket if possible
    async with active_client_lock:
        if active_client:
            try:
                # Send a cancellation message that the frontend can detect
                await active_client.send_json({
                    "role": "system",
                    "content": "Processing canceled by user."
                })
            except Exception:
                pass
    
    return {"status": "Processing canceled"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global active_client
    
    # First check if there's already a connection before accepting
    async with active_client_lock:
        if active_client:
            # Don't immediately accept if there's already a connection
            # Either reject or queue this connection
            await websocket.close(code=1000, reason="Only one connection allowed")
            return
        
        # Accept the connection only if there isn't an active client
        await websocket.accept()
        active_client = websocket
    
    try:
        while True:
            await websocket.receive_json()
    except WebSocketDisconnect:
        async with active_client_lock:
            if active_client == websocket:
                active_client = None


@app.post("/send_message")
async def send_message(message: Message):
    await _async_update_callback(message.model_dump())