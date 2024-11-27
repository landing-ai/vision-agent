import asyncio
from typing import Any, Dict, List, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from vision_agent.agent import VisionAgentV2
from vision_agent.agent.types import AgentMessage
from vision_agent.utils.execute import CodeInterpreterFactory

app = FastAPI()
DEBUG_HIL = False

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
