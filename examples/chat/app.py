import os
import threading
import time
from pathlib import Path

import streamlit as st
import zmq
from code_editor import code_editor
from streamlit_autorefresh import st_autorefresh

import vision_agent as va

WORKSPACE = Path(os.environ.get("WORKSPACE", ""))
ZMQ_PORT = os.environ.get("ZMQ_PORT", None)
if ZMQ_PORT is None:
    ZMQ_PORT = "5555"
    os.environ["ZMQ_PORT"] = ZMQ_PORT


CACHE = Path(".cache.pkl")
SAVE = {
    "name": "Save",
    "feather": "Save",
    "hasText": True,
    "commands": ["save-state", ["response", "saved"]],
    "response": "saved",
    "style": {"bottom": "calc(50% - 4.25rem", "right": "0.4rem"},
}
# set artifacts remote_path to WORKSPACE
artifacts = va.tools.Artifacts(WORKSPACE / "artifacts.pkl")
if Path("artifacts.pkl").exists():
    artifacts.load("artifacts.pkl")
else:
    artifacts.save("artifacts.pkl")

agent = va.agent.VisionAgent(verbosity=1, local_artifacts_path="artifacts.pkl")

st.set_page_config(layout="wide")

if "file_path" not in st.session_state:
    st.session_state.file_path = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "updates" not in st.session_state:
    st.session_state.updates = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""


def update_messages(messages, lock):
    if Path("artifacts.pkl").exists():
        artifacts.load("artifacts.pkl")
    new_chat, _ = agent.chat_with_code(messages, artifacts=artifacts)
    with lock:
        for new_message in new_chat:
            if new_message not in messages:
                messages.append(new_message)


def get_updates(updates, lock):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://*:{ZMQ_PORT}")

    while True:
        message = socket.recv_json()
        with lock:
            updates.append(message)
        time.sleep(0.1)


def submit():
    st.session_state.input_text = st.session_state.widget
    st.session_state.widget = ""


update_lock = threading.Lock()
message_lock = threading.Lock()

st_autorefresh(interval=1000, key="refresh")


def main():
    st.title("Vision Agent")
    left_column, right_column = st.columns([2, 3])

    with left_column:
        st.title("Chat & Code Execution")
        tabs = st.tabs(["Chat", "Code Execution Logs"])

        with tabs[0]:
            messages = st.container(height=400)
            for message in st.session_state.messages:
                if message["role"] in {"user", "assistant"}:
                    msg = message["content"]
                    msg = msg.replace("<execute_python>", "<execute_python>`")
                    msg = msg.replace("</execute_python>", "`</execute_python>")
                    messages.chat_message(message["role"]).write(msg)
                else:
                    messages.chat_message("observation").text(message["content"])

            st.text_input("Chat here", key="widget", on_change=submit)
            prompt = st.session_state.input_text

            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                messages.chat_message("user").write(prompt)
                message_thread = threading.Thread(
                    target=update_messages,
                    args=(st.session_state.messages, message_lock),
                )
                message_thread.daemon = True
                message_thread.start()
                st.session_state.input_text = ""

        with tabs[1]:
            updates = st.container(height=400)
            for update in st.session_state.updates:
                updates.chat_message("coder").write(update)

    with right_column:
        st.title("File Browser & Code Editor")
        tabs = st.tabs(["File Browser", "Code Editor"])

        with tabs[0]:
            uploaded_file = st.file_uploader("Upload a file")
            if uploaded_file is not None:
                with open(WORKSPACE / uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # make it None so it wont load and overwrite the image
                artifacts.artifacts[uploaded_file.name] = None

            for file in WORKSPACE.iterdir():
                if "__pycache__" not in str(file) and not str(file).startswith("."):
                    if st.button(file.name):
                        st.session_state.file_path = file

            if (
                "file_path" in st.session_state
                and st.session_state.file_path is not None
                and st.session_state.file_path.suffix
                in (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp",
                )
            ):
                st.image(
                    str(WORKSPACE / st.session_state.file_path), use_column_width=True
                )

        with tabs[1]:
            if (
                "file_path" not in st.session_state
                or st.session_state.file_path is None
                or st.session_state.file_path.suffix != ".py"
            ):
                st.write("Please select a python file from the file browser.")
            else:
                with open(WORKSPACE / st.session_state.file_path, "r") as f:
                    code = f.read()

                resp = code_editor(code, lang="python", buttons=[SAVE])
                if resp["type"] == "saved":
                    text = resp["text"]
                    with open(st.session_state.file_path, "w") as f:
                        f.write(text)

    if "update_thread_started" not in st.session_state:
        update_thread = threading.Thread(
            target=get_updates, args=(st.session_state.updates, update_lock)
        )
        update_thread.daemon = True
        update_thread.start()
        st.session_state.update_thread_started = True


if __name__ == "__main__":
    main()
