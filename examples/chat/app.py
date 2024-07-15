import os
import pickle as pkl
from pathlib import Path
from typing import Dict, List

import streamlit as st
from code_editor import code_editor

import vision_agent as va

WORKSPACE = Path(os.environ.get("WORKSPACE", ""))

st.set_page_config(layout="wide")

st.title("Vision Agent")
left_column, right_column = st.columns([2, 3])

CACHE = Path(".cache.pkl")
SAVE = {
    "name": "Save",
    "feather": "Save",
    "hasText": True,
    "commands": ["save-state", ["response", "saved"]],
    "response": "saved",
    "style": {"bottom": "calc(50% - 4.25rem", "right": "0.4rem"},
}
agent = va.agent.VisionAgent(verbosity=1)


def generate_response(chat: List[Dict]) -> List[Dict]:
    return agent.chat_with_code(chat)


if "file_path" not in st.session_state:
    st.session_state.file_path = None


if "messages" not in st.session_state:
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            st.session_state.messages = pkl.load(f)
    else:
        st.session_state.messages = []

with left_column:
    st.title("Chat")

    messages = st.container(height=400)
    for message in st.session_state.messages:
        messages.chat_message(message["role"]).write(message["content"])

    if prompt := st.chat_input("Chat here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        messages.chat_message("user").write(prompt)

        updated_chat = generate_response(st.session_state.messages)
        for chat in updated_chat:
            if chat not in st.session_state.messages:
                st.session_state.messages.append(chat)
                if chat["role"] in {"user", "assistant"}:
                    msg = chat["content"]
                    msg = msg.replace("<execute_python>", "<execute_python>`")
                    msg = msg.replace("</execute_python>", "`</execute_python>")
                    messages.chat_message(chat["role"]).write(msg)
                else:
                    messages.chat_message("observation").text(chat["content"])


with right_column:
    st.title("File & Code Editor")

    tabs = st.tabs(["File Browser", "Code Editor"])

    with tabs[0]:
        uploaded_file = st.file_uploader("Upload a file")
        if uploaded_file is not None:
            with open(WORKSPACE / uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

        for file in WORKSPACE.iterdir():
            if "__pycache__" not in str(file) and not str(file).startswith("."):
                if st.button(file.name):
                    st.session_state.file_path = file

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
