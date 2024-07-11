import subprocess
from pathlib import Path
from typing import List

import vision_agent as va
from vision_agent.tools.tool_utils import get_tool_documentation

CURRENT_FILE = None
CURRENT_LINE = 0
DEFAULT_WINDOW_SIZE = 100


def generate_vision_code(save_file: str, chat: str, media: List[str]) -> str:
    """Generates python code to solve vision based tasks.

    Parameters:
        save_file (str): The file path to save the code.
        chat (str): The chat message from the user.
        media (List[str]): The media files to use.

    Returns:
        str: The generated code.

    Examples
    --------
        >>> generate_vision_code("code.py", "Can you detect the dogs in this image?", ["dog.jpg"])
        from vision_agent.tools import load_image, owl_v2
        def detect_dogs(image_path: str):
            image = load_image(image_path)
            dogs = owl_v2("dog", image)
            return dogs
    """

    agent = va.agent.VisionAgentCoder()
    try:
        fixed_chat = [{"role": "user", "content": chat, "media": media}]
        response = agent.chat_with_workflow(fixed_chat)
        code = response["code"]
        with open(save_file, "w") as f:
            f.write(code)
        code_lines = code.splitlines(keepends=True)
        total_lines = len(code_lines)
        return view_lines(code_lines, 0, total_lines, save_file, total_lines)
    except Exception as e:
        return str(e)


def edit_vision_code(code_file: str, chat_history: List[str], media: List[str]) -> str:
    """Edits python code to solve a vision based task.

    Parameters:
        code_file (str): The file path to the code.
        chat_history (List[str]): The chat history to used to generate the code.

    Returns:
        str: The edited code.

    Examples
    --------
        >>> edit_vision_code(
        >>>     "code.py",
        >>>     ["Can you detect the dogs in this image?", "Can you use a higher threshold?"],
        >>>     ["dog.jpg"],
        >>> )
        from vision_agent.tools import load_image, owl_v2
        def detect_dogs(image_path: str):
            image = load_image(image_path)
            dogs = owl_v2("dog", image, threshold=0.8)
            return dogs
    """

    agent = va.agent.VisionAgentCoder()
    with open(code_file, "r") as f:
        code = f.read()

    # Append latest code to second to last message from assistant
    fixed_chat_history = []
    for i, chat in enumerate(chat_history):
        if i == 0:
            fixed_chat_history.append({"role": "user", "content": chat, "media": media})
        elif i > 0 and i < len(chat_history) - 1:
            fixed_chat_history.append({"role": "user", "content": chat})
        elif i == len(chat_history) - 1:
            fixed_chat_history.append({"role": "assistant", "content": code})
            fixed_chat_history.append({"role": "user", "content": chat})

    try:
        response = agent.chat_with_workflow(fixed_chat_history)
        code = response["code"]
        with open(code_file, "w") as f:
            f.write(code)
        code_lines = code.splitlines(keepends=True)
        total_lines = len(code_lines)
        return view_lines(code_lines, 0, total_lines, code_file, total_lines)
    except Exception as e:
        return str(e)


def format_lines(lines: List[str], start_idx: int) -> str:
    output = ""
    for i, line in enumerate(lines):
        output += f"{i + start_idx}|{line}"
    return output


def view_lines(
    lines: List[str], line_num: int, window_size: int, file_path: str, total_lines: int
) -> str:
    start = max(0, line_num - window_size)
    end = min(len(lines), line_num + window_size)
    return f"[File: {file_path} ({total_lines} lines total)]\n" + format_lines(
        lines[start:end], start
    )


def open_file(file_path: str, line_num: int = 0, window_size: int = 100) -> str:
    file_path_p =  Path(file_path)
    if not file_path_p.exists():
        return f"[File {file_path} does not exist]"

    total_lines = sum(1 for _ in open(file_path_p))
    window_size = window_size // 2
    if line_num - window_size < 0:
        line_num = window_size
    elif line_num >= total_lines:
        line_num = total_lines - 1 - window_size

    global CURRENT_LINE, CURRENT_FILE
    CURRENT_LINE = line_num
    CURRENT_FILE = file_path

    with open(file_path, "r") as f:
        lines = f.readlines()

    return view_lines(lines, line_num, window_size, file_path, total_lines)


def create_file(file_path: str) -> str:
    file_path_p = Path(file_path)
    if file_path_p.exists():
        return f"[File {file_path} already exists]"
    file_path_p.touch()
    return f"[File created {file_path}]"


def scroll_up() -> str:
    if CURRENT_FILE is None:
        return "[No file is open]"

    return open_file(CURRENT_FILE, CURRENT_LINE + DEFAULT_WINDOW_SIZE)


def scroll_down() -> str:
    if CURRENT_FILE is None:
        return "[No file is open]"

    return open_file(CURRENT_FILE, CURRENT_LINE - DEFAULT_WINDOW_SIZE)


def edit_file(file_path: str, start: int, end: int, content: str) -> str:
    file_path_p = Path(file_path)
    if not file_path_p.exists():
        return f"[File {file_path} does not exist]"

    total_lines = sum(1 for _ in open(file_path_p))
    if start < 0 or end < 0 or start > end or end > total_lines:
        return "[Invalid line range]"
    if start == end:
        end += 1

    new_content_lines = content.splitlines(keepends=True)
    new_content_lines = [
        line if line.endswith("\n") else line + "\n" for line in new_content_lines
    ]
    with open(file_path_p, "r") as f:
        lines = f.readlines()
        edited_lines = lines[:start] + new_content_lines + lines[end:]

    cur_line = start + len(content.split("\n")) // 2
    tmp_file = file_path_p.with_suffix(".tmp")
    with open(tmp_file, "w") as f:
        f.writelines(edited_lines)

    process = subprocess.Popen(
        [
            "flake8",
            "--isolated",
            "--select=F821,F822,F831,E111,E112,E113,E999,E902",
            tmp_file,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, _ = process.communicate()
    tmp_file.unlink()
    if stdout != "":
        stdout = stdout.replace(tmp_file.name, file_path)
        error_msg = f"[Edit failed with the following status]\n" + stdout
        original_view = view_lines(
            lines,
            start + ((end - start) // 2),
            DEFAULT_WINDOW_SIZE,
            file_path,
            total_lines,
        )
        total_lines_edit = sum(1 for _ in edited_lines)
        edited_view = view_lines(
            edited_lines, cur_line, DEFAULT_WINDOW_SIZE, file_path, total_lines_edit
        )

        error_msg += f"\n[This is how your edit would have looked like if applied]\n{edited_view}\n\n[This is the original code before your edit]\n{original_view}"
        return error_msg

    with open(file_path_p, "w") as f:
        f.writelines(edited_lines)

    return open_file(file_path, cur_line)


META_TOOL_DOCSTRING = get_tool_documentation([generate_vision_code, edit_vision_code])
