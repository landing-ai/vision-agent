import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Union

import vision_agent as va
from vision_agent.lmm.types import Message
from vision_agent.tools.tool_utils import get_tool_documentation
from vision_agent.tools.tools import TOOL_DESCRIPTIONS

# These tools are adapted from SWE-Agent https://github.com/princeton-nlp/SWE-agent

CURRENT_FILE = None
CURRENT_LINE = 0
DEFAULT_WINDOW_SIZE = 100
ZMQ_PORT = os.environ.get("ZMQ_PORT", None)


def report_progress_callback(port: int, inp: Dict[str, Any]) -> None:
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(f"tcp://localhost:{port}")
    socket.send_json(inp)


def filter_file(file_name: Union[str, Path]) -> bool:
    file_name_p = Path(file_name)
    return (
        file_name_p.is_file()
        and "__pycache__" not in str(file_name_p)
        and file_name_p.suffix in [".py", ".txt"]
        and not file_name_p.name.startswith(".")
    )


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
        >>> generate_vision_code("code.py", "Can you detect the dogs in this image?", ["image.jpg"])
        from vision_agent.tools import load_image, owl_v2
        def detect_dogs(image_path: str):
            image = load_image(image_path)
            dogs = owl_v2("dog", image)
            return dogs
    """

    if ZMQ_PORT is not None:
        agent = va.agent.VisionAgentCoder(
            report_progress_callback=lambda inp: report_progress_callback(
                int(ZMQ_PORT), inp
            )
        )
    else:
        agent = va.agent.VisionAgentCoder()
    try:
        fixed_chat: List[Message] = [{"role": "user", "content": chat, "media": media}]
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
    fixed_chat_history: List[Message] = []
    for i, chat in enumerate(chat_history):
        if i == 0:
            fixed_chat_history.append({"role": "user", "content": chat, "media": media})
        elif i > 0 and i < len(chat_history) - 1:
            fixed_chat_history.append({"role": "user", "content": chat})
        elif i == len(chat_history) - 1:
            fixed_chat_history.append({"role": "assistant", "content": code})
            fixed_chat_history.append({"role": "user", "content": chat})

    try:
        response = agent.chat_with_workflow(fixed_chat_history, test_multi_plan=False)
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
    return (
        f"[File: {file_path} ({total_lines} lines total)]\n"
        + format_lines(lines[start:end], start)
        + ("[End of file]" if end == len(lines) else f"[{len(lines) - end} more lines]")
    )


def open_file(file_path: str, line_num: int = 0, window_size: int = 100) -> str:
    """Opens the file at at the given path in the editor. If `line_num` is provided,
    the window will be moved to include that line. It only shows the first 100 lines by
    default! Max `window_size` supported is 2000. use `scroll up/down` to view the file
    if you want to see more.

    Parameters:
        file_path (str): The file path to open, preferred absolute path.
        line_num (int): The line number to move the window to.
        window_size (int): The number of lines to show above and below the line.
    """

    file_path_p = Path(file_path)
    if not file_path_p.exists():
        return f"[File {file_path} does not exist]"

    total_lines = sum(1 for _ in open(file_path_p))
    window_size = min(window_size, 2000)
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
    """Creates and opens a new file with the given name.

    Parameters:
        file_path (str): The file path to create, preferred absolute path.
    """

    file_path_p = Path(file_path)
    if file_path_p.exists():
        return f"[File {file_path} already exists]"
    file_path_p.touch()
    global CURRENT_FILE
    CURRENT_FILE = file_path
    return f"[File created {file_path}]"


def scroll_up() -> str:
    """Moves the window up by 100 lines."""
    if CURRENT_FILE is None:
        return "[No file is open]"

    return open_file(CURRENT_FILE, CURRENT_LINE + DEFAULT_WINDOW_SIZE)


def scroll_down() -> str:
    """Moves the window down by 100 lines."""
    if CURRENT_FILE is None:
        return "[No file is open]"

    return open_file(CURRENT_FILE, CURRENT_LINE - DEFAULT_WINDOW_SIZE)


def search_dir(search_term: str, dir_path: str) -> str:
    """Searches for search_term in all files in a directory.

    Parameters:
        search_term (str): The search term to look for.
        dir_path (str): The directory path to search in, preferred absolute path.
    """

    dir_path_p = Path(dir_path)
    if not dir_path_p.exists():
        return f"[Directory {dir_path} does not exist]"

    matches = []
    for file in dir_path_p.glob("**/*"):
        if filter_file(file):
            with open(file, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if search_term in line:
                        matches.append(f"{file}:{i}|{line.strip()}\n")
    if not matches:
        return f"[No matches found for {search_term} in {dir_path}]"
    if len(matches) > 100:
        return f"[More than {len(matches)} matches found for {search_term} in {dir_path}. Please narrow your search]"

    return_str = f"[Found {len(matches)} matches for {search_term} in {dir_path}]\n"
    for match in matches:
        return_str += match

    return_str += f"[End of matches for {search_term} in {dir_path}]"
    return return_str


def search_file(search_term: str, file_path: str) -> str:
    """Searches the file for the given search term.

    Parameters:
        search_term (str): The search term to look for.
        file_path (str): The file path to search in, preferred absolute path.
    """

    file_path_p = Path(file_path)
    if not file_path_p.exists():
        return f"[File {file_path} does not exist]"

    with open(file_path_p, "r") as f:
        lines = f.readlines()

    search_results = []
    for i, line in enumerate(lines):
        if search_term in line:
            search_results.append(f"{i}|{line.strip()}\n")

    if not search_results:
        return f"[No matches found for {search_term} in {file_path}]"

    return_str = (
        f"[Found {len(search_results)} matches for {search_term} in {file_path}]\n"
    )
    for result in search_results:
        return_str += result

    return_str += f"[End of matches for {search_term} in {file_path}]"
    return return_str


def find_file(file_name: str, dir_path: str = "./") -> str:
    """Finds all files with the given name in the specified directory.

    Parameters:
        file_name (str): The file name to look for.
        dir_path (str): The directory path to search in, preferred absolute path.
    """

    dir_path_p = Path(dir_path)
    if not dir_path_p.exists():
        return f"[Directory {dir_path} does not exist]"

    files = list(dir_path_p.glob(f"**/*{file_name}*"))
    files = [f for f in files if filter_file(f)]
    if not files:
        return f"[No files found in {dir_path} with name {file_name}]"

    return_str = f"[Found {len(files)} matches for {file_name} in {dir_path}]\n"
    for match in files:
        return_str += str(match) + "\n"

    return_str += f"[End of matches for {file_name} in {dir_path}]"
    return return_str


def edit_file(file_path: str, start: int, end: int, content: str) -> str:
    """Edits the file at the given path with the provided content. The content will be
    inserted between the `start` and `end` line numbers. If the `start` and `end` are
    the same, the content will be inserted at the `start` line number. If the `end` is
    greater than the total number of lines in the file, the content will be inserted at
    the end of the file. If the `start` or `end` are negative, the function will return
    an error message.

    Parameters:
        file_path (str): The file path to edit, preferred absolute path.
        start (int): The line number to start the edit.
        end (int): The line number to end the edit.
        content (str): The content to insert.
    """
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
        error_msg = "[Edit failed with the following status]\n" + stdout
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


def get_tool_descriptions() -> str:
    """Returns a description of all the tools that `generate_vision_code` has access to.
    Helpful for answering questions about what types of vision tasks you can do with
    `generate_vision_code`."""
    return TOOL_DESCRIPTIONS


META_TOOL_DOCSTRING = get_tool_documentation(
    [
        get_tool_descriptions,
        generate_vision_code,
        edit_vision_code,
        open_file,
        create_file,
        scroll_up,
        scroll_down,
        edit_file,
        search_dir,
        search_file,
        find_file,
    ]
)
