import difflib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from IPython.display import display

from vision_agent.tools.tools import get_tools_descriptions as _get_tool_descriptions
from vision_agent.utils.execute import Execution, MimeType
from vision_agent.utils.tools_doc import get_tool_documentation

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


def redisplay_results(execution: Execution) -> None:
    """This function is used to add previous execution results to the current output.
    This is handy if you are inside a notebook environment, call it notebook1, and you
    have a nested notebook environment, call it notebook2, and you want the execution
    results from notebook2 to be included in the execution results for notebook1.
    """
    for result in execution.results:
        if result.text is not None:
            display({MimeType.TEXT_PLAIN: result.text}, raw=True)
        if result.html is not None:
            display({MimeType.TEXT_HTML: result.html}, raw=True)
        if result.markdown is not None:
            display({MimeType.TEXT_MARKDOWN: result.markdown}, raw=True)
        if result.svg is not None:
            display({MimeType.IMAGE_SVG: result.svg}, raw=True)
        if result.png is not None:
            display({MimeType.IMAGE_PNG: result.png}, raw=True)
        if result.jpeg is not None:
            display({MimeType.IMAGE_JPEG: result.jpeg}, raw=True)
        if result.mp4 is not None:
            display({MimeType.VIDEO_MP4_B64: result.mp4}, raw=True)
        if result.latex is not None:
            display({MimeType.TEXT_LATEX: result.latex}, raw=True)
        if result.json is not None:
            display({MimeType.APPLICATION_JSON: result.json}, raw=True)
        if result.artifact is not None:
            display({MimeType.APPLICATION_ARTIFACT: result.artifact}, raw=True)
        if result.extra is not None:
            display(result.extra, raw=True)


class Artifacts:
    """Artifacts is a class that allows you to sync files between a local and remote
    environment. In our case, the remote environment could be where the VisionAgent is
    executing code and as the user adds new images, files or modifies files, those
    need to be in sync with the remote environment the VisionAgent is running in.
    """

    def __init__(self, cwd: Union[str, Path]) -> None:
        """Initializes the Artifacts object with it's remote and local save paths.

        Parameters:
            cwd (Union[str, Path]): The path to save all the chat related files. For example "/home/user/chat_abc/".
        """
        self.cwd = Path(cwd)

    def show(self) -> str:
        """Prints out all the files in the curret working directory"""
        output_str = "[Artifacts loaded]\n"
        for k in self:
            output_str += f"Artifact name: {k}, loaded to path: {str(self.cwd / k)}\n"
        output_str += "[End of artifacts]\n"
        print(output_str)
        return output_str

    def __iter__(self) -> Any:
        return iter(os.listdir(self.cwd))

    def __getitem__(self, name: str) -> Any:
        file_path = self.cwd / name
        if file_path.exists():
            with open(file_path, "r") as file:
                return file.read()
        else:
            raise KeyError(f"File '{name}' not found in artifacts")

    def __setitem__(self, name: str, value: Any) -> None:
        file_path = self.cwd / name
        with open(file_path, "w") as file:
            file.write(value)

    def __contains__(self, name: str) -> bool:
        return name in os.listdir(self.cwd)


def filter_file(file_name: Union[str, Path]) -> Tuple[bool, bool]:
    file_name_p = Path(file_name)
    return (
        file_name_p.is_file()
        and "__pycache__" not in str(file_name_p)
        and not file_name_p.name.startswith(".")
        and file_name_p.suffix
        in [".png", ".jpeg", ".jpg", ".mp4", ".txt", ".json", ".csv"]
    ), file_name_p.suffix in [".png", ".jpeg", ".jpg", ".mp4"]


# These tools are adapted from SWE-Agent https://github.com/princeton-nlp/SWE-agent


def format_lines(lines: List[str], start_idx: int) -> str:
    output = ""
    for i, line in enumerate(lines):
        output += f"{i + start_idx}|{line}"
    return output


def view_lines(
    lines: List[str],
    line_num: int,
    window_size: int,
    name: str,
    total_lines: int,
    print_output: bool = True,
) -> str:
    start = max(0, line_num - window_size)
    end = min(len(lines), line_num + window_size)
    return_str = (
        f"[Artifact: {name} ({total_lines} lines total)]\n"
        + format_lines(lines[start:end], start)
        + (
            "\n[End of artifact]"
            if end == len(lines)
            else f"\n[{len(lines) - end} more lines]"
        )
    )

    if print_output:
        print(return_str)
    return return_str


def check_and_load_image(code: str) -> List[str]:
    if not code.strip():
        return []

    pattern = r"view_media_artifact\(\s*([^\)]+),\s*['\"]([^\)]+)['\"]\s*\)"
    matches = re.findall(pattern, code)
    return [match[1] for match in matches]


def view_media_artifact(artifacts: Artifacts, name: str) -> str:
    """Allows only the agent to view the media artifact with the given name. DO NOT use
    this to show media to the user, the user can already see all media saved in the
    artifacts.

    Parameters:
        artifacts (Artifacts): The artifacts object to show the image from.
        name (str): The name of the image artifact to show.
    """
    if name not in artifacts:
        output_str = f"[Artifact {name} does not exist]"
    else:
        output_str = f"[Image {name} displayed]"
    print(output_str)
    return output_str


def get_tool_descriptions() -> str:
    """Returns a description of all the tools that `generate_vision_code` has access to.
    Helpful for answering questions about what types of vision tasks you can do with
    `generate_vision_code`."""
    return _get_tool_descriptions()


def get_diff(before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True)
        )
    )


def get_diff_with_prompts(name: str, before: str, after: str) -> str:
    diff = get_diff(before, after)
    return f"[Artifact {name} edits]\n{diff}\n[End of edits]"


META_TOOL_DOCSTRING = get_tool_documentation(
    [
        get_tool_descriptions,
        view_media_artifact,
    ]
)
