import difflib
import json
import os
import pickle as pkl
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from IPython.display import display

import vision_agent as va
from vision_agent.clients.landing_public_api import LandingPublicAPI
from vision_agent.lmm.types import Message
from vision_agent.tools.tool_utils import get_tool_documentation
from vision_agent.tools.tools import TOOL_DESCRIPTIONS
from vision_agent.tools.tools_types import BboxInput, BboxInputBase64, PromptTask
from vision_agent.utils.execute import Execution, MimeType
from vision_agent.utils.image_utils import convert_to_b64, numpy_to_bytes
from vision_agent.utils.video import frames_to_bytes

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

    def __init__(self, remote_save_path: Union[str, Path]) -> None:
        self.remote_save_path = Path(remote_save_path)
        self.artifacts: Dict[str, Any] = {}

        self.code_sandbox_runtime = None

    def load(self, file_path: Union[str, Path]) -> None:
        """Loads are artifacts into the remote environment. If an artifact value is None
        it will skip loading it.

        Parameters:
            file_path (Union[str, Path]): The file path to load the artifacts from
        """
        with open(file_path, "rb") as f:
            self.artifacts = pkl.load(f)
        for k, v in self.artifacts.items():
            if v is not None:
                mode = "w" if isinstance(v, str) else "wb"
                with open(self.remote_save_path.parent / k, mode) as f:
                    f.write(v)

    def show(self, uploaded_file_path: Optional[Union[str, Path]] = None) -> str:
        """Shows the artifacts that have been loaded and their remote save paths."""
        loaded_path = (
            Path(uploaded_file_path)
            if uploaded_file_path is not None
            else self.remote_save_path
        )
        output_str = "[Artifacts loaded]\n"
        for k in self.artifacts.keys():
            output_str += f"Artifact {k} loaded to {str(loaded_path / k)}\n"
        output_str += "[End of artifacts]\n"
        print(output_str)
        return output_str

    def save(self, local_path: Optional[Union[str, Path]] = None) -> None:
        save_path = (
            Path(local_path) if local_path is not None else self.remote_save_path
        )
        with open(save_path, "wb") as f:
            pkl.dump(self.artifacts, f)

    def __iter__(self) -> Any:
        return iter(self.artifacts)

    def __getitem__(self, name: str) -> Any:
        return self.artifacts[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self.artifacts[name] = value

    def __contains__(self, name: str) -> bool:
        return name in self.artifacts


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
            "[End of artifact]"
            if end == len(lines)
            else f"[{len(lines) - end} more lines]"
        )
    )

    if print_output:
        print(return_str)
    return return_str


def open_code_artifact(
    artifacts: Artifacts, name: str, line_num: int = 0, window_size: int = 100
) -> str:
    """Opens the provided code artifact. If `line_num` is provided, the window will be
    moved to include that line. It only shows the first 100 lines by default! Max
    `window_size` supported is 2000.

    Parameters:
        artifacts (Artifacts): The artifacts object to open the artifact from.
        name (str): The name of the artifact to open.
        line_num (int): The line number to move the window to.
        window_size (int): The number of lines to show above and below the line.
    """
    if name not in artifacts:
        return f"[Artifact {name} does not exist]"

    total_lines = len(artifacts[name].splitlines())
    window_size = min(window_size, 2000)
    window_size = window_size // 2
    if line_num - window_size < 0:
        line_num = window_size
    elif line_num >= total_lines:
        line_num = total_lines - 1 - window_size

    lines = artifacts[name].splitlines(keepends=True)

    return view_lines(lines, line_num, window_size, name, total_lines)


def create_code_artifact(artifacts: Artifacts, name: str) -> str:
    """Creates a new code artifiact with the given name.

    Parameters:
        artifacts (Artifacts): The artifacts object to add the new artifact to.
        name (str): The name of the new artifact.
    """
    if name in artifacts:
        return_str = f"[Artifact {name} already exists]"
    else:
        artifacts[name] = ""
        return_str = f"[Artifact {name} created]"
    print(return_str)

    display(
        {
            MimeType.APPLICATION_ARTIFACT: json.dumps(
                {
                    "name": name,
                    "content": artifacts[name],
                    "action": "create",
                }
            )
        },
        raw=True,
    )
    return return_str


def edit_code_artifact(
    artifacts: Artifacts, name: str, start: int, end: int, content: str
) -> str:
    """Edits the given code artifact with the provided content. The content will be
    inserted between the `start` and `end` line numbers. If the `start` and `end` are
    the same, the content will be inserted at the `start` line number. If the `end` is
    greater than the total number of lines in the file, the content will be inserted at
    the end of the file. If the `start` or `end` are negative, the function will return
    an error message.

    Parameters:
        artifacts (Artifacts): The artifacts object to edit the artifact from.
        name (str): The name of the artifact to edit.
        start (int): The line number to start the edit.
        end (int): The line number to end the edit.
        content (str): The content to insert.
    """
    # just make the artifact if it doesn't exist instead of forcing agent to call
    # create_artifact
    if name not in artifacts:
        artifacts[name] = ""

    total_lines = len(artifacts[name].splitlines())
    if start < 0 or end < 0 or start > end or end > total_lines:
        print("[Invalid line range]")
        return "[Invalid line range]"
    if start == end:
        end += 1

    new_content_lines = content.splitlines(keepends=True)
    new_content_lines = [
        line if line.endswith("\n") else line + "\n" for line in new_content_lines
    ]
    lines = artifacts[name].splitlines(keepends=True)
    edited_lines = lines[:start] + new_content_lines + lines[end:]

    cur_line = start + len(content.split("\n")) // 2
    with tempfile.NamedTemporaryFile(delete=True) as f:
        with open(f.name, "w") as f:  # type: ignore
            f.writelines(edited_lines)

        process = subprocess.Popen(
            [
                "flake8",
                "--isolated",
                "--select=F821,F822,F831,E111,E112,E113,E999,E902",
                f.name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, _ = process.communicate()

        if stdout != "":
            stdout = stdout.replace(f.name, name)
            error_msg = "[Edit failed with the following status]\n" + stdout
            original_view = view_lines(
                lines,
                start + ((end - start) // 2),
                DEFAULT_WINDOW_SIZE,
                name,
                total_lines,
                print_output=False,
            )
            total_lines_edit = sum(1 for _ in edited_lines)
            edited_view = view_lines(
                edited_lines,
                cur_line,
                DEFAULT_WINDOW_SIZE,
                name,
                total_lines_edit,
                print_output=False,
            )

            error_msg += f"\n[This is how your edit would have looked like if applied]\n{edited_view}\n\n[This is the original code before your edit]\n{original_view}"
            print(error_msg)
            return error_msg

    artifacts[name] = "".join(edited_lines)

    display(
        {
            MimeType.APPLICATION_ARTIFACT: json.dumps(
                {
                    "name": name,
                    "content": artifacts[name],
                    "action": "edit",
                }
            )
        },
        raw=True,
    )
    return open_code_artifact(artifacts, name, cur_line)


def generate_vision_code(
    artifacts: Artifacts,
    name: str,
    chat: str,
    media: List[str],
    test_multi_plan: bool = True,
    custom_tool_names: Optional[List[str]] = None,
) -> str:
    """Generates python code to solve vision based tasks.

    Parameters:
        artifacts (Artifacts): The artifacts object to save the code to.
        name (str): The name of the artifact to save the code to.
        chat (str): The chat message from the user.
        media (List[str]): The media files to use.
        test_multi_plan (bool): Do not change this parameter.
        custom_tool_names (Optional[List[str]]): Do not change this parameter.

    Returns:
        str: The generated code.

    Examples
    --------
        >>> generate_vision_code(artifacts, "code.py", "Can you detect the dogs in this image?", ["image.jpg"])
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

    fixed_chat: List[Message] = [{"role": "user", "content": chat, "media": media}]
    response = agent.chat_with_workflow(
        fixed_chat,
        test_multi_plan=test_multi_plan,
        custom_tool_names=custom_tool_names,
    )
    redisplay_results(response["test_result"])
    code = response["code"]
    artifacts[name] = code
    code_lines = code.splitlines(keepends=True)
    total_lines = len(code_lines)

    display(
        {
            MimeType.APPLICATION_ARTIFACT: json.dumps(
                {
                    "name": name,
                    "content": code,
                    "contentType": "vision_code",
                    "action": "create",
                }
            )
        },
        raw=True,
    )
    return view_lines(code_lines, 0, total_lines, name, total_lines)


def edit_vision_code(
    artifacts: Artifacts,
    name: str,
    chat_history: List[str],
    media: List[str],
    customized_tool_names: Optional[List[str]] = None,
) -> str:
    """Edits python code to solve a vision based task.

    Parameters:
        artifacts (Artifacts): The artifacts object to save the code to.
        name (str): The file path to the code.
        chat_history (List[str]): The chat history to used to generate the code.
        customized_tool_names (Optional[List[str]]): Do not change this parameter.

    Returns:
        str: The edited code.

    Examples
    --------
        >>> edit_vision_code(
        >>>     artifacts,
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
    if name not in artifacts:
        print(f"[Artifact {name} does not exist]")
        return f"[Artifact {name} does not exist]"

    code = artifacts[name]

    # Append latest code to second to last message from assistant
    fixed_chat_history: List[Message] = []
    user_message = "Previous user requests:"
    for i, chat in enumerate(chat_history):
        if i < len(chat_history) - 1:
            user_message += " " + chat
        else:
            fixed_chat_history.append(
                {"role": "user", "content": user_message, "media": media}
            )
            fixed_chat_history.append({"role": "assistant", "content": code})
            fixed_chat_history.append({"role": "user", "content": chat})

    response = agent.chat_with_workflow(
        fixed_chat_history,
        test_multi_plan=False,
        custom_tool_names=customized_tool_names,
    )
    redisplay_results(response["test_result"])
    code = response["code"]
    artifacts[name] = code
    code_lines = code.splitlines(keepends=True)
    total_lines = len(code_lines)

    display(
        {
            MimeType.APPLICATION_ARTIFACT: json.dumps(
                {
                    "name": name,
                    "content": code,
                    "action": "edit",
                }
            )
        },
        raw=True,
    )
    return view_lines(code_lines, 0, total_lines, name, total_lines)


def write_media_artifact(
    artifacts: Artifacts,
    name: str,
    media: Union[str, np.ndarray, List[np.ndarray]],
    fps: Optional[float] = None,
) -> str:
    """Writes a media file to the artifacts object.

    Parameters:
        artifacts (Artifacts): The artifacts object to save the media to.
        name (str): The name of the media artifact to save.
        media (Union[str, np.ndarray, List[np.ndarray]]): The media to save, can either
            be a file path, single image or list of frames for a video.
        fps (Optional[float]): The frames per second if you are writing a video.
    """
    if isinstance(media, str):
        with open(media, "rb") as f:
            media_bytes = f.read()
    elif isinstance(media, list):
        media_bytes = frames_to_bytes(media, fps=fps if fps is not None else 1.0)
    elif isinstance(media, np.ndarray):
        media_bytes = numpy_to_bytes(media)
    else:
        print(f"[Invalid media type {type(media)}]")
        return f"[Invalid media type {type(media)}]"
    artifacts[name] = media_bytes
    print(f"[Media {name} saved]")
    return f"[Media {name} saved]"


def list_artifacts(artifacts: Artifacts) -> str:
    """Lists all the artifacts that have been loaded into the artifacts object."""
    output_str = artifacts.show()
    print(output_str)
    return output_str


def check_and_load_image(code: str) -> List[str]:
    if not code.strip():
        return []

    pattern = r"view_media_artifact\(\s*([^\)]+),\s*['\"]([^\)]+)['\"]\s*\)"
    matches = re.findall(pattern, code)
    return [match[1] for match in matches]


def view_media_artifact(artifacts: Artifacts, name: str) -> str:
    """Allows you to view the media artifact with the given name. This does not show
    the media to the user, the user can already see all media saved in the artifacts.

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
    return TOOL_DESCRIPTIONS


def object_detection_fine_tuning(bboxes: List[Dict[str, Any]]) -> str:
    """DO NOT use this function unless the user has supplied you with bboxes.
    'object_detection_fine_tuning' is a tool that fine-tunes object detection models to
    be able to detect objects in an image based on a given dataset. It returns the fine
    tuning job id.

    Parameters:
        bboxes (List[BboxInput]): A list of BboxInput containing the image path, labels
            and bounding boxes. The coordinates are unnormalized.

    Returns:
        str: The fine tuning job id, this id will used to retrieve the fine tuned
            model.

    Example
    -------
        >>> fine_tuning_job_id = object_detection_fine_tuning(
            [{'image_path': 'filename.png', 'labels': ['screw'], 'bboxes': [[370, 30, 560, 290]]},
             {'image_path': 'filename.png', 'labels': ['screw'], 'bboxes': [[120, 0, 300, 170]]}],
             "phrase_grounding"
        )
    """
    task = "phrase_grounding"
    bboxes_input = [BboxInput.model_validate(bbox) for bbox in bboxes]
    task_type = PromptTask[task.upper()]
    fine_tuning_request = [
        BboxInputBase64(
            image=convert_to_b64(bbox_input.image_path),
            filename=Path(bbox_input.image_path).name,
            labels=bbox_input.labels,
            bboxes=bbox_input.bboxes,
        )
        for bbox_input in bboxes_input
    ]
    landing_api = LandingPublicAPI()
    fine_tune_id = str(
        landing_api.launch_fine_tuning_job("florencev2", task_type, fine_tuning_request)
    )
    print(f"[Fine tuning id: {fine_tune_id}]")
    return fine_tune_id


def get_diff(before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True)
        )
    )


def get_diff_with_prompts(name: str, before: str, after: str) -> str:
    diff = get_diff(before, after)
    return f"[Artifact {name} edits]\n{diff}\n[End of edits]"


def use_extra_vision_agent_args(
    code: str,
    test_multi_plan: bool = True,
    customized_tool_names: Optional[List[str]] = None,
) -> str:
    """This is for forcing arguments passed by the user to VisionAgent into the
    VisionAgentCoder call.

    Parameters:
        code (str): The code to edit.
        test_multi_plan (bool): Do not change this parameter.
        customized_tool_names (Optional[List[str]]): Do not change this parameter.

    Returns:
        str: The edited code.
    """
    generate_pattern = r"generate_vision_code\(\s*([^\)]+)\s*\)"

    def generate_replacer(match: re.Match) -> str:
        arg = match.group(1)
        out_str = f"generate_vision_code({arg}, test_multi_plan={test_multi_plan}"
        if customized_tool_names is not None:
            out_str += f", custom_tool_names={customized_tool_names})"
        else:
            out_str += ")"
        return out_str

    edit_pattern = r"edit_vision_code\(\s*([^\)]+)\s*\)"

    def edit_replacer(match: re.Match) -> str:
        arg = match.group(1)
        out_str = f"edit_vision_code({arg}"
        if customized_tool_names is not None:
            out_str += f", custom_tool_names={customized_tool_names})"
        else:
            out_str += ")"
        return out_str

    new_code = re.sub(generate_pattern, generate_replacer, code)
    new_code = re.sub(edit_pattern, edit_replacer, new_code)
    return new_code


def use_object_detection_fine_tuning(
    artifacts: Artifacts, name: str, fine_tune_id: str
) -> str:
    """Replaces calls to 'owl_v2_image', 'florence2_phrase_detection' and
    'florence2_sam2_image' with the fine tuning id. This ensures that the code utilizes
    the fined tuned florence2 model. Returns the diff between the original code and the
    new code.

    Parameters:
        artifacts (Artifacts): The artifacts object to edit the code from.
        name (str): The name of the artifact to edit.
        fine_tune_id (str): The fine tuning job id.

    Examples
    --------
        >>> diff = use_object_detection_fine_tuning(artifacts, "code.py", "23b3b022-5ebf-4798-9373-20ef36429abf")
    """

    if name not in artifacts:
        output_str = f"[Artifact {name} does not exist]"
        print(output_str)
        return output_str

    code = artifacts[name]

    patterns_with_fine_tune_id = [
        (
            r'florence2_phrase_grounding\(\s*["\']([^"\']+)["\']\s*,\s*([^,]+)(?:,\s*["\'][^"\']+["\'])?\s*\)',
            lambda match: f'florence2_phrase_grounding("{match.group(1)}", {match.group(2)}, "{fine_tune_id}")',
        ),
        (
            r'florence2_phrase_grounding_video\(\s*["\']([^"\']+)["\']\s*,\s*([^,]+)(?:,\s*["\'][^"\']+["\'])?\s*\)',
            lambda match: f'florence2_phrase_grounding_video("{match.group(1)}", {match.group(2)}, "{fine_tune_id}")',
        ),
        (
            r'owl_v2_image\(\s*["\']([^"\']+)["\']\s*,\s*([^,]+)(?:,\s*["\'][^"\']+["\'])?\s*\)',
            lambda match: f'owl_v2_image("{match.group(1)}", {match.group(2)}, "{fine_tune_id}")',
        ),
        (
            r'florence2_sam2_image\(\s*["\']([^"\']+)["\']\s*,\s*([^,]+)(?:,\s*["\'][^"\']+["\'])?\s*\)',
            lambda match: f'florence2_sam2_image("{match.group(1)}", {match.group(2)}, "{fine_tune_id}")',
        ),
    ]

    new_code = code
    for (
        pattern_with_fine_tune_id,
        replacer_with_fine_tune_id,
    ) in patterns_with_fine_tune_id:
        if re.search(pattern_with_fine_tune_id, new_code):
            new_code = re.sub(
                pattern_with_fine_tune_id, replacer_with_fine_tune_id, new_code
            )

    if new_code == code:
        output_str = (
            f"[No function calls to replace with fine tuning id in artifact {name}]"
        )
        print(output_str)
        return output_str

    artifacts[name] = new_code

    diff = get_diff_with_prompts(name, code, new_code)
    print(diff)

    display(
        {
            MimeType.APPLICATION_ARTIFACT: json.dumps(
                {"name": name, "content": new_code, "action": "edit"}
            )
        },
        raw=True,
    )
    return diff


META_TOOL_DOCSTRING = get_tool_documentation(
    [
        get_tool_descriptions,
        open_code_artifact,
        create_code_artifact,
        edit_code_artifact,
        generate_vision_code,
        edit_vision_code,
        write_media_artifact,
        view_media_artifact,
        object_detection_fine_tuning,
        use_object_detection_fine_tuning,
        list_artifacts,
    ]
)
