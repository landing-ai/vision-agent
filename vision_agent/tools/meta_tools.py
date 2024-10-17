import difflib
import json
import os
import pickle as pkl
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import libcst as cst
from IPython.display import display

import vision_agent as va
from vision_agent.clients.landing_public_api import LandingPublicAPI
from vision_agent.lmm.types import Message
from vision_agent.tools.tool_utils import get_tool_documentation
from vision_agent.tools.tools import TOOL_DESCRIPTIONS
from vision_agent.tools.tools_types import BboxInput, BboxInputBase64, PromptTask
from vision_agent.utils.execute import Execution, MimeType
from vision_agent.utils.image_utils import convert_to_b64

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

    def __init__(
        self, remote_save_path: Union[str, Path], local_save_path: Union[str, Path]
    ) -> None:
        """Initializes the Artifacts object with it's remote and local save paths.

        Parameters:
            remote_save_path (Union[str, Path]): The path to save the artifacts in the
                remote environment. For example "/home/user/artifacts.pkl".
            local_save_path (Union[str, Path]): The path to save the artifacts in the
                local environment. For example "/Users/my_user/workspace/artifacts.pkl".
        """
        self.remote_save_path = Path(remote_save_path)
        self.local_save_path = Path(local_save_path)
        self.artifacts: Dict[str, Any] = {}

        self.code_sandbox_runtime = None

    def load(
        self,
        artifacts_path: Union[str, Path],
        load_to_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Loads are artifacts into the load_to_dir directory. If load_to_dir is None,
        it will load into remote_save_path directory. If an artifact value is None it
        will skip loading it.

        Parameters:
            artifacts_path (Union[str, Path]): The file path to load the artifacts from.
                If you are in the remote environment this would be remote_save_path, if
                you are in the local environment this would be local_save_path.
            load_to_dir (Optional[Union[str, Path]): The directory to load the artifacts
                into. If None, it will load into remote_save_path directory.
        """
        with open(artifacts_path, "rb") as f:
            self.artifacts = pkl.load(f)

        load_to_dir = (
            self.remote_save_path.parent if load_to_dir is None else Path(load_to_dir)
        )

        for k, v in self.artifacts.items():
            if v is not None:
                mode = "w" if isinstance(v, str) else "wb"
                with open(load_to_dir / k, mode) as f:
                    f.write(v)

    def show(self, uploaded_file_dir: Optional[Union[str, Path]] = None) -> str:
        """Prints out the artifacts and the directory they have been loaded to. If you
        pass in upload_file_dir, it will show the artifacts have been loaded to the
        upload_file_dir directory. If you don't pass in upload_file_dir, it will show
        the artifacts have been loaded to the remote_save_path directory.

        Parameters:
            uploaded_file_dir (Optional[Union[str, Path]): The directory the artifacts
                have been loaded to.
        """
        loaded_path = (
            Path(uploaded_file_dir)
            if uploaded_file_dir is not None
            else self.remote_save_path.parent
        )
        output_str = "[Artifacts loaded]\n"
        for k in self.artifacts.keys():
            output_str += (
                f"Artifact name: {k}, loaded to path: {str(loaded_path / k)}\n"
            )
        output_str += "[End of artifacts]\n"
        print(output_str)
        return output_str

    def save(self, local_path: Optional[Union[str, Path]] = None) -> None:
        """Saves the artifacts to the local_save_path directory. If local_path is None,
        it will save to the local_save_path directory.
        """
        save_path = Path(local_path) if local_path is not None else self.local_save_path
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


def filter_file(file_name: Union[str, Path]) -> Tuple[bool, bool]:
    file_name_p = Path(file_name)
    return (
        file_name_p.is_file()
        and "__pycache__" not in str(file_name_p)
        and not file_name_p.name.startswith(".")
        and file_name_p.suffix
        in [".png", ".jpeg", ".jpg", ".mp4", ".txt", ".json", ".csv"]
    ), file_name_p.suffix in [".png", ".jpeg", ".jpg", ".mp4"]


def capture_files_into_artifacts(artifacts: Artifacts) -> None:
    """This function is used to capture all files in the current directory into an
    artifact object. This is useful if you want to capture all files in the current
    directory and use them in a different environment where you don't have access to
    the file system.

    Parameters:
        artifact (Artifacts): The artifact object to save the files to.
    """
    for file in Path(".").glob("**/*"):
        usable_file, is_media = filter_file(file)
        mode = "rb" if is_media else "r"
        if usable_file:
            file_name = file.name
            if file_name.startswith(str(Path(artifacts.remote_save_path).parents)):
                idx = len(Path(artifacts.remote_save_path).parents)
                file_name = file_name[idx:]
            with open(file, mode) as f:
                artifacts[file_name] = f.read()


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
        start (int): The line number to start the edit, can be in [-1, total_lines]
            where -1 represents the end of the file.
        end (int): The line number to end the edit, can be in [-1, total_lines] where
            -1 represents the end of the file.
        content (str): The content to insert.
    """
    # just make the artifact if it doesn't exist instead of forcing agent to call
    # create_artifact
    if name not in artifacts:
        artifacts[name] = ""

    total_lines = len(artifacts[name].splitlines())
    if start == -1:
        start = total_lines
    if end == -1:
        end = total_lines

    if start < 0 or end < 0 or start > end or end > total_lines:
        print("[Invalid line range]")
        return "[Invalid line range]"

    new_content_lines = content.splitlines(keepends=True)
    new_content_lines = [
        line if line.endswith("\n") else line + "\n" for line in new_content_lines
    ]
    lines = artifacts[name].splitlines(keepends=True)
    lines = [line if line.endswith("\n") else line + "\n" for line in lines]
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


def generate_vision_plan(
    artifacts: Artifacts,
    name: str,
    chat: str,
    media: List[str],
    test_multi_plan: bool = True,
    custom_tool_names: Optional[List[str]] = None,
) -> str:
    """Generates a plan to solve vision based tasks.

    Parameters:
        artifacts (Artifacts): The artifacts object to save the plan to.
        name (str): The name of the artifact to save the plan context to.
        chat (str): The chat message from the user.
        media (List[str]): The media files to use.
        test_multi_plan (bool): Do not change this parameter.
        custom_tool_names (Optional[List[str]]): Do not change this parameter.

    Returns:
        str: The generated plan.

    Examples
    --------
        >>> generate_vision_plan(artifacts, "plan.json", "Can you detect the dogs in this image?", ["image.jpg"])
        [Start Plan Context]
        plan1: This is a plan to detect dogs in an image
        -load image
        -detect dogs
        -return detections
        [End Plan Context]
    """

    # verbosity is set to 0 to avoid adding extra content to the VisionAgent conversation
    if ZMQ_PORT is not None:
        agent = va.agent.VisionAgentPlanner(
            report_progress_callback=lambda inp: report_progress_callback(
                int(ZMQ_PORT), inp
            ),
            verbosity=0,
        )
    else:
        agent = va.agent.VisionAgentPlanner(verbosity=0)

    fixed_chat: List[Message] = [{"role": "user", "content": chat, "media": media}]
    response = agent.generate_plan(
        fixed_chat,
        test_multi_plan=test_multi_plan,
        custom_tool_names=custom_tool_names,
    )
    if response.test_results is not None:
        redisplay_results(response.test_results)
    response.test_results = None
    artifacts[name] = response.model_dump_json()

    output_str = f"[Start Plan Context, saved at {name}]"
    for plan in response.plans.keys():
        output_str += f"\n{plan}: {response.plans[plan]['thoughts'].strip()}\n"  # type: ignore
        output_str += "    -" + "\n    -".join(
            e.strip() for e in response.plans[plan]["instructions"]
        )

    output_str += f"\nbest plan: {response.best_plan}\n"
    output_str += "thoughts: " + response.plan_thoughts.strip() + "\n"
    output_str += "[End Plan Context]"
    print(output_str)
    return output_str


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
    # verbosity is set to 0 to avoid adding extra content to the VisionAgent conversation
    if ZMQ_PORT is not None:
        agent = va.agent.VisionAgentCoder(
            report_progress_callback=lambda inp: report_progress_callback(
                int(ZMQ_PORT), inp
            ),
            verbosity=0,
        )
    else:
        agent = va.agent.VisionAgentCoder(verbosity=0)

    fixed_chat: List[Message] = [{"role": "user", "content": chat, "media": media}]
    response = agent.generate_code(
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
    custom_tool_names: Optional[List[str]] = None,
) -> str:
    """Edits python code to solve a vision based task.

    Parameters:
        artifacts (Artifacts): The artifacts object to save the code to.
        name (str): The file path to the code.
        chat_history (List[str]): The chat history to used to generate the code.
        custom_tool_names (Optional[List[str]]): Do not change this parameter.

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

    # verbosity is set to 0 to avoid adding extra content to the VisionAgent conversation
    agent = va.agent.VisionAgentCoder(verbosity=0)
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

    response = agent.generate_code(
        fixed_chat_history,
        test_multi_plan=False,
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
                    "action": "edit",
                }
            )
        },
        raw=True,
    )
    return view_lines(code_lines, 0, total_lines, name, total_lines)


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
    code: Optional[str],
    test_multi_plan: bool = True,
    custom_tool_names: Optional[List[str]] = None,
) -> Optional[str]:
    """This is for forcing arguments passed by the user to VisionAgent into the
    VisionAgentCoder call.

    Parameters:
        code (str): The code to edit.
        test_multi_plan (bool): Do not change this parameter.
        custom_tool_names (Optional[List[str]]): Do not change this parameter.

    Returns:
        str: The edited code.
    """
    if code is None:
        return None

    class VisionAgentTransformer(cst.CSTTransformer):
        def __init__(
            self, test_multi_plan: bool, custom_tool_names: Optional[List[str]]
        ):
            self.test_multi_plan = test_multi_plan
            self.custom_tool_names = custom_tool_names

        def leave_Call(
            self, original_node: cst.Call, updated_node: cst.Call
        ) -> cst.Call:
            # Check if the function being called is generate_vision_code or edit_vision_code
            if isinstance(updated_node.func, cst.Name) and updated_node.func.value in [
                "generate_vision_code",
                "edit_vision_code",
            ]:
                # Add test_multi_plan argument to generate_vision_code calls
                if updated_node.func.value == "generate_vision_code":
                    new_arg = cst.Arg(
                        keyword=cst.Name("test_multi_plan"),
                        value=cst.Name(str(self.test_multi_plan)),
                        equal=cst.AssignEqual(
                            whitespace_before=cst.SimpleWhitespace(""),
                            whitespace_after=cst.SimpleWhitespace(""),
                        ),
                    )
                    updated_node = updated_node.with_changes(
                        args=[*updated_node.args, new_arg]
                    )

                # Add custom_tool_names if provided
                if self.custom_tool_names is not None:
                    list_arg = []
                    for i, tool_name in enumerate(self.custom_tool_names):
                        if i < len(self.custom_tool_names) - 1:
                            list_arg.append(
                                cst._nodes.expression.Element(
                                    value=cst.SimpleString(value=f'"{tool_name}"'),
                                    comma=cst.Comma(
                                        whitespace_before=cst.SimpleWhitespace(""),
                                        whitespace_after=cst.SimpleWhitespace(" "),
                                    ),
                                )
                            )
                        else:
                            list_arg.append(
                                cst._nodes.expression.Element(
                                    value=cst.SimpleString(value=f'"{tool_name}"'),
                                )
                            )
                    new_arg = cst.Arg(
                        keyword=cst.Name("custom_tool_names"),
                        value=cst.List(list_arg),
                        equal=cst.AssignEqual(
                            whitespace_before=cst.SimpleWhitespace(""),
                            whitespace_after=cst.SimpleWhitespace(""),
                        ),
                    )
                    updated_node = updated_node.with_changes(
                        args=[*updated_node.args, new_arg]
                    )

            return updated_node

    # Parse the input code into a CST node
    tree = cst.parse_module(code)

    # Apply the transformer to modify the CST
    transformer = VisionAgentTransformer(test_multi_plan, custom_tool_names)
    modified_tree = tree.visit(transformer)

    # Return the modified code as a string
    return modified_tree.code


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
        view_media_artifact,
        object_detection_fine_tuning,
        use_object_detection_fine_tuning,
        list_artifacts,
    ]
)
