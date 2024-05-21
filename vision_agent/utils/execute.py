"""This code is adapted from MetaGPT's https://github.com/geekan/MetaGPT/blob/main/metagpt/actions/di/execute_nb_code.py
"""

import base64 as b64
import io
import re
from time import sleep
from typing import Dict, List, Tuple

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellTimeoutError, DeadKernelError
from nbclient.util import run_sync
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell
from PIL import Image


def remove_escape_and_color_codes(input_str: str) -> str:
    pattern = re.compile(r"\x1b\[[0-9;]*[mK]")
    result = pattern.sub("", input_str)
    return result


def parse_outputs(outputs: List[Dict]) -> Tuple[bool, str]:
    success, parsed_output = True, []
    for output in outputs:
        # TODO: add parse image data
        if output["output_type"] == "stream":
            parsed_output.append(output["text"])
        elif output["output_type"] == "text/plain":
            parsed_output.append(output["data"]["text/plain"])
        elif output["output_type"] == "display_data":
            if "image/png" in output["data"]:
                image_bytes = b64.b64decode(output["data"]["image/png"])
                Image.open(io.BytesIO(image_bytes)).show()
        elif output["output_type"] == "error":
            success = False
            output_text = remove_escape_and_color_codes("\n".join(output["traceback"]))
            parsed_output.append(output_text)

    return success, ",".join(parsed_output)


class Execute:
    def __init__(self, timeout: int = 600) -> None:
        self.nb = nbformat.v4.new_notebook()
        self.timeout = timeout
        self.nb_client = NotebookClient(self.nb, timeout=self.timeout)

    def build(self) -> None:
        if self.nb_client.kc is None or not run_sync(self.nb_client.kc.is_alive)():  # type: ignore
            self.nb_client.create_kernel_manager()
            self.nb_client.start_new_kernel()
            self.nb_client.start_new_kernel_client()

    def terminate(self) -> None:
        if self.nb_client.km is not None and run_sync(self.nb_client.km.is_alive)():  # type: ignore
            run_sync(self.nb_client.km.shutdown_kernel)(now=True)
            run_sync(self.nb_client.km.cleanup_resources)()

            channels = [
                self.nb_client.kc.stdin_channel,
                self.nb_client.kc.hb_channel,
                self.nb_client.kc.control_channel,
            ]

            for ch in channels:
                if ch.is_alive():
                    ch.stop()

            self.nb_client.kc = None
            self.nb_client.km = None

    def reset(self) -> None:
        self.terminate()
        self.nb = nbformat.v4.new_notebook()
        self.nb_client = NotebookClient(self.nb, timeout=self.timeout)
        sleep(1)
        self.build()

    def run_cell(self, cell: NotebookNode, cell_index: int) -> Tuple[bool, str]:
        try:
            self.nb_client.execute_cell(cell, cell_index)
            return parse_outputs(self.nb.cells[-1].outputs)
        except CellTimeoutError:
            run_sync(self.nb_client.km.interrupt_kernel)()  # type: ignore
            sleep(1)
            return False, "Cell execution timed out."
        except DeadKernelError:
            self.reset()
            return False, "DeadKernelError"
        except Exception:
            return parse_outputs(self.nb.cells[-1].outputs)

    def add_code_cell(self, code: str) -> None:
        self.nb.cells.append(new_code_cell(code))

    def run_additional(self, code: str) -> Tuple[bool, str]:
        self.build()
        self.add_code_cell(code)
        return self.run_cell(self.nb.cells[-1], len(self.nb.cells) - 1)

    def run_isolation(self, code: str) -> Tuple[bool, str]:
        self.reset()
        self.add_code_cell(code)
        return self.run_cell(self.nb.cells[-1], len(self.nb.cells) - 1)
