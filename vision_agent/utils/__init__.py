import os
from .execute import (
    CodeInterpreter,
    CodeInterpreterFactory,
    Error,
    Execution,
    Logs,
    Result,
)
from .sim import AzureSim, OllamaSim, Sim, load_sim, merge_sim


def should_report_tool_traces() -> bool:
    return bool(os.environ.get("REPORT_TOOL_TRACES", False))
