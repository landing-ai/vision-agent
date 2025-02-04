import inspect
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


def get_tool_documentation(funcs: List[Callable[..., Any]]) -> str:
    docstrings = ""
    for func in funcs:
        docstrings += f"{func.__name__}{inspect.signature(func)}:\n{func.__doc__}\n\n"

    return docstrings


def get_tool_descriptions(funcs: List[Callable[..., Any]]) -> str:
    descriptions = ""
    for func in funcs:
        description = func.__doc__
        if description is None:
            description = ""

        if "Parameters:" in description:
            description = (
                description[: description.find("Parameters:")]
                .replace("\n", " ")
                .strip()
            )

        description = " ".join(description.split())
        descriptions += f"- {func.__name__}{inspect.signature(func)}: {description}\n"
    return descriptions


def get_tool_descriptions_by_names(
    tool_name: Optional[List[str]],
    funcs: List[Callable[..., Any]],
    util_funcs: List[
        Callable[..., Any]
    ],  # util_funcs will always be added to the list of functions
) -> str:
    if tool_name is None:
        return get_tool_descriptions(funcs + util_funcs)

    invalid_names = [
        name for name in tool_name if name not in {func.__name__ for func in funcs}
    ]

    if invalid_names:
        raise ValueError(f"Invalid customized tool names: {', '.join(invalid_names)}")

    filtered_funcs = (
        funcs
        if not tool_name
        else [func for func in funcs if func.__name__ in tool_name]
    )
    return get_tool_descriptions(filtered_funcs + util_funcs)


def get_tools_df(funcs: List[Callable[..., Any]]) -> pd.DataFrame:
    data: Dict[str, List[str]] = {"desc": [], "doc": [], "name": []}

    for func in funcs:
        desc = func.__doc__
        if desc is None:
            desc = ""
        desc = desc[: desc.find("Parameters:")].replace("\n", " ").strip()
        desc = " ".join(desc.split())

        doc = f"{func.__name__}{inspect.signature(func)}:\n{func.__doc__}"
        data["desc"].append(desc)
        data["doc"].append(doc)
        data["name"].append(func.__name__)

    return pd.DataFrame(data)  # type: ignore


def get_tools_info(funcs: List[Callable[..., Any]]) -> Dict[str, str]:
    data: Dict[str, str] = {}

    for func in funcs:
        desc = func.__doc__
        if desc is None:
            desc = ""

        data[func.__name__] = f"{func.__name__}{inspect.signature(func)}:\n{desc}"

    return data
