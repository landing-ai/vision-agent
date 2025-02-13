import inspect
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


def get_tool_documentation(funcs: List[Callable[..., Any]]) -> str:
    """Generate a formatted string of documentation for a list of functions.

    Args:
        funcs (List[Callable[..., Any]]): A list of functions for which to retrieve and format documentation.

    Returns:
        str: A formatted string containing the name, signature, and docstring of each function in the input list."""
    docstrings = ""
    for func in funcs:
        docstrings += f"{func.__name__}{inspect.signature(func)}:\n{func.__doc__}\n\n"

    return docstrings


def get_tool_descriptions(funcs: List[Callable[..., Any]]) -> str:
    """Generates a summary of tool descriptions from a list of functions.

    This function extracts and formats the docstring summaries of given
    functions, omitting any parameter details, and returns a consolidated
    string describing each function with its name and signature.

    Args:
        funcs (List[Callable[..., Any]]): A list of functions to extract
            descriptions from.

    Returns:
        str: A formatted string containing the name, signature, and a brief
        description of each function. The description is derived from the
        function's docstring, truncated before any 'Parameters:' section.
        Each function's entry is prefixed with a dash and followed by a newline."""
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
    """Generates a description of tools based on their names.

    This function returns a string containing descriptions of specified tools.
    If no specific tool names are provided, it returns descriptions for all
    available tools combined with utility functions.

    Args:
        tool_name (Optional[List[str]]): A list of tool names to filter by.
            If None, all tools will be included.
        funcs (List[Callable[..., Any]]): A list of callable functions representing tools.
        util_funcs (List[Callable[..., Any]]): A list of utility functions that are always included.

    Returns:
        str: A string with the descriptions of the selected tools and utility functions.

    Raises:
        ValueError: If any of the provided tool names do not match the available tools."""
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
    """Generate a DataFrame summarizing documentation for a list of functions.

    This function takes a list of callable objects and creates a Pandas DataFrame
    containing a brief description, full documentation, and the name of each function.

    Args:
        funcs: A list of callable objects (functions) to be documented.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - 'desc': A short description of each function, extracted from the docstring
              before the "Parameters:" section.
            - 'doc': The full signature and docstring of each function.
            - 'name': The name of each function."""
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
