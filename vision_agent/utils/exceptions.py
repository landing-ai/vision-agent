"""Vision Agent exceptions."""


class InvalidApiKeyError(Exception):
    """Exception raised when the an invalid API key is provided. This error could be raised from any SDK code, not limited to a HTTP client."""

    def __init__(self, message: str):
        self.message = f"""{message}
For more information, see https://landing-ai.github.io/landingai-python/landingai.html#manage-api-credentials"""
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class RemoteToolCallFailed(Exception):
    """Exception raised when an error occurs during a tool call."""

    def __init__(self, tool_name: str, status_code: int, message: str):
        self.message = (
            f"""Tool call ({tool_name}) failed due to {status_code} - {message}"""
        )


class RemoteSandboxError(Exception):
    """Exception related to remote sandbox."""

    is_retryable = False


class RemoteSandboxCreationError(RemoteSandboxError):
    """Exception raised when failed to create a remote sandbox.
    This could be due to the remote sandbox service is unavailable.
    """

    is_retryable = False


class RemoteSandboxExecutionError(RemoteSandboxError):
    """Exception raised when failed in a remote sandbox code execution."""

    is_retryable = False


class RemoteSandboxClosedError(RemoteSandboxError):
    """Exception raised when a remote sandbox is dead.
    This is retryable in the sense that the user can try again with a new sandbox. Can't be retried in the same sandbox.
    When this error is raised, the user should retry by create a new VisionAgent (i.e. a new sandbox).
    """

    is_retryable = True


class FineTuneModelIsNotReady(Exception):
    """Exception raised when the fine-tune model is not ready.
    If this is raised, it's recommended to wait 5 seconds before trying to use
    the model again.
    """


class FineTuneModelNotFound(Exception):
    """Exception raised when the fine-tune model is not found.
    If this is raised, it's recommended to try another model id.
    """
