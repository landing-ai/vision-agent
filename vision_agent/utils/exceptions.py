"""Vision Agent exceptions."""


class InvalidApiKeyError(Exception):
    """Exception raised when the an invalid API key is provided. This error could be raised from any SDK code, not limited to a HTTP client."""

    def __init__(self, message: str):
        self.message = f"""{message}
For more information, see https://landing-ai.github.io/landingai-python/landingai.html#manage-api-credentials"""
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


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
