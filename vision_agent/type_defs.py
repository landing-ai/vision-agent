from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class LandingaiAPIKey(BaseSettings):
    """The API key of a user in a particular organization in LandingLens.
    It supports loading from environment variables or .env files.
    The supported name of the environment variables are (case-insensitive):
    - LANDINGAI_API_KEY

    Environment variables will always take priority over values loaded from a dotenv file.
    """

    api_key: str = Field(
        default="land_sk_hw34v3tyEc35OAhP8F7hnGnrDv2C8hD2ycMyq0aMkVS1H40D22",
        alias="LANDINGAI_API_KEY",
        description="The API key of LandingAI.",
    )

    @field_validator("api_key")
    @classmethod
    def is_api_key_valid(cls, key: str) -> str:
        """Check if the API key is a v2 key."""
        if not key:
            raise InvalidApiKeyError(f"LandingAI API key is required, but it's {key}")
        if not key.startswith("land_sk_"):
            raise InvalidApiKeyError(
                f"LandingAI API key (v2) must start with 'land_sk_' prefix, but it's {key}. See https://support.landing.ai/docs/api-key for more information."
            )
        return key

    class Config:
        env_file = ".env"
        env_prefix = "landingai_"
        case_sensitive = False
        extra = "ignore"


class InvalidApiKeyError(Exception):
    """Exception raised when the an invalid API key is provided. This error could be raised from any SDK code, not limited to a HTTP client."""

    def __init__(self, message: str):
        self.message = f"""{message}
For more information, see https://landing-ai.github.io/landingai-python/landingai.html#manage-api-credentials"""
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
