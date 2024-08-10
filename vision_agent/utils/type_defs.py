from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from vision_agent.utils.exceptions import InvalidApiKeyError


class LandingaiAPIKey(BaseSettings):
    """The API key of a user in a particular organization in LandingLens.
    It supports loading from environment variables or .env files.
    The supported name of the environment variables are (case-insensitive):
    - LANDINGAI_API_KEY

    Environment variables will always take priority over values loaded from a dotenv file.
    """

    api_key: str = Field(
        default="land_sk_zKvyPcPV2bVoq7q87KwduoerAxuQpx33DnqP8M1BliOCiZOSoI",
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
