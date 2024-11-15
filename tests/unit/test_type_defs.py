import os
from unittest import mock

from .fixtures import mock_openai_embedding

# mock the embedding model for Sim class
with mock.patch("vision_agent.utils.sim.OpenAI", mock_openai_embedding()):
    from vision_agent.utils.type_defs import LandingaiAPIKey


def test_load_api_credential_from_env_var():
    actual = LandingaiAPIKey()
    assert actual.api_key is not None

    os.environ["landingai_api_key"] = "land_sk_123"
    actual = LandingaiAPIKey()
    assert actual.api_key == "land_sk_123"
    del os.environ["landingai_api_key"]
