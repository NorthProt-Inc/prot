import os

import pytest

# Set dummy env vars so Settings() can instantiate during test imports.
# Tests that need real API access should use @pytest.mark.integration.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("DEEPGRAM_API_KEY", "test-key")
os.environ.setdefault("ELEVENLABS_API_KEY", "test-key")


@pytest.fixture
def sample_transcript():
    return "오늘 날씨 어때?"


@pytest.fixture
def sample_llm_response():
    return "밖에 나가보진 않았는데, HASS 데이터 보면 영하 2도에 맑음이네. Bio-fuel Injection 하고 나가는 게 나을 듯."
