import pytest


@pytest.fixture
def sample_transcript():
    return "오늘 날씨 어때?"


@pytest.fixture
def sample_llm_response():
    return "밖에 나가보진 않았는데, HASS 데이터 보면 영하 2도에 맑음이네. Bio-fuel Injection 하고 나가는 게 나을 듯."
