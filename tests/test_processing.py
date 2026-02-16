import pytest
from prot.processing import sanitize_for_tts, ensure_complete_sentence, chunk_sentences


class TestSanitizeForTts:
    def test_strips_markdown_bold(self):
        assert sanitize_for_tts("이건 **중요한** 내용이야") == "이건 중요한 내용이야"

    def test_strips_markdown_headers(self):
        assert sanitize_for_tts("# 제목\n내용") == "제목\n내용"

    def test_strips_numbered_list(self):
        assert sanitize_for_tts("1. 첫째\n2. 둘째") == "첫째\n둘째"

    def test_strips_bullets(self):
        assert sanitize_for_tts("- 항목\n• 항목") == "항목\n항목"

    def test_strips_code_backticks(self):
        assert sanitize_for_tts("이건 `코드` 임") == "이건 코드 임"

    def test_passthrough_clean_text(self):
        text = "오늘 날씨 좋다."
        assert sanitize_for_tts(text) == text


class TestEnsureCompleteSentence:
    def test_truncates_at_last_period(self):
        assert ensure_complete_sentence("안녕하세요. 오늘은") == "안녕하세요."

    def test_truncates_at_question_mark(self):
        assert ensure_complete_sentence("뭐해? 나는") == "뭐해?"

    def test_truncates_at_exclamation(self):
        assert ensure_complete_sentence("좋아! 그러면") == "좋아!"

    def test_returns_full_if_complete(self):
        assert ensure_complete_sentence("완전한 문장이야.") == "완전한 문장이야."

    def test_returns_text_if_no_punctuation(self):
        assert ensure_complete_sentence("문장부호없음") == "문장부호없음"


class TestChunkSentences:
    def test_splits_on_period(self):
        sentences, remainder = chunk_sentences("첫 문장. 두번째 문장.")
        assert sentences == ["첫 문장.", "두번째 문장."]
        assert remainder == ""

    def test_splits_on_question_mark(self):
        sentences, remainder = chunk_sentences("뭐해? 나는 잘 지내.")
        assert sentences == ["뭐해?", "나는 잘 지내."]
        assert remainder == ""

    def test_handles_empty_string(self):
        sentences, remainder = chunk_sentences("")
        assert sentences == []
        assert remainder == ""

    def test_preserves_single_sentence(self):
        sentences, remainder = chunk_sentences("하나의 문장만.")
        assert sentences == ["하나의 문장만."]
        assert remainder == ""

    def test_retains_incomplete_trailing_text(self):
        sentences, remainder = chunk_sentences("완성된 문장. 미완성 텍스트")
        assert sentences == ["완성된 문장."]
        assert remainder == "미완성 텍스트"

    def test_all_incomplete(self):
        sentences, remainder = chunk_sentences("문장 종결 없는 텍스트")
        assert sentences == []
        assert remainder == "문장 종결 없는 텍스트"

    def test_multiple_with_trailing(self):
        sentences, remainder = chunk_sentences("첫째. 둘째! 셋째는 아직")
        assert sentences == ["첫째.", "둘째!"]
        assert remainder == "셋째는 아직"
