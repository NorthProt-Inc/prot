import pytest
from prot.processing import chunk_sentences, MAX_BUFFER_CHARS


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

    def test_force_flush_on_oversized_remainder(self):
        """chunk_sentences should force-flush remainder exceeding MAX_BUFFER_CHARS."""
        long_text = "가" * (MAX_BUFFER_CHARS + 100)
        sentences, remainder = chunk_sentences(long_text)
        assert len(remainder) <= MAX_BUFFER_CHARS
        assert len(sentences) >= 1

    def test_normal_remainder_not_flushed(self):
        """Remainder under MAX_BUFFER_CHARS should be preserved."""
        text = "완성. 미완성"
        sentences, remainder = chunk_sentences(text)
        assert remainder == "미완성"
        assert sentences == ["완성."]
