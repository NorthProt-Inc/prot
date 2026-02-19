import pytest
from prot.processing import chunk_sentences, is_tool_result_message, sanitize_for_tts, MAX_BUFFER_CHARS


class TestSanitizeForTts:
    def test_injects_space_before_bracket(self):
        assert sanitize_for_tts("비효율이지[sarcastic].") == "비효율이지 [sarcastic]."

    def test_injects_space_after_bracket(self):
        assert sanitize_for_tts("[sarcastic]다음") == "[sarcastic] 다음"

    def test_no_space_between_bracket_and_punctuation(self):
        assert sanitize_for_tts("[sarcastic].") == "[sarcastic]."

    def test_already_spaced(self):
        assert sanitize_for_tts("비효율이지 [sarcastic].") == "비효율이지 [sarcastic]."

    def test_multiple_tags(self):
        assert sanitize_for_tts("하나[sighs]. 둘[laughs]!") == "하나 [sighs]. 둘 [laughs]!"

    def test_plain_text_unchanged(self):
        assert sanitize_for_tts("일반 한국어 텍스트.") == "일반 한국어 텍스트."

    def test_empty_string(self):
        assert sanitize_for_tts("") == ""

    def test_tag_at_start(self):
        assert sanitize_for_tts("[sighs] 또 그 얘기야.") == "[sighs] 또 그 얘기야."

    def test_tag_mid_sentence_both_sides(self):
        assert sanitize_for_tts("이건[whispers]비밀이야.") == "이건 [whispers] 비밀이야."


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


class TestIsToolResultMessage:
    def test_tool_result_only(self):
        msg = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
        ]}
        assert is_tool_result_message(msg) is True

    def test_multiple_tool_results(self):
        msg = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            {"type": "tool_result", "tool_use_id": "t2", "content": "done"},
        ]}
        assert is_tool_result_message(msg) is True

    def test_mixed_content_blocks(self):
        msg = {"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
        ]}
        assert is_tool_result_message(msg) is False

    def test_text_only(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        assert is_tool_result_message(msg) is False

    def test_string_content(self):
        msg = {"role": "user", "content": "hello"}
        assert is_tool_result_message(msg) is False

    def test_empty_content_list(self):
        msg = {"role": "user", "content": []}
        assert is_tool_result_message(msg) is False

    def test_no_content_key(self):
        msg = {"role": "user"}
        assert is_tool_result_message(msg) is False
