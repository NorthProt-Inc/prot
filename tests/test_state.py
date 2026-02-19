import pytest
from prot.state import StateMachine, State


class TestStateMachine:
    def test_initial_state_is_idle(self):
        sm = StateMachine()
        assert sm.state == State.IDLE

    def test_speech_detected_in_idle_goes_to_listening(self):
        sm = StateMachine()
        sm.on_speech_detected()
        assert sm.state == State.LISTENING

    def test_utterance_complete_goes_to_processing(self):
        sm = StateMachine()
        sm.on_speech_detected()
        sm.on_utterance_complete()
        assert sm.state == State.PROCESSING

    def test_tts_started_goes_to_speaking(self):
        sm = StateMachine()
        sm.on_speech_detected()
        sm.on_utterance_complete()
        sm.on_tts_started()
        assert sm.state == State.SPEAKING

    def test_tts_complete_goes_to_active(self):
        sm = StateMachine()
        sm.on_speech_detected()
        sm.on_utterance_complete()
        sm.on_tts_started()
        sm.try_on_tts_complete()
        assert sm.state == State.ACTIVE

    def test_active_timeout_goes_to_idle(self):
        sm = StateMachine()
        sm.on_speech_detected()
        sm.on_utterance_complete()
        sm.on_tts_started()
        sm.try_on_tts_complete()
        sm.on_active_timeout()
        assert sm.state == State.IDLE

    def test_speech_during_speaking_goes_to_interrupted(self):
        sm = StateMachine()
        sm.on_speech_detected()
        sm.on_utterance_complete()
        sm.on_tts_started()
        sm.on_speech_detected()  # barge-in
        assert sm.state == State.INTERRUPTED

    def test_interrupt_handled_goes_to_listening(self):
        sm = StateMachine()
        sm.on_speech_detected()
        sm.on_utterance_complete()
        sm.on_tts_started()
        sm.on_speech_detected()  # barge-in
        sm.on_interrupt_handled()
        assert sm.state == State.LISTENING

    def test_speech_in_active_goes_to_listening(self):
        sm = StateMachine()
        sm.on_speech_detected()  # IDLE -> LISTENING
        sm.on_utterance_complete()  # -> PROCESSING
        sm.on_tts_started()  # -> SPEAKING
        sm.try_on_tts_complete()  # -> ACTIVE
        sm.on_speech_detected()  # ACTIVE -> LISTENING
        assert sm.state == State.LISTENING

    def test_vad_threshold_normal_in_idle(self):
        sm = StateMachine(vad_threshold_normal=0.5, vad_threshold_speaking=0.8)
        assert sm.vad_threshold == 0.5

    def test_vad_threshold_elevated_in_speaking(self):
        sm = StateMachine(vad_threshold_normal=0.5, vad_threshold_speaking=0.8)
        sm.on_speech_detected()
        sm.on_utterance_complete()
        sm.on_tts_started()
        assert sm.vad_threshold == 0.8

    def test_try_on_tts_complete_from_speaking(self):
        sm = StateMachine()
        sm.on_speech_detected()
        sm.on_utterance_complete()
        sm.on_tts_started()
        assert sm.state == State.SPEAKING
        assert sm.try_on_tts_complete() is True
        assert sm.state == State.ACTIVE

    def test_try_on_tts_complete_from_interrupted(self):
        sm = StateMachine()
        sm.on_speech_detected()
        sm.on_utterance_complete()
        sm.on_tts_started()
        sm.on_speech_detected()  # barge-in -> INTERRUPTED
        assert sm.state == State.INTERRUPTED
        assert sm.try_on_tts_complete() is False
        assert sm.state == State.INTERRUPTED

    def test_invalid_transition_raises(self):
        sm = StateMachine()
        with pytest.raises(ValueError):
            sm.on_utterance_complete()  # can't go from IDLE -> PROCESSING directly

    def test_tool_iteration_speaking_to_processing(self):
        sm = StateMachine()
        sm.on_speech_detected()       # IDLE -> LISTENING
        sm.on_utterance_complete()    # -> PROCESSING
        sm.on_tts_started()           # -> SPEAKING
        sm.on_tool_iteration()        # -> PROCESSING
        assert sm.state == State.PROCESSING

    def test_tool_iteration_invalid_from_active(self):
        sm = StateMachine()
        sm.on_speech_detected()
        sm.on_utterance_complete()
        sm.on_tts_started()
        sm.try_on_tts_complete()          # -> ACTIVE
        with pytest.raises(ValueError):
            sm.on_tool_iteration()
