from enum import Enum


class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ACTIVE = "active"
    INTERRUPTED = "interrupted"


class StateMachine:
    VALID_TRANSITIONS = {
        State.IDLE: {State.LISTENING},
        State.LISTENING: {State.PROCESSING},
        State.PROCESSING: {State.SPEAKING},
        State.SPEAKING: {State.ACTIVE, State.INTERRUPTED},
        State.ACTIVE: {State.IDLE, State.LISTENING},
        State.INTERRUPTED: {State.LISTENING},
    }

    def __init__(
        self,
        vad_threshold_normal: float = 0.5,
        vad_threshold_speaking: float = 0.8,
    ):
        self._state = State.IDLE
        self._vad_normal = vad_threshold_normal
        self._vad_speaking = vad_threshold_speaking

    @property
    def state(self) -> State:
        return self._state

    @property
    def vad_threshold(self) -> float:
        if self._state == State.SPEAKING:
            return self._vad_speaking
        return self._vad_normal

    def _transition(self, to: State) -> None:
        valid = self.VALID_TRANSITIONS.get(self._state, set())
        if to not in valid:
            raise ValueError(f"Invalid transition: {self._state.value} -> {to.value}")
        self._state = to

    def on_speech_detected(self) -> None:
        if self._state in (State.IDLE, State.ACTIVE):
            self._transition(State.LISTENING)
        elif self._state == State.SPEAKING:
            self._transition(State.INTERRUPTED)
        else:
            raise ValueError(f"Invalid transition: {self._state.value} -> speech_detected")

    def on_utterance_complete(self) -> None:
        self._transition(State.PROCESSING)

    def on_tts_started(self) -> None:
        self._transition(State.SPEAKING)

    def on_tts_complete(self) -> None:
        self._transition(State.ACTIVE)

    def try_on_tts_complete(self) -> bool:
        """SPEAKING -> ACTIVE, returns False if state changed (e.g., interrupted)."""
        if self._state == State.SPEAKING:
            self._state = State.ACTIVE
            return True
        return False

    def on_active_timeout(self) -> None:
        self._transition(State.IDLE)

    def on_interrupt_handled(self) -> None:
        self._transition(State.LISTENING)
