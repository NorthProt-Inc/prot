from pathlib import Path

PERSONA_PATH = Path(__file__).parent.parent.parent / "data" / "axel.json"


def load_persona() -> str:
    """Load Axel persona from file."""
    if PERSONA_PATH.exists():
        return PERSONA_PATH.read_text(encoding="utf-8")
    return ""
