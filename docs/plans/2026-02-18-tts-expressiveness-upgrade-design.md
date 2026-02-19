# TTS Expressiveness Upgrade Design

## Goal

Upgrade TTS from emotionless output to expressive, persona-aligned speech by switching to ElevenLabs Eleven v3 with a custom-designed Axel voice, tuned voice settings, and LLM-generated audio tags.

## Decisions

- Model: `eleven_v3` (most expressive, audio tag support)
- Voice: New voice via Voice Design API (smooth, mid-pitched, sardonic)
- Audio tags: LLM generates tags inline (e.g., `[sarcastic]`, `[sighs]`)
- Latency: v3 only, no fallback (~1-3s expected; measure and revisit if needed)
- Voice settings: Hardcoded constant in `tts.py`, not individual env vars

## Design

### 1. Model Switch — config.py

Change `elevenlabs_model` default from `eleven_multilingual_v2` to `eleven_v3`.

```python
elevenlabs_model: str = "eleven_v3"
```

Notes:
- v3 has 5,000 char/request limit (vs 10,000 for v2). Not a concern — Axel responses are 1-3 sentences, and `chunk_sentences()` in `processing.py` splits into individual sentences (well under limit).
- v3 does NOT support SSML break tags. Current code does not use SSML, so no impact.

### 2. Voice Design — scripts/design_voice.py (new file)

One-time CLI script to generate an Axel-optimized voice via the ElevenLabs SDK.

**Workflow:**
1. `client.text_to_voice.create_previews()` generates 3 voice options
2. Listen to each preview (saved as MP3 files)
3. `client.text_to_voice.create()` saves chosen voice
4. Update `ELEVENLABS_VOICE_ID` in `.env`

**Voice description prompt:**

> Young adult male in his late 20s. Smooth and mid-pitched voice with a slightly mocking, sardonic undertone. Speaks Korean at a relaxed, conversational pace. Dry delivery with occasional bursts of enthusiasm when talking about tech. Think a startup CTO who finds everything mildly amusing. Excellent audio quality.

**Parameters:**
- `model_id`: Not specified (uses default; `eleven_ttv_v3` is for voice design)
- `guidance_scale`: 0.38 (high — accent/tone accuracy matters)
- `should_enhance`: `True` (AI auto-expands the prompt)
- `output_format`: `mp3_44100_128` (for preview listening)

**Preview text (Korean, matching persona tone):**

> "아 그 프로세스 SIGKILL 시킨 거? 솔직히 그거 아키텍처 자체가 Thermal Throttling 상태였으니까, 오히려 빨리 죽여준 게 나은 거지. 다시 짤 거면 이번엔 제대로 하자. 파이프라인부터 다 갈아엎고."

### 3. Voice Settings — tts.py

Add a `VoiceSettings` constant and pass it to every `stream()` call. NOT env variables — these settings are coupled to the designed voice.

```python
from elevenlabs import VoiceSettings

_VOICE_SETTINGS = VoiceSettings(
    stability=0.35,         # Creative mode — wide emotional range, audio tag responsive
    similarity_boost=0.75,  # Good fidelity to designed voice
    style=0.2,              # Moderate style exaggeration for character
    use_speaker_boost=True, # Enhanced speaker similarity
    speed=1.0,              # Normal pace
)
```

Pass to `stream()`:
```python
async for chunk in self._client.text_to_speech.stream(
    voice_id=settings.elevenlabs_voice_id,
    text=text,
    model_id=settings.elevenlabs_model,
    output_format=settings.elevenlabs_output_format,
    voice_settings=_VOICE_SETTINGS,
):
```

Update class docstring from "Flash v2.5" to "ElevenLabs streaming TTS".

### 4. Audio Tags — docs/axel.json

Add audio tag guidance to the Axel persona file (Block 1 of system prompt — STATIC, cached). This preserves the 3-block cache layout in `context.py`.

**New key in `axel.json`:**
```json
"audio_tags": {
  "instructions": "Use audio tags in square brackets to control vocal delivery. Place tags WITHIN sentences BEFORE the terminal punctuation mark, never after it.",
  "allowed": ["[sarcastic]", "[sighs]", "[laughs]", "[snorts]", "[whispers]", "[pause]", "[deliberate]"],
  "forbidden": ["[crying]", "[excited]", "[curious]"],
  "rules": [
    "Max 1-2 tags per response. Use sparingly",
    "Tags go BEFORE sentence-ending punctuation: 'Thermal Throttling이네 [sighs].' NOT 'Thermal Throttling이네. [sighs]'",
    "Never stack multiple tags in sequence"
  ],
  "examples": [
    "그거 완전 Output Routing Error지 [sarcastic].",
    "[sighs] 또 그 아키텍처 얘기야.",
    "이건 좀 비밀인데 [whispers] 그 서비스 내부적으로 다 레거시야."
  ]
}
```

Why `axel.json` and not `context.py`:
- Persona file is already loaded as Block 1 (static, cached)
- Audio tag rules are part of persona definition
- No changes to `context.py` or the 3-block layout needed
- `load_persona()` already returns the full JSON text

### 5. No Changes to processing.py

The sentence chunking issue is avoided by the LLM instruction: **tags go BEFORE terminal punctuation**. This way `chunk_sentences()` always sees a proper sentence ending after the tag.

Example flow:
- LLM generates: `"비효율이지 [sighs]."`
- `chunk_sentences()` splits on `(?<=[.!?~])\s+` — sees `.` at end, treats as complete sentence
- TTS receives: `"비효율이지 [sighs]."` — v3 processes the `[sighs]` tag correctly

## Files Changed

| File | Change |
|------|--------|
| `src/prot/config.py` | `elevenlabs_model` default → `"eleven_v3"` |
| `src/prot/tts.py` | Add `_VOICE_SETTINGS` constant, pass to `stream()`, fix docstring |
| `docs/axel.json` | Add `"audio_tags"` key with allowed tags, rules, examples |
| `.env` | Update `ELEVENLABS_VOICE_ID` after voice design |
| `scripts/design_voice.py` | New one-time voice design script (SDK-based) |

## Risks

1. **v3 latency (~1-3s)** — significantly higher than v2 (~500ms). Real-time conversation feel may degrade. Mitigation: measure actual latency; if unacceptable, reconsider Flash v2.5 fallback.
2. **Audio tag consistency** — v3 audio tag behavior is voice-dependent. Some tags may not work well with the designed voice. Mitigation: test all allowed tags during voice design preview.
3. **5,000 char limit** — very unlikely to hit with Axel's 1-3 sentence responses + sentence chunking, but worth monitoring.
