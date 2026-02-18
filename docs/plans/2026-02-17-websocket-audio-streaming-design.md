# WebSocket PCM Audio Streaming

## Problem
prot's TTS output goes to local paplay only. Need to stream audio to remote devices
(Windows PC on LAN) without additional software installation.

## Decision
WebSocket binary streaming of raw PCM data. Browser client uses Web Audio API for playback.

## Architecture

```
ElevenLabs TTS → PCM chunks → audio_q → player.play_chunk(data)
                                              ├─ paplay stdin (local)
                                              └─ ws.send_bytes (remote browsers)
                                                      │
                                                  Web Audio API AudioWorklet
                                                      │
                                                  Remote speakers
```

## Changes

### playback.py — Add WebSocket broadcast
- Maintain `set[WebSocket]` of connected clients
- `play_chunk()`: write to paplay + broadcast to all WS clients
- `register(ws)` / `unregister(ws)` methods

### app.py — Add `/ws/audio` endpoint
- Accept WebSocket connection
- Register with AudioPlayer
- Keep alive until disconnect
- No auth (LAN-only, same as existing endpoints)

### static/audio.html — Browser client (new)
- Connect to `/ws/audio`
- Receive binary PCM frames (s16le, 24kHz, mono)
- Play via AudioWorklet (s16le → float32 conversion)
- Volume slider with software amplification
- Connection status indicator

## Latency Budget
| Segment         | Added latency |
|-----------------|---------------|
| WS send_bytes   | <1ms          |
| LAN transfer    | ~1-5ms        |
| Audio buffering | ~20-50ms      |
| **Total**       | **~25-55ms**  |

## Format
- PCM signed 16-bit little-endian
- 24000 Hz sample rate
- 1 channel (mono)
