# Current Work

*Last updated: 2026-02-17*

## Active Branch
`main`

## Latest Work
- Hot-path performance optimizations (plan in `docs/plans/2026-02-17-hot-path-perf-optimizations.md`)
  - Parallelize neighbor queries
  - Pre-compute STT audio template
  - Remove dead code

## Recently Completed
- Agentic tool loop (up to 3 iterations, Home Assistant + web search)
- STT WebSocket connection reuse between utterances
- TTS HTTP connection pool pre-warming during startup
- Initial project documentation (README, operation.md)
- Audio device validation to prevent SEGV on invalid mic index

## Recent Commits
```
d35ae35 3947198UMM
762742d docs: add initial project documentation
8626f07 merge: fix journal log issues (SEGV, tool loop, TTS cold start, STT reconnect)
88a1a2f feat(stt): reuse existing WebSocket connection between utterances
63edc52 feat(tts): pre-warm HTTP connection pool during startup
ee1e027 feat(pipeline): implement agentic tool loop
```

---
**Note:** Update this memory at the end of each work session.
