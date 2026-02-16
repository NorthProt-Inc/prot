---
paths:
  - "**/*"
---

# prot — Real-time Voice Conversation Architecture

## Project Overview
Building a real-time voice conversation architecture leveraging SOTA models.

## Core Goals
- Ultra-low latency real-time voice conversation
- SOTA STT/LLM/TTS pipeline integration
- Production-grade architecture design

## Design Principles
- Latency is the top priority — must be measurable at every component
- Modular pipeline design (STT ↔ LLM ↔ TTS independently swappable)
- Streaming-first: chunk-based processing to minimize TTFB

## Code Conventions
- Async-first design
- Each pipeline stage must have a well-defined interface
- Configuration via environment variables or config files

## Rules
- Conversation with Claude: Korean
- All documentation, comments, commit messages, and technical terms: English
- Record architectural decisions with rationale in docs/
- Performance benchmarks must include measurable metrics
