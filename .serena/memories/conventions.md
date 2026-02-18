# Project Conventions

## Language
- **Conversation with Claude**: Korean
- **All code, documentation, comments, commit messages, technical terms**: English

## Design Principles
- **Latency is top priority** — must be measurable at every component
- **Async-first** design throughout the codebase
- **Streaming-first** — chunk-based processing to minimize TTFB
- **Modular pipeline** — each stage (STT, LLM, TTS) independently swappable
- **Well-defined interfaces** per pipeline stage

## Configuration
- Environment variables via `pydantic-settings` (`Settings` class in `config.py`)
- `.env` file for local development

## Commit Style
- **Conventional Commits** format: `feat`, `fix`, `refactor`, `perf`, `docs`, `chore`, `merge`
- Scope in parentheses when applicable: `feat(stt):`, `fix(pipeline):`

## Documentation
- Architectural decisions with rationale in `docs/plans/`
- Performance benchmarks must include measurable metrics

## Code Style
- No typechecker or linter currently configured
- Python 3.12+ features allowed
- Type hints used but not enforced
