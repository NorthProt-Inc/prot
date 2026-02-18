# Build and Test Commands

## Package Manager
- **uv** 0.10.2
- **Python** 3.12.3

## Install
```bash
uv sync                # production dependencies only
uv sync --extra dev    # with dev dependencies (pytest, pytest-asyncio, pytest-cov)
```
**Note:** `--group dev` is WRONG; dev deps are in `[project.optional-dependencies]`, not `[dependency-groups]`.

## Test
```bash
uv run pytest                                    # unit tests only (integration deselected by default)
uv run pytest -m integration                     # integration tests (requires API keys)
uv run pytest --cov=prot --cov-report=term-missing  # with coverage
```

## Test Configuration
- pytest-asyncio with `asyncio_mode = "auto"` (no need for `@pytest.mark.asyncio` on async tests)
- Integration tests deselected by default via `addopts = "-m 'not integration'"`
- Test directory: `tests/`

## Dev Server
```bash
uv run uvicorn prot.app:app --host 0.0.0.0 --port 8000 --reload
```

## Production
- systemd user service: `deploy/prot.service`
- Tunnel: cloudflared QUIC

## Suggested Commands
```bash
# Development workflow
uv sync --extra dev          # install with dev deps
uv run pytest                # run tests
uv run pytest -v             # verbose test output

# Git
git status
git log --oneline -10
git diff

# System
ls src/prot/                 # list source files
ls tests/                    # list test files
```
