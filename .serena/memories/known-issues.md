# Known Issues

*Last updated: 2026-02-17*

## P0 — Critical
- **prot.service inactive (dead)**: enabled but not running since 01:07:46 PST
- **Health endpoint unreachable**: connection refused (cascade of service being down)

## P1 — High
- **README wrong install command**: documents `uv sync --group dev` but correct command is `uv sync --extra dev` (pyproject.toml uses `[project.optional-dependencies]`, not `[dependency-groups]`)

## P2 — Medium
- **cloudflared timeouts**: QUIC stream timeouts with no upstream while prot.service is down (cascade)
- **pytest-asyncio warnings**: 3 PytestWarnings in test_playback.py — `@pytest.mark.asyncio` on non-async test functions (lines 68, 72, 76)
- **.env.example missing LOG_LEVEL**: variable documented in README and used in code (config.py, log.py)

## P3 — Low
- **No typechecker configured**: no mypy, pyright, or pytype
- **No linter configured**: no ruff, flake8, or pylint
- **Coverage flag inconsistency**: testing section includes `--cov-report=term-missing` not shown in Dev Commands table in docs
