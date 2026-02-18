# Clean Skill v2 — System Garbage Cleaner Design

## Overview

기존 프로젝트 캐시 청소 스킬을 시스템 전역 가비지 청소기로 확장.
레벨 기반 범위 제어 + 스캔-확인-삭제 흐름의 하이브리드 방식.

## CLI Interface

```
/clean                    → 현재 프로젝트 캐시 (스캔 → 확인 → 삭제)
/clean --dev              → + 글로벌 개발도구 캐시
/clean --system           → + OS 레벨 (journalctl, apt, snap, tmp, trash)
/clean --deep             → + Docker, ML 모델, 백업/로그
/clean --nuke             → 전부 (반드시 확인 프롬프트)
/clean --dry-run          → 어떤 레벨이든 스캔만
/clean --category=docker  → 특정 카테고리만 지정
/clean [path]             → 특정 경로 프로젝트 캐시
/clean --all              → ~/projects 하위 전체 프로젝트
```

플래그 조합 가능: `/clean --dev --dry-run`, `/clean --deep --category=docker`

## Category Layers (6 Layers)

### Layer 0 — Project Cache (default, always safe)

| Type | Targets | Auto-detect |
|------|---------|-------------|
| Python | `__pycache__/`, `*.pyc`, `*.pyo`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.coverage`, `htmlcov/`, `.tox/`, `*.egg-info/` | `pyproject.toml`, `setup.py` |
| Node.js | `node_modules/.cache/`, `.turbo/`, `.next/`, `.nuxt/`, `.svelte-kit/`, `node_modules/.vitest/` | `package.json` |
| Rust | `target/debug/`, `target/release/incremental/` | `Cargo.toml` |
| Go | project build artifacts | `go.mod` |
| Java | `build/`, `.gradle/`, `target/` (Maven) | `build.gradle`, `pom.xml` |
| General | `.DS_Store`, `Thumbs.db`, `*.swp`, `*~` | always |

### Layer 1 — Global Dev Tool Cache (`--dev`)

| Target | Path |
|--------|------|
| pip | `~/.cache/pip/` |
| uv | `~/.cache/uv/` |
| npm | `~/.npm/_cacache/` |
| pnpm | `~/.local/share/pnpm/store/` |
| yarn | `~/.cache/yarn/` |
| cargo | `~/.cargo/registry/cache/` |
| go | `~/.cache/go-build/` |
| gradle | `~/.gradle/caches/` |
| maven | `~/.m2/repository/` (confirm required) |

### Layer 2 — System Maintenance (`--system`, may require sudo)

| Target | Command |
|--------|---------|
| journalctl | `journalctl --vacuum-time=7d` |
| apt cache | `apt clean` |
| snap | `snap list --all` → remove disabled revisions |
| /tmp | files older than 7 days only |
| trash | `~/.local/share/Trash/` |
| thumbnail cache | `~/.cache/thumbnails/` |

### Layer 3 — Docker (`--deep` or `--category=docker`)

| Target | Command |
|--------|---------|
| dangling images | `docker image prune` |
| build cache | `docker builder prune` |
| stopped containers | `docker container prune` |
| unused volumes | `docker volume prune` (confirm required) |
| unused networks | `docker network prune` |

### Layer 4 — ML Model Cache (`--deep` or `--category=ml`)

| Target | Path |
|--------|------|
| HuggingFace | `~/.cache/huggingface/` |
| PyTorch | `~/.cache/torch/` |
| transformers | `~/.cache/huggingface/transformers/` |

### Layer 5 — Backups/Logs (`--deep` or `--category=logs`)

| Target | Pattern |
|--------|---------|
| log files | `*.log` (in project, older than 30 days) |
| backup files | `*.bak`, `*.backup`, `*.orig` |
| core dumps | `core.*`, `*.core` |
| editor backups | `*~`, `*.swp`, `*.swo`, `#*#` |

## Execution Flow

```
1. Parse arguments (level, flags, path)
2. Scan targets for applicable layers
3. Sum sizes per category & sort (largest first)
4. Display table (Category | Items | Size | Risk)
5. User confirmation (stop here if --dry-run)
6. Execute deletion
7. Report results (before/after comparison)
```

## Safety Rules

### Never Delete (Hardcoded)
- `site-packages/`, `.venv/`, `venv/`, `env/`, `node_modules/` (itself)
- `~/.ssh/`, `~/.gnupg/`, `~/.config/`
- `.git/` directories

### Confirm Required (Interactive Prompt)
- Docker volumes (`--deep`)
- ML model cache (can be tens of GB)
- Maven `~/.m2/repository/` (may break builds)
- `--nuke` mode entirely

### Sudo Handling
- Layer 2 items note sudo requirement
- Process non-sudo items first, then list sudo-required items separately

## Output Format

```markdown
## System Cleanup Report

### Scan Results (--dev level)
| Category          | Items | Size    | Risk  |
|-------------------|-------|---------|-------|
| __pycache__       | 45    | 12.3 MB | Safe  |
| .pytest_cache     | 3     | 1.2 MB  | Safe  |
| pip cache         | 892   | 2.1 GB  | Safe  |
| uv cache          | 234   | 890 MB  | Safe  |
| npm cache         | 1203  | 1.5 GB  | Safe  |
| **Total**         |       | **4.5 GB** |    |

### Deleted
- 2,377 items removed
- 4.5 GB freed

### Skipped (requires confirmation)
- ~/.m2/repository: 3.2 GB (use --category=maven to clean)

### After
- Disk usage: 45.2 GB → 40.7 GB
```

## Level Inclusion Matrix

| Flag | L0 Project | L1 Dev | L2 System | L3 Docker | L4 ML | L5 Logs |
|------|-----------|--------|-----------|-----------|-------|---------|
| (default) | O | | | | | |
| --dev | O | O | | | | |
| --system | O | O | O | | | |
| --deep | O | O | O | O | O | O |
| --nuke | O | O | O | O | O | O |
| --category=X | | | | only X | only X | only X |
