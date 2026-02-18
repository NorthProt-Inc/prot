# Clean Skill v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** `~/.claude/commands/clean.md`를 6-layer 시스템 가비지 청소기로 재작성

**Architecture:** 단일 커맨드 스킬 파일을 레이어 기반 구조로 확장. 스캔-확인-삭제 흐름, 자동 프로젝트 타입 감지, 안전 규칙을 포함.

**Tech Stack:** Claude Code command skill (Markdown), Bash

---

### Task 1: Frontmatter 및 CLI 인터페이스 섹션 작성

**Files:**
- Modify: `~/.claude/commands/clean.md` (전체 재작성)

**Step 1: 기존 파일 백업**

```bash
cp ~/.claude/commands/clean.md ~/.claude/commands/clean.md.bak
```

**Step 2: 새 파일 작성 — Frontmatter + Usage + Arguments + Level Matrix**

`~/.claude/commands/clean.md`를 아래 내용으로 시작:

```markdown
---
description: System garbage cleaner (project cache, dev tools, Docker, system, ML, logs)
argument-hint: [--dev|--system|--deep|--nuke|--dry-run|--category=X|path]
allowed-tools: [Bash]
---

# /clean - System Garbage Cleaner

## Usage
- `/clean` - Current project cache (scan → confirm → delete)
- `/clean --dev` - + Global dev tool caches
- `/clean --system` - + OS-level maintenance (may need sudo)
- `/clean --deep` - + Docker, ML models, backups/logs
- `/clean --nuke` - Everything (confirmation required)
- `/clean --dry-run` - Scan only, no deletion
- `/clean --category=X` - Specific category only (docker, ml, logs, maven)
- `/clean [path]` - Project cache at specified path
- `/clean --all` - All projects under ~/projects

Flags are combinable: `/clean --dev --dry-run`

## Arguments: $ARGUMENTS

## Level Inclusion Matrix
| Flag | L0 Project | L1 Dev | L2 System | L3 Docker | L4 ML | L5 Logs |
|------|-----------|--------|-----------|-----------|-------|---------|
| (default) | O | | | | | |
| --dev | O | O | | | | |
| --system | O | O | O | | | |
| --deep | O | O | O | O | O | O |
| --nuke | O | O | O | O | O | O |
| --category=X | | | | only X | only X | only X |
```

**Step 3: 검증**

파일 상단이 올바르게 작성되었는지 확인:
```bash
head -30 ~/.claude/commands/clean.md
```

---

### Task 2: Layer 0 — Project Cache 섹션 작성

**Files:**
- Modify: `~/.claude/commands/clean.md` (이어서 추가)

**Step 1: Layer 0 내용 추가**

```markdown
## Layer 0: Project Cache (Default)

Automatically detect project type and clean relevant caches.

### Auto-Detection
Check these files at TARGET path to determine project type:
- `pyproject.toml`, `setup.py`, `requirements.txt` → Python
- `package.json` → Node.js
- `Cargo.toml` → Rust
- `go.mod` → Go
- `build.gradle`, `build.gradle.kts` → Gradle
- `pom.xml` → Maven
- Always: General junk files

### Python Cache
```bash
find TARGET -type d -name "__pycache__" $EXCLUDE -exec rm -rf {} + 2>/dev/null
find TARGET \( -name "*.pyc" -o -name "*.pyo" \) $EXCLUDE -delete 2>/dev/null
find TARGET -type d \( -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" -o -name ".tox" \) $EXCLUDE -exec rm -rf {} + 2>/dev/null
find TARGET -type f -name ".coverage" $EXCLUDE -delete 2>/dev/null
find TARGET -type d -name "htmlcov" $EXCLUDE -exec rm -rf {} + 2>/dev/null
find TARGET -type d -name "*.egg-info" $EXCLUDE -exec rm -rf {} + 2>/dev/null
```

### Node.js Cache
```bash
find TARGET -type d -path "*/node_modules/.cache" -exec rm -rf {} + 2>/dev/null
find TARGET -type d -path "*/node_modules/.vitest" -exec rm -rf {} + 2>/dev/null
find TARGET -type d \( -name ".turbo" -o -name ".next" -o -name ".nuxt" -o -name ".svelte-kit" \) $EXCLUDE -exec rm -rf {} + 2>/dev/null
```

### Rust Cache
```bash
find TARGET -type d -path "*/target/debug/incremental" -exec rm -rf {} + 2>/dev/null
find TARGET -type d -path "*/target/release/incremental" -exec rm -rf {} + 2>/dev/null
find TARGET -type d -path "*/target/debug/build" -exec rm -rf {} + 2>/dev/null
find TARGET -type d -path "*/target/debug/deps" -exec rm -rf {} + 2>/dev/null
```

### Go Cache
```bash
find TARGET -type d -name "__debug_bin*" -exec rm -rf {} + 2>/dev/null
```

### Gradle Cache
```bash
find TARGET -type d -name "build" -not -path "*/node_modules/*" -not -path "*/.git/*" $EXCLUDE -exec rm -rf {} + 2>/dev/null
find TARGET -type d -name ".gradle" $EXCLUDE -exec rm -rf {} + 2>/dev/null
```

### Maven Cache
```bash
find TARGET -type d -name "target" -not -path "*/node_modules/*" -not -path "*/rust/*" $EXCLUDE -exec rm -rf {} + 2>/dev/null
```

### General Junk (Always)
```bash
find TARGET \( -name ".DS_Store" -o -name "Thumbs.db" -o -name "*.swp" -o -name "*.swo" -o -name "*~" -o -name "#*#" \) $EXCLUDE -delete 2>/dev/null
```
```

---

### Task 3: Layer 1 — Global Dev Tool Cache 섹션 작성

**Files:**
- Modify: `~/.claude/commands/clean.md` (이어서 추가)

**Step 1: Layer 1 내용 추가**

```markdown
## Layer 1: Global Dev Tool Cache (--dev)

These are global caches in the user's home directory. Safe to delete — tools will re-download as needed.

### Scan Commands (sizes)
```bash
echo "=== Global Dev Tool Caches ==="
du -sh ~/.cache/pip 2>/dev/null || true
du -sh ~/.cache/uv 2>/dev/null || true
du -sh ~/.npm/_cacache 2>/dev/null || true
du -sh ~/.local/share/pnpm/store 2>/dev/null || true
du -sh ~/.cache/yarn 2>/dev/null || true
du -sh ~/.cargo/registry/cache 2>/dev/null || true
du -sh ~/.cache/go-build 2>/dev/null || true
du -sh ~/.gradle/caches 2>/dev/null || true
```

### Deletion Commands
```bash
rm -rf ~/.cache/pip 2>/dev/null
rm -rf ~/.cache/uv 2>/dev/null
rm -rf ~/.npm/_cacache 2>/dev/null
rm -rf ~/.local/share/pnpm/store 2>/dev/null
rm -rf ~/.cache/yarn 2>/dev/null
rm -rf ~/.cargo/registry/cache 2>/dev/null
rm -rf ~/.cache/go-build 2>/dev/null
rm -rf ~/.gradle/caches 2>/dev/null
```

### Confirm Required
- `~/.m2/repository/` — Maven local repo, may break builds. Only delete with `--category=maven` or `--nuke`.
```bash
du -sh ~/.m2/repository 2>/dev/null
# Only on explicit confirmation:
rm -rf ~/.m2/repository 2>/dev/null
```
```

---

### Task 4: Layer 2 — System Maintenance 섹션 작성

**Files:**
- Modify: `~/.claude/commands/clean.md` (이어서 추가)

**Step 1: Layer 2 내용 추가**

```markdown
## Layer 2: System Maintenance (--system)

Some commands require sudo. Process non-sudo items first, then list sudo-required items.

### No Sudo Required
```bash
echo "=== Trash ==="
du -sh ~/.local/share/Trash 2>/dev/null
rm -rf ~/.local/share/Trash/* 2>/dev/null

echo "=== Thumbnail cache ==="
du -sh ~/.cache/thumbnails 2>/dev/null
rm -rf ~/.cache/thumbnails/* 2>/dev/null

echo "=== /tmp old files (7+ days, owned by current user) ==="
find /tmp -maxdepth 1 -user "$(whoami)" -mtime +7 -exec rm -rf {} + 2>/dev/null
```

### Sudo Required (inform user, do not auto-execute)
Present these as suggestions, execute only with user confirmation:
```bash
echo "=== journalctl (requires sudo) ==="
journalctl --disk-usage 2>/dev/null
# sudo journalctl --vacuum-time=7d

echo "=== apt cache (requires sudo) ==="
du -sh /var/cache/apt/archives 2>/dev/null
# sudo apt clean

echo "=== snap old revisions (requires sudo) ==="
snap list --all 2>/dev/null | awk '/disabled/{print $1, $3}'
# sudo snap remove <snap> --revision=<rev>
```
```

---

### Task 5: Layer 3 — Docker 섹션 작성

**Files:**
- Modify: `~/.claude/commands/clean.md` (이어서 추가)

**Step 1: Layer 3 내용 추가**

```markdown
## Layer 3: Docker (--deep or --category=docker)

Only execute if `docker` command is available. Check first:
```bash
command -v docker &>/dev/null || { echo "Docker not installed, skipping"; }
```

### Scan Commands
```bash
echo "=== Docker disk usage ==="
docker system df 2>/dev/null

echo "=== Dangling images ==="
docker images -f "dangling=true" -q 2>/dev/null | wc -l

echo "=== Stopped containers ==="
docker ps -a -f "status=exited" -q 2>/dev/null | wc -l

echo "=== Build cache ==="
docker builder du 2>/dev/null | tail -1
```

### Deletion Commands
```bash
echo "=== Pruning dangling images ==="
docker image prune -f 2>/dev/null

echo "=== Pruning stopped containers ==="
docker container prune -f 2>/dev/null

echo "=== Pruning build cache ==="
docker builder prune -f 2>/dev/null

echo "=== Pruning unused networks ==="
docker network prune -f 2>/dev/null
```

### Confirm Required (volumes)
```bash
# DANGER: May delete persistent data. Only with --nuke or explicit confirmation.
echo "=== Unused volumes ==="
docker volume ls -f "dangling=true" -q 2>/dev/null | wc -l
# docker volume prune -f
```
```

---

### Task 6: Layer 4 — ML Model Cache 섹션 작성

**Files:**
- Modify: `~/.claude/commands/clean.md` (이어서 추가)

**Step 1: Layer 4 내용 추가**

```markdown
## Layer 4: ML Model Cache (--deep or --category=ml)

WARNING: These can be tens of GB. Always show size first, always confirm before deleting.

### Scan Commands
```bash
echo "=== ML Model Caches ==="
du -sh ~/.cache/huggingface 2>/dev/null || true
du -sh ~/.cache/torch 2>/dev/null || true
du -sh ~/.cache/huggingface/hub 2>/dev/null || true
du -sh ~/.cache/huggingface/transformers 2>/dev/null || true
```

### Deletion Commands (confirm required)
```bash
# Only after explicit user confirmation due to large re-download cost:
rm -rf ~/.cache/huggingface 2>/dev/null
rm -rf ~/.cache/torch 2>/dev/null
```
```

---

### Task 7: Layer 5 — Backups/Logs 섹션 작성

**Files:**
- Modify: `~/.claude/commands/clean.md` (이어서 추가)

**Step 1: Layer 5 내용 추가**

```markdown
## Layer 5: Backups & Logs (--deep or --category=logs)

### Scan Commands
```bash
echo "=== Log files (30+ days old) ==="
find TARGET -name "*.log" -mtime +30 $EXCLUDE 2>/dev/null | head -20
find TARGET -name "*.log" -mtime +30 $EXCLUDE 2>/dev/null | wc -l

echo "=== Backup files ==="
find TARGET \( -name "*.bak" -o -name "*.backup" -o -name "*.orig" \) $EXCLUDE 2>/dev/null | head -20
find TARGET \( -name "*.bak" -o -name "*.backup" -o -name "*.orig" \) $EXCLUDE 2>/dev/null | wc -l

echo "=== Core dumps ==="
find TARGET \( -name "core" -o -name "core.*" -o -name "*.core" \) -type f $EXCLUDE 2>/dev/null | head -10
find TARGET \( -name "core" -o -name "core.*" -o -name "*.core" \) -type f $EXCLUDE 2>/dev/null | wc -l
```

### Deletion Commands
```bash
echo "=== Deleting old logs ==="
find TARGET -name "*.log" -mtime +30 $EXCLUDE -delete 2>/dev/null

echo "=== Deleting backup files ==="
find TARGET \( -name "*.bak" -o -name "*.backup" -o -name "*.orig" \) $EXCLUDE -delete 2>/dev/null

echo "=== Deleting core dumps ==="
find TARGET \( -name "core" -o -name "core.*" -o -name "*.core" \) -type f $EXCLUDE -delete 2>/dev/null
```
```

---

### Task 8: Safety Rules, Execution Flow, Output Format 섹션 작성

**Files:**
- Modify: `~/.claude/commands/clean.md` (마무리 섹션)

**Step 1: Safety Rules 추가**

```markdown
## Safety Rules (MANDATORY)

### Never Delete (Hardcoded exclusions for ALL operations)
- `site-packages/`, `.venv/`, `venv/`, `env/` — Python virtual environments
- `node_modules/` itself — Only `node_modules/.cache/` and `node_modules/.vitest/`
- `~/.ssh/`, `~/.gnupg/`, `~/.config/` — Security-critical
- `.git/` — Version control
- `/usr/`, `/bin/`, `/sbin/`, `/lib/` — System paths

### Exclusion Patterns (applied to ALL find commands)
```
EXCLUDE='-not -path "*/site-packages/*" -not -path "*/.venv/*" -not -path "*/venv/*" -not -path "*/env/*" -not -path "*/.git/*"'
```

### Confirm Required (interactive prompt before deletion)
- Docker volumes (data loss risk)
- ML model caches (large re-download)
- Maven `~/.m2/repository/` (build breakage)
- `--nuke` mode entirely

### Sudo Handling
- Never auto-execute sudo commands
- Present as suggestions with `# sudo ...` prefix
- User must explicitly confirm each sudo operation
```

**Step 2: Execution Flow 추가**

```markdown
## Execution Flow

### Step 1: Parse Arguments
Determine from $ARGUMENTS:
1. Level flags: `--dev`, `--system`, `--deep`, `--nuke`
2. Modifiers: `--dry-run`, `--category=X`, `--all`
3. Path: explicit path or default to current working directory
4. If `--all` → TARGET=`~/projects`
5. If path given → TARGET=that path
6. Otherwise → TARGET=current working directory

### Step 2: Scan Phase (ALWAYS runs first)
For each applicable layer, run scan commands and collect:
- Category name
- Item count (directories/files found)
- Total size (via du -sh or summing)
- Risk level (Safe / Confirm / Sudo)

### Step 3: Display Results Table
```bash
echo "=== Disk usage before ==="
du -sh TARGET 2>/dev/null
df -h . 2>/dev/null | tail -1
```

Present as:
| Category | Items | Size | Risk |
|----------|-------|------|------|

Sort by size descending.

### Step 4: Confirmation
- If `--dry-run`: Stop here, display table only
- If `--nuke`: Require explicit "yes" from user before ANY deletion
- Otherwise: Proceed with Safe items, ask for Confirm items individually

### Step 5: Execute Deletion
Run deletion commands for confirmed categories only.

### Step 6: Report
```bash
echo "=== Disk usage after ==="
du -sh TARGET 2>/dev/null
df -h . 2>/dev/null | tail -1
```
```

**Step 3: Output Format 추가**

```markdown
## Output Format

```
## System Cleanup Report

### Target
- Path: /home/user/projects/myapp
- Level: --dev
- Mode: actual deletion

### Disk Before
- Project: 2.1 GB
- Disk free: 45.2 GB

### Scan Results
| Category          | Items | Size     | Risk    | Action  |
|-------------------|-------|----------|---------|---------|
| __pycache__       | 45    | 12.3 MB  | Safe    | Deleted |
| .pytest_cache     | 3     | 1.2 MB   | Safe    | Deleted |
| .coverage         | 1     | 0.5 MB   | Safe    | Deleted |
| pip cache         | 892   | 2.1 GB   | Safe    | Deleted |
| uv cache          | 234   | 890 MB   | Safe    | Deleted |
| .m2/repository    | 4521  | 3.2 GB   | Confirm | Skipped |
| **Total cleaned** |       | **3.0 GB** |       |         |

### Disk After
- Project: 2.0 GB
- Disk free: 48.2 GB
- Freed: 3.0 GB

### Skipped
- ~/.m2/repository: 3.2 GB (use --category=maven)

### Sudo Required (not executed)
- journalctl: ~500 MB (run: sudo journalctl --vacuum-time=7d)
- apt cache: ~200 MB (run: sudo apt clean)
```
```

---

### Task 9: 통합 검증 — dry-run 테스트

**Step 1: `/clean --dry-run` 테스트**

현재 프로젝트에서 `/clean --dry-run` 실행.
Expected: 프로젝트 캐시 스캔 결과 테이블 출력, 삭제 없음.

**Step 2: `/clean --dev --dry-run` 테스트**

Expected: Layer 0 + Layer 1 스캔 결과 표시.

**Step 3: `/clean --deep --dry-run` 테스트**

Expected: 전체 레이어 스캔 결과 표시.

**Step 4: 문제가 있으면 수정, 없으면 진행**

---

### Task 10: 백업 파일 정리 및 커밋

**Step 1: 백업 제거**

```bash
rm ~/.claude/commands/clean.md.bak
```

**Step 2: 커밋**

```bash
git add docs/plans/2026-02-17-clean-skill-v2-design.md docs/plans/2026-02-17-clean-skill-v2.md
git commit -m "docs: add clean skill v2 design and implementation plan"
```

Note: `~/.claude/commands/clean.md`는 프로젝트 외부 경로이므로 별도 관리.
