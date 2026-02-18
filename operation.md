# prot 운영 가이드

---

## 설치

### 사전 요구사항

- Python 3.12 이상
- [uv](https://docs.astral.sh/uv/) 패키지 매니저
- PulseAudio (`paplay` 오디오 출력용)
- PostgreSQL 15+ with pgvector (memory 기능 사용 시)

### 의존성 설치

```bash
uv sync              # 프로덕션 의존성
uv sync --extra dev  # 개발 의존성 포함 (pytest, pytest-asyncio, pytest-cov)
```

---

## 설정

### 환경 변수

```bash
cp .env.example .env
```

`.env` 파일을 편집하여 API 키를 설정한다.

#### API Keys (필수)

| 변수 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API 키 (Claude) |
| `ELEVENLABS_API_KEY` | Yes | — | ElevenLabs API 키 (STT + TTS) |

#### Audio / VAD

| 변수 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `MIC_DEVICE_INDEX` | No | (system default) | PyAudio 입력 장치 인덱스 |
| `SAMPLE_RATE` | No | `16000` | 오디오 샘플레이트 (Hz) |
| `CHUNK_SIZE` | No | `512` | 오디오 청크 크기 |
| `VAD_THRESHOLD` | No | `0.5` | VAD 음성 감지 임계값 (IDLE/ACTIVE) |
| `VAD_THRESHOLD_SPEAKING` | No | `0.8` | VAD 임계값 (SPEAKING 상태, barge-in) |

#### STT / LLM / TTS

| 변수 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `STT_LANGUAGE` | No | `ko` | STT 인식 언어 |
| `CLAUDE_MODEL` | No | `claude-opus-4-6` | Claude 모델 ID |
| `CLAUDE_MAX_TOKENS` | No | `1500` | Claude 최대 출력 토큰 |
| `CLAUDE_EFFORT` | No | `medium` | Claude thinking effort (low/medium/high) |
| `ELEVENLABS_VOICE_ID` | No | `Fahco4VZzobUeiPqni1S` | ElevenLabs voice ID |
| `ELEVENLABS_MODEL` | No | `eleven_multilingual_v2` | ElevenLabs TTS 모델 |
| `ELEVENLABS_OUTPUT_FORMAT` | No | `pcm_24000` | TTS 출력 오디오 포맷 |

#### Home Assistant

| 변수 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `HASS_URL` | No | `http://localhost:8123` | Home Assistant URL |
| `HASS_TOKEN` | No | — | Home Assistant long-lived access token |

#### Database / Memory

| 변수 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `DATABASE_URL` | No | `postgresql://prot:prot@localhost:5432/prot` | PostgreSQL 연결 문자열 |
| `DB_POOL_MIN` | No | `2` | DB 커넥션 풀 최소 크기 |
| `DB_POOL_MAX` | No | `10` | DB 커넥션 풀 최대 크기 |
| `DB_EXPORT_DIR` | No | `data/db` | DB 종료 시 CSV 내보내기 디렉토리 |
| `VOYAGE_API_KEY` | No | — | Voyage AI 임베딩 API 키 |
| `VOYAGE_MODEL` | No | `voyage-4-lite` | Voyage 임베딩 모델 |
| `VOYAGE_DIMENSION` | No | `1024` | 임베딩 벡터 차원 |
| `MEMORY_EXTRACTION_MODEL` | No | `claude-haiku-4-5-20251001` | Memory 추출용 모델 |
| `RAG_CONTEXT_TARGET_TOKENS` | No | `3000` | RAG 컨텍스트 목표 토큰 수 |
| `RAG_TOP_K` | No | `10` | RAG 검색 상위 결과 수 |
| `COMMUNITY_REBUILD_INTERVAL` | No | `5` | Community detection 재구축 간격 (추출 횟수) |
| `COMMUNITY_MIN_ENTITIES` | No | `5` | Community detection 최소 엔티티 수 |

#### Logging / Timers

| 변수 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `LOG_LEVEL` | No | `INFO` | 로그 레벨 (DEBUG, INFO, WARNING, ERROR) |
| `ACTIVE_TIMEOUT` | No | `30` | ACTIVE 상태 유지 시간 (초) |

### 데이터베이스 설정 (선택)

Memory/GraphRAG 기능 사용 시 PostgreSQL + pgvector가 필요하다.

```bash
# PostgreSQL에 데이터베이스 생성
createdb prot
psql prot -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql prot -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
```

스키마는 애플리케이션 시작 시 자동 생성되거나, 수동으로 적용할 수 있다:

```bash
psql prot < src/prot/schema.sql
```

### 오디오 장치 확인

```bash
# 사용 가능한 입력 장치 목록
python3 -c "
import pyaudio
pa = pyaudio.PyAudio()
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f'{i}: {info[\"name\"]}')
pa.terminate()
"
```

`MIC_DEVICE_INDEX`를 원하는 장치 번호로 설정한다.

### 출력 볼륨 조절

이어폰 사용 시 청각 보호를 위해 볼륨을 낮춰 설정한다.

```bash
# 현재 볼륨 확인
pactl get-sink-volume @DEFAULT_SINK@

# 볼륨 설정 (이어폰 권장: 20~30%)
pactl set-sink-volume @DEFAULT_SINK@ 25%

# 뮤트 토글
pactl set-sink-mute @DEFAULT_SINK@ toggle

# 출력 장치 목록 및 활성 포트 확인
pactl list sinks | grep -E "Name:|Description:|Active Port:|State:"
```

---

## 실행

### 개발 모드

```bash
# Dev launcher — 기존 포트 프로세스 자동 정리 후 시작
./scripts/run.sh

# 또는 수동 실행
uv run uvicorn prot.app:app --host 0.0.0.0 --port 8000 --reload
```

`scripts/run.sh`는 지정된 포트(기본 8000)에 남아있는 프로세스를 자동으로 종료한 뒤 uvicorn을 실행한다. `PORT` 환경변수로 포트를 변경할 수 있다.

### 프로덕션 모드

```bash
uv run uvicorn prot.app:app --host 0.0.0.0 --port 8000 --log-level info
```

### systemd (user service)

```bash
# 서비스 파일 복사
cp deploy/prot.service ~/.config/systemd/user/

# 서비스 등록 및 시작
systemctl --user daemon-reload
systemctl --user enable --now prot

# 상태 확인
systemctl --user status prot

# 로그 확인
journalctl --user -u prot -f
```

> `deploy/prot.service`는 `pipewire-pulse.service`에 의존한다. PulseAudio/PipeWire가 실행 중이어야 한다.

---

## 디버그

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok", "state": "idle"}
```

### 상태 확인

```bash
curl http://localhost:8000/state
# {"state": "idle"}
```

### 런타임 진단

```bash
curl http://localhost:8000/diagnostics
# {"state": "idle", "background_tasks": 0, "active_timeout": false, ...}
```

### 메모리 프로파일링

```bash
PROT_TRACEMALLOC=1 uv run uvicorn prot.app:app --port 8000
curl http://localhost:8000/memory
```

### 로그 레벨 변경

`.env`에서 `LOG_LEVEL=DEBUG`로 변경 후 서비스 재시작.

---

## 운영

### 상태 머신 플로우

```text
IDLE → LISTENING → PROCESSING → SPEAKING → ACTIVE → IDLE
                                    ↓
                              INTERRUPTED → LISTENING (barge-in)
                              PROCESSING  ← (tool loop)
```

- **IDLE**: 대기 상태. VAD가 음성을 감지하면 LISTENING으로 전환.
- **LISTENING**: STT WebSocket 연결, 음성을 텍스트로 변환 중.
- **PROCESSING**: LLM이 응답 생성 중.
- **SPEAKING**: TTS 오디오 재생 중. barge-in 감지 시 중단 가능.
- **ACTIVE**: 응답 완료 후 후속 발화 대기 (30초 timeout).
- **INTERRUPTED**: 사용자가 말하는 중 끼어들었을 때. STT 재연결 후 LISTENING으로.

### DB 종료 시 CSV 내보내기

애플리케이션 종료(shutdown) 시 `DB_EXPORT_DIR` (기본: `data/db`)로 테이블을 CSV로 자동 내보낸다.

### 대화 로그

대화 내역은 `data/conversations/` 에 날짜별 JSONL 파일(`YYYY-MM-DD.jsonl`)로 기록된다.

### 주요 모니터링 포인트

1. `/health` 엔드포인트로 서비스 생존 확인
2. `/diagnostics`의 `background_tasks` 수 — 비정상적 증가 시 메모리 누수 의심
3. `/diagnostics`의 `db_pool_free` — 0이면 DB 커넥션 고갈
4. systemd 로그에서 `STT connect failed` / `TTS stream failed` 빈도 확인

### 서비스 재시작

```bash
systemctl --user restart prot
```

### 데이터베이스 백업

```bash
pg_dump prot > prot_backup_$(date +%Y%m%d).sql
```

---

## 트러블슈팅 체크리스트

| 증상 | 확인 사항 |
|------|-----------|
| 서비스 시작 안 됨 | `.env` 파일에 `ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY` 설정 확인 |
| 마이크 입력 없음 | `MIC_DEVICE_INDEX` 유효성 확인, PulseAudio 실행 여부 |
| 음성 인식 안 됨 | ElevenLabs API 키 유효성, 네트워크 연결, `VAD_THRESHOLD` 값 조정 |
| 응답 소리 안 남 | `paplay` 설치 확인, PulseAudio 출력 장치 확인 |
| DB 연결 실패 | `DATABASE_URL` 확인, PostgreSQL 실행 여부, pgvector 확장 설치 |
| Memory 기능 비활성 | `VOYAGE_API_KEY` 미설정 시 정상 — DB 없이도 기본 대화 가능 |
| Barge-in 불안정 | `VAD_THRESHOLD_SPEAKING` 값 조정 (높을수록 barge-in 어려움) |
| 응답 지연 | `CLAUDE_EFFORT` 값 확인 (`low`/`medium`/`high`), 네트워크 상태 |
| `SEGV` / crash | `journalctl --user -u prot` 로그 확인, PyAudio 장치 충돌 가능성 |
| 포트 충돌 | `./scripts/run.sh` 사용 (자동 정리) 또는 `lsof -ti :8000` 수동 확인 |
