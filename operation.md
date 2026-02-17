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
uv sync --group dev  # 개발 의존성 포함
```

---

## 설정

### 환경 변수

```bash
cp .env.example .env
```

`.env` 파일을 편집하여 API 키를 설정한다.

| 변수 | 필수 | 기본값 | 설명 |
| ------ | ------ | -------- | ------ |
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API 키 (Claude) |
| `ELEVENLABS_API_KEY` | Yes | — | ElevenLabs API 키 (STT + TTS) |
| `ELEVENLABS_VOICE_ID` | No | `Fahco4VZzobUeiPqni1S` | ElevenLabs voice ID |
| `VOYAGE_API_KEY` | No | — | Voyage AI 임베딩 API 키 |
| `DATABASE_URL` | No | `postgresql://prot:prot@localhost:5432/prot` | PostgreSQL 연결 문자열 |
| `HASS_URL` | No | `http://localhost:8123` | Home Assistant URL |
| `HASS_TOKEN` | No | — | Home Assistant long-lived access token |
| `MIC_DEVICE_INDEX` | No | `11` | PyAudio 입력 장치 인덱스 |
| `LOG_LEVEL` | No | `INFO` | 로그 레벨 (DEBUG, INFO, WARNING, ERROR) |
| `ACTIVE_TIMEOUT` | No | `30` | ACTIVE 상태 유지 시간 (초) |
| `CLAUDE_MODEL` | No | `claude-opus-4-6` | Claude 모델 ID |
| `CLAUDE_MAX_TOKENS` | No | `1500` | Claude 최대 출력 토큰 |
| `CLAUDE_EFFORT` | No | `medium` | Claude thinking effort |
| `VAD_THRESHOLD` | No | `0.5` | VAD 음성 감지 임계값 (IDLE/ACTIVE) |
| `VAD_THRESHOLD_SPEAKING` | No | `0.8` | VAD 임계값 (SPEAKING 상태, barge-in) |

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

---

## 실행

### 개발 모드

```bash
uv run uvicorn prot.app:app --host 0.0.0.0 --port 8000 --reload
```

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
| ------ | ----------- |
| 서비스 시작 안 됨 | `.env` 파일에 `ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY` 설정 확인 |
| 마이크 입력 없음 | `MIC_DEVICE_INDEX` 유효성 확인, PulseAudio 실행 여부 |
| 음성 인식 안 됨 | ElevenLabs API 키 유효성, 네트워크 연결, `VAD_THRESHOLD` 값 조정 |
| 응답 소리 안 남 | `paplay` 설치 확인, PulseAudio 출력 장치 확인 |
| DB 연결 실패 | `DATABASE_URL` 확인, PostgreSQL 실행 여부, pgvector 확장 설치 |
| Memory 기능 비활성 | `VOYAGE_API_KEY` 미설정 시 정상 — DB 없이도 기본 대화 가능 |
| Barge-in 불안정 | `VAD_THRESHOLD_SPEAKING` 값 조정 (높을수록 barge-in 어려움) |
| 응답 지연 | `CLAUDE_EFFORT` 값 확인 (`low`/`medium`/`high`), 네트워크 상태 |
| `SEGV` / crash | `journalctl --user -u prot` 로그 확인, PyAudio 장치 충돌 가능성 |
