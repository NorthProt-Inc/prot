# Pipeline Data/Memory Audit Design

**Date**: 2026-02-16
**Branch**: fix/pipeline-robustness
**Scope**: 전체 포괄 분석 (정적 감사 + 테스트 검증 + 런타임 프로파일링)

## 목적

prot 실시간 음성 대화 시스템의 데이터/메모리 파이프라인 전체를 체계적으로 감사하여:
- 메모리 누수, Task 누실, 리소스 미정리 문제 식별
- 데이터 정합성 및 동시성 안전성 검증
- 장시간 운영 시 안정성 확보를 위한 개선점 도출

## 아키텍처 요약

```
PyAudio Thread → VAD → STT(ElevenLabs WS) → LLM(Claude) → TTS(ElevenLabs) → paplay
                                                    ↓
                                            GraphRAG(pgvector) → Memory extraction
```

핵심 파일: `pipeline.py`(316 LOC), `stt.py`(147), `tts.py`(37), `llm.py`(93),
`vad.py`(45), `audio.py`(58), `playback.py`(78), `state.py`(76),
`memory.py`(119), `graphrag.py`(168), `db.py`(52), `embeddings.py`(48)

## Phase 1: 정적 코드 감사 (7축 분석)

### 축 1. asyncio Task 생명주기
- **대상**: `pipeline.py`, `stt.py`, `memory.py`
- **체크리스트**:
  - [ ] `create_task()` / `ensure_future()` 전수 조사
  - [ ] 각 Task 참조 저장 여부
  - [ ] cancel 경로 존재 여부
  - [ ] done callback 또는 await 존재 여부
  - [ ] shutdown 시 모든 Task 정리 확인
  - [ ] fire-and-forget 패턴 식별 및 누수 위험 평가

### 축 2. Queue/버퍼 바운딩
- **대상**: `pipeline.py`, `processing.py`
- **체크리스트**:
  - [ ] `asyncio.Queue` maxsize 설정 확인
  - [ ] LLM 스트림 buffer 누적 상한 확인
  - [ ] Producer > Consumer 속도일 때 백프레셔 동작 확인
  - [ ] Sentinel 패턴 정확성 (종료 시그널)
  - [ ] barge-in 시 큐 잔여 데이터 처리

### 축 3. WebSocket/스트림 정리
- **대상**: `stt.py`, `tts.py`, `llm.py`
- **체크리스트**:
  - [ ] connect/disconnect 대칭성
  - [ ] 에러 시 finally 블록의 리소스 정리
  - [ ] 재연결 시 이전 리소스 완전 해제
  - [ ] async context manager 사용 여부
  - [ ] WebSocket close code/reason 처리

### 축 4. 스레드↔이벤트루프 안전성
- **대상**: `pipeline.py`, `audio.py`
- **체크리스트**:
  - [ ] `run_coroutine_threadsafe` 사용 패턴 검증
  - [ ] 공유 변수에 대한 스레드 안전성
  - [ ] PyAudio 콜백 스레드 → asyncio 전달 경로
  - [ ] 이벤트루프 참조(`_loop`) 유효성

### 축 5. DB 연결 풀/트랜잭션
- **대상**: `db.py`, `graphrag.py`, `memory.py`, `embeddings.py`
- **체크리스트**:
  - [ ] acquire/release 대칭 (context manager)
  - [ ] 트랜잭션 에러 시 롤백 보장
  - [ ] 풀 close 경로 완전성
  - [ ] 동시 쿼리 수 제한 (Semaphore)
  - [ ] 커넥션 타임아웃 설정

### 축 6. 데이터 정합성
- **대상**: `processing.py`, `pipeline.py`, `state.py`
- **체크리스트**:
  - [ ] 문장 청킹 정규식 패턴 정확성
  - [ ] barge-in 시 LLM/TTS/Queue 데이터 유실 처리
  - [ ] 상태 전이와 데이터 처리 동기화
  - [ ] 전사(transcript) 누적의 정확성
  - [ ] 빈 문장 / 공백 처리

### 축 7. 프로세스/리소스 정리
- **대상**: `playback.py`, `audio.py`, `pipeline.py`
- **체크리스트**:
  - [ ] paplay 서브프로세스 좀비화 방지 (wait 호출)
  - [ ] PyAudio 스트림 정리 (stop_stream, close, terminate)
  - [ ] shutdown 경로의 완전성 (모든 리소스 해제 순서)
  - [ ] 시그널 핸들링 (SIGTERM, SIGINT)

## Phase 2: 테스트 실행 및 검증

- [ ] `pytest` 전체 테스트 스위트 실행
- [ ] `pytest --cov` 커버리지 측정
- [ ] 실패 테스트 분석 및 원인 파악
- [ ] 메모리/데이터 관련 테스트 누락 식별
- [ ] 에지 케이스 테스트 존재 여부 확인:
  - 빈 입력, 초장문 입력, 종결부호 없는 입력
  - 연결 실패/재연결 시나리오
  - barge-in 타이밍 에지 케이스
  - 동시 shutdown 시나리오

## Phase 3: 런타임 프로파일링 설계

- [ ] `tracemalloc` 기반 메모리 프로파일링 코드 작성
- [ ] `asyncio` debug mode (`PYTHONASYNCIODEBUG=1`) 활성화 계획
- [ ] Queue 크기 추적 로깅 (`qsize()`)
- [ ] Task 수 모니터링 (`asyncio.all_tasks()`)
- [ ] 장시간(30분+) 실행 후 메모리 스냅샷 비교 계획
- **참고**: 외부 서비스(ElevenLabs, Claude API, PostgreSQL) 연결 필요

## 결과물

1. **분석 보고서**: 발견된 문제점 (심각도별), 재현 조건, 영향 범위, 수정 제안
2. **수정 구현 계획**: writing-plans 스킬로 전환하여 단계별 수정 계획 작성

## 심각도 분류 기준

| 심각도 | 기준 |
|--------|------|
| **Critical** | 프로덕션 크래시 또는 데이터 손실 유발 |
| **High** | 장시간 운영 시 메모리 누수 또는 리소스 고갈 |
| **Medium** | 에지 케이스에서 비정상 동작 가능 |
| **Low** | 코드 품질/방어적 프로그래밍 개선 사항 |
