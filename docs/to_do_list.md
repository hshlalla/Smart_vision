# Smart Vision To-Do (Updated)

본 문서는 기존 `to_do_list.md`를 현재 코드베이스 기준으로 재정리한 실행 계획이다.  
우선순위는 `High / Medium / Low`, 진행 구간은 `Now / Next / Later`로 관리한다.

## 0) Recent Completed (반영 완료)

- API 실행 스크립트 안정화 (`.env` source 로딩, unbound 변수 방지)
- 업로드 관측성 강화 (API/프론트 업로드 로그)
- Agent 이미지 fallback 보강 (tool-call 생략 시 `hybrid_search` fallback)
- API + Model `pytest` 테스트 기본 세트 구축
- CI 테스트 워크플로우 추가 (`.github/workflows/tests.yml`)
- 프론트 업로드 UX 보강 (용량 제한/리사이즈/인코딩 진행률)

## 1) Now (이번 스프린트)

### 1.1 Evaluation Set 구축 (High)
- 목표: 실제 문제를 재현하는 벤치마크 셋 확보
- 범위:
- 명판 있음/없음 분리
- 카테고리 균형 샘플링
- 한글/영문 혼합 질의 포함
- 산출물:
- `data/` 하위 평가셋 메타 CSV
- 정답 라벨 규칙 문서
- 지표:
- `Recall@1`, `Recall@5`, `MRR`

### 1.2 Fusion Weight 자동 탐색 (High)
- 목표: 수동 α/β/γ를 데이터 기반으로 튜닝
- 범위:
- grid/random search로 `image/ocr/caption/text` 가중치 탐색
- 평가셋 기준 최고 조합 저장
- 산출물:
- 실험 스크립트 + 결과 리포트
- 추천 기본값(.env 또는 설정) 반영안
- 지표:
- baseline 대비 `Recall@5` 개선율

### 1.3 OCR 후처리 강화 (High)
- 목표: 파트넘버/라벨 인식 정확도 개선
- 범위:
- 문자열 정규화 룰(대소문자/특수문자)
- 레이아웃 기반 그룹화(라인/블록)
- part-number 후보 우선 추출
- 산출물:
- 후처리 모듈 + 테스트 케이스
- 지표:
- 파트넘버 매칭 정확도, false positive 비율

### 1.4 1000건 오류 분석 파이프라인 (High)
- 목표: 모듈별 병목/오류 원인 가시화
- 범위:
- OCR/캡션/임베딩/랭킹 단계별 실패 코드 분류
- 샘플 1000건 통계 리포트 자동 생성
- 산출물:
- 분석 스크립트 + 에러 taxonomy 문서
- 지표:
- 상위 5개 오류군 비중, 개선 후 재측정

## 2) Next (다음 스프린트)

### 2.1 Cross-Encoder Reranker PoC (High)
- 목표: top-k 재정렬 정확도 향상
- 후보:
- `bge-reranker` 계열 또는 동급 경량 모델
- 범위:
- top-20 재정렬 후 top-5 비교
- 지표:
- `NDCG@5`, `Recall@5`, latency 증가량

### 2.2 Milvus 파라미터 튜닝 (Medium)
- 목표: 정확도/지연시간 균형 최적화
- 범위:
- HNSW `M`, `efConstruction`, query `ef` 실험
- modality별 최적 값 후보 도출
- 지표:
- `p95 latency`, `Recall@5`

### 2.3 Captioner 운영 옵션 재검토 (Medium)
- 목표: 속도 병목 완화
- 범위:
- Qwen 계열 경량 대체안, 토큰 동적 조절
- 품질 유지 가능한 최소 설정 탐색
- 지표:
- 평균/`p95` 응답시간, 정확도 하락폭

### 2.4 API 테스트 확장 (Medium)
- 목표: 회귀 방지 강화
- 범위:
- `catalog/index_pdf`, `hybrid/index` 경계 케이스
- 대용량/비정상 파일 입력 테스트
- 지표:
- 엔드포인트별 테스트 커버리지

## 3) Later (연구/중장기)

### 3.1 Domain Finetuning (Medium)
- LoRA/contrastive 학습 실험
- 데이터 준비 완료 후 진행

### 3.2 BGE-VL Text Tower 통합 PoC (Low)
- BGE-M3 대체 가능성 검증
- 롱폼 입력 제한 대응 전략 포함

### 3.3 Learn-to-Rank 정식 도입 (Low)
- Pointwise/Pairwise/Listwise 중 데이터 규모 맞춰 선택

### 3.4 운영 고도화 (Medium)
- 작업 큐(Celery/Redis), 모니터링(Prometheus), DR 백업 정책

## 4) Execution Rules

- 각 항목은 반드시 “지표 + 완료조건(DoD)”를 함께 정의한다.
- 실험 백로그와 프로덕션 백로그를 분리해 관리한다.
- 릴리즈 반영 시 `docs/releasenote_*.md`에 동일 변경을 동기화한다.
