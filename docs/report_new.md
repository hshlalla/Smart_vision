University of London  
Bachelor in Computer Science

Final Project (Updated Final Report)  
Smart Image Part Identifier for Secondhand Platforms  
CM3020 Artificial Intelligence, "Orchestrating AI models to achieve a goal"

Name: SuHun Hong  
Email: hshlalla@naver.com

## 1. Report Purpose

본 문서는 초기 제출본 `docs/Report.md` 이후 실제 구현이 크게 확장된 내용을 최종보고서 제출용으로 정리한 업데이트 문서다.  
핵심은 단순 이미지 유사도 검색에서 다음으로 확장되었다는 점이다.

1. 멀티모달 하이브리드 검색: 이미지 + OCR + 텍스트 + 캡션 + 스펙 토큰
2. 내부 문서 기반 Catalog RAG: PDF 인덱싱/검색
3. Tool-calling Agent: 필요 시 하이브리드 검색/카탈로그 검색/웹 검색/메타 업데이트 실행
4. 운영 가능한 모노레포 구조: `apps/*`, `packages/*`, `data/*`

## 2. Initial Requirements Traceability (Report.md 기준)

### 2.1 Domain Requirements

| ID | 초기 요구사항 | 현재 반영 상태 | 구현 근거 |
|---|---|---|---|
| R1 | Open-world inventory | 구현 | 분류기 재학습이 아닌 Milvus 인덱싱 기반 (`/api/v1/hybrid/index`) |
| R2 | Fine-grained ambiguity 대응 | 구현 | dense + lexical + spec 융합 점수, part number match 반영 |
| R3 | OCR 불확실성 대응 | 구현 | PaddleOCRVL -> PaddleOCR fallback, OCR 없음 시 zero/우회 처리 |
| R4 | 구조화 출력 + 근거 | 구현 | Top-K + score + metadata, catalog source/page 근거 반환 |

### 2.2 User Requirements

| ID | 초기 요구사항 | 현재 반영 상태 | 구현 근거 |
|---|---|---|---|
| U1 | 비전문 사용자 friction 감소 | 구현 | Web UI Search/Index/Chat/Catalog 탭 |
| U2 | Top-K shortlist | 구현 | `top_k` 지원 (`hybrid/search`, `catalog/search`) |
| U3 | 투명성 + 수정 가능 | 부분구현 | Agent 단계별 debug/tool 결과 제공, upsert 가능. UI의 명시적 accept/edit 플로우는 추가 여지 |

### 2.3 Objectives (O1~O5)

| ID | 목표 | 상태 | 코멘트 |
|---|---|---|---|
| O1 | End-to-end MVP | 구현 | 업로드 -> 인덱싱 -> 검색/채팅 동작 |
| O2 | Retrieval 정량평가 | 부분구현 | 운영 파이프라인은 완성, 평가 자동화 리포트는 후속 |
| O3 | OCR 품질 정량평가 | 부분구현 | OCR 경로/로그/fallback 구현, CER 자동집계는 후속 |
| O4 | Latency 측정 | 구현 | 전처리/임베딩/삽입 단계 timing 로그 존재 |
| O5 | 사용자 효율/신뢰 검증 | 부분구현 | 기능 준비 완료, 사용자 연구 재실행 필요 |

## 3. Final Architecture

최종 폴더 구조는 다음과 같다.

- `apps/web`: React + Mantine 기반 모바일 친화 UI
- `apps/api`: FastAPI (auth, hybrid, catalog, agent)
- `apps/demo`: Gradio 디버그 UI
- `packages/model`: `smart_match` 하이브리드 검색 코어
- `data/raw`: 데이터셋 루트
- `docs`: 보고서/릴리즈노트/머메이드 다이어그램

전체/세부 흐름도는 `docs/ARCHITECTURE_MERMAID.md`를 기준 문서로 사용한다.

## 4. Implemented Features (Code-based)

### 4.1 Web Frontend (`apps/web`)

구현된 화면/기능:

1. `SearchPage` (`/app/search`)
- 텍스트 질의 + 이미지 업로드 동시 지원
- 업로드 이미지는 브라우저에서 base64로 변환 후 JSON으로 전송
- API: `POST /api/v1/hybrid/search`

2. `IndexPage` (`/app/index`)
- 이미지 + 메타데이터 업로드
- API: `POST /api/v1/hybrid/index` (`multipart/form-data`)

3. `AgentChatPage` (`/app/chat`)
- 질문 + 선택 이미지 입력
- tool 실행 결과(debug), sources, identified 정보 노출
- API: `POST /api/v1/agent/chat`

4. `CatalogPage` (`/app/catalog`)
- PDF 업로드 인덱싱
- 카탈로그 질의/필터(`model_id`, `part_number`) 검색
- API: `POST /api/v1/catalog/index_pdf`, `POST /api/v1/catalog/search`

5. 인증 흐름
- `auth status -> login -> me` 체크
- `AUTH_ENABLED=false`면 익명 허용

### 4.2 API Layer (`apps/api`)

등록된 라우터:

- `auth`: `/api/v1/auth/*`
- `hybrid`: `/api/v1/hybrid/index`, `/api/v1/hybrid/search`
- `catalog`: `/api/v1/catalog/index_pdf`, `/api/v1/catalog/search`
- `agent`: `/api/v1/agent/chat`

핵심 동작:

1. 하이브리드 인덱싱
- `model_id` 필수 검증
- 이미지 전처리(OCR/임베딩/캡션) 후 Milvus upsert

2. 하이브리드 검색
- `query_text`, `image_base64`, `part_number`, `top_k`
- 이미지 없이 텍스트만 검색 가능
- 이미지 있으면 이미지+텍스트 멀티채널 검색

3. Catalog RAG
- PDF 텍스트 추출 + OCR fallback
- chunk 임베딩 후 `catalog_chunks` 저장
- dense + lexical + spec 혼합 점수 반환

4. Agent 응답 후처리
- catalog 사용 시 source/page 근거 강제
- web_search 사용 시 source URL 추출

### 4.3 Model Core (`packages/model/smart_match`)

핵심 컴포넌트:

1. `HybridSearchOrchestrator`
- 인덱싱/검색/모델메타 upsert 단일 진입점

2. 전처리 파이프라인
- OCR: `PaddleOCRVLPipeline` (fallback to PaddleOCR)
- Image embedding: BGE-VL
- Text embedding: BGE-M3
- Captioning: GPT/Qwen (환경변수 기반 on/off/백엔드 선택)

3. 랭킹/퓨전
- dense score(image/ocr/caption/text_query)
- lexical score
- spec match score (예: `16V`, `3A`, part number)
- 최종 blend 점수로 Top-K 정렬

4. Milvus 컬렉션
- `image_parts`, `text_parts`, `attrs_parts`, `model_texts`, `caption_parts`
- Catalog: `catalog_chunks`
- Counter: `sv_counters`

5. 안정성 강화
- 컬렉션 vector field 누락 시 조기 감지 및 명확 에러
- `COUNTERS_COLLECTION` 명 충돌 방지
- 구 스키마 컬렉션과 payload mismatch 대응 로직 강화

#### 4.3.1 Hybrid Search Orchestration (Pseudocode, English)

The implementation is retrieval-first and model-centric. The orchestrator collects candidates from multiple channels, merges them by `model_id`, and then performs two-stage scoring.

```text
Algorithm HybridSearch(query_image, query_text, top_k, part_number):
    if query_image is empty and query_text is empty:
        return []

    initialize model_scores = {}
      # model_id -> {image_sims, ocr_sims, caption_sims, text_query_sims}
    initialize model_images = {}
      # model_id -> image-level evidence list
    search_k = max(top_k, min(100, top_k * 3))

    if query_image exists:
        query_record = preprocess(query_image)

        image_hits = search_images(query_record.image_vector, top_k=search_k)
        for hit in image_hits:
            model_id = get_model_id(hit)
            sim = distance_to_similarity(hit.distance)
            model_scores[model_id].image_sims.append(sim)
            model_images[model_id].append(hit metadata)

        if query_record.ocr_text exists:
            ocr_vec = encode_text(query_record.ocr_text)
            ocr_hits = search_texts(ocr_vec, top_k=search_k)
            for hit in ocr_hits:
                model_scores[get_model_id(hit)].ocr_sims.append(distance_to_similarity(hit.distance))

        if query_record.caption_text exists:
            cap_vec = encode_text(query_record.caption_text)
            cap_hits = search_captions(cap_vec, top_k=search_k)
            for hit in cap_hits:
                model_scores[get_model_id(hit)].caption_sims.append(distance_to_similarity(hit.distance))

    if query_text exists:
        text_vec = encode_text(query_text)
        text_hits = search_models(text_vec, top_k=search_k)
        for hit in text_hits:
            model_scores[str(hit.id)].ocr_sims.append(distance_to_similarity(hit.distance))
            model_scores[str(hit.id)].text_query_sims.append(distance_to_similarity(hit.distance))

    if model_scores is empty:
        return []

    return rank_and_finalize(model_scores, model_images, query_text, part_number, top_k)
```

```text
Algorithm rank_and_finalize(model_scores, model_images, query_text, part_number, top_k):
    for each model_id:
        image_score   = max(image_sims)   or 0
        ocr_score     = max(ocr_sims)     or 0
        caption_score = max(caption_sims) or 0

        # Stage 1: dense fusion using available channels only
        dense = (alpha*image + beta*ocr + gamma*caption) / active_weight_sum

        lexical_score = lexical_match(query_text, metadata+ocr+caption+description)
        spec_score    = spec_match(query_text, part_number, electrical_tokens)
        lexical_hit   = substring_hit(query_text, evidence_text)

        # Stage 2: final blend
        final = 0.65*dense + 0.20*lexical_score + 0.15*spec_score
        if lexical_hit:
            final += 0.05
        final = min(1.0, final)

    if part_number filter is provided:
        keep exact part_number matches when available

    sort by (lexical_hit, final) descending
    return top_k
```

This logic is implemented in `HybridSearchOrchestrator.search()` with explicit score decomposition fields (`image_score`, `ocr_score`, `caption_score`, `lexical_score`, `spec_match_score`) for transparency and UI evidence rendering.

### 4.4 Catalog RAG (Internal Knowledge)

Catalog RAG 구현 내용:

1. 인덱싱
- `pypdf` 기반 페이지 추출
- 텍스트 없는 페이지는 OCR fallback
- chunk 분할 후 BGE-M3 임베딩
- 문서 메타(`source`, `page`, `model_id`, `part_number`, `maker`) 저장

2. 검색
- 질의 임베딩 기반 ANN 검색
- lexical/spec 재점수화
- 결과에 source/page/chunk text 포함

3. Agent 통합
- 내부 부품 매뉴얼/카탈로그 질의 시 `catalog_search` 우선 호출 정책

### 4.5 Agent Tool Orchestration Policy

`apps/api/smart_vision_api/services/agent.py` 기준 툴셋:

- `hybrid_search`
- `vision_identify`
- `web_search`
- `extract_prices`
- `gparts_search_prices`
- `catalog_search`
- (옵션) `allocate_model_id`, `upsert_model_metadata` (`update_milvus=true`일 때)

실행 원칙:

1. 이미지가 있으면 `hybrid_search` 우선
2. 내부 문서형 질문은 `catalog_search` 우선
3. 외부 시세/웹 정보는 `web_search` 계열 사용
4. 매칭 실패 + 업데이트 허용 시 신규 `model_id` 할당 후 메타 업서트
5. catalog 사용 시 답변에 source/page 근거 포함

정리하면, LLM은 항상 툴을 호출하는 것이 아니라 질의 성격에 따라 툴 호출 여부를 결정한다.

### 4.6 Demo (`apps/demo`)

Gradio 데모는 API를 통하지 않고 오케스트레이터를 직접 호출한다.

주요 탭:

- `single_asset_indexing`
- `Search`
- `ocr_preview`
- `ocr_markdown`
- `Milvus Status`

역할:

- API/웹 이슈와 분리해 모델 파이프라인 자체를 독립 검증
- 개발 중 스키마/인덱싱/검색 상태 점검

### 4.7 Runtime/DevOps Improvements

1. 경로 정규화
- 구경로(`smart-vision-api`, `smart-vision-model`, `smart-vision-demo`) -> canonical 구조 이동

2. 실행 스크립트 안정화
- `run_dev.sh`, `run_demo.sh`의 `PYTHONPATH` unbound 에러 방지
- `python -m uvicorn` 방식으로 실행 안정성 향상

3. 의존성 충돌 완화
- `numpy<2` 제약(일부 torch ABI 이슈 회피)
- API/Demo/Model requirements 정합성 보강

4. Milvus 운영 이슈 대응
- schema mismatch/No vector field 에러 원인 가시화
- 오래된 컬렉션 사용 시 drop/recreate 가이드 필요성 문서화

### 4.8 Reliability/Testing/Observability Updates (2026-02-22)

최근 개발 단계에서 API/모델 패키지에 다음 안정화 작업을 추가했다.

1. 실행 스크립트의 `.env` 로딩 안정화
- `apps/api/scripts/run_dev.sh`, `run_prod.sh`, `run_docker.sh`에서 `.env`를 `source` 기반으로 로딩하도록 정리
- `PYTHONPATH`/`IMAGE_NAME`/`CONTAINER_NAME` 미정의로 인한 `set -u` 종료 리스크를 제거
- `MILVUS_URI`는 강제 덮어쓰지 않고 미설정 시 기본값 사용

2. API 이미지 업로드 가시성 강화
- `POST /api/v1/agent/chat`, `POST /api/v1/hybrid/search`, `POST /api/v1/hybrid/index`에 수신/완료/실패 로그를 추가
- 이미지 포함 여부, base64 길이, 결과 건수 등 운영 확인에 필요한 핵심 필드를 로깅
- 프론트(`SearchPage`, `AgentChatPage`)에도 업로드 직전/요청 완료 콘솔 로그를 추가

3. Agent 이미지 처리 내결함성 보강
- 이미지가 업로드된 턴에서 LLM이 툴 호출을 생략하는 경우를 대비해 `hybrid_search` 1회 fallback 실행 로직을 추가
- `request_id`가 존재하면 이미지 재업로드를 요구하지 않도록 시스템 규칙을 강화

4. 테스트 체계 확장 (`pytest`)
- `apps/api/tests/test_agent_chat_api.py` 추가
- `packages/model/tests/test_metadata_normalizer.py` 추가
- `packages/model/tests/test_tracker_dataset.py` 추가
- API 에러(detail) 반환, source dedupe, catalog 근거 보강, CSV 파싱/정규화 등 핵심 회귀 포인트를 자동 검증

5. API 입력 검증 및 예외 처리 개선
- `MAX_IMAGE_BASE64_LENGTH` 설정을 도입해 과대 이미지 payload를 사전에 차단
- `POST /api/v1/agent/chat`, `POST /api/v1/hybrid/search`에서 제한 초과 시 `413`을 명시적으로 반환
- `HTTPException`은 500으로 재래핑하지 않도록 분리해 오류 의미를 유지

6. 로깅 중복 출력 완화
- 커스텀 로거에서 `logger.propagate = False`를 적용해 uvicorn/root logger로의 중복 전파를 차단
- 운영 로그 가독성을 개선하고 동일 이벤트의 중복 라인 출력을 줄임

7. 프론트 업로드 UX 개선
- 이미지 base64 인코딩 전 대용량 파일 자동 리사이즈(해상도/품질 옵션) 적용
- 인코딩 진행률 표시(퍼센트)와 용량 제한 메시지 추가
- 업로드 실패 시 사용자에게 즉시 원인 피드백 제공

8. CI 자동 테스트 파이프라인 추가
- `.github/workflows/tests.yml` 추가
- `apps/api/tests`, `packages/model/tests`를 push/PR 시 자동 실행
- 로컬/원격 테스트 결과 일관성을 확보

9. 작업계획 문서 정리
- 기존 루트 `to_do_list.md`를 `docs/to_do_list.md`로 이관
- 우선순위를 `Now / Next / Later`로 재구성해 실행 가능 항목 중심으로 정리

## 5. Important Behavioral Clarifications

### 5.1 Hybrid Search 이미지 입력 방식

- `index`: 파일 업로드(`multipart`)가 맞다.
- `search`: 서비스에서 이미지를 base64(JSON)로 전송하는 방식이 맞다.
- 따라서 Swagger에서 파일 업로드 입력이 안 보일 수 있으나, 실제 웹 서비스 검색은 이미지 지원된다.

### 5.2 다중 이미지 상품 처리

- 동일 `model_id`에 여러 이미지를 인덱싱 가능
- 검색 시 모델 단위로 후보를 집계/통합해 반환
- 정면/측면 등 다양한 관점 이미지를 같은 상품 증거로 축적 가능

### 5.3 Caption이 안 들어가 보이는 경우

아래 상황이면 캡션이 비활성/약화될 수 있다.

1. `ENABLE_CAPTIONER=false`
2. `CAPTIONER_BACKEND` 설정 및 런타임 의존성 부재
3. OpenAI key 없음 + CUDA 없음 환경에서 로컬 캡셔너 비활성

즉 캡셔닝은 "항상 필수"가 아니라 환경 기반 optional channel이다.

### 5.4 Auth disabled 메시지

`"Auth is disabled on this server."`는 인증이 꺼진 개발 모드에서 정상 동작 메시지다.  
`AUTH_ENABLED=true`로 켜면 로그인 토큰 흐름이 활성화된다.

## 6. Known Gaps / Future Work (최종보고서에 명시 권장)

1. 정량평가 자동화
- Acc@1/5, CER, p95 latency, 사용자 효율 지표 자동 리포트 파이프라인 추가

2. Human-in-the-loop UI 강화
- 후보 확정/수정/반려를 명시적으로 기록하는 승인 UX 추가

3. Reranker 고도화
- cross-encoder 재랭킹 및 confidence calibration

4. 운영 배포 체계
- stage/prod 분리, 모니터링, tracing, 에러 대시보드

5. 데이터 거버넌스
- 내부 문서/부품 메타의 버전관리, 접근권한, 감사로그 강화

## 7. Conclusion

현재 시스템은 초기 제안의 "사진 기반 부품 식별" MVP를 넘어 다음을 충족한다.

1. 멀티모달 하이브리드 검색 엔진 구현
2. 내부 문서 기반 Catalog RAG 구현
3. Tool-calling Agent 기반 자동 질의응답/근거 제시
4. 웹/API/모델/데모가 분리된 재사용 가능한 구조 완성

최종보고서 본문에는 최소한 다음을 포함하는 것이 바람직하다.

1. 요구사항 추적표(R1~R4, U1~U3, O1~O5)
2. 구현 기능 상세(웹/API/모델/RAG/Agent)
3. 한계 및 후속 계획(정량평가/운영화/UX 개선)

## Appendix A. Endpoint Summary

- `POST /api/v1/hybrid/index` (multipart: image + model_id + metadata)
- `POST /api/v1/hybrid/search` (json: query_text, image_base64, top_k, part_number)
- `POST /api/v1/catalog/index_pdf` (multipart: pdf + optional metadata)
- `POST /api/v1/catalog/search` (json)
- `POST /api/v1/agent/chat` (json: message, image_base64, update_milvus)
- `GET /api/v1/auth/status`, `POST /api/v1/auth/login`, `GET /api/v1/auth/me`

## Appendix B. Supporting Documents

- 초기 보고서: `docs/Report.md`
- 업데이트 보고서: `docs/report_new.md`
- 아키텍처 다이어그램: `docs/ARCHITECTURE_MERMAID.md`
- 릴리즈노트:
- `docs/releasenote_api.md`
- `docs/releasenote_model.md`
- `docs/releasenote_demo.md`
- 개발 작업계획: `docs/to_do_list.md`
