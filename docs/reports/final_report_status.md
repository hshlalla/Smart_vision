University of London  
Bachelor in Computer Science

Final Project (Updated Final Report)  
Smart Image Part Identifier for Secondhand Platforms  
CM3020 Artificial Intelligence, "Orchestrating AI models to achieve a goal"

Name: SuHun Hong  
Email: hshlalla@naver.com

## 0. Document Role

이 문서는 최종보고서 제출 파일 자체가 아니라, 현재 코드베이스와 피드백을 기준으로 최종보고서에 반영할 내용을 정리하는 작업용 상태 문서다.

최종보고서 정리 시 기준 문서는 아래와 같다.

1. 제출본 기준선: `submission/reports/Draft.docx`
2. 피드백: `submission/feedback/preliminary_feedback.md`, `submission/feedback/draft_feedback.md`
3. 제출 지침: `submission/guides/final_report_guide.md`
4. 재현 증적: `submission/evidence/report_support_2026-03-10/`
5. 실험 상태 요약: `experiments/CURRENT_EXPERIMENT_STATUS.md`

`docs/reports/report_working_reference.md`는 과거 작업용 장문 reference이고, 현재 문서가 실제 최종정리용 기준이다.

## 1. Report Purpose

본 문서는 초기 작업용 보고서 reference(`docs/reports/report_working_reference.md`) 이후 실제 구현이 크게 확장된 내용을 최종보고서 제출용으로 정리한 업데이트 문서다.  
핵심은 단순 이미지 유사도 검색에서 다음으로 확장되었다는 점이다.

1. 멀티모달 하이브리드 검색: 이미지 + OCR + 텍스트 + 캡션 + 스펙 토큰
2. 내부 문서 기반 Catalog RAG: PDF 인덱싱/검색
3. Tool-calling Agent: 필요 시 하이브리드 검색/카탈로그 검색/웹 검색/메타 업데이트 실행
4. 운영 가능한 모노레포 구조: `apps/*`, `packages/*`, `data/*`

최종 제출 단계에서는 “무엇을 더 만들 수 있었는가”보다, 현재 프로토타입이 원래 목표를 얼마나 잘 달성했는지를 평가하는 서술이 더 중요하다. 이는 final report와 video demo가 함께 채점되고 서로 cross-reference된다는 점에서도 중요하다.

## 2. Initial Requirements Traceability (작업용 초기 보고서 기준)

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
| O1 | End-to-end MVP | 구현 | 업로드 -> 인덱싱 -> 검색/채팅 동작. `experiments/CURRENT_EXPERIMENT_STATUS.md` 기준 current-index suite에서 `8/8` scenario success 확보 |
| O2 | Retrieval 정량평가 | 부분구현 | current-index sanity benchmark와 sampled holdout `C1/C3` 비교는 완료. 다만 `C2`, `C4`, 더 큰 controlled ablation은 후속 |
| O3 | OCR 품질 정량평가 | 부분구현 | OCR-only `10`-sample pilot 완료. 다만 full OCR+Qwen benchmark와 OCR-on retrieval comparison은 후속 |
| O4 | Latency 측정 | 부분구현 | `C1`, `C3` 단계별 latency와 p50/p90/p95는 확보. 다만 indexing benchmark와 peak resource summary는 후속 |
| O5 | 사용자 효율/신뢰 검증 | 부분구현 | 기능 준비 완료, pilot protocol 정의 완료. aggregate 사용자 연구 결과는 후속 |

## 3. Final Architecture

최종 폴더 구조는 다음과 같다.

- `apps/web`: React + Mantine 기반 모바일 친화 UI
- `apps/api`: FastAPI (auth, hybrid, catalog, agent)
- `apps/demo`: Gradio 디버그 UI
- `packages/model`: `smart_match` 하이브리드 검색 코어
- `data/raw`: 데이터셋 루트
- `docs`: 보고서/릴리즈노트/머메이드 다이어그램
- `data/datasets/unified_v1`: 최종 통합 dataset 및 train/test split, eval manifest

전체/세부 흐름도는 `docs/architecture/ARCHITECTURE_MERMAID.md`를 기준 문서로 사용한다.

## 4. Implemented Features (Code-based)

### 4.1 Web Frontend (`apps/web`)

구현된 화면/기능:

1. `SearchPage` (`/app/search`)
- 텍스트 질의 + 이미지 업로드 동시 지원
- 업로드 이미지는 브라우저에서 base64로 변환 후 JSON으로 전송
- API: `POST /api/v1/hybrid/search`

2. `IndexPage` (`/app/index`)
- 하나 이상의 이미지 업로드
- GPT 기반 metadata preview 후 사용자 수정/confirm
- 실제 저장은 confirm 이후 비동기 인덱싱
- API: `POST /api/v1/hybrid/index/preview`, `POST /api/v1/hybrid/index/confirm`

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
- preview 단계에서 GPT metadata draft 생성
- preview 단계에서 part number 기반 duplicate candidate 확인 가능
- confirm 단계에서 `model_id` 자동 할당 가능
- 사용자가 기존 모델 append 또는 새 모델 유지 선택 가능
- 여러 장 이미지 전처리(OCR/임베딩/캡션) 후 모델 단위 Milvus upsert

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
- 반복 인입 시 기존 모델 메타 merge와 새 이미지 append를 지원

2. 전처리 파이프라인
- OCR: `PaddleOCRVLPipeline` (fallback to PaddleOCR)
- Image embedding: `Qwen3-VL-Embedding-2B`
- Text embedding: `BGE-M3`
- Reranker: `Qwen3-VL-Reranker-2B`
- Captioning: `Qwen3-VL-2B-Instruct` (환경변수 기반 on/off/백엔드 선택)

3. 랭킹/퓨전
- dense score(image/ocr/caption/text_query)
- lexical score
- spec match score (예: `16V`, `3A`, part number)
- 최종 blend 점수 + `Qwen3-VL-Reranker-2B` top-N 재정렬

7. 최신 모델 스택 반영(2026-03-13)
- 초기 구현은 `BGE-VL-large` + `BGE-M3` 중심이었지만, CPU/저사양 환경에서 지연이 커 실사용 응답성이 떨어졌다.
- 이에 따라 하이브리드 검색 스택을 `Qwen3-VL-Embedding-2B`(image embedding), `BGE-M3`(text embedding), `Qwen3-VL-Reranker-2B`(re-ranking), `Qwen3-VL-2B-Instruct`(captioning) 조합으로 재구성했다.
- 이 변경은 최신 Qwen3-VL 계열을 이미지 표현과 cross-modal reranking에 사용하되, 텍스트 채널은 기존에 검증된 `BGE-M3`를 유지해 OCR/metadata 검색 안정성을 지키려는 절충 설계를 반영한 것이다.
- 모델 교체로 인해 Milvus 벡터 공간과 차원이 일부 바뀌므로 기존 BGE-VL 기반 이미지 컬렉션은 재사용하지 않고, 새 컬렉션(`qwen3_vl_image_parts`, `bge_m3_*`)으로 분리해 재인덱싱하는 전략을 채택했다.
- 최종보고서에는 단순히 "최신 모델을 사용했다"가 아니라, 왜 BGE 스택에서 Qwen3-VL 스택으로 옮겼는지, 그리고 운영 제약(CPU latency, model family consistency, reranker integration) 때문에 어떤 설계 결정을 했는지를 명시해야 한다.

8. 설계 질의응답 기반 보강 근거(2026-03-13)
- 질문 1: "기존 파이프라인에서 실제 reranker를 쓰고 있었는가?"  
  답: 아니었다. 기존 구조는 `BGE-VL-large` 이미지 임베딩 + `BGE-M3` 텍스트 임베딩 + lexical/spec 보정으로 정렬했고, `fusion_retriever`의 rerank 인터페이스는 있었지만 실제 cross-encoder는 `_noop_cross_encoder`라 실질적인 2-stage re-ranking은 없었다.
- 질문 2: "이미 `Qwen3-VL-2B-Instruct`를 캡셔닝에 쓰고 있었다면 변화 폭이 큰가?"  
  답: 캡션 경로 자체는 이미 Qwen 계열을 쓰고 있었으므로 변화의 핵심은 captioner가 아니라 image embedding과 reranking 단계였다. 즉 체감상 중요한 변화는 `BGE-VL` 제거와 실제 multimodal reranker 도입이다.
- 질문 3: "텍스트 인코더도 Qwen3-VL 계열로 통일해야 하는가?"  
  답: 반드시 그렇지는 않다. OCR, maker, part number, description처럼 exact lexical evidence가 중요한 부품 도메인에서는 `BGE-M3`를 유지하는 편이 안전하다. 따라서 현재 설계는 image 쪽은 Qwen3-VL, text 쪽은 `BGE-M3`를 유지하는 혼합 전략을 채택했다.
- 질문 4: "Qwen3-VL이 OCR에 강하다면 OCR 모델을 없애도 되는가?"  
  답: 바로 단정하지 않았다. Qwen3-VL은 OCR/문서 이해 능력이 강하지만, 부품 검색에서는 serial, model name, maker text 같은 exact string이 중요하므로 OCR 제거 여부는 실험으로 판단해야 한다. 현 시점에서는 OCR을 보조 evidence/fallback으로 유지하는 편이 더 방어적이다.
- 질문 5: "YOLO로 물체를 크롭한 뒤 OCR을 써야 하는가?"  
  답: 단일 부품 사진 중심이라면 처음부터 YOLO를 필수로 넣는 것은 과설계일 수 있다. 복수 물체/배경 잡음/라벨 영역이 매우 작은 경우에만 후보 실험으로 두는 것이 적절하다.

9. 비교 실험 계획(설계 선택을 위한 ablation)
- 실험 A: `Qwen3-VL-Embedding-2B` + `Qwen3-VL-Reranker-2B` + `Qwen3-VL-2B-Instruct`를 중심으로 사용하고, OCR은 제거하거나 최소한의 fallback으로만 유지한다.
- 실험 B: `Qwen3-VL-Embedding-2B` + `Qwen3-VL-Reranker-2B` + `Qwen3-VL-2B-Instruct` + `BGE-M3` + OCR을 함께 유지한다.
- 평가 데이터셋: 유사한 외형의 제품 1000개를 고정하고, 900개는 index set, 100개는 held-out test/query set으로 사용한다.
- 비교 지표:
  - Top-K retrieval quality
  - exact part number / maker / short label match recall
  - latency (indexing/search)
  - 실사용 관점의 응답 안정성
- 기대 차이:
  - 실험 A는 파이프라인 단순화와 모델 패밀리 일관성이 강점이다.
  - 실험 B는 exact text evidence를 더 안정적으로 활용할 가능성이 높다.
- 보고서 서술 포인트:
  - "Qwen3-VL이 최신 SOTA 계열이라 무조건 교체"가 아니라,
  - "부품 검색 도메인에서 exact OCR evidence의 가치와 multimodal consistency 사이의 trade-off를 비교 실험으로 검증한다"는 점을 명시해야 한다.

6. 검색 랭킹 보정(2026-03-13)
- 한글 질의 토큰화 지원을 추가했다. 기존 lexical 토크나이저가 영문/숫자만 인식해 `홍수훈` 같은 한국어 질의는 exact/substring 보정이 거의 반영되지 않았다.
- exact substring match의 가중치를 높였다. 실제 사용자 기대는 "등록된 설명/제조사/모델명과 질의가 그대로 일치하면 상위로 와야 한다"는 것이므로, dense similarity만으로 순위를 정하는 것은 UX와 맞지 않았다.
- `model_id`, `maker`, `part_number`, `description` 필드에 대한 exact/partial match boost를 추가했다. 이는 구조화 메타데이터를 단순 저장 용도뿐 아니라 ranking evidence로 활용하도록 한 조치다.
- 매우 낮은 점수 후보를 제거하는 최소 점수 컷오프를 추가했다. 이전에는 절대 점수가 0.12 수준이어도 반환 후보군 내부 상대순위 때문에 1위가 될 수 있었는데, 이는 사용자가 "관련 없는 결과가 1위"라고 느끼게 만드는 원인이었다.
- 이 변경은 멀티모달 dense retrieval만으로는 open-world 장비 검색의 사용자 기대를 충족하기 어렵고, 특히 사람 이름/브랜드명/짧은 부품명처럼 exact lexical evidence가 강한 질의에서는 명시적 규칙 기반 보정이 필요하다는 구현 교훈을 반영한다.

4. Milvus 컬렉션
- `qwen3_vl_image_parts`, `bge_m3_text_parts`, `attrs_parts_v2`, `bge_m3_model_texts`, `bge_m3_caption_parts`
- Catalog: `bge_m3_catalog_chunks`
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

### 4.9 Dataset Consolidation and Evaluation Preparation (2026-03-17)

- 최종 통합 데이터셋 `unified_v1` 구축 완료
- 전체 항목 수: `1491`
- split: `train 1192 / test 299`
- 도메인:
  - `auto_part`
  - `semiconductor_equipment_part`
- retrieval evaluation 입력 생성 완료:
  - `index_manifest.jsonl`
  - `query_manifest.jsonl`
  - `eval_summary.json`
- 실험 runner가 수치만 채우면 바로 평가 가능한 상태로 정리됨
- 주의: 이는 평가 **입력 준비 완료**를 의미하며, E1/E2/E3/E4의 정량 실험이 이미 완료되었다는 뜻은 아님
- 결과 입력은 `docs/reports/final_experiment_results_fill_template_ko.md`를 기준으로 채운다
- 실제 실행된 current-index / sampled / reliability run 상태는 `experiments/CURRENT_EXPERIMENT_STATUS.md`를 기준으로 보완한다

### 4.10 Usability Pilot Preparation

- 사용성 평가는 아직 완료되지 않았으나, 소규모 pilot 계획은 정의됨
- 3개 task 기반 절차:
  - image-based identification
  - shortlist judgement
  - metadata preview/edit/confirm
- 측정 항목:
  - task time
  - correctness
  - edit count
  - external search usage
  - self-reported trust/usability
- post-task Google Form 문항 초안 작성 완료

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
- 기존 루트 `to_do_list.md`를 `docs/planning/to_do_list.md`로 이관
- 우선순위를 `Now / Next / Later`로 재구성해 실행 가능 항목 중심으로 정리

## 4.11 Experiment Runner Status (2026-03-21)

`experiments/` 폴더에는 최종보고서용 재현 실험 러너와 상태 요약이 정리되어 있다. 다만 raw run output(`experiments/runs/...`)은 실행 머신의 local-output 성격이라 현재 git 작업트리에는 포함되지 않을 수 있다. 따라서 보고서 문구는 raw path가 아니라 `experiments/CURRENT_EXPERIMENT_STATUS.md`에 정리된 수치와 결론을 기준으로 맞춰야 한다.

현재 확인된 실행 상태는 다음과 같다.

- `E0`: current-index suite 기준 `8/8` scenario success
- `E1`: current-index sanity benchmark(`Accuracy@1=0.9732`, `Hit@5=1.0`, `MRR=0.9855`)와 sampled holdout `C3` baseline(`item_id@1=0.9667`) 확보
- `E3`: current-index warm total mean 약 `653.65 ms`, sampled `C3` warm total mean 약 `731.13 ms` 확보
- `E5`: reliability refresh 기준 API/model tests, frontend build, retrieval-eval input generation 정상

다만 아래 항목은 아직 최종 completed evidence로 쓰면 안 된다.

- full controlled `C1/C2/C3/C4` retrieval comparison
- OCR CER / exact-match aggregate benchmark
- latency percentile/resource summary
- human-review pilot aggregate results

즉 현재 상태는 “아무 실험도 안 했다”가 아니라, operational sanity / sampled baseline / engineering reliability 증거는 이미 있고, stricter controlled comparison이 남아 있는 상태다.

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

- 작업용 장문 보고서 reference: `docs/reports/report_working_reference.md`
- 현재 최종보고서 상태 문서: `docs/reports/final_report_status.md`
- 아키텍처 다이어그램: `docs/architecture/ARCHITECTURE_MERMAID.md`
- 릴리즈노트:
- `docs/release_notes/releasenote_api.md`
- `docs/release_notes/releasenote_model.md`
- `docs/release_notes/releasenote_demo.md`
- 개발 작업계획: `docs/planning/to_do_list.md`
- 제출본/피드백/가이드/증적: `submission/README.md` 참고
