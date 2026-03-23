# 4. Implementation

이 장은 3장에서 제시한 설계가 실제 코드 수준에서 어떻게 구현되었는지를 설명한다. runtime environment, storage setup, model integration, asynchronous indexing backend, 그리고 개발 과정에서 마주친 주요 engineering challenge와 대응 방안을 다룬다.

## 4.1 Development Environment and Tech Stack

프로젝트는 monorepo 형태로 구현되었다. 이를 통해 web application, API service, reusable model package가 함께 진화하면서도 논리적으로는 분리된 상태를 유지할 수 있었다.

- **Frontend (`apps/web`)**  
  주 사용자 인터페이스는 React, TypeScript, Vite를 사용해 구현했다. UI는 Mantine 기반 컴포넌트를 활용하여 이미지 업로드, metadata preview, Top-K result inspection, catalog interaction, agent-assisted workflow를 지원한다.

- **Backend Orchestrator (`apps/api`)**  
  백엔드는 Python과 FastAPI로 구현되었다. request validation, authentication, preview-confirm indexing, asynchronous task status reporting, hybrid search routing을 담당한다.

- **AI Package (`packages/model`)**  
  재사용 가능한 `smart_match` 패키지 안에 multimodal retrieval pipeline을 구현했다. 이 계층은 metadata normalisation, OCR, embedding generation, score fusion, reranking, Milvus integration을 포함한다.

[Insert Figure 4.1: Repository or module overview]  
*(작성 가이드: repository structure 또는 module relationship diagram 삽입)*

**Figure 4.1. `apps/web`, `apps/api`, `packages/model`이 통합 prototype 안에서 어떻게 연결되는지를 보여주는 implementation-level module overview.**

이 구현 방식의 장점은 multimodal retrieval logic을 web layer와 독립적으로 테스트할 수 있으면서도, 동시에 end-to-end 통합 시스템으로 작동하게 할 수 있다는 점이다.

## 4.2 Vector Database Setup (Milvus)

Milvus는 단일 벡터만 검색하는 저장소가 아니라, 서로 다른 종류의 표현을 함께 검색해야 하는 본 시스템의 핵심 retrieval store로 사용되었다. 구현에서는 다음과 같은 신호를 separate collection 형태로 저장한다.

1. **Image vectors**  
   Qwen3-VL image encoder가 생성한 벡터
2. **OCR text vectors**  
   BGE-M3가 임베딩한 OCR-derived text 벡터
3. **Caption vectors**  
   caption text를 BGE-M3로 임베딩한 벡터
4. **Attribute records**  
   `maker`, `part_number`, `category`, `image_path` 같은 structured metadata
5. **Model-level text vectors**  
   part/model 단위로 병합된 metadata와 text evidence

즉 현재 구현은 단일 inventory table이 아니라 multi-collection Milvus layout을 사용한다. 구현 수준의 컬렉션 구조는 Table 4.1에 정리했다.

**Table 4.1. 하이브리드 retrieval 파이프라인에서 사용하는 Milvus 컬렉션 구조**

| Collection role | 현재 collection 계열 | 주요 내용 | Vector/index 전략 |
| --- | --- | --- | --- |
| Image retrieval | `qwen3_vl_image_parts` | Qwen3-VL 기반 primary image embedding | 런타임에 결정되는 image embedding dimension, `HNSW + COSINE` |
| OCR/text retrieval | `bge_m3_text_parts` | OCR-derived 또는 metadata-derived text embedding | `1024`차원 text vector, `HNSW + COSINE` |
| Caption retrieval | `bge_m3_caption_parts` | caption text embedding | `1024`차원 text vector, `HNSW + COSINE` |
| Model-level retrieval | `bge_m3_model_texts` | model 단위로 병합된 metadata text, caption text, OCR text | `1024`차원 text vector, `HNSW + COSINE` |
| Structured attributes | `attrs_parts_v2` | `maker`, `part_number`, `category`, `image_path` 등 구조화 메타데이터 | 작은 vector placeholder를 포함한 attribute collection, `FLAT + L2` |

*Table note:* image-vector 차원은 active Qwen3-VL encoder가 런타임에 결정하며, BGE-M3 기반 text path는 `1024`차원 embedding space를 사용한다.

이 구조 덕분에 backend는 visually similar item을 찾는 동시에, lexical 또는 identifier evidence가 맞는 후보에 가중치를 줄 수 있다. 또한 text-only search path는 full multimodal stack을 호출하지 않고 model-level text collection을 직접 조회할 수 있어 더 가볍게 동작한다.

추가적으로 `model_id` counter는 Milvus 내부가 아니라 lightweight SQLite-backed namespace로 관리되도록 구현했다. 이는 single-node 실험 환경에서 구현 복잡성을 줄이는 데 도움이 되었다.

## 4.3 AI Model Integration and Orchestration Pipeline

단일 user-generated image만으로 산업 부품을 식별하는 일은 단순한 classification 문제가 아니다. 실제로는 global visual form, embedded text, normalized metadata, caption-like description, user query terms에 단서가 분산되어 있다. 따라서 모델 계층은 하나의 monolithic classifier가 아니라, `packages/model` 내부의 orchestrated pipeline으로 구현되었다.

이 orchestration layer는 다음 핵심 구성요소를 통합한다.

1. **Qwen3-VL Image Encoder (Primary Visual Perception)**  
   Qwen3-VL은 primary visual engine으로 사용된다. hybrid retrieval에 사용되는 dense image representation을 생성하며, global shape, material cue, layout, visually embedded identifier region을 함께 반영한다.

2. **BGE-M3 Text Encoder (Semantic and Lexical Embedding)**  
   BGE-M3는 OCR-derived text, user-facing metadata, merged model text, caption, text-only search를 모두 처리한다. 이를 통해 image similarity뿐 아니라 짧은 alphanumeric identifier와 listing-oriented metadata field도 함께 검색할 수 있다.

3. **PaddleOCR (Conditional Text Extraction)**  
   OCR은 고정된 mandatory stage가 아니라 optional component로 구현되었다. 런타임에서는 indexing-time OCR과 query-time OCR을 각각 제어할 수 있어, 운영 모드에 따라 evidence recovery와 noise/latency 사이의 균형을 조정할 수 있다.

4. **Contextual Captioning Layer**  
   caption generation은 runtime mode와 environment configuration에 따라 GPT 또는 local Qwen captioner를 사용한다. 생성된 caption은 다시 BGE-M3로 임베딩되어 human-readable semantic signal로 활용된다.

핵심 engineering challenge는 여러 모델을 단순히 불러오는 것이 아니라, 이를 하나의 preprocessing/search stack으로 동기화하는 일이었다. 현재 구현에서는 `PreprocessingPipeline`이 central coordinator 역할을 수행한다. 즉, image를 visual encoder에 전달하고, 필요한 경우 caption generation과 OCR을 호출하며, 추출된 field를 정규화하고, 최종적으로 Milvus indexing 또는 query에 사용할 standardised record를 반환한다. 이러한 orchestration 구조는 산업 부품 식별에 필요한 evidence가 visual form, embedded text, metadata, user query text에 분산되어 있기 때문에 필수적이었다.

[Insert Figure 4.2: Code snippet illustrating the multimodal retrieval stack]

```python
# Simplified orchestration setup
self.vision_encoder = Qwen3VLImageEncoder()
self.text_encoder = BGEM3TextEncoder()

if self.enable_ocr_indexing or self.enable_ocr_query:
    self.ocr_engine = PaddleOCRVLPipeline()
else:
    self.ocr_engine = None

self.preprocessing = PreprocessingPipeline(
    vision_encoder=self.vision_encoder,
    ocr_engine=self.ocr_engine,
    text_encoder=self.text_encoder,
    metadata_normalizer=self.metadata_normalizer,
    captioner=self.captioner,
)
```

**Figure 4.2. Qwen3-VL image encoding, BGE-M3 text encoding, conditional OCR, caption-aware preprocessing을 포함하는 multimodal preprocessing stack의 단순화된 orchestration setup.**

## 4.4 Asynchronous Indexing and the Preview-Confirm Backend Architecture

4장의 중요한 구현 과제 중 하나는 무거운 machine-learning inference와 웹 애플리케이션이 요구하는 responsiveness를 동시에 만족시키는 것이었다. image decoding, caption generation, embedding creation, Milvus upsert는 일반적인 request validation이나 CRUD-style API 처리보다 훨씬 느리다. 만약 이 모든 작업을 confirm request 안에서 inline으로 처리하면, 인터페이스가 느려지고 반복 업로드 상황에서 indexing workflow의 안정성도 떨어진다.

이를 해결하기 위해 백엔드는 human-in-the-loop workflow와 직접 맞물리는 decoupled preview-confirm architecture로 구현되었다.

### 4.4.1 Preview-Confirm Workflow

인덱싱 흐름은 두 개의 backend phase로 나뉜다.

- **preview step**에서는 업로드 이미지를 임시로 decode하고, draft metadata를 생성하며, 초안 식별자가 기존 indexed part와 맞는 경우 possible duplicate candidate를 함께 반환할 수 있다.
- **confirm step**에서는 사용자가 검토한 metadata를 정규화한 뒤, 더 무거운 indexing job을 inline 실행하지 않고 asynchronous queue에 넣는다.
- 이후 frontend는 task endpoint를 polling하면서 작업이 terminal state에 도달할 때까지 상태를 추적한다.

이 구조는 실제 데이터가 index에 기록되기 전에 review 가능한 human-in-the-loop checkpoint를 유지하면서도 UI responsiveness를 확보한다.

### 4.4.2 Queueing Model and Task-State Tracking

구현은 단순한 `FastAPI BackgroundTasks` 패턴이 아니라 internal `ThreadPoolExecutor` 기반 queue와 in-memory task registry를 사용한다. prototype 환경에서는 별도의 broker-backed task system을 두지 않고도 non-blocking confirm 동작을 구현할 수 있다는 점에서 이 방식이 충분히 실용적이었다.

```python
task_id = uuid.uuid4().hex
self._set_task(
    _IndexTask(
        task_id=task_id,
        status="queued",
        model_id=model_id,
        detail="Indexing job queued.",
        created_at=time.time(),
        updated_at=time.time(),
    )
)
self._executor.submit(
    self._run_confirm_index_job,
    task_id,
    list(image_b64_list),
    dict(cleaned),
    ocr_image_indices,
)
return {"status": "queued", "model_id": model_id, "task_id": task_id}
```

**Figure 4.3. API가 즉시 `task_id`를 반환하고 frontend가 task polling으로 진행 상태를 추적하는 confirm-stage asynchronous queueing logic.**

## 4.5 Search Implementation and Hybrid Retrieval Routing

애플리케이션의 핵심 동작은 검색 요청이 retrieval stack 안에서 어떻게 라우팅되는가에 달려 있다. 실제 사용자 입력은 매우 다양하다. 어떤 사용자는 흐릿한 사진만 올리고, 어떤 사용자는 이미지와 drafted metadata를 함께 제공하며, 어떤 사용자는 이미 알고 있는 part number만으로 검색한다. 이 모든 요청을 가장 비싼 경로로 강제하지 않기 위해, 구현은 두 가지 큰 검색 경로를 제공한다.

- **Full multimodal search (heavy path)**  
  이미지 기반 요청을 처리하며, 필요하면 query text도 함께 반영한다. 이 경로는 image embedding, optional OCR text, caption-derived semantic signal, lexical overlap, exact identifier boost를 결합한다.

- **Lightweight text-only search (fast path)**  
  시각 입력 없이 텍스트 기반 요청만 있을 때 사용된다. 이 경우 API는 full multimodal preprocessing stack을 거치지 않고, 더 작은 BGE-M3-driven model-level text collection을 직접 조회한다.

구현에는 explicit ranking control도 포함된다. lexical matching 개선, exact substring boosting, low-score filtering 같은 조정은, 산업 부품 retrieval이 약한 embedding뿐 아니라 exact identifier가 존재하는데도 ranking에서 충분히 반영되지 않는 문제를 해결하는 데 중요했다.

### 4.5.1 Hybrid Scoring Logic

상위 수준에서 보면 retrieval pipeline은 여러 evidence channel에서 candidate model을 모은 뒤, 먼저 dense base score를 계산하고 그 위에 metadata-aware adjustment를 적용한다. base dense score는 image, OCR-text, caption similarity의 가중 결합으로 계산된다.

\[
\text{dense\_score} =
\frac{\alpha \cdot s_{\text{image}} + \beta \cdot s_{\text{ocr}} + \gamma \cdot s_{\text{caption}}}
{\alpha + \beta + \gamma}
\]

현재 구현에서는 해당 evidence가 존재할 때 `alpha = 0.5`, `beta = 0.3`, `gamma = 0.2`를 사용한다. 이는 visual signal을 중심에 두되, 필요할 때 OCR-derived 및 caption-derived evidence도 보조적으로 반영하려는 설계 의도를 보여준다. 이후 lexical overlap, specification-aware matching, `maker`, `part_number`, `model_id` 같은 structured field에 대한 exact-field boost를 추가로 계산해 dense score를 보정한다.

단순화하면 최종 ranking 단계는 다음과 같이 요약할 수 있다.

\[
\text{final\_score} =
0.45 \cdot \text{dense\_score}
+ 0.20 \cdot \text{lexical\_score}
+ 0.15 \cdot \text{spec\_match}
+ 0.20 \cdot \text{exact\_field\_boost}
\]

여기에 direct lexical hit나 exact field match가 감지되면 추가 bonus adjustment가 적용된다. 즉 global semantic similarity가 초기 candidate neighbourhood를 형성하고, 필요할 경우 exact identifier evidence가 purely visual closeness를 넘어 ranking을 재정렬한다. 반대로 lexical이나 exact-field support 없이 minimum threshold를 넘지 못하는 후보는 최종 Top-K ranking 전에 제거된다.

### 4.5.2 Simplified Search Pseudocode

```text
Algorithm 1. Hybrid multimodal retrieval and scoring pipeline
입력: query_image?, query_text?, top_k

1. query_image가 있으면:
   a. image preprocessing -> image embedding, optional OCR text, optional caption 생성
   b. image collection 검색
   c. OCR text가 있으면 text collection 검색
   d. caption이 있으면 caption collection 검색

2. query_text가 있으면:
   a. BGE-M3로 query text 인코딩
   b. model-level text collection 검색

3. model_id 기준으로 candidate 병합

4. 각 candidate에 대해:
   a. image/OCR/caption similarity로 dense fusion score 계산
   b. query text와 model text field의 lexical score 계산
   c. exact-field boost와 specification-match score 계산
   d. 여러 신호를 결합해 final_score 계산
   e. minimum threshold를 넘지 못하는 low-confidence candidate 제거

5. 남은 candidate를 final_score 기준으로 정렬

6. 필요하면 상위 후보에만 Qwen3-VL reranker 적용

7. evidence field와 representative image를 포함한 Top-K 결과 반환
```

[Insert Figure 4.4: Example of hybrid score decomposition and ranked output]  
**Figure 4.4. exact identifier evidence가 시각적으로 유사하지만 lexical evidence가 약한 후보를 넘어 올바른 후보를 상위로 끌어올리는 hybrid score decomposition 예시.**

## 4.6 Technical Challenges and Engineering Mitigations

### 4.6.1 Local Compute Constraints

**Challenge:** 무거운 multimodal model은 반복적 개발 과정에서 실행 비용이 크며, 특히 non-CUDA local environment에서는 OCR, image embedding, reranking이 모두 병목이 될 수 있다.

**Mitigation:** expensive component를 toggle하거나 isolate할 수 있도록 설계했다. OCR은 disable 가능하고, reranking도 optional로 둘 수 있으며, metadata preview는 runtime mode에 따라 다른 backend를 선택할 수 있다. 이러한 modularity 덕분에 full heavy path를 매 iteration마다 실행하기 어려운 상황에서도 개발과 디버깅을 지속할 수 있었다.

### 4.6.2 Balancing Speed and Accuracy

**Challenge:** listing assistant는 interactive use에 충분할 만큼 빠르게 응답해야 한다. 그러나 OCR, image embedding, captioning, reranking을 매 요청마다 모두 실행하면 latency가 과도하게 커질 수 있다.

**Mitigation:** 구현 단계에서 여러 runtime configuration을 지원하도록 했다. 이를 통해 OCR-rich path와 faster vision-dominant path를 실제로 비교하고, accuracy 하나가 아니라 quality-latency trade-off를 바탕으로 operating recommendation을 도출할 수 있었다. Chapter 5의 결과는 stronger evidence recovery가 특별히 필요한 상황이 아니라면, 더 빠른 C3-style operating mode가 practical default에 가깝다는 점을 보여준다.

### 4.6.3 Viewpoint Variance and Incomplete Visual Evidence

**Challenge:** 앞면 이미지에는 side view, rear view, label view가 가진 식별 정보가 없을 수 있다. 이로 인해 indexed image와 query image 사이에 mismatch가 생긴다.

**Mitigation:** multi-image indexing을 도입해 하나의 item이 여러 complementary view로 표현될 수 있게 했다. 또한 metadata-aware 및 text-aware retrieval signal이 visual similarity만으로 부족한 경우를 보완하도록 했다.

### 4.6.4 Repeated Ingestion and Duplicate Management

**Challenge:** 실제 listing workflow에서는 동일한 part가 더 좋은 이미지나 더 풍부한 metadata와 함께 다시 업로드될 수 있다. 이를 모두 새로운 item으로 처리하면 index가 파편화된다.

**Mitigation:** duplicate-aware ingestion을 구현했다. preview 단계에서 backend는 drafted identifier를 기준으로 duplicate candidate를 반환할 수 있고, confirm 단계에서는 사용자가 기존 `model_id`에 merge할지 별도 항목으로 둘지를 선택할 수 있다. 이는 ambiguous case를 human review로 남기면서도 index consistency를 유지하는 데 도움이 된다.

## 4.7 Implementation Status Summary

현재 prototype 단계에서 구현된 시스템은 multimodal search, duplicate-aware indexing, catalog retrieval, agent-assisted evidence expansion을 연결하는 end-to-end workflow를 지원한다. 아직 production-grade marketplace platform은 아니지만, controlled experiment, user-facing demonstration, 그리고 본 프로젝트의 핵심 architectural decision에 대한 비판적 평가를 수행하기에는 충분한 완성도를 갖추고 있다.
