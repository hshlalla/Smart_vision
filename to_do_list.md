# To-Do List (Pipelines & Models)

우선순위는 “🔥 High”, “⚙️ Medium”, “🧊 Low” 로 표시했습니다.

## 1. Preprocessing & Pipeline Orchestration
- 🔥 **비동기 파이프라인 설계**: 이미지 임베딩, OCR, 캡션을 순차 처리하고 있어 GPU/CPU 자원이 놀고 있다. asyncio 또는 멀티프로세싱 워커를 활용해 I/O 바운드 단계(OCR, 캡션)와 GPU 단계(비전 인코더)를 병렬화.

## 2. OCR (PaddleOCR-VL)

- ⚙️ **후처리 개선**: 현재 단순 문자열 조합 → 라벨 정규화(대소문자/특수문자) + 레이아웃 기반 그룹화를 도입해 파트 넘버 후보를 우선 추출.


- 🔥 **Domain Finetuning**: 반도체 장비 데이터로 Low-Rank Adapter(LoRA) 미세조정 실험. 샘플 수 적을 경우 contrastive learning으로 이미지-메타 쌍을 추가 학습.
- 🔥 **Text Tower 통합 실험**: BGE-VL은 `get_text_features()`를 제공하므로 OCR/메타 텍스트를 동일한 임베딩 공간으로 사영할 수 있다. `BGEVLTextEncoder`(신규)로 텍스트 임베딩을 생성하고, Milvus `text_collection`을 재색인해 BGE-M3 의존성을 제거하는 PoC 진행.
- ⚙️ **Resolution 최적화**: 현재 기본 해상도 사용. 384→512 등 업샘플 시 성능/속도 trade-off 측정, GPU 메모리 기반 자동 선택 기능 추가.
- 🧊 **양자화 옵션**: INT8/FP8 변환으로 추론 속도 최적화. 성능 저하 허용 범위를 벤치마크.

## 4. Captioner (Qwen3-VL)
- 한번에 여러장을 보고 캡션이 가능한지 확인할것. vit 계열에 fetch를 사용하는 모델을 찾아보면 있을 수 있음.



- 🔥 **Prompt 엔지니어링**: 산업 장비 특화 키워드(Labels, Ports, Condition)를 강조하는 시스템 프롬프트 세트 실험, 현재 설명을 시켰을때 너무 자세하게 설명하기때문에 텍스트길이가 길고 속도가 느린점, 가장 적은말로 설명하고 정확도를높이는것이 필요.
- 🔥 **추론 지연 최적화**: Qwen3-VL-8B 추론이 병목이라 현재 프로덕션에서 비활성화되어 있으므로, (1) 8bit quantization/ 적용으로 Latency를 줄이거나 (2) 더 작은 Qwen2-VL 2B·Phi-3 Vision·LLaVA-NeXT 등 경량 모델을 대체 옵션으로 검증.
- ⚙️ **Multimodal Judge 실험**: Qwen3-VL을 captioner뿐 아니라 judge 모델로도 활용해 `image + OCR 텍스트 + metadata` 조합이 일관된지 검증. 색인 전/후 샘플을 평가해 잘못된 매칭(메타 오류, OCR 누락)을 자동 감지하는 파이프라인 PoC 진행.
- ⚙️ **Fallback 품질 개선**: Qwen3 로드 실패 시 현재 Qwen2 7B만 사용 → local lightweight 모델(Blip2, MiniGPT-4)도 옵션으로 추가해 모델 추가실험 필요
- ⚙️ **토큰/시간 최적화**: `max_new_tokens=256` 고정 → 이미지 복잡도/메타데이터 길이에 따라 동적으로 조정하여 평균 처리시간 감소.

## 5. Text Encoder (BGE-M3)
캡셔닝, 텍스트가 sementic search 가 되는지 확인할것. report 필요. 유명하지 않은 회사들까지 커버가능한지 한번 확인해볼것, 영어로 표현되는것, 한글로표현되는것이 같이 표현되고 있는지
- 언어 교차 질문 가능한지 확인할것. 


- 🔥 **Dual-encoder 재학습**: OCR 텍스트·캡션과 Tracker 메타 텍스트를 Positive pair로 사용해 domain finetuning 이후 Milvus 리랭크 정확도 측정.
- ⚙️ **특정 필드 Weighting**: `maker`, `part_number` 등 필드를 재조합해 text_corpus에 삽입 시 가중치 태그(`[part_number]`)로 검색 시 중요도를 제어.
- 🧊 **Long-form Handling**: 4K 토큰 넘는 설명을 요약 후 인코딩할 Summarizer 모듈 추가.
- 🧊 **Long-form 유지 근거**: BGE-VL 텍스트 타워는 512 토큰 제한이라 긴 OCR/설명 텍스트에 불리함. 따라서 BGE-M3를 계속 사용해 롱폼 입력을 안정적으로 커버하고, BGE-VL 통합은 요약/슬라이딩 윈도우 전략과 함께 단계적으로 검토.

## 6. Retrieval & Ranking
- 🔥 **FusionWeight 학습**: 현재 수동 α/β/γ → 라벨된 검색 세트로 grid search 또는 Learn-to-Rank 모델을 학습해 자동 최적화.
- 🔥 **Cross-Encoder 도입**: place-holder인 `noop` 대신 BAAI/bge-reranker-large 혹은 Cohere ReRank를 붙여 top-k 재정렬. (이부분은 gpt가 개선안으로 뽑아준것인데 이해하지 못함. 공부필요)

reranking 논문 찾아볼것

- ⚙️ **Milvus 파라미터 최적화**: HNSW `M`, `efConstruction`, 검색 시 `ef` 값을 워크로드별로 튜닝하고, 컬렉션별로 다른 파라미터(이미지 vs 텍스트)를 허용.
- HNSW (Hierarchical Navigable Small World): 그래프 기반 근사 최근접 탐색(ANN) 인덱스입니다. 벡터를 여러 층의 스몰월드 네트워크로 구성해 빠르게 이웃 후보를 찾습니다. 대량 벡터(수십만~수천만)에서도 높은 검색 속도/정확도를 유지하는 Milvus 기본 옵션 중 하나입니다.
- M: 그래프에서 각 노드(벡터)가 몇 개의 이웃과 연결될지 결정합니다. 값이 크면 연결이 촘촘해져 정확도는 올라가지만 메모리/빌드 시간이 늘어납니다. 일반적으로 16~64 사이에서 데이터 규모에 맞춰 조정합니다.
- efConstruction: 인덱스를 만들 때 탐색할 후보 수입니다. 크면 그래프 품질이 좋아져 검색 정확도가 향상되지만, 인덱스 구축 시간이 길어집니다.
- ef (검색 시): 쿼리할 때 몇 개의 후보를 살펴볼지 정하는 값입니다. 높게 잡으면 더 많은 후보를 확인하므로 정확도가 올라가지만 응답 시간이 늘어나죠. 요청 유형(실시간 API vs 배치 분석)에 따라 다르게 설정할 수 있습니다.

- ⚙️ **단일 컬렉션 실험**: BGE-VL 이미지/텍스트 벡터 차원이 동일해지면 `hybrid_collection` 하나에 modality 필드를 두고 저장 가능. 이 경우 인덱스 수가 줄어드는 대신 검색 시 modality별 top-k 뽑는 로직(파티션 or 필터)이 필요하므로 PoC 후 성능/운영 이점 비교.
- 🧊 **Learn-to-Rank 도입 메모**: 
- Pointwise: 각 문서가 얼마나 relevant한지 회귀/분류로 예측하고 점수를 정렬.
- Pairwise: 두 문서 중 어느 쪽이 더 relevant인지 (승/패) 학습, 예: RankNet.
- Listwise: 전체 리스트 단위로 NDCG 같은 지표를 직접 최적화, 예: LambdaMART, ListNet.
Pointwise/Pairwise/Listwise 접근으로 검색 결과를 재정렬하는 모델을 의미. 이미지·OCR·캡션 스코어, 메타 필터 여부 등을 feature로 삼고, 사람이 라벨한 랭킹(정답 순위)을 학습 데이터로 만들어 자동으로 최적 가중치를 학습한다.


## 8. API / 서비스 계층
- ⚙️ **배치 모드 최적화**: `HybridSearchOrchestrator`에 bulk 색인 시 progress callback, 재시작 지점 저장 기능 추가.
- ⚙️ **스루풋 모니터링**: FastAPI + Celery/Redis 조합으로 async 작업 큐 도입, Prometheus exporter로 단계별 처리시간 수집.

## 9. 운영/배포
- 🔥 **모델 캐시 관리**: Qwen/BGE 체크포인트를 shared volume에 캐시하고, 버전 업데이트 시 checksum 검사로 재다운로드 방지.
- ⚙️ **GPU 메모리 자동 감지**: 실행 시 VRAM 용량을 읽어 모델 로딩 정책(fp16/32, batch size)을 자동 조정.
- 🧊 **Disaster Recovery**: Milvus 컬렉션 백업/복원 스크립트와 S3 snapshot 정책 문서화.




순위.
1. ocr 마키나락스 담당
2. test datset 구성 -> 학습데이터 추후 구성 필요.
3. bge-vl, bge-m3, qwen3 vl, reranking
4. 결과값 퓨전하는것을 찾아봐야함.
5. 단위별 성능테스트 필요.
6. demo 최적화 (직관적으로 사용할수 있도록 예시 넣어주기)
7. 유저용 데모 필요.
8. 오류 분석. 1000개 정도 진행하고 결과보는것이 필요. 각 모듈별로 어디가 문제인지 파악하는것 필요.




명판이 없는 case 별로 test dataset를 만들고 성능분석
3가지 버전 
1. 카테고리별로 sampling 한 후에 dateset구성




- 상호 보환적인 부분이 있을수 있고, 둘다 못하는게 있을수 있음.
- 어떻게 fusion을 할수 있는지 확인해야한다.
- embedding 
- meta search 방법 찾아볼것.
- 이미지 

목표
1. 점수상승
2. responds time reduce
3. model 최적화 (gpu)

