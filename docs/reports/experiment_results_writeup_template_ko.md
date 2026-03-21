# 실험 결과 작성 템플릿

이 문서는 최종 리포트 5장 Evaluation 파트에 바로 반영할 수 있도록 만든 결과 작성 템플릿이다.  
실험이 끝나면 수치만 채워 넣고, 필요하면 문장을 조금 다듬어서 [`final_report_docx_ready_ko.md`](/Users/mac/project/Smart_vision/docs/reports/final_report_docx_ready_ko.md)에 반영하면 된다.

## 1. Evaluation Opening Paragraph

아래 문단은 Evaluation 장 첫머리에 넣는 기본 문장이다.

> 평가는 이 시스템의 실제 목적, 즉 사용자가 업로드한 부품 이미지로부터 관련 후보를 shortlist 안에서 찾고, 그 근거를 이해하며, listing workflow에 활용할 수 있는지를 중심으로 설계되었다. 이에 따라 평가는 retrieval effectiveness, OCR robustness, latency and resource cost, engineering reliability, 그리고 pilot usability의 다섯 영역으로 구성되었다. 다만 최종 구현 단계에서 current-index sanity benchmark와 sampled image-holdout benchmark를 분리해 해석하였다. 전자는 운영 상태 점검용이고, 후자는 더 엄격한 sampled controlled comparison 용도로 사용하였다.

## 2. Dataset Paragraph

> 평가에는 통합 데이터셋 `unified_v1`을 사용하였다. 전체 데이터 수는 **1491개**이며, 이 중 **1192개**를 index set, **299개**를 query/test set으로 사용하는 current-index sanity benchmark를 먼저 수행하였다. 다만 reranker/OCR 비교에는 stricter image-holdout protocol이 더 적절하다고 판단하여, 별도 sampled benchmark도 추가로 수행하였다. 이 sampled benchmark는 `30`개 item을 뽑아 각 item에서 `1`장의 held-out query image와 `1`장의 indexed image를 사용하였다.

## 3. E1 Retrieval Effectiveness

### 3.1 목적 문단

> Retrieval effectiveness 실험의 목적은 multimodal hybrid pipeline이 단일 신호 기반 baseline보다 더 나은 shortlist 품질을 제공하는지 평가하는 것이다. 특히 본 실험은 OCR 사용 여부, reranker 사용 여부, text channel 유지 여부가 검색 정확도에 어떤 영향을 주는지 비교하는 데 초점을 둔다.

### 3.2 설정 문단

> 비교 구성은 다음과 같다. `C1`은 OCR 없이 `Qwen3-VL image embedding + BGE-M3 text channel + reranker`를 사용하는 mixed pipeline이다. `C2`는 여기에 OCR evidence를 추가한 구성으로 계획되었다. `C3`는 `C1`에서 reranker를 제거한 ablation이다. `C4`는 text channel을 최소화한 image-heavy/light-text 구성으로 계획되었다. 현재 완료된 controlled comparison은 `C1`과 `C3`이며, `C2`와 `C4`는 후속 과제로 남았다.

### 3.3 결과 표

표 제목:

`Table X. Retrieval effectiveness comparison across hybrid-search configurations.`

권장 열:
- Configuration
- Accuracy@1
- Accuracy@5
- Recall@5
- MRR
- Exact identifier hit rate

해석 문단 템플릿:

> 표 X는 retrieval effectiveness 비교 결과를 보여준다. sampled image-holdout benchmark에서 `C1`과 `C3`는 모두 Accuracy@1, Accuracy@5, Recall@5, MRR에서 `1.0`을 기록했다. 그러나 exact item-level identifier hit rate는 두 구성 모두 `0.9667`로 동일했다. 즉 이번 sampled task에서는 reranker를 추가해도 retrieval quality 자체는 향상되지 않았다. 오히려 유일한 오차는 두 구성 모두 동일한 `group_key` 내 visually similar variant confusion에서 발생했으며, 이는 reranker가 이 near-duplicate pair를 분리하지 못했음을 보여준다.

### 3.4 핵심 해석 문단

> 이 결과는 본 프로젝트의 핵심 주장, 즉 중고 부품 식별은 단순 이미지 분류가 아니라 retrieval-first, evidence-aware ranking 문제라는 관점을 지지한다. 특히 sampled holdout에서 `C3` baseline만으로도 shortlist 품질이 매우 높게 유지되었고, reranker는 추가적인 품질 향상 없이 큰 비용만 초래했다. 따라서 현 단계에서는 “더 복잡한 후처리”보다 “충분히 강한 first-stage retrieval + human review shortlist”가 더 실용적이라는 해석이 가능하다.

## 4. E2 OCR Robustness

### 4.1 목적 문단

> OCR robustness 실험은 OCR을 primary signal로 유지해야 하는지, 아니면 uncertain evidence로 제한해야 하는지를 판단하기 위해 수행되었다. 초기 full OCR+Qwen pilot은 local time budget 안에 완료되지 못했기 때문에, 먼저 `10`-sample OCR-only pilot을 수행해 PaddleOCR-VL의 identifier recovery 정도를 확인하였다.

### 4.2 결과 표

표 제목:

`Table X. OCR robustness results on the identifier-visible subset.`

권장 열:
- Method
- CER
- Exact full-string match
- Part number recall
- Maker recall

해석 문단 템플릿:

> 표 X에서 볼 수 있듯이 OCR-only pilot에서 PaddleOCR-VL의 part number exact match와 recall은 각각 `0.1`에 머물렀고, maker recall도 `0.1` 수준이었다. 조건을 단순화해 보면 OCR text가 실제로 보이는 `3`개 샘플에서는 exact match가 `0.3333`까지 올라갔지만, blur 또는 image-block markdown이 우세한 `7`개 샘플에서는 exact match와 recall이 모두 `0.0`이었다. 이는 OCR을 단독 identifier source로 사용하기 어렵고, uncertain evidence 또는 fallback signal로 제한하는 현재 설계 방향이 더 타당함을 시사한다.

### 4.3 실패 사례 문단

> 정성적 실패 사례를 보면 문제는 단순 문자 인식 실패만이 아니었다. 여러 자동차 부품 샘플에서는 OCR 결과가 실제 label text 대신 `img_in_image_box` markdown이나 흐릿한 숫자열 위주로 반환되었다. 반면 산업용 차단기처럼 라벨이 크고 정면에 가까운 사례에서는 `20811M-081`, `Fuji Electric` 같은 문자열이 안정적으로 추출되었다. 따라서 향후 개선 방향은 OCR 모델 교체만이 아니라, label-focused crop, multi-view capture, evidence fusion 개선을 포함해야 한다.

## 5. E3 Latency and Resource Benchmark

### 5.1 목적 문단

> Latency benchmark의 목적은 제안한 파이프라인이 실제 interactive workflow에 사용할 수 있을 정도의 응답성을 제공하는지 평가하는 것이다. 이를 위해 indexing과 query latency를 각각 측정하고, cold/warm 조건과 resource cost를 함께 기록하였다.

### 5.2 결과 표

표 제목:

`Table X. Latency and resource benchmark for indexing and query execution.`

권장 열:
- Operation
- Condition
- Mean
- p50
- p90
- p95
- Notes

권장 행:
- Indexing
- Query total
- Query preprocessing
- Query embedding
- Retrieval
- Rerank

해석 문단 템플릿:

> 표 X에 따르면 `C3` baseline의 warm query 전체 평균 시간은 `731.13 ms`, p95는 `751.85 ms`였고, 가장 큰 병목은 query preprocessing(`698.96 ms`)이었다. 반면 `C1` reranker-on의 warm query 전체 평균 시간은 `89337.71 ms`, p95는 `96477.25 ms`로 급증했으며, reranker가 지배하는 finalize 단계만 평균 `65199.13 ms`를 차지했다. 이는 현재 시스템이 reranker를 제외한 baseline 경로에서는 prototype 수준의 interactive support에 도달했지만, 현재 local Apple Silicon 환경에서 reranker-on은 production-grade responsiveness와 거리가 멀다는 점을 보여준다.

## 6. E4 Pilot Usability

### 6.1 목적 문단

> Pilot usability study의 목적은 시스템이 실제 listing assistance 관점에서 유용한지, 그리고 사용자가 shortlist와 evidence를 신뢰하고 활용할 수 있는지를 확인하는 것이다. 참가자는 주어진 task를 수행한 뒤 task time, edit count, external search usage를 기록하고, 이후 Likert-scale 설문에 응답하였다.

### 6.2 결과 표

표 제목:

`Table X. Pilot usability study summary.`

권장 열:
- Metric
- Result

권장 행:
- Participants
- Mean task completion time
- Mean manual edit count
- External search usage rate
- Mean trust score
- Mean usability score

해석 문단 템플릿:

> Pilot usability 결과는 사용자가 시스템을 완전 자동 판정기로 보지 않더라도, shortlist narrowing과 metadata drafting 보조 도구로는 충분한 가치를 느꼈음을 보여준다. 특히 **[예: metadata preview usefulness]** 항목의 평균 점수가 높게 나타난 반면, **[예: trust in final answer]**는 상대적으로 낮아, human confirmation 중심 설계가 여전히 필요함을 시사한다.

## 7. E5 Engineering Reliability

### 7.1 문단 템플릿

> Retrieval 및 usability 중심 평가와 별도로, engineering reliability 근거도 함께 제시하였다. API regression test는 `21 passed`, model regression test는 `11 passed`를 통과하였고, frontend production build도 성공하였다. 또한 evaluation input generation은 `index 1192 / query 299` manifest를 정상 생성하였다. 이는 recent changes, especially preview-confirm indexing flow, writeback safety, and experiment input generation이 기본 회귀 수준에서 안정적임을 보여준다.

## 8. Critical Discussion

### 8.1 성공 요약 문단

> 종합적으로 본 실험 결과는 본 시스템이 단순 이미지 분류기보다 retrieval-first identification assistant로서 더 적절하다는 점을 지지한다. 현재 sampled holdout 결과는 baseline `C3`만으로도 매우 강한 shortlist 품질을 달성했으며, 사용자의 실제 워크플로우를 고려하면 자동 정답 하나보다 evidence-backed shortlist가 더 현실적인 지원 형태임을 확인할 수 있었다. 반면 reranker는 이번 local setup에서 정확도 향상 없이 latency cost만 크게 증가시켰다.

### 8.2 한계 문단

> 동시에 한계도 분명했다. OCR은 여전히 노이즈와 blur에 취약했고, local-only prototype 환경에서는 reranker latency가 매우 컸으며, 일부 multimodal 경로는 Apple Silicon에서 `mps` 대신 `cpu` fallback을 필요로 했다. 또한 usability 평가는 아직 수행되지 않았기 때문에, 더 큰 사용자 집단에 대한 일반화에는 주의가 필요하다.

### 8.3 개선 방향 문단

> 향후에는 region-focused OCR, multi-view evidence aggregation, more stable GPU deployment, and audited writeback workflow를 우선 개선 대상으로 삼을 수 있다. 특히 OCR은 전체 장면을 그대로 읽기보다 label-centric crop이 필요하며, reranker는 현재 local hardware에서는 비효율적이므로 더 작은 모델이나 다른 deployment target을 검토할 필요가 있다. 이 방향은 단순히 accuracy를 높이는 것뿐 아니라, 실제 listing workflow에서의 trust와 operability를 함께 개선하는 데 중요하다.
