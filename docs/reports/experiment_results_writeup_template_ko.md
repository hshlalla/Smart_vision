# 실험 결과 작성 템플릿

이 문서는 최종 리포트 5장 Evaluation 파트에 바로 반영할 수 있도록 만든 결과 작성 템플릿이다.  
실험이 끝나면 수치만 채워 넣고, 필요하면 문장을 조금 다듬어서 [`final_report_docx_ready_ko.md`](/Users/mac/project/Smart_vision/docs/reports/final_report_docx_ready_ko.md)에 반영하면 된다.

## 1. Evaluation Opening Paragraph

아래 문단은 Evaluation 장 첫머리에 넣는 기본 문장이다.

> 평가는 이 시스템의 실제 목적, 즉 사용자가 업로드한 부품 이미지로부터 관련 후보를 shortlist 안에서 찾고, 그 근거를 이해하며, listing workflow에 활용할 수 있는지를 중심으로 설계되었다. 이에 따라 평가는 retrieval effectiveness, OCR robustness, latency and resource cost, engineering reliability, 그리고 pilot usability의 다섯 영역으로 구성되었다. 모든 retrieval 실험은 고정된 unified dataset split 위에서 수행되었으며, index set과 query set은 사전에 분리되어 재현 가능하도록 manifest 형태로 고정하였다.

## 2. Dataset Paragraph

> 평가에는 통합 데이터셋 `unified_v1`을 사용하였다. 전체 데이터 수는 **1491개**이며, 이 중 **1192개**를 index set, **299개**를 query/test set으로 사용하였다. 도메인 분포는 자동차 부품 `1012개`, 반도체 장비용 부품 `479개`이다. split은 단순 무작위가 아니라 `part_number`와 유사 제품군 기준으로 그룹화하여 동일 또는 거의 동일한 항목이 train/test 양쪽에 동시에 등장하지 않도록 구성하였다.

## 3. E1 Retrieval Effectiveness

### 3.1 목적 문단

> Retrieval effectiveness 실험의 목적은 multimodal hybrid pipeline이 단일 신호 기반 baseline보다 더 나은 shortlist 품질을 제공하는지 평가하는 것이다. 특히 본 실험은 OCR 사용 여부, reranker 사용 여부, text channel 유지 여부가 검색 정확도에 어떤 영향을 주는지 비교하는 데 초점을 둔다.

### 3.2 설정 문단

> 비교 구성은 다음과 같다. `C1`은 OCR 없이 `Qwen3-VL image embedding + GPT/Qwen-derived text + BGE-M3 text channel + reranker`를 사용하는 mixed pipeline이다. `C2`는 여기에 OCR evidence를 추가한 구성이다. `C3`는 `C1`에서 reranker를 제거한 ablation이다. `C4`는 text channel을 최소화한 image-heavy/light-text 구성이다.

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

> 표 X는 retrieval effectiveness 비교 결과를 보여준다. 가장 중요한 지표인 Accuracy@1은 **[best config]**에서 가장 높게 나타났고, Accuracy@5와 Recall@5 역시 같은 구성에서 가장 우수했다. 이는 **[핵심 원인: 예: OCR evidence / reranker / text channel]**가 shortlist 품질 개선에 실질적으로 기여했음을 시사한다. 반면 **[worst config]**는 visually similar 항목 구분에서 상대적으로 약했으며, 이는 단일 신호만으로는 fine-grained identification을 안정적으로 수행하기 어렵다는 점을 보여준다.

### 3.4 핵심 해석 문단

> 이 결과는 본 프로젝트의 핵심 주장, 즉 중고 부품 식별은 단순 이미지 분류가 아니라 retrieval-first, evidence-aware ranking 문제라는 관점을 지지한다. 특히 Top-1보다 Top-5 계열 지표가 중요한 이유는 실제 사용자 워크플로우가 단일 자동 정답보다 shortlist 기반 human review를 전제로 하기 때문이다.

## 4. E2 OCR Robustness

### 4.1 목적 문단

> OCR robustness 실험은 OCR을 primary signal로 유지해야 하는지, 아니면 uncertain evidence로 제한해야 하는지를 판단하기 위해 수행되었다. 이 실험은 identifier visibility가 있는 subset을 사용해 part number와 maker 추출 성능을 정량화한다.

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

> 표 X에서 볼 수 있듯이 OCR 성능은 clean single-object 조건에서는 비교적 양호했지만, blur, glare, 작은 라벨, partial occlusion이 있는 경우 급격히 저하되었다. 특히 **[가장 취약한 조건]**에서 exact match가 크게 하락하였다. 이는 OCR을 단독 ground truth로 사용하는 대신, multimodal retrieval evidence의 한 요소로 취급하는 현재 설계 방향이 타당함을 뒷받침한다.

### 4.3 실패 사례 문단

> 정성적 실패 사례를 보면 문제는 단순 문자 인식 실패만이 아니었다. 실제로는 stylised label, background clutter, reflective surface, multi-object scene이 복합적으로 작용하여 identifier extraction을 어렵게 만들었다. 따라서 향후 개선 방향은 OCR 모델 교체뿐 아니라 region focus, multi-view capture, evidence fusion 개선을 포함해야 한다.

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

> 표 X에 따르면 warm query 조건에서 전체 응답 시간은 **[p50]** 수준이었고, p95는 **[value]**까지 증가하였다. 가장 큰 병목은 **[예: embedding / rerank / OCR]** 단계였다. 이는 현재 시스템이 prototype 수준의 interactive support에는 도달했지만, production-grade responsiveness를 위해서는 추가 최적화가 필요함을 보여준다.

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

> Retrieval 및 usability 중심 평가와 별도로, engineering reliability 근거도 함께 제시하였다. API regression test는 **[value]**, model regression test는 **[value]**를 통과하였다. 이는 최근 변경사항, 특히 preview-confirm indexing flow, writeback safety, dataset preparation, evaluation input generation이 기본 회귀 수준에서 안정적임을 보여준다.

## 8. Critical Discussion

### 8.1 성공 요약 문단

> 종합적으로 본 실험 결과는 본 시스템이 단순 이미지 분류기보다 retrieval-first identification assistant로서 더 적절하다는 점을 지지한다. 특히 multimodal evidence fusion과 reranking은 shortlist 품질 향상에 실질적으로 기여했으며, 사용자 입장에서는 자동 정답 하나보다 evidence-backed shortlist가 더 현실적인 지원 형태임을 확인할 수 있었다.

### 8.2 한계 문단

> 동시에 한계도 분명했다. OCR은 여전히 노이즈에 취약했고, latency는 local-only prototype 환경에서 병목이 존재했으며, 일부 환경에서는 runtime dependency 제약이 관찰되었다. 또한 usability 평가는 pilot 규모에 머물렀기 때문에, 더 큰 사용자 집단에 대한 일반화에는 주의가 필요하다.

### 8.3 개선 방향 문단

> 향후에는 region-focused OCR, multi-view evidence aggregation, more stable GPU deployment, and audited writeback workflow를 우선 개선 대상으로 삼을 수 있다. 이 방향은 단순히 accuracy를 높이는 것뿐 아니라, 실제 listing workflow에서의 trust와 operability를 함께 개선하는 데 중요하다.

