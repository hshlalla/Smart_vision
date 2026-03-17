# CM3070 기출문제 답안집 (한국어 버전)

## 0. 프로젝트 핵심 요약

이 문서는 **Smart Image Part Identifier for Secondhand Platforms** 프로젝트를 기준으로 작성한 CM3070 온라인 시험 대비용 한국어 답안 초안이다.
이 프로젝트는 **CM3020 Artificial Intelligence, "Orchestrating AI models to achieve a goal"** 템플릿을 따른다.
핵심 아이디어는 중고 플랫폼에서 산업용/전자 부품을 사진으로 식별하는 문제를 **closed-set classification** 문제가 아니라, **retrieval-first, human-in-the-loop 의사결정 지원 문제**로 다루는 것이다.

현재 시스템은 다음 요소를 결합한다.

- 이미지 기반 검색
- OCR 기반 텍스트 증거 추출
- 텍스트 및 메타데이터 검색
- 캡션 기반 검색
- Milvus 벡터 저장소
- PDF 기반 카탈로그 검색
- 여러 도구를 오케스트레이션하고 근거를 노출하는 agent 계층

현재 결과를 가장 정직하게 표현하면 다음과 같다.

> 이 프로젝트는 중고 부품 식별 지원을 위한, 기술적으로 도전적인 end-to-end 프로토타입이며, 평가 근거는 의미 있게 존재하지만 아직 일부는 진행 중이다.

## 여러 문제에 공통으로 재사용할 핵심 포인트

### 강점

- 문제 적합성이 높다. 개방형(open-world), 미세 구분(fine-grained), 텍스트 민감(text-sensitive)한 부품 식별은 실제로 어려운 문제다.
- 시스템 차원의 기여가 분명하다. OCR, 멀티모달 검색, 카탈로그 검색, agent orchestration을 하나의 워크플로우로 묶었다.
- 설계 framing이 정직하다. 자동 확정 시스템이 아니라 shortlist + evidence + user verification 구조를 택했다.
- 현장형 safeguard가 있다. 중복처럼 보이는 재등록 입력을 자동 폐기하지 않고 review 후 merge할 수 있다.
- 웹, API, 모델 계층을 포함하는 실제 동작 프로토타입이 있다.
- 최근 작업으로 안전성(safety)과 테스트 가능성(testability)이 개선되었다.

### 한계

- OCR은 glare, blur, 작은 글자, 가림(occlusion)에 여전히 취약하다.
- 전체 accept/edit/reject 검수 워크플로우는 아직 완성되지 않았다.
- CER/WER 집계, latency percentile, 전체 ablation 실험은 아직 완전히 끝나지 않았다.
- 따라서 현재 평가는 의미는 있지만, 최종 production 수준 주장을 뒷받침하기엔 아직 불완전하다.

### 기술적 도전과제 요약 문장

가장 어려운 기술적 문제는 서로 다른 형태의 라벨을 안정적으로 읽고,
그 불완전한 OCR/VLM 신호와 본체 이미지 신호를 함께 결합해 올바른 후보를 고르는 일이었다.
문서용 OCR은 라벨 식별에 바로 최적화되어 있지 않았고, VLM을 도입해도 훼손되거나 지워진 라벨은 여전히 어려웠다.
게다가 본체 모양까지 거의 동일하고 라벨만 다른 경우에는 이미지 단서만으로는 분별력이 약했고,
여러 모델의 weighted sum은 정확도뿐 아니라 latency까지 함께 고민해야 하는 문제였다.


### 평가 관련 요약 문장

현재 확보된 평가 근거는 다음과 같다.

- retrieval-first 접근의 정성적/기초적 근거
- API 및 model pytest 기반 회귀 검증(regression evidence)
- hybrid search 경로에 대한 latency instrumentation
- limitation과 next step을 포함한 비판적 분석

아직 미완료인 항목은 다음과 같다.

- aggregate CER benchmark
- p50/p90/p95 latency summary
- large controlled hybrid ablation
- user study rerun

---

# 1. CM3070 Exam Mar 2025

## Q1. Project process

### (a) 프로젝트 수행의 네 단계(the four D’s)는 무엇인가?

프로젝트의 네 단계는 다음과 같이 정리할 수 있다.

1. **Discover**: 문제, 사용자, 도메인 제약, 관련 선행연구를 이해하는 단계
2. **Design**: 요구사항, 아키텍처, 평가 기준, 구현 계획으로 구체화하는 단계
3. **Develop**: 프로토타입을 구현하고, 테스트하고, 통합하며, 반복적으로 개선하는 단계
4. **Discuss**: 결과를 비판적으로 평가하고, 한계와 기여, 향후 확장 가능성을 정리하는 단계

내 프로젝트에서는 이 네 단계가 최종 보고서 구조와도 자연스럽게 연결된다. Discover는 문제 정의와 literature review에 대응하고, Design은 retrieval-first 아키텍처와 evaluation strategy를 정리하며, Develop은 실제 웹/API/모델 시스템 구현으로 이어졌고, Discuss는 evaluation 및 conclusion 장에서 나타난다.

### (b) 각 단계에서 내 프로젝트에서 무엇을 했는가?

**Discover 단계**에서는 중고 플랫폼에서 산업용/전자 부품을 식별하는 문제를 분석했다. 이 도메인은 open-world이며, visually similar한 부품이 많고, 실제 구분 단서는 part number나 model code 같은 작은 텍스트인 경우가 많다는 점을 파악했다. 또한 이미지 검색, OCR, 멀티모달 임베딩, 벡터 검색, marketplace assistance와 관련된 선행연구를 검토했다. 이 과정을 통해 단순한 closed-set classifier는 적합하지 않다고 판단했다.

**Design 단계**에서는 문제 분석을 구체적인 요구사항으로 변환했다. 시스템은 이미지 기반 검색을 지원해야 하고, OCR이 실패해도 쓸모가 있어야 하며, Top-K shortlist 형태로 결과를 보여줘야 하고, 사용자가 근거를 확인할 수 있어야 한다고 정리했다. 이를 바탕으로 React frontend, FastAPI backend, model package, hybrid retrieval pipeline, Milvus vector collections, catalog retrieval, agent layer로 구성된 모듈형 아키텍처를 설계했다. 또한 retrieval, OCR, latency, engineering stability를 어떻게 평가할지도 설계 단계에 포함했다.

**Develop 단계**에서는 웹, API, 모델 계층을 아우르는 end-to-end workflow를 구현했다. 사용자는 이미지를 업로드하고 필요시 텍스트 질의를 함께 넣을 수 있으며, 시스템은 hybrid retrieval을 수행하고 근거가 포함된 결과를 반환한다. 이후 catalog retrieval과 agent orchestration도 추가했다. 마지막 단계에서는 writeback을 기본 opt-in으로 바꾸고, 테스트를 보강하고, latency instrumentation을 넣는 등 안정성도 강화했다.

**Discuss 단계**에서는 이 시스템이 실제로 무엇을 보여주는지 비판적으로 정리했다. 결론적으로 이 프로젝트는 retrieval-first assistant로서는 성공적이지만, fully autonomous identifier라고 말하기에는 아직 이르다. 이미 확보된 근거와 아직 partial 또는 future work인 평가를 구분해 서술했고, 다시 한다면 quantitative evaluation automation을 더 이른 시점에 넣었을 것이라고 반성할 수 있다.

## Q2. Presentation of the Project

### (a) 최종 제출물(report와 video)을 어떻게 구성했는가?

보고서는 Introduction, Literature Review, Design, Implementation, Evaluation, Conclusion의 6장 구조로 구성했다. 가장 중요한 결정은 보고서가 하나의 일관된 이야기를 갖도록 만드는 것이었다. 즉, 왜 이 문제가 어려운지, 왜 retrieval-first 접근이 적절한지, 아키텍처가 어떻게 이를 해결하는지, 그리고 현재 어떤 근거까지 확보되었는지를 자연스럽게 연결하려 했다. 또한 시스템을 autonomous classifier가 아니라 **human-in-the-loop assistant**로 재정의한 것도 중요한 구성상의 결정이었다. 이 표현이 현재 확보된 증거 수준과 더 잘 맞기 때문이다.

비디오의 경우 가장 중요한 목적은 “정말 동작하는 시스템인지”를 보여주는 것이다. 따라서 architecture를 길게 설명하기보다, 웹 UI에서 이미지를 올리고 shortlist가 반환되며, 결과에 evidence가 함께 표시되고, agent 또는 catalog path가 어떻게 동작하는지를 짧고 명확하게 보여주는 것이 가장 효과적이다.

또 하나의 중요한 발표 전략은 **무엇을 빼느냐**이다. 모든 코드 세부사항을 다 보여주기보다, architecture, hybrid retrieval rationale, safety decision, evaluation honesty 같은 핵심만 남기는 편이 훨씬 설득력 있다.

### (b) 30분 발표를 해야 한다면 어떻게 구성할 것인가?

적절한 발표 맥락은 중고거래 플랫폼이나 산업용 부품 이커머스 회사의 내부 product/research review라고 생각한다. 이 프로젝트는 실제 사용자 워크플로우 문제를 해결하려는 시스템이기 때문에 이런 맥락이 가장 자연스럽다.

발표는 약 14~16장의 슬라이드로 구성할 것이다.

1. 문제와 동기
2. 왜 중고 부품 식별이 어려운가
3. 프로젝트 목표와 요구사항
4. 왜 closed-set classification이 충분하지 않은가
5. retrieval-first 시스템 개요
6. hybrid search pipeline
7. OCR 및 metadata fusion
8. web/API/model 계층 아키텍처
9. catalog retrieval과 agent orchestration
10. 데모 스크린샷 또는 짧은 라이브 데모
11. evaluation strategy
12. 현재 확보된 근거
13. 한계와 리스크
14. future work 및 기대 효과

시각자료로는 architecture diagram, UI screenshot, retrieval evidence 예시, 그리고 supported claim과 in-progress work를 구분한 evaluation summary를 포함할 것이다. 반대로 low-level code detail은 Q&A로 넘기고 본 발표에서는 생략하는 편이 좋다. 30분 발표의 목적은 모든 구현 세부를 설명하는 것이 아니라, 문제 적합성, 시스템 설계, 그리고 evidence-based contribution을 설득력 있게 전달하는 것이기 때문이다.

## Q3. Reflection and continuation

### (a) 프로젝트 전체를 평가하라.

이 프로젝트의 가장 큰 강점은 문제 적합성, 시스템 통합, 그리고 정직한 framing이다. 사용자 사진, OCR 노이즈, open-world inventory, listing-oriented output이라는 실제 문제를 잘 반영하고 있다. 가장 큰 기여는 새로운 foundation model을 제안한 것이 아니라, OCR, multimodal retrieval, vector search, catalog lookup, agent tooling을 실제로 사용할 수 있는 workflow로 통합한 점이다. 또한 자동 확정이 아니라 evidence-backed shortlist를 제공하는 형태로 uncertainty를 설계에 반영한 점도 강점이다.

반면 가장 큰 약점은 evaluation의 완성도와 workflow maturity에 있다. OCR은 현실적인 이미지 품질 저하에 여전히 취약하다. accept/edit/reject review flow도 아직 완전하지 않다. CER 집계, latency percentile, retrieval variant 간 ablation 같은 정량 평가도 아직 충분히 마무리되지 않았다. 따라서 이 프로젝트는 architecture 측면에서는 강하지만, quantitative validation 측면에서는 아직 확장 여지가 많다.

### (b) 미래 학생이 이 프로젝트를 이어받는다면?

가장 적절한 후속 프로젝트 중 하나는 **human-reviewed listing completion with audited feedback loops**이다. 현재 시스템은 retrieval과 evidence 제공까지는 잘 되어 있으므로, 그 다음 단계로 사용자가 결과를 accept/edit/reject하고 수정된 metadata를 안전하게 반영하는 전체 review workflow를 완성하는 것이 자연스럽다. 이는 현재 프로젝트의 강점을 실사용 수준으로 끌어올리는 방향이다.

또 다른 강력한 후속 주제는 **controlled multimodal ablation and region-focused retrieval**이다. 예를 들어 Qwen 중심 파이프라인과 OCR+BGE를 유지하는 mixed pipeline을 비교하고, 필요시 region-focused OCR이나 object cropping을 넣어 hard case를 분석할 수 있다. 이 방향은 시스템의 과학적 검증을 강화해 준다.

미래 학생에게 주고 싶은 조언은 세 가지다. 첫째, 현재의 frontend/API/model 분리를 유지할 것. 둘째, 새 기능을 구현하기 전에 평가 기준을 먼저 정할 것. 셋째, 모델 복잡도를 늘리는 속도보다 evidence를 축적하는 속도를 더 중요하게 볼 것. 가장 큰 함정은 아키텍처를 더 복잡하게 만들면서도 실제로 무엇이 좋아졌는지 증명하지 못하는 상황이다.

---

# 2. CM3070 Past Exam 2024-09

## Q1. Project management and execution

### (a) 계획 단계에서 세운 주요 목표는 무엇이었는가?

내가 계획 단계에서 세운 주요 목표는 다음과 같다.

- 사진 기반 부품 식별을 위한 end-to-end prototype을 만드는 것
- closed-set classification이 아니라 open-world retrieval을 지원하는 것
- 이미지 하나에만 의존하지 않고 image + OCR + text evidence를 함께 활용하는 것
- 단순 similarity score가 아니라 listing workflow에 쓸 수 있는 output을 제공하는 것
- 나중에 정직하게 평가할 수 있도록 instrumentation과 evidence를 남기는 것

이 목표들이 적절했던 이유는, 새로운 foundation model을 만드는 것이 아니라 실제 문제를 해결하는 시스템 통합 프로젝트에 집중했기 때문이다.

### (b) 초기 단계를 어떻게 조직했는가?

초기 단계는 네 가지 축으로 나눠 조직했다. 첫째, literature와 problem framing. 둘째, architecture design. 셋째, prototype implementation. 넷째, evaluation planning이다. 주요 자원으로는 프로젝트 템플릿, 이전 제출 피드백, 선택한 모델/데이터베이스 스택, 실제 인덱싱 및 테스트에 사용할 데이터가 있었다. 주요 작업은 요구사항 정리, retrieval-first 아키텍처 선택, API와 UI 흐름 설계, indexing/search 구현, evidence 수집 계획 수립 등이었다.

타임라인은 완전히 선형적이지 않고 반복적(iterative)이었다. planning이 끝난 뒤 implementation이 시작된 것이 아니라, 구현을 하면서 OCR reliability나 open-world retrieval 문제를 더 잘 이해하게 되었고, 그 이해가 다시 설계에 반영되었다.

### (c) 초기 계획의 효과를 분석하라.

초기 계획은 예상보다 잘 작동했다. 핵심 문제 정의가 비교적 일찍 고정되었고,
초기에 세운 주요 기능 목표도 생각보다 빠르게 달성했다. 그 덕분에 나중에는 agent 기능과 catalog 기능 같은
확장 요소까지 추가할 수 있었다. 이 점에서 보면 초기 계획은 프로젝트의 기반을 빠르게 확보하는 데 효과적이었다.

하지만 hindsight로 보면, 약점은 목표 미달보다는 scope control에 있었다.
핵심 목표를 빨리 달성한 이후 계속해서 추가 기능 아이디어가 떠올랐고,
무엇을 이번 프로젝트 범위 안에 포함할지 우선순위를 정하는 일이 더 어려워졌다.
즉, 계획의 문제는 “기본 목표를 못 달성했다”가 아니라,
“성공 이후 확장을 어떻게 제한할 것인가”에 더 가까웠다.

또한 이 과정에서 프로젝트의 표현도 더 정교해졌다.
처음에는 “AI part identification”처럼 넓게 말할 수 있었지만,
점차 이를 “evidence-backed shortlist assistant”라는 더 방어적이고 정확한 주장으로 다듬게 되었다.

그리고 핵심 목표를 빨리 달성했다고 해서 곧바로 충분하다고 보지는 않았다.
결국 정확하게 판별되지 않으면 아무도 쓰지 않는 시스템이 되기 때문에,
나는 이후 단계에서 모델 성능을 실제로 끌어올리는 데 최대한 집중했다.
즉 확장 기능을 붙이는 것과 별개로, 정확도는 끝까지 가장 중요한 기준이었다.


## Q2. Technical challenge and solution

### (a) 가장 중요한 기술적 도전과제는 무엇이었는가?

가장 중요한 기술적 도전과제는 라벨 인식 문제를 안정적으로 처리하는 것이었다.
처음에는 OCR 중심으로 접근했지만, 일반적인 OCR은 문서에 더 특화되어 있었고,
실제 중고 부품 라벨은 모양이 제각각이며 오염, 훼손, 지워짐, 가림 현상이 많았다.
그 때문에 특정 라벨 유형마다 맞춤 코드를 작성하는 방식은 현실적으로 유지하기 어려웠고,
오류에 대응하는 것도 쉽지 않았다.

이후 VLM을 도입하면서 많은 문제가 완화되었지만,
훼손되거나 일부 지워진 라벨, 부정확하게 보이는 문자, 희미한 part number 같은 경우는 여전히 어려웠다.
그 결과 다른 제품이 검색되는 문제가 남았다. 이를 보완하기 위해 상품 본체의 시각적 특징을 더 활용하는 방향도 추가했지만,
본체 모양까지 거의 동일하고 라벨만 다른 제품군에서는 이미지 단서만으로는 분별력이 약했다.

또 다른 큰 어려움은 fusion scoring이었다. 여러 모델을 한 번에 쓰면 신호를 어떻게 결합할지가 중요해지는데,
weighted sum으로 점수를 계산할 때 특정 신호에 가중치를 높이면 다른 failure case에서 오히려 잘못된 결과를 강화할 수 있었다.
예를 들어 label signal에 높은 가중치를 주면, 라벨이 가려져 있거나 안 보이는 경우 잘못된 제품으로 더 강하게 끌리는 문제가 있었다.
게다가 여러 모델을 동시에 사용하면 속도도 느려져 latency 제약도 함께 생겼다.

### (b) 이를 해결하기 위해 어떤 접근을 취했는가?

나는 이 문제를 pure OCR 방식으로 밀어붙이기보다,
OCR + VLM + 본체 이미지 특징 + metadata를 함께 쓰는 hybrid retrieval 구조로 전환했다.
즉 OCR은 여전히 중요한 신호로 유지하되,
그것만 믿지 않고 VLM 기반 이미지 이해와 시각적 retrieval, 텍스트 검색, metadata evidence를 함께 결합했다.

구체적으로는 사용자 이미지와 metadata를 받아 preprocessing, OCR, embedding 생성,
다중 Milvus collection 검색, weighted fusion, ranking을 수행하는 파이프라인을 설계했다.
또한 다양한 실험을 반복하면서 이 프로젝트에 맞는 weighting과 검색 흐름을 조정했고,
필요할 때는 catalog retrieval과 agent orchestration을 붙여 직접적인 식별이 약한 경우에도
다른 근거를 활용할 수 있도록 했다.

속도 문제에 대해서는 모든 경우에 무거운 경로를 쓰지 않도록 경량 텍스트 경로와 옵션화된 구성도 두었다.
즉 이 접근은 단일 모델 하나로 문제를 해결하려 하기보다,
불완전한 신호들을 실제 도메인에 맞게 조합해 보완하는 방향이었다.

### (c) 그 접근의 효과를 평가하라.

이 접근은 분명히 효과가 있었다. 특히 VLM을 도입한 뒤에는 OCR만 썼을 때보다 많은 문제 상황이 완화되었고,
여러 실험을 통해 프로젝트에 맞게 조금씩 튜닝하면서 성능을 끌어올릴 수 있었다.
이 점은 내가 이 프로젝트에서 잘한 부분이기도 하다.

하지만 한계도 분명하다. 훼손되거나 지워진 라벨은 여전히 어렵고,
본체가 거의 같은데 라벨만 다른 경우는 시각 정보만으로 충분히 구분되지 않는다.
또 weighted sum 기반 fusion은 hand-tuning의 한계가 있었고,
정확도와 속도 사이의 trade-off도 남았다.

그래서 이 접근은 “문제를 많이 해결한 현실적인 prototype”으로는 성공적이었지만,
완전히 근본적인 해결이라고 보기는 어렵다. 다시 한다면 orchestration만으로 해결하려 하기보다,
라벨이나 도메인 특성에 더 특화된 핵심 모델 또는 fine-tuning을 강화하고,
fusion weight도 더 체계적으로 학습 또는 보정하는 방향을 고려할 것이다.


## Q3. Background literature

### Reference 1: CLIP 계열 / image-text embedding 문헌

내 프로젝트에서 가장 중요한 참고문헌 중 하나는 CLIP 계열의 vision-language representation learning 연구다. 이 연구들의 핵심 기여는 이미지와 텍스트를 공유 임베딩 공간에 매핑하여 cross-modal retrieval이 가능하게 했다는 점이다.

이 문헌이 중요한 이유는, 내 프로젝트가 왜 closed-set classification보다 retrieval을 중심에 두어야 하는지를 정당화해 주었기 때문이다. 사용자의 이미지와 텍스트 질의를 하나의 broader retrieval architecture 안에서 다룰 수 있다는 설계 발상에 직접적인 영향을 주었다. 이런 문헌은 널리 인용되고 후속 연구에 큰 영향을 준 foundational work이므로 신뢰성과 품질이 매우 높다고 평가할 수 있다.

### Reference 2: BGE-M3 / multilingual retrieval 관련 문헌

또 다른 중요한 참고문헌은 BGE-M3 및 multilingual retrieval 관련 연구다. 이 연구의 중요한 점은 다국어 환경에서 강한 text retrieval 성능을 제공하고, 여러 retrieval mode를 지원한다는 것이다.

이 프로젝트에서는 OCR 텍스트, metadata, 한국어 검색어, 영어 브랜드명, shorthand identifier가 섞여 등장할 수 있기 때문에, 강한 text retrieval 채널을 유지하는 것이 중요했다. 따라서 이 문헌은 image-only similarity에서 벗어나 hybrid retrieval로 가는 설계 결정에 영향을 주었다. 이 역시 현재 실무와 연구에서 영향력이 큰 모델 계열에 관한 기술 문헌이므로 품질과 신뢰성이 높다고 볼 수 있다.

## Q4. Future directions

### (a) 중기 및 장기 가능성은 무엇인가?

중기적으로는 이 프로젝트가 중고 플랫폼에서 technical item listing assistance를 개선할 수 있다. 사용자는 identification effort를 줄일 수 있고, listing metadata의 일관성도 높일 수 있다. 장기적으로는 같은 아키텍처를 equipment maintenance, spare-part lookup, circular economy workflow, industrial inventory discovery 같은 영역으로 확장할 수 있다.

이 주장이 가능한 이유는, 이 시스템이 특정 UI나 좁은 데이터셋에만 묶여 있지 않고, open-world identification을 위한 multimodal evidence orchestration pattern 자체를 제안하고 있기 때문이다.

### (b) 미래 학생이나 실무자가 어떻게 확장할 수 있는가?

가장 구체적인 확장 방향은 다음과 같다.

- accept/edit/reject를 포함한 full audited human-review workflow
- retrieval, OCR, latency, usability에 대한 stronger evaluation automation
- hard case를 위한 region-focused OCR 또는 object detection
- richer catalog grounding과 document retrieval
- confirmed user correction을 활용하는 active learning

이 확장들은 현재 시스템을 갈아엎는 것이 아니라, 이미 있는 architecture를 기반으로 자연스럽게 발전시키는 방향이다.

### (c) 후속 작업자에게 줄 조언은 무엇인가?

내가 후속 작업자에게 가장 먼저 하고 싶은 말은,
처음부터 문제 정의를 아주 좁고 명확하게 잡으라는 것이다.
AI 모델은 매우 빠르게 발전하고 있기 때문에,
현재 특별해 보이는 기능도 금방 기본 기능이 될 수 있다.
그래서 중요한 것은 기능의 개수를 늘리는 것이 아니라,
정확히 어떤 문제를 어떤 제약 아래에서 풀 것인지 정하는 일이다.

이 프로젝트의 경우 회사 정보가 외부로 나가면 안 된다는 제약 때문에
on-premise와 local 기반 모델 사용이 매우 중요했다.
이런 상황에서는 단순히 여러 모델을 orchestration 해서 weighted sum으로 묶는 것보다,
한 개의 모델이 더 좁은 scope에서 전문가처럼 강하게 동작하도록 만드는 것이
오히려 더 핵심 경쟁력이 될 수 있다.

따라서 후속 작업자는 agent 파이프라인을 더 화려하게 만드는 것 자체보다,
핵심 가치가 어디에 있는지 먼저 정하고,
필요하다면 domain-specific fine-tuning이나 specialist component를 더 강화하는 방향을 우선적으로 검토하는 것이 좋다.


## Q5. Presentation

### (a) 10분 발표에 무엇을 넣을 것인가?

10분 발표에는 문제 동기, retrieval-first framing, architecture overview, 대표적인 search flow, UI demo, evaluation summary를 넣을 것이다. 여기에 limitation slide를 하나 반드시 넣는 것이 좋다. 이 프로젝트의 설득력은 어디까지가 현재 근거로 뒷받침되고, 어디부터가 future work인지 정직하게 보여주는 데 있기 때문이다.

반대로 너무 자세한 코드 내부 구조, 모든 모델 비교, 아직 검증되지 않은 주장은 빼는 편이 좋다. 10분 발표에서는 completeness보다 clarity가 더 중요하다.

### (b) 발표를 어떻게 설계할 것인가?

8~10장 정도의 짧은 슬라이드 덱을 사용할 것이다. architecture diagram, screenshot, score/evidence example을 포함하고, 흐름은 problem → requirements → system design → implementation → demo → evaluation → limitation → future work 순서로 잡는다. 발표 톤은 순수 academic style보다는 product + engineering style이 더 적절하다. 왜냐하면 이 프로젝트의 가치는 실제 user workflow에 연결할 때 가장 잘 드러나기 때문이다.

이 발표 방식의 한계는 일부 기술적 nuance를 단순화한다는 점이다. 예를 들어 모든 collection 구조나 model trade-off를 자세히 설명하기는 어렵다. 하지만 10분이라는 제약 안에서는 이 정도 단순화가 오히려 장점이 된다.

---

# 3. CM3070 Past Exam 2024-03

## Q1. Project report

### (a) literature review의 목적은 무엇인가?

literature review의 목적은 단순히 “읽은 것이 많다”는 것을 보여주는 데 있지 않다. 더 중요한 목적은 프로젝트를 기존 연구와 실무 사례 속에 위치시키고, 무엇이 이미 알려져 있는지, 어떤 한계와 gap이 있는지, 그리고 왜 내가 특정한 설계 결정을 했는지를 정당화하는 데 있다.

내 프로젝트에서는 literature review가 retrieval-first framing, OCR을 uncertain evidence로 보는 관점, fine-grained domain에서의 multimodal retrieval 필요성, user verification 중심 설계를 정당화하는 역할을 했다.

### (b) 어떤 자료를 literature review에 넣고, 어떤 자료는 빼는가?

literature review에는 프로젝트의 핵심 설계 결정을 정당화하는 데 오래 쓸 수 있는 자료를 넣는 것이 좋다.
예를 들어 multimodal retrieval, OCR uncertainty, vector search, human-in-the-loop assistance처럼
상대적으로 개념 수준에서 유효한 문헌이 중심이 되어야 한다.

반대로 너무 빠르게 가치가 떨어지는 자료는 literature review의 중심에서 비중을 낮추는 것이 좋다.
특정 시점의 모델 버전 비교, 짧은 기간에 뒤집히는 benchmark 결과, 구현 팁 위주의 자료는
개발에는 참고가 될 수 있어도 핵심 academic background로는 약할 수 있다.
실제로 AI 발전 속도가 매우 빨라서, 몇 달 전 실험 결과보다 새 버전 모델 결과가 더 높게 나오는 경우가 있었고,
이 때문에 일부 초반 비교 자료는 최종 설계 설명의 중심 근거로 쓰기엔 적합하지 않았다.

따라서 이 프로젝트의 literature review에서는 오래 유지되는 개념적 근거를 중심에 두고,
빠르게 변하는 모델 비교 자료는 implementation decision을 보조하는 용도로만 제한하는 것이 가장 적절하다.


### (c) appendix에는 어떤 자료가 들어가는가?

appendix에는 본문 흐름을 방해할 수 있지만 재현성이나 보조 설명에는 중요한 자료가 들어간다. 예를 들어 extended table, 추가 screenshot, 상세 test output, 추가 qualitative example, prompt, supplementary implementation note 등이 appendix에 적절하다.

### (d) 내가 appendix를 넣는다면 무엇을 담겠는가?

이 프로젝트에서 appendix에 넣기 좋은 자료는 OCR failure example, 추가 retrieval output, artifact bundle 구조, environment/test 기록 등이다. 이런 자료는 본문을 과도하게 길게 만들지 않으면서도 reproducibility와 transparency를 높여준다.

## Q2. Project design

### (i) 어떤 template를 선택했고 왜 그렇게 했는가?

나는 **CM3020 “Orchestrating AI models to achieve a goal”** 템플릿을 선택했다. 이 프로젝트의 강점은 하나의 새 모델을 학습하는 데 있지 않고, OCR, multimodal retrieval, vector search, metadata processing, catalog search, agent orchestration을 하나의 practical system으로 묶는 데 있기 때문에 이 템플릿이 가장 적절했다.

### (ii) 기술적 도전과제를 요약하라.

기술적 도전과제는 imperfect secondhand photo로부터 fine-grained part를 open-world 환경에서 식별하는 것이었다. 비슷하게 생긴 부품들이 많고, 실제 차이는 작은 텍스트 표식에 숨어 있는 경우가 많았다.

### (iii) 해결책을 짧게 요약하라.

해결책은 image, OCR, text, metadata, caption signal을 함께 활용하는 hybrid retrieval-first architecture를 설계하고, 이들을 vector collection에 저장해 evidence-backed Top-K candidate를 반환하는 것이었다.

### (iv) 기술적 해결책을 자세히 설명하라.

고수준에서 보면 사용자는 web interface를 통해 search, indexing, chat, catalog 기능을 사용한다. 웹 클라이언트는 FastAPI backend로 요청을 보낸다. API는 authentication, hybrid indexing/search, catalog PDF indexing/search, agent chat을 위한 route를 제공한다. API 계층은 model package에 위임하고, 그 안의 hybrid-search orchestrator가 preprocessing, OCR, embedding generation, candidate retrieval, fusion, ranking을 담당한다.

Milvus는 하나의 거대한 테이블이 아니라 여러 collection으로 벡터와 metadata를 저장한다. 이를 통해 image retrieval, text retrieval, caption retrieval, metadata-aware lookup을 분리해 다룰 수 있다. catalog path는 PDF chunk를 인덱싱하여 내부 문서 검색을 지원하고, agent path는 여러 tool을 호출하면서 evidence source를 함께 반환할 수 있다.

이 설계의 중요한 특징은 modularity다. frontend는 사용자 경험을 담당하고, API는 stable interface를 제공하며, model layer는 내부적으로 발전할 수 있다. 이 구조는 후속 실험과 기능 확장에도 유리하다.

## Q3. Evaluation

### (a) 평가를 어떻게 수행했는가?

이 프로젝트의 평가는 prototype testing, retrieval-oriented analysis, implementation validation, critical reflection을 결합한 형태였다. 이는 이 프로젝트가 단순 알고리즘 실험이 아니라 실제 동작하는 시스템이기 때문에 적절한 방식이었다. 사용한 방법에는 API/model regression test, retrieval 관련 관찰 및 baseline evidence, latency instrumentation, 그리고 아직 미완료인 부분까지 포함한 structured critical discussion이 있었다.

결과적으로 시스템이 end-to-end prototype으로 동작한다는 점과 retrieval-first assistant라는 framing이 타당하다는 점은 보여줬다. 하지만 여전히 aggregate CER, latency percentile, controlled hybrid ablation은 더 강화되어야 한다. 만약 개선한다면 이들 정량 평가를 더 일찍 자동화할 것이다.

평가 결과는 supported claim, partial evidence, future work를 구분하는 table로 제시하는 것이 가장 적절하다. 그렇게 해야 evidence boundary가 명확해지고, 과장 없이 프로젝트의 현재 상태를 보여줄 수 있기 때문이다.

### (b) 프로젝트 자료를 어떻게 관리했는가?

프로젝트 자료는 web, API, model, docs, submission artifact를 분리한 저장소 구조로 관리했다. 버전 관리를 사용해 코드 변경 이력을 추적했고, 제출 기준선과 현재 작업 상태를 혼동하지 않도록 submission 폴더와 docs 폴더를 구분했다.

다시 한다면 artifact automation을 더 강화할 것이다. 이 프로젝트에서 얻은 중요한 교훈은 코드 버전 관리만으로는 충분하지 않고, evaluation artifact도 의도적으로 정리·보관해야 한다는 점이다.

## Q4. Self-reflection

### (i) 무엇을 달성했는가?

나는 계획했던 핵심 목표를 비교적 빠르게 달성했고,
그 위에 agent 기능과 catalog 기능까지 추가할 수 있었다.
즉 단순한 검색 실험을 넘어서,
web, API, model 계층을 갖춘 end-to-end prototype을 만들었다는 점이 가장 큰 성과다.

### (ii) 가장 잘한 부분은 무엇인가?

가장 잘한 부분은 여러 모델을 실제로 많이 써보고,
반복적인 실험을 통해 이 프로젝트에 맞게 최적화하려고 노력한 점이라고 생각한다.
단순히 모델을 붙여놓는 데서 끝나지 않고,
실험 결과를 바탕으로 조금씩 튜닝하면서 성능을 끌어올린 부분은 스스로 가장 칭찬하고 싶은 부분이다.

### (iii) 가장 부족한 부분과 개선 방향은 무엇인가?

가장 약한 부분은 orchestration으로 문제를 해결하려 한 반면,
핵심 fine-tuning 또는 domain-specific specialist model을 충분히 강화하지 못한 점이라고 생각한다.
여러 모델을 조합해 보완하는 방식은 도움이 되었지만,
장기적인 경쟁력은 결국 핵심 모델이 특정 문제를 전문가처럼 얼마나 잘 해결하느냐에 더 달려 있다고 본다.

다시 한다면 이 부분을 더 고도화할 것이다.
즉 weighted-sum orchestration을 계속 개선하는 것에 그치지 않고,
라벨 인식이나 도메인 특성에 더 맞는 fine-tuning, specialist component,
그리고 더 체계적인 score calibration 쪽을 핵심 개선 방향으로 두고 싶다.


## Q5. Further work

### (i) 전체 rewrite가 필요한가?

전체 rewrite는 필요하지 않다고 본다. 현재 frontend/API/model의 모듈 분리는 강한 기반이 된다. 다만 evaluation automation이나 reviewed writeback workflow 같은 부분은 targeted refactoring이 필요할 수 있다.

### (ii) 후속 학생이 추가하면 좋은 두 가지는 무엇인가?

두 가지 중요한 추가 방향은 다음과 같다.

- accept/edit/reject를 갖춘 audited human-review workflow
- controlled multimodal benchmarking 및 region-focused retrieval 실험

이 두 방향은 각각 product usability와 scientific validation을 동시에 강화해 준다.

### (iii) 그 작업을 어떻게 진행하라고 조언하겠는가?

후속 학생에게는 현재 architecture를 유지하고, 구현 전에 measurable success criteria를 먼저 정하고, extra model complexity를 넣기 전에 clear evaluation plan을 세우라고 조언하겠다. 또한 data version, collection schema, artifact generation을 철저하게 관리해야 한다.

---

# 4. CM3070 Past Exam 2023-09

## Q1. Project Approach

### (a) 어떤 template를 선택했고 왜 그렇게 했는가?

나는 **CM3020 “Orchestrating AI models to achieve a goal”** 템플릿을 선택했다. 이 프로젝트는 하나의 새로운 모델을 발명하는 것보다, 여러 AI 구성요소를 실제 목적에 맞게 조합하는 데 더 큰 가치가 있기 때문이다.

### (b) 요구사항을 달성하기 위해 어떤 route를 택했는가?

나는 retrieval-first route를 택했다. fixed classifier를 학습하는 대신, image evidence, OCR text, caption, metadata, vector search를 결합하는 hybrid search pipeline을 설계했다. 여기에 frontend, backend API, model-side orchestration, 그리고 later extension으로 catalog 및 agent 기능까지 포함했다.

### (c) 다른 접근은 무엇이 가능했는가?

대안으로는 curated dataset 위에 supervised closed-set classification system을 학습하는 방법이 있었다.

### (d) 내 접근과 대안을 비교하라.

내가 retrieval-first route를 택한 이유는 open-world domain에 더 잘 맞기 때문이다. 중고 플랫폼에서는 새로운 item, 희귀 item, 단종 부품이 계속 등장하므로, closed classifier는 유지보수 비용이 크고 쉽게 brittle해진다.

내 접근의 장점은 flexibility, evidence exposure, realistic problem fit이다. 단점은 architecture가 더 복잡하고, ranking/evaluation을 더 신중하게 설계해야 한다는 점이다. 반면 closed-set classifier는 좁은 데이터셋에서는 더 단순하게 설명하고 benchmark하기 쉬울 수 있지만, 실제 문제에는 덜 적합하다.

## Q2. Evaluation and Testing

### (a) 어떤 testing과 evaluation을 했는가?

나는 engineering validation과 retrieval-oriented evaluation을 함께 사용했다. engineering validation에는 safety 및 testability 관련 최근 변경사항에 대한 API/model regression test가 포함되었다. 또한 hybrid search path에 latency instrumentation을 넣어 각 단계 소요시간을 측정할 수 있게 했다. retrieval evaluation 측면에서는 image-first assistance가 유용하지만 그 자체로 충분하지 않다는 qualitative/baseline evidence를 정리했다.

### (b) 다른 evaluation/testing 방법은 무엇이 가능했는가?

더 큰 고정 benchmark를 두고 explicit Top-K accuracy를 측정하거나, 더 체계적인 OCR CER 실험, 더 넓은 latency analysis, trust/effort reduction을 보는 user study를 할 수도 있었다. 이런 방법들은 더 강한 quantitative evidence를 제공했을 것이다. 내가 이를 전부 완료하지 못한 이유는 시간과 통합 복잡도 때문이지만, hindsight로 보면 더 이르게 operationalise했어야 했다.

## Q3. Self Reflection

### (a) 무엇을 달성했는가?

나는 search, indexing, evidence-backed retrieval, catalog search, agent orchestration을 포함하는 working prototype을 만들었다.

### (b) 가장 좋은 두 부분은 무엇인가?

첫째, architectural framing이 강하다. 문제를 closed-set recognition이 아니라 open-world retrieval로 다뤘다는 점이 중요하다. 둘째, 여러 modality와 계층을 아우르는 end-to-end integration을 실제로 구현했다는 점이 강점이다.

### (c) 다른 한 측면에서 무엇을 더 할 수 있었는가?

evaluation automation과 user-facing review workflow를 더 발전시킬 수 있었다. 이 두 영역은 scientific confidence와 product maturity를 동시에 높여줄 수 있기 때문에 가장 명확한 개선 지점이다.

## Q4. Video

### (a) 비디오의 주요 목적은 무엇인가?

비디오의 주요 목적은 시스템이 실제로 동작한다는 점을 보여주고, architecture를 구체적인 사용자 흐름을 통해 이해 가능하게 만드는 것이다. 즉 보고서를 반복하는 것이 아니라, 실제 동작을 통해 achievement를 입증하는 것이 핵심이다.

### (b) 비디오는 어떻게 구성할 것인가?

voice-over가 있는 screen recording 형태가 가장 적절하다. 웹 인터페이스에서 이미지 업로드, shortlist 반환, evidence 표시, agent 또는 catalog 사용 예시를 보여주면 된다. 이 방식이 좋은 이유는 동작하는 시스템과 usability를 동시에 명확하게 보여줄 수 있기 때문이다.

### (c) 5분은 짧은가 긴가?

모든 기능을 다 보여주려면 5분은 짧다. 하지만 잘 구조화하면 충분하다. 핵심은 하나의 search scenario, 하나의 evidence example, 하나의 architecture summary, 그리고 limitation에 대한 짧은 언급만 남기는 것이다. minor implementation detail을 빼면 오히려 영상이 더 명확해진다.

## Q5. Development

### (a) 원래 가지고 있던 능력 중 중요한 두 가지는 무엇인가?

내가 원래 갖고 있던 능력 중 중요한 두 가지는 Python 기반 backend 개발 능력과 일반적인 machine-learning/data workflow 이해였다. 이 능력들 덕분에 API integration, data handling, model orchestration, debugging을 빠르게 진행할 수 있었다. 만약 이런 기반이 없었다면 프로젝트 특화 문제로 들어가기 전에 기본 setup에 훨씬 더 많은 시간을 써야 했을 것이다.

### (b) 프로젝트 중 새롭게 배운 두 가지는 무엇인가?

프로젝트 중 새롭게 배운 중요한 두 가지는 multimodal retrieval system design과, 구현만큼이나 evaluation evidence 관리가 중요하다는 점이다. 또한 OCR uncertainty가 retrieval ranking에 실제로 어떻게 영향을 주는지에 대해서도 더 깊이 이해하게 되었다. 이런 것들은 논문과 기술문서, 코드 실험, 피드백 반영 과정을 통해 습득했다.

---

## 실제 시험 직전 최종 주의사항

실제 시험 답안으로 바꿀 때는 다음 표현을 일관되게 유지하는 것이 좋다.

- 시스템을 **retrieval-first, human-in-the-loop identification assistant**로 설명할 것
- full quantitative evaluation이 끝났다고 과장하지 말 것
- **implemented**, **instrumented**, **future work**를 명확히 구분할 것
- 기여를 mainly **system orchestration and practical integration**로 설명할 것
- 기술적 선택을 항상 도메인 특성과 연결할 것: open-world inventory, fine-grained ambiguity, OCR uncertainty, user need for verifiable results
