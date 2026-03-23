# 2. Literature Review

이 장은 visual product retrieval, closed-set object detection에서 open-world hybrid search로의 전환, 산업 환경에서의 OCR 한계, Vision-Language Model(VLM)의 부상, human-in-the-loop(HITL) 시스템 설계라는 여러 축을 중심으로 관련 연구와 산업 사례를 비판적으로 검토한다. 이를 통해 본 프로젝트의 아키텍처 선택과 engineering decision이 왜 필요한지를 이론적·실증적으로 정당화한다.

## 2.1 The Evolution of Visual Product Retrieval and Object Detection

텍스트 기반 질의에서 이미지 기반 retrieval로의 전환은 전자상거래와 산업 환경 모두에서 사용자 상호작용 방식을 크게 바꾸어 놓았다. 과거 product retrieval은 keyword indexing과 manually curated metadata에 크게 의존했다. 그러나 카탈로그 규모가 커질수록 vocabulary mismatch 문제가 심각해졌다. 특히 비전문 판매자는 산업 부품을 설명하거나 검색하는 데 필요한 정확한 기술 용어를 알지 못하는 경우가 많다.

이 semantic gap을 줄이기 위해 CBIR(Content-Based Image Retrieval) 시스템이 활발히 연구·상용화되었다. 초기 CBIR은 SIFT, HOG와 같은 handcrafted visual feature에 의존했다. 이러한 방법은 통제된 조명 환경에서 duplicate-like matching에는 어느 정도 유효했지만, 고수준 semantic understanding은 부족했다. 이후 ResNet 계열과 같은 deep learning 기반 CNN이 등장하면서 raw pixel에서 더 풍부한 semantic embedding을 추출할 수 있게 되었다. eBay의 visual search rollout은 deep learning이 marketplace 규모에서도 listing 및 search friction을 줄일 수 있음을 보여주는 대표적 사례다.

산업 환경에서는 visual identification이 object detection이나 image classification 문제로 프레이밍된 경우도 많다. YOLO와 같은 모델은 part detection, defect localisation, controlled-factory recognition task에 널리 사용되었다. 최근에는 CLIP, DINOv2와 같은 self-supervised Vision Transformer 계열이 exhaustive manual labelling 없이도 강한 generic visual representation을 제공하면서 널리 활용되고 있다.

**비판적 평가:** 그러나 이러한 접근은 secondhand industrial domain에서 구조적 한계를 가진다. 첫째, closed-set object detector는 predefined class space를 가정한다. 중고 시장에서 자주 등장하는 rare, discontinued, previously unseen part는 전체 재학습 없이는 다루기 어렵다. 둘째, 산업용·전자 부품은 fine-grained하고 visually homogeneous하다. 서로 다른 부품 변형이 거의 동일한 macro-appearance를 가지면서, 실제 차이는 작은 식별자에만 존재하는 경우가 많다. 따라서 generic visual embedding은 올바른 semantic neighbourhood는 찾을 수 있어도 정확한 variant 식별에는 실패하기 쉽다.

## 2.2 From Closed-Set Detection to Open-World Hybrid Search

이러한 closed-set 분류의 한계를 넘기 위해, 최근 retrieval architecture는 dense vector embedding space와 ANN(Approximate Nearest Neighbour) search로 이동하고 있다. 이 접근에서는 이미지를 정적 class 중 하나로 분류하는 대신, 고차원 벡터로 변환한 뒤 embedding space에서 가장 가까운 이웃을 찾는다. FAISS와 같은 foundational system은 대규모 similarity search를 실용화했고, Milvus와 같은 vector database는 이를 확장 가능한 data-management layer로 제품화했다.

이 전환은 open-world domain에서 특히 중요하다. indexed inventory가 계속 증가해도, 핵심 embedding model을 매번 재학습할 필요가 없기 때문이다. 그러나 pure visual vector search만으로는 여전히 fine-grained textual identifier를 처리하기 어렵다.

이를 해결하기 위해 state-of-the-art retrieval은 hybrid vector search로 이동하고 있다. hybrid search는 dense visual similarity와 lexical, sparse, scalar evidence를 retrieval 및 ranking 단계에서 함께 결합한다. 즉 “겉보기로 비슷한 후보를 먼저 찾고, 이후 maker·part number·drafted metadata가 맞는 항목을 상위로 밀어 올리는” 방식이다. 이때 BGE-M3와 같은 강력한 text embedding model은 산업 부품 번호처럼 짧고 밀도 높은 alphanumeric identifier를 표현하는 데 특히 유용하다.

**비판적 평가:** pure visual vector search는 종종 올바른 part neighbourhood까지는 도달하지만, textual identifier가 최종 구분 단서인 상황에서 exact variant 식별에는 부족하다. 따라서 hybrid search는 단순한 성능 향상이 아니라, 산업 부품 retrieval에서 구조적으로 필요한 조건에 가깝다. 강력한 visual encoder, dense text representation, 그리고 hybrid retrieval을 지원하는 vector database를 함께 사용해야 visually homogeneous하지만 textually distinctive한 도메인을 제대로 다룰 수 있다.

## 2.3 The Limitations of OCR in Industrial Imagery

순수 visual retrieval이 exact part number나 manufacturer code 같은 fine-grained textual identifier를 안정적으로 회수하지 못하기 때문에, 많은 시스템은 visual model과 OCR을 함께 사용해 왔다. 실무에서 널리 쓰이는 OCR 파이프라인은 보통 두 단계로 구성된다. 먼저 text detection stage가 이미지 안의 텍스트 영역을 찾고, 이어지는 text recognition stage가 잘라낸 영역을 문자열로 복원한다. PaddleOCR는 이러한 구조를 실용적으로 구현한 대표적 사례다.

이러한 파이프라인은 문서 이미지, 영수증, 정면에서 촬영된 라벨처럼 텍스트가 주요 시각 대상인 경우에는 강하게 작동한다. 따라서 산업용 retrieval에서도 모든 텍스트를 적극적으로 추출하면 exact identifier를 더 잘 회수할 수 있을 것이라는 가정이 자연스럽게 등장한다.

**비판적 평가:** 그러나 open-world, user-generated industrial imagery에서는 이 가정이 잘 성립하지 않는다. 실제 중고 플랫폼 사진은 금속 표면의 glare, 마모된 label, cluttered background, low contrast, skewed viewpoint, mixed text orientation을 자주 포함한다. 이런 조건은 detection과 recognition 단계를 모두 약화시킨다.

더 근본적인 문제는 OCR의 **semantic blindness**다. OCR은 어떤 문자열이 part identity와 중요한 관련이 있는지를 판단하지 않고, 보이는 문자열을 가능한 한 모두 추출하도록 설계되어 있다. 산업용 부품에는 전압, 주파수, 안전 경고, 제조국, 포장재 문구처럼 실제 identity와 직접 관련 없는 텍스트가 많이 포함된다. 이러한 문자열이 그대로 embedding 및 retrieval에 들어가면, vector space는 진짜 part identifier보다 자주 등장하는 specification text에 의해 왜곡될 수 있다.

그 결과 OCR-heavy pipeline은 동일한 전압이나 저항값을 공유하는 unrelated part를 잘못 끌어오는 false positive를 만들 수 있다. 여기에 고해상도 이미지에 대해 detection과 recognition을 순차 실행하는 비용까지 더해지면, interactive listing workflow에 필요한 latency도 크게 악화된다. 따라서 문헌과 본 프로젝트의 실험 결과는 모두 OCR을 기본 retrieval path가 아니라, 선택적으로 사용하는 supporting evidence 또는 verification signal로 해석하는 편이 더 타당함을 시사한다.

## 2.4 Multimodal Embeddings and Vision-Language Models (VLMs)

전통적인 OCR 파이프라인의 순차적 병목과 semantic blindness를 극복하기 위해, 최근 연구는 unified Vision-Language Model(VLM)로 빠르게 이동하고 있다. LLaVA와 Qwen-VL 계열 같은 모델은 visual encoder와 language-model backbone을 결합하고, cross-modal alignment를 통해 이미지와 텍스트를 공유 semantic space 안에서 함께 해석한다. 이는 vision과 text를 서로 분리된 algorithmic silo로 처리하는 방식과는 다르다.

이 broader multimodal trend는 embodied AI나 physical-world AI의 흐름과도 맞닿아 있다. 물론 본 프로젝트는 robotics system은 아니지만, 그 연결점은 여전히 의미가 있다. 산업 부품 식별 역시 object를 isolated crop으로만 보는 것이 아니라, label hierarchy, visual wear, surrounding clutter를 포함한 physical context 속에서 이해해야 하기 때문이다.

한편 textual metadata retrieval 측면에서는 BGE-M3와 같은 text embedding model이 다국어 및 multi-granularity 표현을 제공하며, 짧고 밀도 높은 alphanumeric identifier matching에 적합하다.

**비판적 평가:** 이 도메인에서 VLM의 핵심 장점은 단순한 caption generation이 아니라, identity-relevant evidence를 더 문맥적으로 걸러낼 수 있다는 점이다. VLM은 어떤 텍스트가 prominent manufacturer mark인지, 어떤 문자열이 generic warning인지, 어떤 label이 product identity와 구조적으로 더 관련 있는지를 구분할 수 있다. 따라서 standalone OCR을 primary signal로 사용하는 것보다, VLM을 중심에 두고 text embedding model을 결합하는 편이 open-world, fine-grained industrial-part retrieval에 더 적합하다. 본 프로젝트가 OCR-heavy extraction에서 vision-language-dominant pipeline으로 pivot한 것도 바로 이러한 문헌적 배경과 실험적 증거를 반영한 설계 전환이다.

## 2.5 Critical Evaluation: The Limits of One-Shot VLM Identification

최근 Vision-Language Model의 추론 및 정보 추출 능력이 강해지면서, “강력한 VLM 하나에 이미지를 넣고 exact metadata를 바로 생성하게 하면 되지 않을까”라는 engineering hypothesis가 자연스럽게 등장한다. 이런 구조에서는 벡터 데이터베이스나 hybrid retrieval 없이, 모델에게 부품 번호·제조사·카테고리를 end-to-end로 바로 생성하도록 요구하게 된다.

**비판적 평가:** 그러나 LLM/VLM reliability 관련 문헌은 이러한 접근이 high-stakes, fine-grained domain에서는 위험하다는 점을 보여준다. 특히 세 가지 구조적 문제가 중요하다.

1. **Identifier hallucination**  
   생성형 모델은 verified fact가 아니라 probable token을 출력한다. reliability 연구가 보여주듯, 이러한 모델은 매우 그럴듯하지만 잘못된 identifier를 생성할 수 있다. 산업 부품 listing에서는 부품 번호 한 글자 오류만으로도 전체 listing이 무효가 된다.

2. **Open-world domain과 static parametric knowledge의 충돌**  
   VLM의 내부 지식은 training distribution과 cut-off에 제한된다. 따라서 특정 판매자의 현재 inventory나, 학습 데이터에서 희소했던 rare/discontinued part를 안정적으로 안다고 가정하기 어렵다.

3. **Grounded evidence의 부족**  
   one-shot generated answer는 외부 evidence를 자동으로 제공하지 않는다. 결과가 맞는 것처럼 보여도, 사용자는 왜 그런 판단이 나왔는지 검증하기 어렵다.

이러한 한계는 retrieval-first architecture의 필요성을 강화한다. RAG는 parametric reasoning과 external knowledge access를 분리하는 것이 얼마나 중요한지 보여준 대표적 사례다. 본 프로젝트 역시 유사한 원리를 multimodal setting에 적용한다. 즉 모델은 useful feature와 metadata cue를 추출하는 데 집중하고, 최종 candidate suggestion은 external indexed database에서 grounded retrieval을 통해 제시된다. 이 구조는 hallucinated identifier의 위험을 줄이고, 결과가 실제 database 안의 inspectable evidence에 기반하도록 만든다.

## 2.6 Human-in-the-Loop (HITL) and Interactive Relevance Feedback

retrieval 품질이 높더라도, 잘못된 listing이 금전적 손실, failed search, return friction, buyer trust 저하로 이어질 수 있는 상업적 환경에서 fully autonomous identification은 여전히 위험하다. human-centred AI 문헌은 precision, accountability, user confidence가 중요한 영역에서 over-automation을 경계해야 한다고 반복적으로 지적한다.

human-in-the-loop 설계는 이러한 문제에 대한 실질적 대응이다. 여기서 AI는 decision-making authority가 아니라 decision-support mechanism으로 위치한다. 정보 검색 분야의 relevance feedback 개념도 이 맥락에서 중요하다. 시스템은 plausible candidate를 제시하고, 사용자는 이를 검토하며, 상호작용을 통해 최종 결과를 확정한다. Dong et al. (2021)은 explainable, interactive image-retrieval system이 evidence를 노출함으로써 사용자 신뢰와 과업 성능을 높일 수 있음을 보여주었다.

**비판적 평가:** 산업 부품 식별에서는 이 설계 논리가 특히 설득력 있다. Top-K shortlist는 최종 grounding source가 여전히 인간 사용자, 즉 실제 물건을 손에 들고 있는 판매자라는 사실을 존중한다. 사용자는 candidate를 시각적으로 비교하고, evidence를 검증하며, 필요하면 metadata를 직접 수정할 수 있다. 이는 시스템이 unchecked classifier가 아니라 cognitive and workflow aid로 작동함을 의미한다. 동시에 confirm과 correction을 거친 결과는 시간이 지날수록 indexed data의 질을 높이는 간접적 효과도 제공한다.

문헌 검토에서 도출된 접근법 비교와 본 프로젝트의 설계 입장은 **표 2**에 요약했다.

**Table 2. Comparison of retrieval approaches and their implications for industrial-part identification**

| Approach | Strengths | Limitations in industrial context | Project's stance |
| --- | --- | --- | --- |
| Traditional Visual Search (e.g., CLIP, DINOv2) | 전반적 형상과 색상 표현에 강하고 빠르다 | 서로 거의 같은 외형을 가진 fine-grained 부품 변형을 구분하지 못한다 | 단독으로는 불충분함 |
| Pipeline: Vision + OCR | 텍스트를 추출해 visual search를 보강할 수 있다 | 노이즈, glare, 복잡한 배경에 취약하고 `12V` 같은 irrelevant specs를 과도하게 추출한다 | OCR은 primary search가 아니라 secondary verification에 사용 |
| Vision-Language Models (e.g., Qwen3-VL) | 문맥 기반 텍스트 이해가 가능하고 layout과 logo를 함께 해석한다 | 계산 비용이 더 크고 occasional hallucination 가능성이 있다 | primary retrieval engine으로 채택 |
| Human-in-the-Loop (HITL) | 사용자 신뢰가 높고 수동 검증·수정이 가능하다 | 사용자 상호작용이 필요하며 완전 자동은 아니다 | Top-K shortlist와 evidence를 제공하는 핵심 설계 철학 |

*Table note:* 이 표는 benchmark 결과표가 아니라 문헌 검토의 핵심 논점을 분석적으로 정리한 것이다. 각 접근법이 본 프로젝트의 아키텍처 선택에 어떤 영향을 주었는지를 한눈에 보여주기 위해 사용한다.

## 2.7 Summary of Literature and Architectural Justification

이 장의 문헌 검토는 visual retrieval과 OCR이 모두 잘 확립된 접근이지만, noisy하고 fine-grained하며 open-world인 산업 부품 도메인에서는 둘을 단순 결합하는 것만으로는 충분하지 않음을 보여준다. generic visual embedding은 micro-level distinction에서 약하고, closed-set detector는 dynamic inventory에서 한계를 보이며, 전통적 OCR 파이프라인은 semantic noise와 latency를 동시에 만든다. 여기에 one-shot generative VLM inference는 hallucination과 weakly grounded output이라는 추가 위험을 갖는다.

이러한 비판적 검토의 수렴점은 분명하다. 효과적인 산업 부품 식별 시스템은 Vision-Language Model을 context-aware feature extraction에 사용하고, visual evidence와 textual evidence를 hybrid retrieval로 결합하며, candidate suggestion을 external database에 grounded시키고, 최종적으로는 human-in-the-loop verification interface 안에서 결과를 확인하도록 해야 한다. 이것이 바로 이후 장에서 설계·구현·평가하는 retrieval-first, selective-OCR, evidence-backed architecture의 직접적인 이론적 근거다.
