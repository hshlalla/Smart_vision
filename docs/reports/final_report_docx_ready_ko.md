# Smart Image Part Identifier for Secondhand Platforms (한국어 버전)

University of London  
Bachelor in Computer Science

Final Project  
CM3020 Artificial Intelligence, "Orchestrating AI models to achieve a goal"

Name: SuHun Hong  
Email: hshlalla@naver.com

> 작업 메모: 이 문서는 `docs/reports/final_report_docx_ready.md`의 한국어 대응본이다. 제출 직전에는 영문 최종본을 기준으로 정리하고, 이 파일은 내부 검토와 한국어 참조용으로 사용한다.

## 1. Introduction

이 프로젝트는 CM3020 템플릿 **“Orchestrating AI models to achieve a goal”**를 따른다. 핵심 목표는 중고 플랫폼의 등록 워크플로우에서 사용자가 업로드한 사진만으로 산업용·전자 부품의 제조사, 모델명, 부품 번호를 더 쉽게 식별할 수 있도록 돕는 것이다.

이 문제는 중요하다. 일반 소비재는 대략적인 외형만으로도 검색이 가능한 경우가 있지만, 산업용 부품은 잘못 식별될 경우 가격 책정 오류, 검색 실패, 거래 실패로 이어질 수 있다. 또한 실제 구분 단서는 전체 외형이 아니라 작은 모델 코드, 시리얼 번호, 명판(nameplate)인 경우가 많다. 그런데 이러한 단서는 사용자가 촬영한 이미지에서 glare, blur, 저해상도, 마모, 부분 가림 등의 이유로 잘 보이지 않는 경우가 많다. 그 결과 비전문 판매자는 매뉴얼, 카탈로그, 검색엔진을 수동으로 대조하며 시간을 많이 쓰게 된다.

산업 동향을 보면 사진 기반 보조 기능은 등록 마찰을 줄이는 데 도움이 된다. eBay의 이미지 검색은 사용자가 적절한 키워드를 모르더라도 사진으로 후보를 찾을 수 있음을 보여주었고, 한국의 일부 중고 플랫폼도 업로드 이미지를 바탕으로 상품 정보를 추정하는 AI 보조 기능을 도입했다. 그러나 이런 사례들은 주로 일반 소비재를 대상으로 한다. 부품 식별은 시각적으로 매우 비슷한 변형들이 많고, 실제로 중요한 정보가 작고 노이즈가 많은 텍스트인 경우가 많기 때문에 더 까다롭다. 따라서 이 문제는 단순한 이미지 유사도 검색 이상으로 다뤄져야 한다.

이 프로젝트의 중심 주장은, 중고 부품 식별은 **closed-set classification**이 아니라 **retrieval-first, human-in-the-loop decision-support** 문제로 보는 것이 더 적절하다는 것이다. 유용한 시스템이라면 이미지와 텍스트 증거를 함께 활용할 수 있어야 하고, OCR이 실패해도 어느 정도 작동해야 하며, listing workflow에 맞는 구조화된 결과를 반환해야 하고, 사용자가 결과를 검증할 수 있도록 근거를 제시해야 한다.

이러한 framing은 도메인 제약과 사용자 요구를 모두 반영한다. 도메인 측면에서 중고 부품 인벤토리는 open-world이며 long-tail 특성을 가진다. 희귀 모델, 단종품, 신규 품목이 계속 등장하므로 고정된 클래스 집합에 기반한 분류기는 적합하지 않다. 또한 이 도메인은 매우 fine-grained하다. 외형은 비슷하지만 작은 표기 차이로 구분되는 경우가 많다. 이미지 품질 역시 일정하지 않기 때문에 OCR은 유용하지만 ground truth로 취급할 수는 없다. 마지막으로 사용자는 단순히 “비슷한 이미지”가 아니라 maker, part number, category, supporting evidence처럼 실제 등록에 활용할 수 있는 필드를 원한다.

사용자 측면의 요구도 같은 방향을 가리킨다. 중고 플랫폼 경험이 있는 참여자 6명을 대상으로 한 간단한 요구사항 조사에서, 상세 스펙을 텍스트로 작성하는 것이 어렵고 수동 검색 의존도가 높으며, 사진 기반 보조 기능에 대한 수요가 있음을 확인했다. 동시에 참여자들은 AI 결과를 무조건 신뢰하지 않았다. 반복적으로 강조된 것은 **투명성**, **검증 가능성**, **수정 가능성**이었다. 따라서 시스템은 최종 답을 독단적으로 제시하기보다 후보군을 좁혀주고 근거를 제공하는 shortlist 기반 워크플로우를 지향해야 한다.

[Insert Table 1-1 here: survey 결과와 requirement implication 요약 표 삽입]

이러한 관찰을 바탕으로 본 프로젝트의 목표는, 사용자가 업로드한 부품 사진을 입력으로 받아 Top-5 후보 shortlist와 listing-oriented summary, 그리고 supporting evidence를 함께 제공하는 end-to-end prototype을 구현·평가하는 것이다. 세부 목표는 end-to-end 워크플로우 구현, retrieval effectiveness 평가, OCR robustness 분석, interactive feasibility를 위한 latency instrumentation 추가, 그리고 실제 listing assistance 관점에서의 유용성 평가로 정리할 수 있다.

이 보고서의 나머지 구성은 다음과 같다. 2장은 visual retrieval, OCR, multimodal embeddings, vector databases, interactive feedback 관련 선행연구를 검토한다. 3장은 시스템 설계를 설명하며 evaluation strategy를 설계 안에 포함한다. 4장은 web, API, model 계층에 걸친 구현 내용을 설명한다. 5장은 현재 확보된 근거를 바탕으로 evaluation을 수행하되, 이미 측정된 항목과 아직 계획·부분 완료 상태인 항목을 구분한다. 6장은 기여, 한계, 향후 과제를 정리한다.

## 2. Literature Review

시각 검색은 handcrafted descriptor에서 시작해, 대규모 semantic search를 지원하는 deep feature embedding으로 발전해 왔다. eBay image search와 같은 소비자용 시스템은 사용자가 정확한 키워드를 알지 못하더라도 이미지 기반 질의가 실용적일 수 있음을 보여준다. Google Lens와 유사한 시스템 역시 이러한 패턴을 더 넓은 객체군으로 확장한다. 하지만 이들 사례는 대체로 전체 소비재 수준에서의 검색을 다루며, 미세한 차이를 구별해야 하는 산업용 부품 문제와는 차이가 있다.

산업용 부품이나 세부 식별 문제를 다룬 선행연구는 본 프로젝트에 더 직접적인 근거를 제공한다. 이러한 연구들은 이미지 유사도만으로는 충분하지 않으며, 텍스트 단서와 구조화된 메타데이터가 함께 고려되어야 한다는 점을 보여준다. 특히 OCR과 visual retrieval을 결합하는 접근은 외형이 유사하지만 식별 텍스트가 중요한 도메인에서 유효하다.

OCR 관련 문헌은 텍스트 추출이 유용하지만 불안정하다는 점을 반복적으로 보여준다. 특히 glare, blur, 왜곡, 작은 글씨, worn-out label은 OCR 정확도를 떨어뜨린다. 이는 본 프로젝트에서 OCR을 “핵심 보조 증거”로 사용하되, 유일한 정답 근거로 보지 않는 설계로 이어졌다.

멀티모달 임베딩과 텍스트 검색 관련 문헌은 이미지와 텍스트를 공동 임베딩 공간에서 다루는 방법, 그리고 다국어·다기능 텍스트 검색의 중요성을 제시한다. 이는 이미지 기반 질의와 OCR 텍스트, 메타데이터, 사용자 텍스트 질의를 함께 활용하는 hybrid retrieval 구조를 정당화한다.

벡터 데이터베이스와 hybrid search 관련 선행사례는 여러 종류의 벡터와 메타데이터를 분리 저장하고, 단계적으로 후보를 수집·재정렬하는 방식이 대규모 시스템에 적합하다는 점을 보여준다. 이는 Milvus를 이용한 multi-collection 설계와 ranking fusion 로직의 근거가 된다.

마지막으로 interactive feedback과 user-centred AI literature는, 사용자가 결과를 검증하고 수정할 수 있을 때 시스템에 대한 신뢰가 높아진다는 점을 시사한다. 이는 본 프로젝트에서 evidence-backed shortlist와 safer writeback policy를 채택하게 된 중요한 배경이다.

## 3. Design

본 시스템의 설계는 open-world, fine-grained, text-sensitive한 부품 식별 문제를 해결하도록 구성되었다. 전체 구조는 web frontend, FastAPI backend, hybrid-search model package, Milvus vector storage, catalog retrieval, agent orchestration으로 나뉜다.

시스템의 핵심 흐름은 다음과 같다. 사용자가 이미지와 선택적 텍스트 질의를 입력하면, 시스템은 OCR, embedding generation, candidate retrieval, evidence fusion, ranking을 거쳐 Top-K 후보를 반환한다. 인덱싱 단계에서는 이미지, 캡션, OCR 텍스트, 메타데이터가 저장되며, 검색 단계에서는 다양한 신호를 결합해 사용자에게 근거 있는 결과를 제시한다.

[Insert Figure 3-1 here: 전체 시스템 아키텍처]

요구사항과 컴포넌트의 대응 관계도 중요하다. open-world inventory는 index-based retrieval로 대응하고, fine-grained ambiguity는 hybrid ranking과 metadata-aware boost로 대응하며, OCR uncertainty는 fallback path와 evidence exposure로 대응한다.

[Insert Table 3-1 here: requirement-to-component traceability]

검색 시점의 hybrid flow는 단일 이미지 유사도에 의존하지 않는다. OCR, caption, metadata, optional text query가 함께 후보군과 순위에 영향을 준다. 이 설계는 사용자가 왜 특정 결과가 상위에 왔는지 이해할 수 있도록 하는 데도 중요하다.

[Insert Figure 3-2 here: query-time hybrid search flow]

Agent와 catalog path는 직접적인 image retrieval만으로 답하기 어려운 상황을 지원한다. 예를 들어 특정 모델의 매뉴얼, PDF 문서, 구조화된 출처 근거가 필요한 경우 내부 catalog 검색과 tool orchestration을 통해 보조 정보를 제공할 수 있다.

[Insert Figure 3-3 here: agent and catalog orchestration path]

Human review는 설계의 중심 원칙이지만, 현재 구현 상태에 맞게 정확히 서술해야 한다. 현재 시스템에서 human-in-the-loop behaviour는 shortlist 자체, OCR 및 score evidence 노출, safer writeback 기본값을 통해 부분적으로 구현되어 있다. 최종 검증 단계에서 agent의 Milvus writeback은 operator가 명시적으로 켜지 않으면 비활성화되도록 변경되었다. 이는 불확실한 식별 결과가 자동으로 새로운 지식이 되는 것을 막는 중요한 설계 수정이다. 반면, production-grade accept/edit/reject workflow와 audit logging은 아직 완전하지 않다. 따라서 본 보고서에서는 이를 “부분 구현” 또는 “planned extension”으로 표현한다.

평가 전략 또한 설계의 일부로 포함되었다. Retrieval effectiveness는 shortlist usefulness에 맞춰 Accuracy@1, Accuracy@5 중심으로 본다. OCR robustness는 character-sensitive domain이라는 특성을 반영해 character-level metric과 qualitative failure analysis로 연결한다. Interactive feasibility는 p50, p90, p95 분석이 가능하도록 component-level timing capture를 설계에 포함한다. 또한 시스템 프로젝트라는 특성을 반영해, engineering reliability 역시 evaluation의 일부로 다룬다. 즉 broken default, fragile import, inconsistent API behaviour 같은 문제도 시스템 성패에 직접 영향을 준다.

## 4. Implementation

구현은 monorepo 구조 위에서 이뤄졌다. `apps/web`은 React 기반 사용자 인터페이스를 제공하고, `apps/api`는 FastAPI route를 통해 인증, hybrid search, catalog search, agent chat을 제공한다. `packages/model`은 실제 hybrid retrieval logic, OCR, embedding, ranking, fusion 로직을 담고 있다.

[Insert Figure 4-1 here: repository structure or module overview]

Frontend는 이미지 업로드, 텍스트 질의 입력, Top-K 결과 표시, agent interaction, catalog interaction을 담당한다. API layer는 이 흐름을 endpoint 단위로 노출하며, backend와 model package 사이의 계약 경계를 형성한다. 이러한 분리는 향후 내부 모델 교체나 ranking 변경이 있어도 UI를 안정적으로 유지할 수 있게 한다.

[Insert Figure 4-2 here: web search UI]

Hybrid search core는 preprocessing, OCR, image embedding, text embedding, optional captioning, vector search, candidate fusion, reranking으로 구성된다. 최근 구현에서는 exact/partial lexical match boost, low-score cutoff, metadata-aware ranking이 강화되었고, Qwen3-VL 계열 image retrieval/reranking과 BGE-M3 텍스트 채널을 함께 쓰는 mixed design 방향이 정리되었다.

[Insert Figure 4-3 here: hybrid score decomposition example]

Catalog RAG 기능은 PDF 문서 chunk를 인덱싱하고 내부 문서 검색을 지원한다. Agent integration은 hybrid search, catalog search, external reasoning path를 하나의 conversational interface로 묶는다. 이때 결과에는 source/page 같은 evidence가 포함되어 사용자가 출처를 검증할 수 있다.

[Insert Figure 4-4 here: agent UI with writeback toggle]

최종 검증 단계에서는 reliability, safety, observability 개선도 수행했다. 대표적으로 agent writeback 기본값을 `false`로 바꾸어 안전성을 높였고, lightweight pytest import가 가능하도록 lazy import를 정리했으며, hybrid path에 structured latency instrumentation을 추가했다. 이 작업은 단순 기능 추가보다 “보고 가능한 품질 개선”이라는 의미가 크다.

## 5. Evaluation

평가는 이 시스템의 실제 목적, 즉 사용자가 올바른 부품을 shortlist 안에서 찾고 그 근거를 이해할 수 있는지에 맞춰 설계되었다. 따라서 evaluation은 retrieval effectiveness, OCR robustness, latency/interactivity, engineering reliability의 네 영역으로 나뉜다.

[Insert Table 5-1 here: evaluation overview and evidence status]

현재 바로 사용할 수 있는 근거 중 하나는 image-only baseline retrieval 결과다. 이는 vision alone이 완전히 무의미하지 않음을 보여주지만, 동시에 그것만으로는 충분하지 않다는 점도 시사한다. 유사 외형 부품이 많은 도메인에서는 OCR과 metadata-aware ranking, lexical boost가 필요하다.

[Insert Table 5-2 here: image-only baseline retrieval results]

정성적 failure analysis는 glare, blur, small text, stylised label, visually similar variants 같은 패턴을 보여준다. 이러한 failure는 왜 OCR을 uncertain evidence로 다뤄야 하는지 설명해 준다.

[Insert Figure 5-1 here: OCR and retrieval failure examples]

OCR robustness에 대해서는 아직 aggregate CER benchmark가 완성되지 않았다. 현재는 프로토콜과 계측 구조는 정의되어 있지만, 최종 보고서에서는 이를 “conducted evaluation”로 쓰면 안 된다. 더 정확한 표현은 protocol defined, instrumentation ready, quantitative benchmark in progress에 가깝다.

Latency evaluation 역시 마찬가지다. Hybrid search path에는 단계별 instrumentation이 구현되어 있어 p50/p90/p95를 계산할 준비가 되어 있다. 그러나 batch summary가 아직 생성되지 않았다면, 이미 latency study가 완료되었다고 주장해서는 안 된다.

반면 engineering validation 측면에서는 더 구체적인 근거가 있다. 최종 artifact bundle에는 다음 회귀 테스트 결과가 기록되어 있다.

- API pytest: `12 passed, 1 warning in 5.37s`
- model pytest: `4 passed in 0.09s`

이 결과는 최근 변경사항, 특히 safer writeback default와 lightweight import/testability 개선이 실제로 동작했음을 보여준다. 이는 retrieval benchmark 전체를 대체하진 않지만, 시스템 품질과 안정성 측면의 중요한 근거다.

[Insert Table 5-3 here: objective status and critical assessment]

종합적으로 보면, 본 프로토타입은 retrieval-first identification assistant로서는 분명히 성공적이다. 동시에 현재 근거는 OCR CER, hybrid ablation, latency percentile, usability outcome까지 완전하게 뒷받침하지는 못한다. 따라서 최종 결론은 “완전 자동 식별 시스템이 이미 검증되었다”가 아니라, “기술적으로 도전적이고 현실적이며, 의미 있는 근거를 가진 identification assistant prototype이 구현되었다”가 되어야 한다.

## 6. Conclusion

본 프로젝트는 중고 플랫폼의 부품 식별 문제를 위해, OCR, multimodal embeddings, vector search, catalog retrieval, agent orchestration을 하나의 실용적 workflow로 통합한 시스템 수준의 기여를 제시했다. 핵심은 새로운 foundation model을 제안한 것이 아니라, 여러 AI 구성요소를 실제 문제 맥락에 맞게 오케스트레이션한 데 있다.

이 보고서는 중고 부품 식별을 retrieval-first, human-in-the-loop 문제로 보는 것이 적절하다고 주장했다. 이는 도메인의 open-world, fine-grained, text-sensitive 특성과, 사용자들이 transparency와 control을 원한다는 점을 모두 반영한 framing이다. 따라서 설계는 Top-K retrieval, evidence-backed result, graceful fallback, listing-oriented structured output을 강조한다.

구현 결과는 이러한 설계가 web, API, model 계층에 걸친 working prototype으로 실현될 수 있음을 보여준다. 또한 최근 validation 작업을 통해 writeback safety, regression coverage, latency instrumentation 측면의 개선도 이루어졌다.

반면 한계도 분명하다. OCR은 현실적인 노이즈에 여전히 취약하며, 전체 review-and-writeback workflow는 아직 완성되지 않았고, hybrid benchmark, OCR CER, latency percentile에 대한 정량 자동화도 완료되지 않았다. 이 한계는 프로젝트의 가치를 부정하는 것이 아니라, 현재 근거가 어디까지 도달했는지를 명확히 보여주는 경계선이다.

따라서 본 시스템을 가장 적절하게 표현하는 말은 **`retrieval-first, human-in-the-loop identification assistant`**이다. 이것이 현재 증거 수준과 가장 잘 맞는 표현이며, 동시에 기술적으로 충분히 의미 있는 시스템 통합 프로젝트의 성과라고 볼 수 있다. 향후 과제로는 evaluation automation 완성, audited review workflow 추가, Qwen-centred와 mixed OCR-plus-text pipeline 비교, hard case에 대한 region focus 실험, user-centred validation 강화 등이 있다.

## Final Assembly Note

최종 DOCX 조립 시에는 다음을 수행해야 한다.

1. insertion marker를 실제 figure/table로 교체한다.
2. 참고문헌은 `submission/reports/Draft.docx`를 기준으로 정리·정비한다.
3. citation spacing과 formatting을 일관되게 맞춘다.
4. 모든 claim이 현재 evidence 수준과 정합적인지 다시 확인한다.
5. 이 작업 메모와 내부 drafting annotation은 제거한다.
