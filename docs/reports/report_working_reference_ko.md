# 보고서 작업용 레퍼런스 (한국어 대응본)

> 작업용 reference only.
> 이 문서는 `docs/reports/report_working_reference.md`의 한국어 대응용 companion note다.
> 원문은 과거 작업용 장문 레퍼런스이며, 현재 실제 최종 정리 기준은 `docs/reports/final_report_status.md`와 `docs/reports/final_report_docx_ready.md`다.

University of London  
Bachelor in Computer Science

Final Project  
Smart Image Part Identifier for Secondhand Platforms  
CM3020 Artificial Intelligence, "Orchestrating AI models to achieve a goal"

Name: SuHun Hong  
Email: hshlalla@naver.com

## 문서 성격

이 문서는 과거에 작성된 장문 영문 reference의 한국어 대응본이다.  
다만 원문이 매우 길고 현재 최종 보고서 기준 문서가 아니기 때문에, 여기서는 **직역본**이 아니라 **장별 핵심 내용과 논리 구조를 빠르게 파악할 수 있는 한국어 companion summary** 형태로 정리한다.

## 핵심 문제의식

프로젝트는 중고 플랫폼에서 산업용·전자 부품을 사진으로 식별하는 문제를 다룬다. 이 문제는 일반 소비재 검색보다 어렵다. 그 이유는 다음과 같다.

- 부품은 open-world inventory를 가진다.
- visually similar한 변형이 많아 fine-grained ambiguity가 크다.
- 실제 구분 단서는 작은 텍스트(label, serial, model code)인 경우가 많다.
- 사용자가 올리는 이미지는 blur, glare, occlusion 등으로 품질이 일정하지 않다.
- 사용자는 정답 하나보다 shortlist와 검증 가능한 evidence를 필요로 한다.

## 원문 reference의 주요 논리 흐름

### 1. Introduction 성격의 내용

원문은 먼저 이 문제의 실무적 중요성을 설명한다. 중고 거래 환경에서는 식별 실패가 listing quality와 거래 성공에 직접 영향을 준다. 따라서 단순한 시각 유사도보다 evidence-backed identification support가 필요하다고 본다.

### 2. Literature Review 성격의 내용

원문 reference가 검토하는 선행연구 축은 대략 다음과 같다.

- content-based image retrieval
- industrial part retrieval
- OCR for noisy identifiers
- multimodal embedding and cross-modal retrieval
- vector database and scalable retrieval
- human-centred interaction and feedback

핵심 결론은 단일 접근만으로는 부족하다는 것이다. 이미지 검색은 useful하지만 세부 부품 식별에는 불충분할 수 있고, OCR은 결정적 단서를 줄 수 있지만 noisy하다. 따라서 hybrid system design이 필요하다.

### 3. Design 성격의 내용

원문은 시스템을 classification보다 retrieval 중심으로 설계하려는 방향을 강조한다. 주요 아이디어는 다음과 같다.

- 이미지, OCR, 메타데이터를 함께 활용
- Milvus와 같은 vector storage 활용
- Top-K shortlist 반환
- 사용자에게 근거와 설명 제공
- uncertainty를 고려한 workflow 설계

### 4. Implementation 성격의 내용

과거 reference는 prototype을 구현하는 데 필요한 구성요소를 폭넓게 다뤘다. 여기에는 image processing, OCR integration, embedding, search backend, UI flow, metadata handling 등이 포함된다. 현재 기준으로 보면 일부는 실제 구현으로 이어졌고, 일부는 초기 설계 아이디어 수준에 머문 부분도 있다.

### 5. Evaluation 성격의 내용

원문 reference는 retrieval quality, OCR robustness, latency, usability 같은 평가 항목이 왜 중요한지 설명한다. 다만 현재 최종 보고서 기준에서는, 그 당시 reference의 일부 표현이 실제 완료된 증거 수준보다 앞서 있었을 가능성이 있으므로 그대로 제출본 표현으로 쓰면 안 된다.

### 6. Future Work 성격의 내용

향후 과제로는 다음과 같은 방향이 반복적으로 나타난다.

- OCR robustness 강화
- region focus 또는 detection 기반 접근
- richer metadata integration
- better human feedback loop
- stronger quantitative evaluation

## 현재 문서와의 관계

이 한국어 대응본은 과거 장문 reference를 이해하기 쉽게 풀어주는 용도이며, 실제 제출본 작성 기준은 아니다. 현재는 아래 문서들을 우선적으로 본다.

1. `docs/reports/final_report_status.md`
2. `docs/reports/final_report_revision_checklist.md`
3. `docs/reports/final_report_docx_ready.md`
4. `docs/reports/final_report_figures_tables_plan.md`

## 사용 팁

이 문서는 다음 상황에서 유용하다.

- 과거 reference가 어떤 문제의식에서 출발했는지 빠르게 복기할 때
- 최신 최종본이 왜 retrieval-first / human-in-the-loop framing으로 정리되는지 이해할 때
- 예전 장문 원문을 전부 다시 읽지 않고도 핵심 배경을 파악하고 싶을 때

반대로, 아래 용도로는 이 문서를 직접 기준으로 삼지 않는 것이 좋다.

- 최종 제출 문장 복사
- 현재 구현 상태 주장
- 최신 evaluation 근거 확인

이런 용도에는 최신 상태 문서와 DOCX-ready 원고를 우선 사용해야 한다.
