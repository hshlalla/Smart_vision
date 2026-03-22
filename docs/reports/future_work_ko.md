# Future Work 정리

이 문서는 현재 프로토타입에서 **지금 당장 구현하지는 않지만**, 이후 확장이나 제품화 단계에서 중요한 개선 항목들을 정리한 문서다.  
최종 보고서의 Future Work, Limitations, Improvement Plan 파트에 직접 활용할 수 있도록 작성했다.

## 1. Metadata-Only Draft Registration

현재 인덱싱 흐름은 이미지 업로드를 전제로 한다.  
즉 사용자는 이미지를 업로드한 뒤 metadata preview를 생성하고, 수정 후 confirm을 눌러 저장하게 된다.

향후에는 다음과 같은 기능을 추가할 수 있다.

- 이미지 없이도 draft item 생성
- maker, part number, category, description만 먼저 등록
- 이후 이미지가 준비되면 later attachment 방식으로 추가 업로드
- 기존 draft item에 이미지 임베딩과 텍스트 임베딩을 병합 반영

이 기능은 실제 중고 등록 워크플로우에서 유용할 수 있다. 예를 들어 판매자가 제품 정보는 먼저 알고 있지만, 이미지 촬영은 나중에 하려는 경우에 도움이 된다.

다만 현재 구조에서는 다음 변경이 필요하다.

- draft/completed 상태 관리
- image-less item 허용 데이터 모델
- later image indexing API
- 기존 item 업데이트/병합 로직

따라서 이 기능은 유용하지만, 현재 프로토타입 단계에서는 안정성보다 구조 변경 비용이 더 크므로 future work로 두는 것이 적절하다.

## 2. Audited Review-and-Writeback Workflow

현재 시스템은 `preview -> edit -> confirm` 구조를 통해 human-in-the-loop 인덱싱을 제공한다.  
그러나 더 완전한 production-grade workflow를 위해서는 다음이 필요하다.

- accept / edit / reject 상태 구분
- review history 저장
- audit logging
- 누가 어떤 메타데이터를 수정했는지 기록
- rollback 가능한 writeback history

이러한 기능은 실제 서비스 운영과 데이터 품질 관리 측면에서 중요하다.

## 3. Selective OCR Verification Policy

실험 결과상 OCR은 retrieval의 primary driver로 사용하기보다, verification signal로 제한하는 편이 더 실용적일 수 있다.  
향후에는 다음과 같은 선택적 OCR 정책을 적용할 수 있다.

- 기본 검색 경로에서는 OCR 비중 축소
- low-confidence case에서만 OCR 활성화
- OCR 결과를 ranking signal보다 verification evidence로 사용
- label region만 선택적으로 OCR 수행

이 방향은 latency와 robustness를 동시에 개선할 가능성이 있다.

## 4. Region-Focused and Multi-View Processing

현재도 여러 장 이미지 입력은 지원하지만, 향후에는 더 정교한 region-focused 설계가 가능하다.

- 라벨 영역 자동 탐지
- 포트/버튼/로고 등 key region focus
- multi-view aggregation scoring
- 한 제품의 여러 이미지에서 evidence를 구조적으로 통합

이 기능은 fine-grained product identification에서 중요한 개선 포인트다.

## 5. Stronger OCR and Document Parsing Integration

OCR과 catalog 기능은 이미 존재하지만, 향후에는 더 긴밀하게 통합할 수 있다.

- catalog PDF와 OCR evidence joint reasoning
- OCR text와 catalog candidate 간 direct alignment
- part number 후보를 catalog 문서 안에서 즉시 검증
- document parsing 결과를 retrieval ranking에 반영

이러한 방향은 단순 검색을 넘어서 더 강한 evidence chain을 제공할 수 있게 한다.

## 6. Better Deployment and Runtime Portability

현재 실험 과정에서 환경 제약, 특히 로컬 하드웨어와 dependency 호환성 문제가 관찰되었다.  
향후에는 다음이 필요하다.

- Linux GPU 기준 표준 실행 환경
- containerised deployment
- stable model/runtime compatibility matrix
- local development mode와 production mode 분리

이는 실험 재현성과 운영 안정성을 높이는 데 중요하다.

## 7. Extended Retrieval Benchmarking

현재 실험은 retrieval, OCR, latency, usability를 포함하지만, 향후에는 더 체계적인 benchmark 확장이 가능하다.

- larger-scale held-out benchmark
- more domain-balanced split
- difficult-case benchmark set
- cross-domain generalisation benchmark
- repeated-run variance analysis

이는 결과의 일반화 가능성을 더 강하게 뒷받침할 수 있다.

## 8. Larger-Scale User Study

현재 usability 평가는 pilot 규모에 적합하다.  
향후에는 다음을 포함한 더 큰 사용자 연구가 필요하다.

- 더 많은 participant 수
- novice vs experienced seller 비교
- manual baseline 대비 task-time 절감 비교
- trust and adoption intention 분석
- real listing scenario 기반 longitudinal evaluation

이러한 평가가 있으면 prototype의 실제 사용 가치를 더 강하게 입증할 수 있다.

## 9. Productisation-Oriented Improvements

향후 제품 수준으로 확장하려면 다음 기능도 고려할 수 있다.

- duplicate item detection
- inventory linking
- seller-facing suggestion history
- automatic listing draft export
- marketplace integration

이는 현재의 research prototype을 실사용 도구로 확장하는 방향이다.

## 10. Report-Friendly Summary Paragraph

아래 문단은 최종 보고서 Future Work 부분에 바로 붙여 넣을 수 있다.

> Several extensions remain for future work. First, the indexing workflow could be expanded to support metadata-only draft registration followed by later image attachment, which would better reflect real marketplace listing behaviour. Second, the current preview-confirm flow could be extended into a fully audited accept/edit/reject workflow with revision history and rollback support. Third, OCR should be refined into a selective verification layer rather than a default high-cost retrieval component, potentially through region-focused OCR and multi-view evidence aggregation. Additional future directions include tighter catalog-document integration, more stable Linux/GPU deployment, larger and more difficult retrieval benchmarks, and a larger-scale usability study with novice and experienced sellers.

## 11. Korean Short Summary

짧게 쓰고 싶으면 아래처럼 줄여도 된다.

> 향후 과제로는 metadata-only draft registration, audited review-and-writeback workflow, selective OCR verification policy, region-focused multi-view processing, catalog-document integration, 안정적인 Linux/GPU 배포, 더 큰 retrieval benchmark, 그리고 larger-scale usability study가 있다.
