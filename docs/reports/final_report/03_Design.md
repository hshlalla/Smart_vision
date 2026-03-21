# 3. Design

본 시스템은 React 기반 웹 인터페이스, FastAPI 기반 API 계층, 그리고 hybrid retrieval orchestration을 담당하는 model 계층으로 구성된다. 전체 목표는 사용자가 업로드한 부품 이미지를 입력으로 넣었을 때, image evidence, OCR evidence, metadata text, caption text를 함께 사용해 관련 후보를 수집하고, 최종적으로 evidence-backed shortlist를 반환하는 것이다.

[Insert Figure 3-1 here: overall system architecture]

상위 수준에서 보면 시스템은 세 개의 상호작용 루프를 가진다. 첫째는 **query-time retrieval loop**다. 사용자는 이미지만 올리거나, 이미지와 텍스트를 함께 입력할 수 있다. 둘째는 **indexing loop**다. 새로운 아이템을 등록할 때 메타데이터 초안을 생성하고, 사용자가 수정한 뒤 확정 저장한다. 셋째는 **evidence expansion loop**다. 필요하면 catalog search나 외부 검색을 통해 추가 근거를 제공한다.

Query-time retrieval path의 핵심은 여러 신호를 따로 계산한 뒤 최종적으로 fusion하는 구조다. 이미지는 image embedding collection으로 검색되고, OCR이나 metadata 기반 텍스트는 text collection과 caption collection을 통해 검색된다. 이후 각 후보의 image score, text score, caption score, lexical signal, exact identifier boost를 조합해 최종 점수를 계산한다. 이 방식은 visually similar but textually distinct 항목을 구분하는 데 유리하다.

[Insert Figure 3-2 here: query-time hybrid retrieval flow]

인덱싱 경로는 human-in-the-loop를 전제로 설계되었다. 사용자가 이미지를 업로드하면 시스템은 곧바로 확정 저장하지 않고, 먼저 GPT 기반 metadata preview를 생성한다. 사용자는 여기서 maker, part number, category, description 등을 확인하고 수정할 수 있다. 실제 저장은 confirm 이후에만 수행된다. 이 구조는 자동 writeback보다 안전하며, 실제 listing workflow에 더 자연스럽다.

또한 본 시스템은 multi-image 입력을 고려한다. 단일 이미지로는 제품의 앞면, 옆면, 라벨면, 포트면을 모두 알 수 없기 때문에, metadata preview는 여러 장을 함께 보고 생성하고, 실제 인덱싱도 여러 장을 같은 item 단위로 반영하도록 설계하였다. 이 점은 fine-grained part identification에서 특히 중요하다.

설계상 중요한 또 다른 요소는 **graceful fallback**이다. OCR은 유용하지만 불안정하다. 따라서 OCR이 실패해도 image retrieval과 metadata-aware ranking이 어느 정도 작동해야 한다. 반대로 visual similarity만으로 부족한 경우에는 exact identifier boost나 text retrieval이 보강 역할을 한다. 이러한 fallback-oriented hybrid design은 실제 noisy image 조건에서 robustness를 높이기 위한 선택이다.

추가적으로 시스템은 catalog retrieval과 agent orchestration도 포함한다. Catalog 경로는 PDF나 reference document를 벡터화하여 내부 문서 검색을 수행할 수 있게 하며, agent 경로는 hybrid search, catalog search, 그리고 외부 web search를 단계적으로 호출해 더 풍부한 근거를 제공한다.

[Insert Figure 3-3 here: agent and catalog orchestration path]

평가 전략도 설계 안에 포함된다. 본 프로젝트는 단순 accuracy 하나로 성능을 주장하지 않는다. 대신 retrieval effectiveness, OCR robustness, latency/interactivity, engineering reliability, usability를 서로 다른 실험 트랙으로 나누어 검증하도록 설계되었다. 이는 시스템이 실제로는 검색 도구이자 listing assistance workflow라는 점을 반영한 것이다.

[Insert Table 3-1 here: requirement-to-component traceability]
