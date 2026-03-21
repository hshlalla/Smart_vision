# 2. Literature Review

시각 검색은 handcrafted descriptor에서 시작해, 대규모 semantic search를 지원하는 deep feature embedding으로 발전해 왔다. eBay image search와 같은 소비자용 시스템은 사용자가 정확한 키워드를 알지 못하더라도 이미지 기반 질의가 실용적일 수 있음을 보여준다. Google Lens와 유사한 시스템 역시 이러한 패턴을 더 넓은 객체군으로 확장한다. 하지만 이들 사례는 대체로 전체 소비재 수준에서의 검색을 다루며, 미세한 차이를 구별해야 하는 산업용 부품 문제와는 차이가 있다.

산업용 부품이나 세부 식별 문제를 다룬 선행연구는 본 프로젝트에 더 직접적인 근거를 제공한다. 이러한 연구들은 이미지 유사도만으로는 충분하지 않으며, 텍스트 단서와 구조화된 메타데이터가 함께 고려되어야 한다는 점을 보여준다. 특히 OCR과 visual retrieval을 결합하는 접근은 외형이 유사하지만 식별 텍스트가 중요한 도메인에서 유효하다.

OCR 관련 문헌은 텍스트 추출이 유용하지만 불안정하다는 점을 반복적으로 보여준다. 특히 glare, blur, 왜곡, 작은 글씨, worn-out label은 OCR 정확도를 떨어뜨린다. 이는 본 프로젝트에서 OCR을 “핵심 보조 증거”로 사용하되, 유일한 정답 근거로 보지 않는 설계로 이어졌다.

멀티모달 임베딩과 텍스트 검색 관련 문헌은 이미지와 텍스트를 공동 임베딩 공간에서 다루는 방법, 그리고 다국어·다기능 텍스트 검색의 중요성을 제시한다. 이는 이미지 기반 질의와 OCR 텍스트, 메타데이터, 사용자 텍스트 질의를 함께 활용하는 hybrid retrieval 구조를 정당화한다.

벡터 데이터베이스와 hybrid search 관련 선행사례는 여러 종류의 벡터와 메타데이터를 분리 저장하고, 단계적으로 후보를 수집·재정렬하는 방식이 대규모 시스템에 적합하다는 점을 보여준다. 이는 Milvus를 이용한 multi-collection 설계와 ranking fusion 로직의 근거가 된다.

마지막으로 interactive feedback과 user-centred AI literature는, 사용자가 결과를 검증하고 수정할 수 있을 때 시스템에 대한 신뢰가 높아진다는 점을 시사한다. 이는 본 프로젝트에서 evidence-backed shortlist와 safer writeback policy를 채택하게 된 중요한 배경이다.

이 문헌 검토를 통해 도출되는 핵심은 다음과 같다. 첫째, 부품 식별은 단순 image classification보다 retrieval-first 구조에 더 잘 맞는다. 둘째, OCR은 필요하지만 완전한 ground truth가 아니라 uncertain evidence로 다뤄야 한다. 셋째, hybrid retrieval과 reranking, 그리고 human confirmation이 함께 들어갈 때 실제 사용 맥락에 더 잘 맞는 시스템이 된다. 이러한 결론은 이후 설계와 구현 파트의 방향을 직접적으로 결정한다.
