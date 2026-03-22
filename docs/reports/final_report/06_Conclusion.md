# 6. Conclusion

## 6.1 Summary of Contributions

본 프로젝트는 중고 플랫폼의 부품 식별 문제를 위해, OCR, multimodal embeddings, vector search, catalog retrieval, agent orchestration을 하나의 실용적 workflow로 통합한 시스템 수준의 기여를 제시했다. 핵심은 새로운 foundation model을 제안한 것이 아니라, 여러 AI 구성요소를 실제 문제 맥락에 맞게 오케스트레이션한 데 있다.

이 보고서는 중고 부품 식별을 retrieval-first, human-in-the-loop 문제로 보는 것이 적절하다고 주장했다. 이는 도메인의 open-world, fine-grained, text-sensitive 특성과, 사용자들이 transparency와 control을 원한다는 점을 모두 반영한 framing이다. 따라서 설계는 Top-K retrieval, evidence-backed result, graceful fallback, listing-oriented structured output을 강조한다.

구현 결과는 이러한 설계가 web, API, model 계층에 걸친 working prototype으로 실현될 수 있음을 보여준다. 또한 최근 validation 작업을 통해 writeback safety, regression coverage, latency instrumentation, unified dataset preparation 측면의 개선도 이루어졌다. 최종 실험 보고서와 로컬 추가 검증을 함께 보면, 보고서 기준 주 비교군에서는 `C4`가 가장 강한 benchmark 결과를 보였고, 실제 운영 권고 구성은 `C3 (OCR off, reranker off)`로 정리된다.

## 6.2 Limitations

반면 한계도 분명하다. OCR은 현실적인 노이즈에 여전히 취약하며, 전체 review-and-writeback workflow는 아직 production-grade 수준으로 완성된 것은 아니다. 또한 현재 실험 결과는 prototype과 local evaluation 환경에서 얻어진 것이므로, 더 넓은 환경과 사용자 집단에 대한 일반화에는 추가 검증이 필요하다. 이 한계는 프로젝트의 가치를 부정하는 것이 아니라, 현재 근거가 어디까지 도달했는지를 명확히 보여주는 경계선이다.

## 6.3 Future Work

향후 과제로는 evaluation automation 완성, audited review workflow 추가, selective OCR verification policy 고도화, hard case에 대한 region focus 실험, user-centred validation 강화 등이 있다. 또한 metadata-only draft registration, later image attachment, stronger catalog-document integration, Linux/GPU deployment stabilisation도 의미 있는 확장 방향이다.

## 6.4 Concluding Remarks

따라서 본 시스템을 가장 적절하게 표현하는 말은 **`retrieval-first, human-in-the-loop identification assistant`**이다. 이것이 현재 증거 수준과 가장 잘 맞는 표현이며, 동시에 기술적으로 충분히 의미 있는 시스템 통합 프로젝트의 성과라고 볼 수 있다.
