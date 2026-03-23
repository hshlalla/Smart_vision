# 6. Conclusion and Future Work

이 장은 프로젝트의 주요 성과를 요약하고, 아키텍처적·기술적 기여를 정리하며, 현재 prototype과 future productisation 또는 research work 사이의 경계를 규정하는 한계를 논의한다.

## 6.1 Project Summary

본 프로젝트는 중고 플랫폼에서 산업용 및 전자 부품을 식별하고 listing하는 과정의 마찰을 줄이는 것을 목표로 했다. “Orchestrating AI models” 템플릿에 따라, 시스템은 비전문 판매자가 사용자 업로드 사진과 관련 evidence를 바탕으로 listing-relevant metadata를 추론할 수 있도록 돕는 방향으로 설계되었다.

평가 과정에서 중요한 전환이 발생했다. 초기에는 OCR-heavy retrieval pipeline이 explicit identifier를 더 잘 회수해 가장 강한 결과를 낼 것이라고 가정할 수 있었다. 그러나 empirical benchmark 결과는 산업 환경에서 전통적 OCR이 severe noise와 substantial latency를 동시에 유발한다는 점을 보여주었다. 그 결과 프로젝트는 Qwen3-VL을 이용한 layout-aware image understanding과 BGE-M3를 이용한 robust text representation을 결합한 vision-dominant hybrid retrieval architecture로 중심을 옮기게 되었다. 이 orchestration은 human-in-the-loop workflow 안에 배치되어, 시스템이 fully autonomous classifier가 아니라 evidence-backed decision-support assistant로 기능하도록 했다.

## 6.2 Key Contributions

본 프로젝트의 주요 기여는 다음과 같이 정리할 수 있다.

1. **OCR 한계에 대한 empirical evidence 제시**  
   OCR을 default first-stage retrieval signal로 사용하는 것이 noisy industrial imagery에서는 retrieval quality를 떨어뜨리고 latency를 증가시킬 수 있다는 구체적 실험 근거를 제시했다.

2. **Retrieval-first vision-language orchestration pipeline 구현**  
   Qwen3-VL 기반 image understanding, BGE-M3 text retrieval, metadata-aware scoring, hybrid ranking을 하나의 operational retrieval workflow로 통합했다. main benchmark에서 vision-dominant configuration은 약 `91%` Accuracy@1을 기록했다.

3. **실용적인 human-in-the-loop prototype 제공**  
   `preview-confirm` workflow를 중심으로 하는 working end-to-end prototype을 구현했다. 이는 transparency, editability, evidence visibility에 대한 사용자 요구를 직접 반영하며, 시스템을 black-box prediction이 아닌 AI-assisted user verification 방향으로 전환한다.

4. **단일 모델 데모를 넘어서는 system-level contribution**  
   search, indexing, catalog retrieval, agent-assisted evidence expansion을 하나의 일관된 workflow로 통합했다. 따라서 핵심 기여는 새로운 foundation model 제안이 아니라, 여러 AI 구성요소를 실제 문제 맥락에 맞게 실용적으로 orchestrate한 데 있다.

## 6.3 Limitations and Future Work

retrieval-first architecture의 타당성은 확인되었지만, 여전히 한계가 남아 있으며 이는 다음 단계의 과제를 규정한다.

- **Workflow enhancements**  
  현재 indexing path는 이미지 업로드에서 시작한다. 향후에는 **metadata-only draft registration**을 지원해 사용자가 known metadata로 listing을 시작하고 나중에 이미지를 붙일 수 있게 하는 것이 유용하다. 또한 현재의 `preview-confirm` sequence는 revision history와 rollback support를 포함하는 fully **audited review-and-writeback workflow**로 확장될 수 있다.

- **Selective OCR and region-focused refinement**  
  평가 결과는 OCR을 indiscriminately 적용해서는 안 된다는 점을 보여준다. 향후 버전에서는 필요한 경우에만 OCR을 호출하는 **selective verification policy**가 필요하다. label-region detection, rotation correction, multi-view evidence aggregation 같은 region-focused processing도 difficult case를 더 잘 다룰 수 있게 해 줄 것이다.

- **Scaling and deployment**  
  더 넓은 운영을 위해서는 local hardware constraint를 줄이고 heavier multimodal path의 재현성을 높일 수 있는 stable Linux/GPU environment로의 이전이 필요하다.

- **Broader validation**  
  현재 실험은 유용한 근거를 제공하지만, 앞으로는 novice seller와 experienced seller를 모두 포함하는 larger-scale user study, stronger hard-case benchmark, broader deployment-oriented validation이 필요하다.

## 6.4 Concluding Remarks

fine-grained secondhand part identification은 일반 소비재 이미지 검색과 근본적으로 다른 문제다. 이 프로젝트는 AI 구성요소를 단순히 많이 쌓는다고 해서 자동으로 더 나은 시스템이 되는 것은 아니라는 점을 보여준다. 실제로 더 효과적인 설계는 visual retrieval, OCR, reranking, latency, user trust 사이의 trade-off를 비판적으로 검증하는 과정에서 도출되었다.

따라서 최종 결과를 가장 잘 설명하는 표현은 **`retrieval-first, human-in-the-loop identification assistant`**이다. 이 표현은 구현된 workflow와 현재 확보된 evidence 수준을 모두 가장 잘 반영한다. 본 프로젝트의 가치는 fully autonomous identification을 과장해서 주장하는 데 있지 않고, state-of-the-art AI component를 어려운 실제 도메인에 맞는 practical, evidence-backed listing assistant로 orchestrate할 수 있음을 보여준 데 있다.
