# 5. Evaluation

평가는 이 시스템의 실제 목적, 즉 사용자가 올바른 부품을 shortlist 안에서 찾고 그 근거를 이해할 수 있는지에 맞춰 설계되었다. 따라서 evaluation은 end-to-end workflow success, retrieval effectiveness, OCR robustness, latency/interactivity, engineering reliability의 다섯 영역으로 나뉜다. 이는 단일 accuracy 수치보다 시스템의 실제 사용 가치를 더 잘 반영한다.

[Insert Table 5-1 here: evaluation overview and evidence status]

현재 바로 사용할 수 있는 근거 중 하나는 image-only baseline retrieval 결과다. 이는 vision alone이 완전히 무의미하지 않음을 보여주지만, 동시에 그것만으로는 충분하지 않다는 점도 시사한다. 유사 외형 부품이 많은 도메인에서는 OCR과 metadata-aware ranking, lexical boost가 필요하다.

[Insert Table 5-2 here: image-only baseline retrieval results]

정성적 failure analysis는 glare, blur, small text, stylised label, visually similar variants 같은 패턴을 보여준다. 이러한 failure는 왜 OCR을 uncertain evidence로 다뤄야 하는지 설명해 준다.

[Insert Figure 5-1 here: OCR and retrieval failure examples]

새로운 평가 작업은 이제 단순 계획 수준을 넘었다. 현재 저장소에는 고정된 unified dataset split, retrieval evaluation input, experiment execution guide, result table template가 준비되어 있다. 따라서 실험 자체가 아직 일부 진행 중이더라도, evaluation input은 ad hoc이 아니라 재현 가능한 형태로 고정되어 있다는 점을 명시할 수 있다.

End-to-end 동작 측면에서는 current-index suite 기준 `8/8` scenario success가 확인되었다. 이는 업로드, 인덱싱, 검색, evidence display까지 이어지는 기본 workflow가 operational level에서 동작함을 보여주는 직접 근거다.

Retrieval effectiveness 측면에서는 과거 image-only baseline 외에도 현재 hybrid retrieval configuration 비교 실험을 수행하도록 설계되어 있다. 이 비교는 `C1` OCR off mixed pipeline, `C2` OCR on mixed pipeline, `C3` reranker 제거 baseline, `C4` text-light baseline을 중심으로 구성된다. 실험 수치가 확정되면 Accuracy@1, Accuracy@5, Recall@5, MRR, exact identifier hit rate를 표로 제시해야 한다.

OCR robustness는 여전히 조심스럽게 서술해야 한다. 현재 시스템에는 mature한 OCR path, explicit OCR on/off control, pilot benchmark script가 존재한다. 이는 OCR brittleness가 프로젝트 동기의 핵심이기 때문에 중요하다. 그러나 controlled identifier set에 대한 aggregate CER / exact-match benchmark는 아직 완성되지 않았다. 따라서 최종 보고서에서는 OCR evaluation이 in progress라고 쓰는 것이 맞다.

Latency evaluation도 이전보다 근거가 강해졌다. Query path 내부에 단계별 instrumentation이 존재하며, total latency뿐 아니라 preprocessing, embedding, retrieval, rerank를 분리해 기록할 수 있다. 다만 최종 보고서에서는 p50, p90, p95와 같은 summary 수치를 실제로 채운 뒤 서술 수위를 조정해야 한다.

사용성 평가는 아직 최종 수치가 없는 영역이다. 다만 현재는 listing-oriented task와 post-task Google Form 기반 pilot protocol이 정의되어 있으므로, 최종 보고서에서는 “pilot protocol defined; aggregate outcomes pending”으로 표현하는 것이 적절하다. 외부 사용자 테스트를 위해 web tunnel 기반 접근 방식도 문서화되어 있다.

Engineering validation 측면에서는 더 구체적인 근거가 있다. March artifact bundle에는 API pytest `12 passed, 1 warning in 5.37s`, model pytest `4 passed in 0.09s`가 기록되어 있다. 이후 reliability refresh에서는 API regression tests, model-package tests, frontend production build, retrieval-eval input generation이 모두 정상으로 확인되었다. 이는 full benchmark suite를 대체하진 않지만, 최근 safety/testability 변경이 실제로 검증되었다는 중요한 근거다.

[Insert Table 5-3 here: objective status and critical assessment]

종합적으로 보면, 본 프로토타입은 retrieval-first identification assistant로서는 분명히 성공적이다. 동시에 현재 근거는 OCR CER, hybrid ablation, latency percentile, usability outcome까지 완전하게 뒷받침하지는 못한다. 따라서 최종 결론은 “완전 자동 식별 시스템이 이미 검증되었다”가 아니라, “기술적으로 도전적이고 현실적이며, 의미 있는 근거를 가진 identification assistant prototype이 구현되었다”가 되어야 한다.

실험 결과가 채워질 때는 반드시 다음 문서와 같이 사용한다.

- 결과표 입력: [`experiment_result_tables_template_ko.md`](/Users/mac/project/Smart_vision/docs/reports/experiment_result_tables_template_ko.md)
- 결과 서술 문장: [`experiment_results_writeup_template_ko.md`](/Users/mac/project/Smart_vision/docs/reports/experiment_results_writeup_template_ko.md)
