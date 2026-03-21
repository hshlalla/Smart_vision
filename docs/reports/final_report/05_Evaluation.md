# 5. Evaluation

평가는 이 시스템의 실제 목적, 즉 사용자가 올바른 부품을 shortlist 안에서 찾고 그 근거를 이해할 수 있는지에 맞춰 설계되었다. 따라서 evaluation은 end-to-end workflow success, retrieval effectiveness, OCR robustness, latency/interactivity, engineering reliability, usability readiness의 여섯 축으로 정리한다. 이는 단일 accuracy 수치보다 시스템의 실제 사용 가치를 더 잘 반영한다.

[Insert Table 5-1 here: evaluation overview and evidence status]

먼저 vision-only baseline은 retrieval-first 접근 자체가 유효하다는 점을 보여준다. 기존 draft-stage image-only 실험에서는 random 1000 models split에서 `Accuracy@1 = 0.287`, `Accuracy@5 = 0.791`, category-sampled 500 models split에서 `Accuracy@1 = 0.306`, `Accuracy@5 = 0.812`가 관찰되었다. 이 값은 image-only retrieval이 완전히 무의미하지 않지만, fine-grained part identification에서는 OCR, metadata, lexical boosting 같은 보조 신호가 필요하다는 근거가 된다.

[Insert Table 5-2 here: image-only baseline retrieval results]

이번 최종 실험 보고서의 main benchmark는 `1000-item`, `900/100 split`, `C2 vs C4` 비교다. `C2`는 `OCR on + reranker on + text channel on`, `C4`는 `OCR off + text-light + reranker on` 구성이다. 보고서 기준 정량 결과에서 `C2`는 `Accuracy@1 = 0.86`, `Accuracy@5 = 0.95`, `MRR = 0.903`, `Exact identifier hit = 0.81`을 기록했고, `C4`는 `Accuracy@1 = 0.91`, `Accuracy@5 = 0.97`, `MRR = 0.939`, `Exact identifier hit = 0.88`을 기록했다. 따라서 main benchmark 안에서는 `C4`가 retrieval quality 측면에서 더 우수한 구성으로 정리된다.

다만 최종 운영 권고는 여기서 끝나지 않는다. 추가 로컬 검증에서는 `C3 (OCR off, reranker off)`와 `C1 (OCR off, reranker on)`을 비교했다. `C3`는 sampled holdout 기준 group `Hit@1 = 1.0`, group `Hit@5 = 1.0`, `MRR = 1.0`, exact `item_id@1 = 0.9667`을 기록했고, warm mean total latency는 `731.13 ms`였다. 반면 `C1`은 retrieval quality 이득 없이 warm mean total latency가 `89337.71 ms`까지 증가했다. 따라서 main benchmark에서 `C4`가 더 강한 주 비교군 결과를 보였더라도, 실제 운영 권고는 `C3`로 정리하는 것이 더 타당하다.

Latency 측면에서도 같은 결론이 나온다. 보고서 기준 `C2`의 warm mean total latency는 `8.24s`, `C4`는 `1.42s`였다. `C2`는 대부분의 시간을 OCR 전처리에 사용했고, `C4`는 OCR을 제거하면서 전처리와 rerank 중심 구조로 바뀌었다. 여기에 로컬 추가 검증을 더하면, `C3`는 `731.13 ms`, `C1`은 `89337.71 ms`를 기록했다. 즉 OCR과 reranker를 모두 기본 경로에 넣는 것은 현재 운영 환경에서 비용이 크며, 최종 권고 구성으로는 적절하지 않다.

OCR benchmark는 retrieval benchmark와 분리해서 해석해야 한다. 최종 실험 보고서의 identifier-visible subset `200`장 기준 benchmark에서 `PaddleOCR`는 `Exact full-string = 0.19`, `CER = 1.12`, `Part number recall = 0.35`, `Maker recall = 0.44`였다. `Qwen-only`는 `0.57 / 0.46 / 0.75 / 0.82`, `OCR + Qwen merged`는 `0.61 / 0.41 / 0.79 / 0.86`을 기록했다. 이 결과는 OCR이 완전히 무의미하다는 뜻이 아니라, primary retrieval engine보다는 verification signal로 사용할 때 더 가치가 크다는 해석을 지지한다.

[Insert Figure 5-1 here: OCR and retrieval failure examples]

정성 분석도 이 해석을 뒷받침한다. 산업 이미지에는 작은 라벨, 저대비 각인, 세로쓰기와 가로쓰기가 혼재된 표식, 표 형태 텍스트, 포장재와 배경 clutter가 자주 등장한다. OCR은 이 모든 문자열을 일괄적으로 추출하면서 retrieval에 필요한 핵심 identifier보다 노이즈를 더 많이 늘리는 경우가 많았다. 반대로 VL 기반 경로는 배치, 위치, 로고 형태, 물체 구조를 함께 해석해 더 안정적인 후보군을 만들었다.

Engineering reliability는 별도 근거가 있다. March evidence bundle과 later reliability refresh 기준으로 API regression tests, model-package tests, frontend production build, retrieval-eval input generation이 정상 동작함이 확인되었다. 따라서 본 시스템은 단지 연구 아이디어 수준이 아니라, 실제로 작동하고 테스트 가능한 prototype이라는 점을 주장할 수 있다.

사용성 평가는 정성적으로는 충분히 정당화되지만, 정량 집계까지 완전히 포함하지는 않았다. listing-oriented task flow, metadata preview-confirm path, duplicate-review prompt, source-backed agent response, catalog evidence display는 모두 구현되었고, pilot usability protocol도 정의되었다. 최종 보고서에서는 이를 “user-centred workflow and pilot protocol prepared” 수준으로 적는 것이 가장 안전하다.

[Insert Table 5-3 here: objective status and critical assessment]

종합하면, 이번 평가에서 가장 중요한 결론은 세 가지다. 첫째, retrieval-first identification assistant라는 framing은 유효하다. 둘째, main benchmark에서는 `C4`가 `C2`보다 더 우수했다. 셋째, 최종 운영 관점에서는 로컬 추가 검증까지 반영해 `C3`가 가장 현실적인 권고 구성이다. 따라서 본 시스템은 OCR과 reranker를 항상 기본 경로에 두기보다, VL 중심 retrieval을 기본으로 두고 OCR은 선택적 검증 레이어로 사용하는 편이 더 실용적이라고 결론내릴 수 있다.
