# 5. Evaluation

이 장은 구현된 Smart Image Part Identifier 시스템이 중고 산업용 및 전자 부품을 위한 강건하고 상호작용 가능하며 실질적으로 반응성 있는 식별 보조 도구라는 프로젝트의 핵심 목표를 얼마나 충족하는지 비판적으로 평가한다. 평가는 세 가지 축으로 구성된다. 첫째, retrieval effectiveness와 computational latency를 다루는 offline benchmark, 둘째, 산업 이미지에서 전통적 OCR이 보이는 failure mode에 대한 qualitative analysis, 셋째, human-in-the-loop (HITL) workflow의 실효성을 점검하는 pilot usability study다.

## 5.1 Benchmark Design, Dataset, and Metrics

평가를 실제 문제 도메인에 최대한 가깝게 유지하기 위해, main offline benchmark는 정확히 `1,000`개의 산업용 및 전자 부품 이미지 데이터셋을 기준으로 구성했다. 따라서 이 보고서에서 말하는 `1000-item benchmark`는 `1,000`장의 이미지 데이터셋과 그 위에서 수행한 `900 / 100` gallery-query split을 의미하며, `1,000`개의 독립 query를 의미하는 것은 아니다.

이 데이터셋은 일반적인 학술용 이미지 데이터셋과 다르다. category가 명확히 분리되고 중심 피사체가 선명한 clean dataset이 아니라, 매우 유사한 산업용 부품을 사용자가 직접 촬영한 이미지들로 구성되어 있다. 따라서 cluttered background, metallic glare, blur, low contrast, mixed text orientation 같은 실제 marketplace 조건을 자연스럽게 포함한다. 이런 특성 때문에 본 benchmark는 clean closed-set classification보다 중고 부품 listing 환경을 더 잘 반영한다.

main benchmark는 다음과 같이 구성되었다.

- **Index gallery:** `900`개 item을 retrieval system에 인덱싱해 기존 marketplace inventory를 모사
- **Held-out query set:** indexing에 사용되지 않은 `100`개 item을 query로 분리해 새로운 user upload를 모사
- **Main benchmark comparison:** `C2` 대 `C4`
- **Supporting local validation:** 최종 운영 권고를 보완하기 위해 `C3`와 `C1`을 별도의 current-index validation setting에서 비교
- **Runtime environment:** Apple Silicon, `32GB` unified memory

이 장에서 사용하는 주요 지표는 다음과 같다.

- **Accuracy@1:** 정답 item이 Top-1 결과에 포함된 query 비율
- **Accuracy@5:** 정답 item이 Top-5 shortlist 안에 포함된 query 비율. 이 지표는 HITL 인터페이스가 single-label 자동 판정이 아니라 shortlist review를 전제로 하기 때문에 특히 중요하다.
- **MRR (Mean Reciprocal Rank):** 정답 item이 순위 상단에 위치할수록 높은 값을 주는 ranking-quality 지표
- **Exact identifier hit:** maker와 part identifier가 정확하게 회수되거나 정렬된 비율
- **CER (Character Error Rate):** OCR-focused sub-benchmark에서 사용하는 정규화된 문자 단위 오류율이며, 낮을수록 extraction error가 적다
- **Mean total latency:** 명시된 하드웨어 환경에서 측정한 query당 end-to-end 평균 실행 시간

추가 결과표, raw summary, protocol note는 Appendix F에 수록한다.

## 5.2 Evaluated Configurations and Main Benchmark Results

이 프로젝트의 핵심 engineering question은 visual understanding, exact text extraction, computational speed를 어떻게 균형 있게 조합할 것인가였다. 따라서 broader evaluation process에서는 세 가지 구성을 고려했지만, 직접 통제된 main benchmark는 `C2`와 `C4`를 중심으로 수행했다.

- **C2 (OCR-heavy legacy hypothesis):** OCR enabled, reranker enabled
- **C4 (vision-dominant main benchmark configuration):** OCR disabled, reranker enabled
- **C3 (fast practical baseline):** OCR disabled, reranker disabled. 이 구성은 main controlled benchmark가 아니라 별도의 local current-index setting에서 검증되었다.

초기 가설은 OCR을 적극적으로 사용하면 part number 같은 explicit identifier를 더 잘 회수할 수 있으므로 retrieval이 개선될 것이라는 것이었다. 그러나 main `900 / 100` benchmark는 이 가설을 지지하지 않았다.

**Table 5.1. Main benchmark retrieval comparison (`900`-item gallery, `100` held-out queries)**

| Configuration | Accuracy@1 | Accuracy@5 | MRR | Exact identifier hit | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| C2 (OCR ON + Reranker ON) | 0.86 | 0.95 | 0.903 | 0.81 | 8.24s |
| C4 (OCR OFF + Reranker ON) | 0.91 | 0.97 | 0.939 | 0.88 | 1.42s |

*Table note:* Table 5.1은 직접 비교 가능한 main benchmark 결과만 포함한다. 두 행 모두 동일한 `900 / 100` split과 동일한 evaluation protocol로 측정되었다.

Table 5.1이 보여주듯이, vision-dominant 구성인 `C4`는 OCR-heavy 구성인 `C2`보다 모든 retrieval-quality 지표에서 더 좋은 결과를 냈고, mean latency도 `8.24s`에서 `1.42s`로 줄였다. 이 결과는 중요하다. 이 도메인에서는 OCR로 추가 텍스트를 더 많이 넣는다고 해서 retrieval이 자동으로 좋아지는 것이 아니라, 오히려 ranking 과정에 noise를 주입할 수 있음을 보여주기 때문이다.

이 차이는 ranking metric에서도 분명하게 드러난다. OCR-heavy path는 `MRR`을 `0.939`에서 `0.903`으로 떨어뜨렸는데, 이는 noisy OCR evidence가 Top-1 정확도뿐 아니라 전체 ranking order 자체를 악화시켰음을 의미한다. `Accuracy@5`도 HITL workflow에서 특히 중요하다. `C4`의 `Accuracy@5 = 0.97`이라는 값은 held-out query의 `97%`에서 정답 item이 review shortlist 안에 포함되었음을 뜻하며, 이는 시스템이 fully automatic classifier가 아니더라도 운영적으로 매우 의미 있는 성능이다.

## 5.3 Supporting Local Validation and Operational Recommendation

최종 운영 권고는 main benchmark만으로 결정하지 않았다. 실제 현재 인덱스 환경에서는 reranker가 practical retrieval quality에 의미 있는 이득을 주는지도 별도로 확인할 필요가 있었다.

**Table 5.2. Supporting local validation for the final operating recommendation**

| Configuration | Group Hit@1 | Group Hit@5 | MRR | Exact item Top-1 | Warm mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| C3 (OCR OFF + Reranker OFF) | 1.0000 | 1.0000 | 1.0000 | 0.9667 | 731.13ms |
| C1 (OCR OFF + Reranker ON) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 89337.71ms |

*Table note:* Table 5.2는 Table 5.1과 다른 local validation setting에서 나온 결과이므로 직접적인 head-to-head benchmark 표로 해석하면 안 된다. 이 표의 목적은 main benchmark를 대체하는 것이 아니라 최종 운영 권고를 정당화하는 데 있다.

local validation 결과는 reranker가 현재 환경에서 practical quality gain을 거의 주지 않으면서도 runtime cost는 극단적으로 증가시켰음을 보여준다. `C1`은 exact item Top-1을 `0.9667`에서 `1.0000`으로 올렸지만, warm mean total latency를 `731.13ms`에서 `89337.71ms`로 증가시켰다. 현재 deployment constraint 아래에서는 이 trade-off를 운영상 받아들이기 어렵다.

**Figure 5.1. Mean end-to-end latency across C2, C4, and C3.**  
*여기에 `C2 (8.24s)`, `C4 (1.42s)`, `C3 (731.13ms)`의 mean total latency를 비교하는 bar chart를 삽입한다. 이 그림은 OCR bottleneck이 얼마나 큰지, 그리고 현재 runtime constraint 아래에서 왜 `C3`가 가장 practical한 default인지 시각적으로 보여주어야 한다.*

latency profiling은 이 비용이 어디서 발생했는지도 보여준다. 보고서 benchmark에서 `C2`는 preprocessing과 OCR 단계만으로 약 `7.11s`를 소비했고, 이것이 전체 지연의 대부분을 차지했다. 반면 `C4`는 always-on OCR을 제거하면서 비용 구조를 크게 줄였고, 이후에는 preprocessing과 reranking이 주요 비용으로 남았다. 이는 전통적 OCR이 legacy pipeline의 지배적 latency bottleneck이라는 점을 분명히 보여준다.

결국 Table 5.1과 Table 5.2를 함께 보면 분명한 engineering conclusion이 나온다. 통제된 main benchmark에서는 `C4`가 가장 강했지만, 실제 운영 관점에서는 `C3`가 local validation에서 strong practical retrieval behaviour를 유지하면서도 interactive 수준의 runtime을 제공했기 때문에 최종 default operating mode로 채택되었다.

## 5.4 Qualitative Analysis of OCR Limitations

OCR-heavy configuration이 왜 약한 성능을 보였는지 이해하기 위해, held-out query set에서 발생한 retrieval failure와 identifier-visible case를 수동으로 검토하는 qualitative error analysis를 수행했다.

반복적으로 관찰된 패턴은 다음 세 가지다.

1. **Irrelevant specification noise**  
   OCR은 `12V`, `50Hz`, 전류값, 저항값, warning label, 표 조각처럼 식별에 직접 도움이 되지 않는 문자열까지 함께 추출했다. 이런 문자열은 텍스트 자체로는 맞더라도 primary identification에는 semantically weak했으며, retrieval signal에 섞일 경우 true identifier의 중요성을 희석시키고 shared specification 기반 false positive를 유발했다.

2. **Layout and orientation failure**  
   산업용 label은 세로쓰기와 가로쓰기가 혼재되거나, 비스듬한 시점, cylindrical placement, partial occlusion을 동반하는 경우가 많았다. 전통적 OCR은 이런 조건에서 intended reading order를 안정적으로 유지하지 못했고, 논리적으로 함께 읽혀야 할 token을 제대로 묶지 못했다.

3. **Logo and context failure**  
   제조사 정보는 plain text가 아니라 stylised logo나 contextual label structure로 표현되는 경우가 있었다. 이런 경우 vision-language modelling은 단순 문자를 해독하려 하기보다 visual form과 layout prominence 자체를 사용할 수 있었기 때문에 text-only extraction보다 더 robust했다.

**Figure 5.2. Representative OCR failure cases in industrial imagery.**  
*여기에 사례 이미지를 삽입한다. 최소한 `12VDC`나 `50Hz` 같은 irrelevant electrical specification, cluttered 또는 reflective label, 세로쓰기와 가로쓰기가 섞인 mixed text layout이 드러나는 예시가 포함되어야 한다. 이 그림은 일부 문자열이 맞게 읽히더라도 raw OCR output 전체는 왜 noisy retrieval signal이 되는지를 보여주어야 한다.*

이러한 failure mode는 raw하고 uncontextualised된 OCR text가 이 도메인에서 reliable primary retrieval signal이 아니었음을 정성적으로 설명해 주며, OCR-heavy path에서 관찰된 `MRR` 하락과도 직접적으로 연결된다.

## 5.5 OCR Benchmark and Verification Role

OCR-specific benchmark는 main retrieval benchmark와 분리해서 해석해야 한다. 이 실험은 end-to-end retrieval quality가 아니라 identifier-visible subset에서의 extraction quality를 측정한다.

**Table 5.3. Identifier extraction benchmark on the identifier-visible subset**

| Method | Exact full-string | CER | Part-number recall | Maker recall |
| --- | ---: | ---: | ---: | ---: |
| PaddleOCR | 0.19 | 1.12 | 0.35 | 0.44 |
| Qwen-only | 0.57 | 0.46 | 0.75 | 0.82 |
| OCR + Qwen merged | 0.61 | 0.41 | 0.79 | 0.86 |

특히 CER 열은 해석상 중요하다. CER은 정규화된 edit-distance 지표이므로 `1.0`을 초과하는 값은 단순한 문자 치환 몇 개가 아니라, target identifier 길이에 비해 심한 extra noise가 예측 결과에 포함되었음을 뜻한다. 실질적으로 `PaddleOCR`의 `CER = 1.12`는 raw OCR output이 irrelevant string과 structural reading error로 크게 오염되어 있었음을 보여준다.

따라서 Table 5.3은 OCR이 완전히 무가치하다는 결론이 아니라, 그 역할을 재배치해야 한다는 결론을 지지한다. OCR 단독은 exact identifier recovery에서 약했고, Qwen-only path는 훨씬 더 나은 identifier quality를 보였다. OCR+Qwen 결합은 가장 높은 score를 소폭 더 끌어올렸지만, 추가 runtime cost가 필요했다. 결국 OCR은 default first-stage retrieval driver가 아니라, 추가 confirmation이 필요할 때 사용하는 supplementary verification evidence로 두는 것이 더 적절하다.

## 5.6 Usability Pilot Results

human-in-the-loop workflow를 검증하기 위해 소규모 post-task usability pilot도 수행했다. 응답 파일에는 총 `6`개의 submission이 있었지만, 그중 하나는 외부 테스트 링크가 오프라인이어서 prototype에 실제 접근하지 못한 무효 응답이었다. 따라서 usable participant count는 `n = 5`다.

**Table 5.4. Pilot usability questionnaire summary (`n = 5` usable responses)**

| Item | Mean score |
| --- | ---: |
| 별도 도움 없이 인터페이스 사용 방법을 이해할 수 있었다 | 4.4 / 5 |
| 검색 결과는 후보 부품을 좁히는 데 유용했다 | 5.0 / 5 |
| 결과와 함께 제시된 evidence는 신뢰 판단에 도움이 되었다 | 4.4 / 5 |
| metadata preview는 listing 정보 준비에 필요한 노력을 줄여주었다 | 4.4 / 5 |
| 유사한 과업에서 완전 수동 검색보다 이 프로토타입을 선호할 것이다 | 4.6 / 5 |
| shortlist를 바탕으로 최종 판단을 내리는 데 자신감이 있었다 | 4.8 / 5 |

*Table note:* raw response file에는 총 여섯 개의 응답이 있었지만, 그중 하나는 외부 테스트 링크가 오프라인이었던 탓에 제외했다. 따라서 usable response count는 `n = 5`다. cleaned raw summary는 Appendix F에 수록한다.

정량적 패턴은 전반적으로 긍정적이다. 참가자들은 shortlist usefulness에 가장 높은 점수(`5.0 / 5`)를 주었고, shortlist를 바탕으로 최종 결정을 내릴 수 있다는 자신감도 높게 평가했다(`4.8 / 5`). 인터페이스 이해 용이성, evidence-based trust, metadata preview usefulness 역시 모두 `4.4 / 5` 이상으로 유지되었다. 이 usability pattern은 vision-dominant retrieval path의 높은 `Accuracy@5`와도 broadly consistent하지만, 표본 수가 작기 때문에 formal causal validation로 해석해서는 안 된다.

정성 피드백도 retrieval-first architecture와 잘 부합했다. 한 참여자는 *“The shortlist of candidate parts was highly relevant. It successfully filtered out the noise and showed exactly what I was looking for.”*라고 응답했다. 다른 참여자는 *“the ‘evidence’ section that explains why a certain part was suggested”*를 가장 유용한 요소로 꼽았다. 전반적으로 참가자들은 shortlist relevance, evidence display의 usefulness, metadata preparation 과정의 manual effort reduction을 반복적으로 언급했다. 반면 개선 요구는 compare-view support, clearer onboarding hint, export integration, confidence-score display처럼 주로 additive refinement 성격이었으며, core workflow failure를 지적하는 내용은 아니었다.

## 5.7 Critical Analysis Against Project Objectives

평가 결과는 AI 구성요소를 더 많이 붙일수록 성능이 자동으로 좋아질 것이라는 초기 가정을 수정하도록 요구한다. 실제로는 vision, OCR, reranking을 naive하게 누적하는 것이 speed와 retrieval quality를 동시에 악화시켰다.

main benchmark는 `C4`가 `C2`보다 우수했음을 보여주었고, 이는 OCR-heavy path가 최선의 설계 선택이 아니었음을 의미한다. supporting local validation은 `C3`가 strong practical retrieval behaviour를 유지하면서도 가장 낮은 acceptable latency를 제공함을 보여주었다. OCR-specific benchmark는 OCR이 여전히 가치가 있음을 인정하되, 그 역할이 always-on first-stage retrieval engine이 아니라 selective verification evidence여야 한다는 점을 분명하게 했다. 마지막으로 pilot usability 결과는 shortlist-and-evidence workflow가 실제 사용자에게 이해 가능하고 유용하다는 점을 보여주었다.

이 결과를 종합하면 최종 설계 결론은 네 가지로 요약된다.

1. **Retrieval-first architecture:** 시스템은 black-box classifier가 아니라 retrieval-first 구조를 유지해야 한다.
2. **Vision-dominant primary path:** vision-language 기반 image understanding이 primary retrieval path가 되어야 한다.
3. **Selective verification:** OCR은 default path가 아니라 selective supporting evidence로 재배치되어야 한다.
4. **Human authority:** 사용자는 human-in-the-loop workflow 안에서 최종 판단 권한을 유지해야 한다.

OCR-heavy pipeline에서 retrieval-first, evidence-backed, human-in-the-loop assistant로의 전환이야말로 이 프로젝트의 가장 중요한 기술적 결론이다. 현재 deployment constraint 아래에서는 `C3`가 가장 practical한 default operating configuration이며, `C4`는 controlled main benchmark에서 가장 강한 결과를 제공한 구성으로 해석할 수 있다.
