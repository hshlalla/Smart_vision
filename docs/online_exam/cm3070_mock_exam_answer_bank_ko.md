# CM3070 Mock Exam 답안집 (한국어 버전)

이 문서는 **Smart Image Part Identifier for Secondhand Platforms** 프로젝트를 기준으로 작성한
CM3070 mock exam용 답안 초안이다.
기존의 `docs/online_exam/cm3070_past_exam_answer_bank_ko.md`를 보완하는 자료이며,
`submission/pastexam/CM3070 mock exam.pdf`에 있는 질문 순서에 맞춰 정리했다.

## 0. 빠른 프로젝트 식별 정보

- **프로젝트 제목:** Smart Image Part Identifier for Secondhand Platforms
- **사용한 템플릿:** `CM3020 Artificial Intelligence, "Orchestrating AI models to achieve a goal"`
- **한 줄 요약:** OCR, 멀티모달 검색, 벡터 검색, 카탈로그 근거, agent orchestration을 결합해
  중고 사진에서 산업용/전자 부품을 식별하도록 돕는 retrieval-first, human-in-the-loop assistant

## 이 mock exam에서 반복해서 사용할 framing

- 이 프로젝트는 **새 foundation model을 만드는 프로젝트**가 아니라, **시스템 통합 프로젝트**로 설명하는 것이 가장 적절하다.
- 가장 방어력 있는 주장은 **evidence-backed shortlist assistance**이지, 완전 자동 식별이 아니다.
- 현재 가장 강하게 말할 수 있는 성과는 다음과 같다.
  - web / API / model 계층을 아우르는 end-to-end prototype
  - OCR + image/text retrieval + Milvus 기반 검색 + catalog grounding
  - regression test와 latency instrumentation
- 반대로 아직 정직하게 partial / in progress로 말해야 하는 항목은 다음과 같다.
  - aggregate OCR CER/WER 리포팅
  - 전체 p50/p90/p95 latency summary
  - 대규모 controlled hybrid ablation
  - 완전한 audited accept/edit/reject workflow
  - 더 넓은 usability validation

---

# 1. Question 1 — Your project

## (a) 프로젝트 제목은 무엇이었는가?

내 프로젝트의 제목은 **Smart Image Part Identifier for Secondhand Platforms**였다.

## (b) 어떤 템플릿을 기반으로 했는가?

이 프로젝트는 **CM3020 Artificial Intelligence** 템플릿인
**“Orchestrating AI models to achieve a goal”**를 기반으로 했다.

## (c) 프로젝트가 무엇에 관한 것이었는지 간단히 설명하라.

내 프로젝트는 중고 플랫폼에 올라온 사진을 바탕으로 산업용 또는 전자 부품을 식별하는 문제를 다뤘다.
이 문제는 많은 부품이 전체적으로는 비슷하게 보이는 반면, 실제 구분 단서는 part number나 model code처럼
작고 읽기 어려운 텍스트에 숨어 있는 경우가 많기 때문에 어렵다.

나는 이 문제를 closed-set classification 문제로 보기보다,
**retrieval-first, human-in-the-loop 의사결정 지원 문제**로 재구성했다.
시스템은 사용자 이미지와 선택적 텍스트 입력을 받아 OCR 근거를 추출하고,
image / text retrieval과 vector search를 수행한 뒤, 근거와 함께 가능성 있는 후보 shortlist를 반환한다.
이 워크플로를 위해 web interface, API layer, model pipeline을 함께 구축했다.

---

# 2. Question 2 — Project process

## (a) 템플릿과의 관계, 유연성, 제약

내 프로젝트는 선택한 템플릿과 매우 잘 맞았다. 이 프로젝트의 핵심 기여는 하나의 새로운 AI 모델을 학습시키는 것이
아니라, **여러 AI 및 검색 컴포넌트를 오케스트레이션해서 실질적으로 유용한 end-to-end 시스템을 만드는 것**이었기 때문이다.
즉 OCR, 멀티모달 검색, 벡터 검색, catalog retrieval, metadata-aware ranking, agent-style interaction layer를
결합해 중고 부품 식별을 돕는 구조라는 점에서 템플릿과 직접 연결된다.

이 템플릿은 상당한 수준의 유연성을 제공했다. 큰 방향성은 “AI 컴포넌트들을 결합해 실제 문제를 해결하라”였지만,
구체적인 아키텍처, 데이터셋, 모델 선택을 고정하지는 않았다. 덕분에 나는 현실적이면서도 기술적으로 흥미로운 문제를
선택할 수 있었고, 프로젝트가 진행되면서 설계를 조정할 수도 있었다. 예를 들어 처음에는 단순 image retrieval에 더
가까운 발상에서 시작했지만, 실제 도메인에서는 OCR 증거와 textual identifier가 매우 중요하다는 점이 분명해지면서
hybrid retrieval system으로 진화시킬 수 있었다.

동시에 이 템플릿은 유용한 제약도 주었다. 오케스트레이션 프로젝트인 만큼, 단순히 여러 도구를 나열하는 것이 아니라
전체 workflow가 왜 일관된 시스템인지 설명해야 했고, 각 구성요소가 왜 필요한지 정당화해야 했으며,
시스템 전체를 평가해야 했다. 이런 제약은 프로젝트를 현실에 붙들어 두는 역할을 했다. 특히 “무조건 정답 하나를 맞히는
자동 식별기”를 주장하기보다, Top-K shortlist와 evidence를 주는 practical system으로 설명하게 만들었다.

유연성의 가장 큰 장점은 실제 도메인의 복잡성에 맞게 프로젝트를 조정할 수 있었다는 점이다. image retrieval, OCR,
metadata processing, catalog evidence를 중고 플랫폼의 현실적인 조건에 맞춰 결합할 수 있었다. 반면 단점은
scope control이 어려워진다는 것이다. 더 강한 OCR, reranking, review flow, evaluation, catalog grounding 등
추가하고 싶은 요소가 많아서 범위가 계속 커질 위험이 있었다.

반대로 제약의 장점은 우선순위를 분명하게 해준다는 점이었다. 먼저 end-to-end prototype을 만들고,
그 다음 정말 필요한 개선을 선택적으로 넣는 방향이 더 자연스러웠다. 그 결과 나는 완전한 production system은 아니지만,
현재 증거 수준에 맞게 방어 가능한 retrieval-first assistant를 만들 수 있었다.

## (b) Inclusive design과 radical inclusion 설명

**Inclusive design**은 능력, 배경, 기기 환경, 도메인 지식, 사용 맥락이 서로 다른 다양한 사람들이
효과적으로 사용할 수 있도록 시스템을 설계하는 접근이다. 컴퓨터 시스템과 앱 설계에서는 “평균 사용자”만 상정하지 않고,
언어 차이, 접근성 요구, 기술 친숙도, 네트워크 환경, 사용 상황의 차이를 함께 고려한다는 뜻이다.
연구 관점에서는 어떤 평가 방식이나 데이터 수집 방식이 특정 사용자 집단을 사실상 배제하고 있지는 않은지도 함께 봐야 한다.

**Radical inclusion**은 여기서 한 걸음 더 나아간다. 단지 더 많은 사람이 “쓸 수 있게” 만드는 것에 그치지 않고,
평소에 배제되기 쉬운 사람들과 edge case를 주변적인 예외가 아니라 중심적인 설계 대상이라고 보는 관점이다.
컴퓨팅 연구에서는 누가 보통 무시되는지, 어떤 데이터가 과소대표되는지, 쉬운 mainstream 사례에서만 잘 작동하는 시스템이
어떤 실패를 가리고 있는지를 묻는 태도와도 연결된다. 즉 설계 과정 자체의 가정과 권력관계를 비판적으로 바라보는 자세다.

컴퓨터 시스템에 적용하면 이런 관점은 “사용자는 항상 완벽한 입력을 준다”, “사용자는 전문가다”,
“사용자는 한 언어만 쓴다”, “불투명한 결과도 그대로 신뢰한다” 같은 가정을 피하게 만든다.
연구에 적용하면 failure case를 더 정직하게 드러내고, 주장 범위를 더 투명하게 제한하며,
현실 세계의 다양성을 더 잘 반영하는 방향으로 이어진다.

## (c) 이 개념들이 내 프로젝트와 어떤 관련이 있었는가?

솔직히 말하면, 나는 프로젝트 초기에 inclusive design이나 radical inclusion이라는 이론적 용어를
의식적으로 출발점으로 삼지는 않았다. 이 프로젝트는 회사에서 일하면서 “이런 기능이 꼭 있었으면 좋겠다”는
실무적 필요에서 시작되었다. 따라서 설계 동기는 이론적 inclusion framework보다 실제 사용 불편을 줄이는 데 더 가까웠다.

그럼에도 결과적으로 몇 가지 설계 선택은 inclusive design과 닿아 있었다. 이 시스템은 비전문 사용자도
쓸 수 있도록 단일 정답을 강요하지 않고 Top-K 후보와 근거를 함께 보여준다. 또한 입력 이미지가 불완전하고,
라벨이 훼손되거나 가려질 수 있다는 현실을 전제로 설계했기 때문에, 깨끗한 입력과 전문가 사용자를 전제로 한
좁은 도구보다는 더 현실적인 방향을 택했다고 볼 수 있다.

즉, 나는 inclusive design을 이론적으로 강하게 의식해서 설계한 것은 아니지만,
실무 문제를 풀기 위해 만든 선택들 가운데 일부가 결과적으로 inclusive한 방향과 맞아떨어졌다고 설명할 수 있다.
예를 들어 evidence를 함께 보여주고, 사용자가 직접 확인할 수 있게 하며,
불확실한 경우 자동 확정 대신 shortlist를 제공하는 구조가 그렇다.

반면 radical inclusion을 충분히 구현했다고 말하기는 어렵다. 정식 accessibility audit를 수행한 것도 아니고,
다양한 사용자 집단을 대상으로 한 체계적인 usability study도 완료하지 않았기 때문이다.
따라서 시험 답안에서는 “이 개념을 명시적으로 이론 프레임으로 채택했다”기보다,
“실무에서 느낀 문제를 해결하는 과정에서 일부 설계가 inclusive한 방향과 연결되었다”는 식으로 정직하게 서술하는 것이 맞다.


---

# 3. Question 3 — Aims, outcomes, and lessons learned

## (a) 최종 결과를 초기 aims / objectives와 비교하라.

내가 보기에 초기 목표 대비 핵심적으로 미달성된 기능은 없다. 오히려 생각보다 초기 계획의 핵심 목표를
빠르게 달성했고, 그 이후 agent 기능과 catalog 기능 같은 추가 요소를 계속 붙이게 되었다.
따라서 이 문항에서 가장 정직한 답은 “목표를 못 이뤘다”기보다,
“핵심 목표는 비교적 빨리 달성했지만 그 이후 scope가 계속 확장되었다”에 가깝다.

초기 목표는 중고 이미지에서 산업용 및 전자 부품을 식별하도록 돕는 AI-assisted system을 만드는 것이었다.
이 목표 자체는 달성되었다. web interface, API layer, model/retrieval pipeline으로 이어지는
end-to-end prototype을 만들었고, 이미지와 텍스트를 받아 후보를 검색하고,
근거를 제시하는 retrieval-first workflow를 구현했다.

또한 예상보다 빠르게 핵심 흐름을 달성한 뒤에는 단순 MVP에 머무르지 않고 agent orchestration,
catalog retrieval, regression test, latency instrumentation 같은 기능을 추가했다.
이 점은 프로젝트가 단순한 search demo를 넘어서 더 넓은 시스템 수준의 prototype로 발전했다는 뜻이다.

하지만 동시에 나는 모델 성능을 끌어올리는 데도 최대한 집중했다.
이유는 핵심 목표가 빠르게 달성되었다고 해도, 결국 정확하게 판별되지 않으면 아무도 쓰지 않는 시스템이 되기 때문이다.
즉 기능을 넓히는 것만으로는 충분하지 않았고, 실제 사용 가능한 수준의 정확도를 확보하는 일이
프로젝트의 가장 중요한 품질 기준이라고 보았다.

그래서 초기 목표와 최종 결과의 차이는 “실패한 목표”보다 “확장된 결과”라는 표현이 더 적절하다.
다만 그 과정에서 새로운 아이디어가 계속 생겨났고, 무엇을 실제 시간 안에 끝낼 수 있는가를
우선순위로 잘라내는 일이 매우 어려웠다. 즉 기능적 측면에서는 핵심 목표를 달성했지만,
프로젝트 관리 측면에서는 prioritisation과 scope control이 더 큰 어려움이 되었다.

## (b) 비슷한 규모의 프로젝트를 다시 한다면 무엇을 다르게 하겠는가? (세 가지 측면)

### Aspect 1 — Problem definition과 핵심 가치 정의

다시 한다면 가장 먼저 바꾸고 싶은 것은 문제 정의를 더 좁고 명확하게 잡는 방식이다.
프로젝트를 진행하면서 느낀 것은, AI 모델은 생각보다 매우 빠르게 좋아지고 있고,
처음에 “새로운 기능”이라고 생각했던 것들이 점점 모델 안에 기본 탑재되어 나온다는 점이다.
그래서 단순히 여러 기능을 붙이는 것보다, 내가 정확히 어떤 문제를 풀고자 하는지,
그리고 이 시스템의 핵심 가치가 무엇인지를 먼저 고정하는 것이 훨씬 중요하다고 느꼈다.

다음 프로젝트에서는 “무엇을 할 수 있는가”보다 “정확히 어떤 문제를 누구를 위해 푸는가”를 먼저 정의할 것이다.
그렇게 해야 나중에 agent, catalog, OCR, reranking 같은 요소가 모두 같은 방향으로 정렬된다.

### Aspect 2 — Core model strategy와 domain specialization

두 번째로 바꾸고 싶은 점은 여러 모델의 orchestration 자체보다,
핵심 모델 하나를 더 강하게 만드는 전략을 더 일찍 고민하는 것이다.
이번 프로젝트에서는 회사 정보가 외부로 나가면 안 된다는 제약 때문에 on-premise와 local 기반 모델을
전제로 두었다. 그런데 local 모델은 GPT보다 성능이 낮은 경우가 많았고,
이 차이를 여러 모델의 weighted sum orchestration으로 보완하려고 했다.

이 방식은 일정 부분 효과가 있었지만 한계도 분명했다. OCR, VLM, body-image retrieval, metadata signal을
모두 함께 쓰다 보니 fusion weight를 어떻게 잡을지가 어려웠고,
특정 신호에 가중치를 주면 다른 failure case에서 오히려 잘못된 결과를 강화하는 문제가 생겼다.
다시 한다면 모델을 여러 개 겹쳐 쓰는 것만으로 해결하려 하기보다,
한 개의 모델이 더 좁은 scope에서 전문가처럼 잘하도록 fine-tuning 하거나,
도메인 특화된 specialist component를 더 강하게 만드는 방향을 더 일찍 검토할 것이다.

### Aspect 3 — Scope control과 우선순위 관리

세 번째로는 scope control을 더 엄격하게 할 것이다.
이번 프로젝트에서는 핵심 목표를 생각보다 빨리 달성했기 때문에,
agent, catalog, 추가 UI, 각종 보조 기능처럼 “넣으면 좋은 것들”이 계속 떠올랐다.
문제는 이런 확장이 프로젝트를 풍부하게 만들기도 했지만,
동시에 무엇을 이번 제출 범위 안에서 반드시 끝내야 하는지 흐리게 만들었다는 점이다.

다음에는 초기 milestone을 더 명확히 나누고,
핵심 문제를 직접 해결하지 않는 확장은 뒤로 미루는 기준을 더 엄격하게 둘 것이다.
그렇게 해야 feature expansion이 프로젝트를 더 좋게 만들면서도,
핵심 품질과 최종 완성도를 해치지 않게 통제할 수 있다.


# 4. Question 4 — Background material and planning your project

## (a) Background reading의 범위와 적절성

내가 느끼기에는 background reading 중 전혀 쓸모없었던 자료가 있었다기보다,
AI 발전 속도가 너무 빨라서 빠르게 가치가 떨어진 자료 유형이 있었다.
특히 특정 시점의 모델 비교 결과, 버전별 benchmark, 구현 팁 같은 자료는 몇 달만 지나도
새로운 모델 버전이 등장하면서 의미가 많이 약해졌다.
실제로 12월에 실험했을 때보다 모델 버전을 새롭게 올린 뒤 결과가 더 높아졌고,
이 때문에 일부 초반 비교 자료는 최종 설계 결정을 뒷받침하는 근거로는 약해졌다.

그렇다고 해서 background reading 전체가 불필요했던 것은 아니다.
retrieval, OCR uncertainty, human review, multimodal search architecture처럼
상대적으로 더 오래 유효한 개념적 문헌은 여전히 중요했다.
문제는 “빠르게 바뀌는 엔지니어링 정보”와 “상대적으로 안정적인 개념 문헌”을
같은 비중으로 보면 안 된다는 점이었다.

지금 다시 한다면 literature review의 중심에는 오래 가는 개념적 근거를 두고,
빠르게 바뀌는 모델 비교 자료는 implementation decision 참고자료로만 더 제한적으로 사용할 것이다.

## (b) 프로젝트 초기에 시간 배분을 어떻게 결정했는가?

프로젝트 초기에 시간 배분이 어려웠던 이유는 단순히 작업이 많아서가 아니라,
어떤 부분이 핵심 문제 해결이고 어떤 부분이 부가 기능인지 초기에 완전히 분리하기 어려웠기 때문이다.
특히 OCR, VLM, retrieval, local model 제약 같은 요소는 실제로 구현하고 실험해 보기 전에는
어느 정도 시간이 들지 정확히 예측하기 힘들었다.

그래서 나는 먼저 “온프레미스 제약 아래에서 실제 식별 문제를 풀 수 있는가”라는 핵심 질문에
시간을 가장 먼저 배정했다. 즉, 문제 정의와 코어 검색 파이프라인을 우선 만들고,
그 위에서 필요한 경우에만 기능을 확장하는 순서를 택했다.
이 판단 자체는 옳았다고 본다. 왜냐하면 핵심 흐름이 먼저 안정돼야
agent나 catalog 같은 확장 기능도 의미 있게 추가할 수 있었기 때문이다.

다만 hindsight로 보면, 초기 계획이 성공적으로 작동하면서 오히려 새로운 기능 아이디어가 많이 떠올랐고,
그 이후에는 시간 배분 문제보다 우선순위 통제가 더 큰 문제가 되었다.
다시 한다면 초반에는 지금처럼 코어 문제 해결에 집중하되,
핵심 milestone 이후에는 어떤 기능까지를 이번 범위로 인정할지 더 강하게 제한할 것이다.
