# CM3070 추가 연습문항 답안집

이 문서는 현재 past exam이나 mock exam PDF에 직접 들어 있지는 않지만,
CM3070 시험 대비에 유용한 추가 연습문항을 프로젝트 맞춤형으로 정리한 자료다.

---

## Q2.1 Proposal과 final report의 차이

### 문제

What is the main difference between the proposal for a project (whether as a video pitch or a written document)
and the final report (again, whether considering the written document or the video aspect of this)?
Using your own project, give examples to illustrate this difference. Also comment on how your proposal
related to your final report.

### 답안

가장 핵심적인 차이는 **proposal은 앞으로 무엇을 하려고 하는지에 대한 계획과 기대를 설명하는 문서**이고,
**final report는 실제로 무엇을 만들었고, 무엇이 바뀌었으며, 무엇을 평가했고, 무엇을 근거로 말할 수 있는지를 정리하는 문서**라는 점이다.

즉 proposal은 prospective한 성격을 가진다. 해결하려는 문제를 정의하고, 왜 이 프로젝트가 가치 있는지 설명하며,
어떤 방향으로 구현할 것인지, 그 방향이 왜 타당해 보이는지, 그리고 시간 안에 실현 가능해 보이는지를 보여주는 것이 목적이다.
그래서 proposal에는 보통 aims, planned methods, expected architecture, rough evaluation plan이 들어간다.
proposal video pitch 역시 같은 역할을 더 짧은 형식으로 수행한다. 즉 아직 완성되지 않은 프로젝트를 “왜 할 만한지” 설득하는 것이다.

반면 final report는 retrospective하고 evidence-based한 성격을 가진다.
실제로 구현된 시스템이 무엇인지, 개발 과정에서 설계가 어떻게 바뀌었는지, 어떤 evaluation을 수행했는지,
그리고 strengths와 limitations가 무엇인지를 비판적으로 설명해야 한다. final project video도 마찬가지로,
proposal pitch처럼 계획을 설명하는 것이 아니라 실제로 동작하는 프로젝트를 보여줘야 한다.

내 프로젝트를 예로 들면, proposal 단계에서는 중고 사진에서 산업용/전자 부품을 식별하는 AI-assisted system을 만들고자 하며,
OCR, multimodal retrieval, vector search, shortlist-based workflow를 활용할 계획이라고 설명했을 것이다.
이 단계에서는 그 요소들이 왜 유망한 방향인지, 왜 기술적으로 흥미로운지, 그리고 왜 실현 가능해 보이는지가 중심이 된다.
즉 핵심은 문제 정의, 동기, 예상 아키텍처, 기대 효과다.

하지만 final report에서는 더 이상 “이렇게 할 것이다”라고만 말할 수 없다.
실제로 무엇이 구현되었는지를 설명해야 한다. 예를 들어 web interface, FastAPI backend,
reusable model package, 여러 신호를 결합한 hybrid retrieval, async indexing, catalog retrieval,
agent-assisted path를 실제로 만들었다는 점을 서술해야 한다. 또한 구현하면서 무엇이 달라졌는지도 설명해야 한다.
예를 들어 OCR은 처음에는 중요한 핵심 신호로 계획되었지만,
실제 구현 과정에서는 훼손되거나 지워진 라벨, 형태가 매우 다양한 라벨 때문에 OCR만으로는 충분히 안정적이지 않다는 것이 드러났다.
그래서 최종 시스템은 VLM signal, body-image feature, metadata, evidence-backed ranking을 함께 쓰는
더 hybrid한 구조로 발전하게 되었다.

또 다른 중요한 변화는 중복 처리 방식이었다. 실제 중고 워크플로에서는 같은 부품이 나중에 더 좋은 라벨 사진,
더 풍부한 description, 추가 각도 이미지와 함께 다시 들어올 수 있다. 그래서 최종 구현은 duplicate-looking registration을
단순 삭제 문제가 아니라 `review + merge` 문제로 다루게 되었다.

이 차이는 video 측면에서도 그대로 나타난다.
proposal video는 프로젝트 개념, 예상 workflow, 기대 impact를 설명하는 데 더 가깝다.
반면 final video는 실제로 시스템이 어떻게 동작하는지 보여줘야 한다.
예를 들어 이미지를 업로드하고, candidate를 검색하고, evidence-backed shortlist를 보여주고,
필요하면 catalog나 agent flow를 짧게 시연하는 식이다.
즉 proposal video는 계획을 설득하는 영상이고, final video는 완성된 시스템을 증명하는 영상이라고 할 수 있다.

내 프로젝트에서 proposal과 final report의 관계는, 같은 핵심 문제와 같은 큰 방향을 공유한다는 점에서 이어져 있다.
둘 다 CM3020의 “orchestrating AI models to achieve a goal”라는 템플릿 아래,
secondhand part identification 문제를 retrieval-first, human-in-the-loop 방식으로 다룬다.
proposal은 이 방향을 처음 제시하는 역할을 하고,
final report는 실제 구현과 evaluation을 거친 뒤 그 방향을 훨씬 더 구체적이고 방어 가능하게 만든다.

따라서 나는 final report가 proposal에서 출발했지만, 그대로 반복된 것은 아니라고 말할 것이다.
핵심 motivation은 유지되었지만, final report는 훨씬 더 넓고, 더 구체적이며, 더 비판적이다.
나중에 추가된 catalog retrieval과 agent orchestration 같은 기능도 포함하고,
반대로 아직 미완성인 OCR aggregate benchmark, latency percentile summary,
완전한 audited review workflow 같은 한계도 함께 설명한다.
바로 이 점 때문에 final report는 proposal보다 더 강하다. proposal이 약속이라면,
final report는 프로젝트가 실제로 무엇이 되었는지를 보여주는 근거 기반 설명이기 때문이다.

---

## Q2.2 윤리와 설계/개발

### 문제

The consideration of ethics is an important part of design and development. What are the most important ethical aspects to consider in software design and development? Which specific ethical considerations did you take into account in your project? Are there other ethical considerations you could or should have taken into account that you decided not to? Explain why.

### 답안

소프트웨어 설계와 개발에서 윤리가 중요한 이유는, 소프트웨어가 단순히 기술적 로직만 구현하는 것이 아니라
사람, 조직, 의사결정에 직접적인 영향을 주기 때문이다. 따라서 일반적으로 가장 중요한 윤리적 고려사항에는
**privacy와 confidentiality**, **safety와 reliability**, **transparency와 accountability**,
**fairness와 bias**, **security**, 그리고 시스템이 불확실할 때도 인간이 의미 있는 통제권을 유지하는 문제가 포함된다.

내 프로젝트에서 가장 중요한 윤리적 고려사항은 **confidentiality**였다.
이 프로젝트는 회사 관련 정보가 외부로 나가면 안 된다는 실제 제약 속에서 진행되었고,
그래서 on-premise와 local model 사용이 설계에서 중요한 의미를 갖게 되었다.
이것은 단순한 편의성 문제가 아니라, 민감한 내부 정보를 외부 서비스에 노출하지 않겠다는 윤리적 판단과 연결된다.

두 번째로 중요한 윤리적 고려사항은 **잘못된 식별이 초래하는 해악**이었다.
시스템이 틀린 부품을 그럴듯하게 확신해서 보여주면 사용자에게 잘못된 판단을 유도하고,
시간을 낭비하게 하며, 시스템 신뢰도도 무너뜨릴 수 있다.
그래서 나는 이 프로젝트를 fully autonomous identifier로 설명하지 않고,
retrieval-first, human-in-the-loop assistant로 설계했다.
즉 단일 정답을 강하게 밀어붙이기보다 shortlist와 evidence를 보여주도록 한 것은
기술적 선택이면서 동시에 윤리적 선택이기도 하다.

세 번째 고려사항은 **safe writeback과 data integrity**였다.
불확실한 모델 결과가 자동으로 저장되어 시스템의 새로운 지식처럼 굳어지면,
시스템은 자기 오류를 스스로 강화할 수 있다. 내 프로젝트에서는 writeback을 기본 opt-in으로 두고,
사용자 확인을 더 중요하게 취급하는 방향으로 이를 완화했다.
이것은 불확실한 결과를 조용히 영구 지식으로 만들어서는 안 된다는 원칙을 반영한다.
같은 원칙은 duplicate handling에도 적용된다. 새 업로드가 기존 부품과 같아 보일 때,
자동 중복 생성이나 무조건 덮어쓰기를 하는 대신 review 후 merge하도록 한 것은 data integrity를 위한 설계 결정이다.

또한 나는 **transparency**도 중요하게 봤다.
이 시스템은 단일한 opaque answer만 주지 않고 evidence, source, ranked candidate를 함께 보여주도록 설계되었다.
이는 사용자가 시스템 출력을 검토하고 반박할 기회를 갖는다는 점에서 윤리적으로도 중요하다.

반면 내가 충분히 다루지 못한 윤리적 고려사항도 있다.
하나는 **inclusive design과 accessibility**다. 나는 프로젝트를 formal accessibility framework에서 출발시키지는 않았고,
다양한 사용자 집단을 대상으로 한 broad usability study도 완료하지 못했다.
또 다른 하나는 **fairness와 dataset bias**다. 어려운 사례와 open-world variability를 고려하긴 했지만,
특정 품목군, 특정 촬영 조건, 특정 사용자 맥락이 체계적으로 불리한지에 대한 full ethical analysis까지는 하지 못했다.
세 번째는 **environmental cost와 efficiency**로, 멀티모달 모델과 반복 실험은 적지 않은 compute를 사용한다.

이 모든 것을 똑같은 수준으로 다루지 못한 이유는 프로젝트 범위 안에서의 우선순위 때문이다.
나는 먼저 기술적 feasibility, confidentiality, 그리고 uncertainty 하에서의 safe use를 가장 시급한 문제로 보았다.
그렇다고 나머지 윤리 문제가 중요하지 않다는 뜻은 아니다. 오히려 시스템이 실제 배포에 가까워질수록,
이 문제들은 다음 단계에서 더 본격적으로 다뤄야 할 영역이라고 생각한다.

---

## Q2.3 Background reading과 literature survey

### 문제

Explain the importance of doing background reading and a literature survey in a computer science project. Describe the most challenging aspect you faced when writing your literature survey, and how you overcame this.

### 답안

background reading과 literature survey는 프로젝트를 단순 구현이 아니라 기존 연구 위에 놓인
학문적 작업으로 만들어 준다. 이를 통해 어떤 문제가 이미 알려져 있고, 어떤 접근이 시도되었으며,
내 프로젝트가 어디에 위치하는지 설명할 수 있다. 또한 설계 선택과 평가 방식의 타당성을 정당화하는 데도 중요하다.
내 프로젝트에서는 왜 이 문제를 단순 closed-set classification이 아니라 retrieval-first로 봐야 하는지,
왜 OCR을 완전한 해답이 아니라 uncertain evidence로 다뤄야 하는지를 설명하는 데 큰 도움이 되었다.

내가 literature survey를 쓰며 가장 어려웠던 점은 AI 분야의 변화 속도가 너무 빨랐다는 것이다.
모델 버전과 benchmark 결과가 빠르게 바뀌다 보니 일부 자료는 짧은 시간 안에 relevance가 떨어졌다.
또 어떤 내용이 literature survey에 들어가야 하고 어떤 내용은 구현 메모에 가까운지도 구분하기 어려웠다.

나는 이를 해결하기 위해 survey의 중심을 오래 유지되는 개념적 주제에 두었다.
즉 retrieval vs classification, OCR uncertainty, multimodal retrieval, human-in-the-loop design을
핵심 축으로 삼고, 최신 모델 비교는 구현 선택을 보조하는 참고 자료로만 사용했다.
이렇게 해서 모델 환경이 바뀌어도 survey 자체의 타당성과 academic value를 유지하려고 했다.

---

## Q2.4 Background material 선택/보고의 윤리

### 문제

Discuss one ethical consideration in selecting and reporting background material, and explain your approach to addressing this consideration in your own project.

### 답안

background material을 선택하고 보고할 때 중요한 윤리적 고려사항 하나는 **정직한 대표성**이다.
즉 내 주장에 유리한 자료만 골라 제시하고, 불리한 근거나 한계는 숨기는 cherry-picking을 피해야 한다.
literature survey는 단순한 설득 문장이 아니라 프로젝트를 정당화하는 academic record의 일부이기 때문이다.

내 프로젝트에서는 이 문제를 줄이기 위해 source의 종류와 evidential strength를 구분했다.
핵심 framing에는 multimodal retrieval, OCR uncertainty, human-in-the-loop assistance처럼
상대적으로 더 오래 가는 conceptual literature를 사용했고, 빠르게 바뀌는 모델 비교나 benchmark 결과는
안정적인 academic foundation처럼 보이지 않도록 더 조심스럽게 다뤘다.

또한 background material이 실제보다 더 많은 것을 증명하는 것처럼 쓰지 않으려고 했다.
예를 들어 이 문헌들이 fully autonomous identification을 이미 정당화한다고 과장하지 않고,
내 프로젝트를 retrieval-first, human-in-the-loop identification assistant로 설명하는 데 필요한 수준으로만
사용했다. 즉 관련성 있고 신뢰할 수 있는 자료를 선택하되, 한계와 claim의 경계도 함께 정직하게 드러내는 방식을 택했다.

---

## Q3.1 Evaluation case study

### 문제

Consider the following case study:
Imagine that your project is the development of a learning analytics algorithm to detect students at risk of failure.
Your evaluation consists of asking reviewers to take a set of inputs to the algorithm and their corresponding outputs,
and answer the following sets of questions.

Set 1 consists of the following questions:
- Does the output help me understand which students are at risk?
- Would this output help a teacher support students better?
- Would this output help you improve your own student retention?

Set 2 consists of the following questions:
- Do you think student A is in need of additional support?
- Do you think student B is in need of additional support?
- Do you think student C is in need of additional support?

Critically discuss the tasks presented, and the questions asked, in the context of the evaluation approach that is being taken.
Include suggestions for additional or different approaches to the evaluation.

### 답안

이 평가는 모델의 실제 타당성보다는 리뷰어의 인상을 측정하는 방식에 가깝다.
Set 1은 출력이 이해 가능하고 유용해 보이는지를 묻기 때문에 face validity와 stakeholder acceptance를 보는 데는 도움이 된다.
하지만 질문이 넓고 주관적이어서 설명 품질, 인터페이스 품질, 예측 정확도를 분리해 평가하지 못한다.
Set 2는 개별 학생에게 추가 지원이 필요한지 판단하게 하므로 조금 더 decision task에 가깝다.
그럼에도 ground truth, 평가 기준, 리뷰어 calibration이 없으면 여전히 correctness보다 opinion을 측정하게 된다.
또한 reviewer bias를 재생산할 수 있고 uncertainty, false positive, false negative 문제도 잘 드러나지 않는다.
더 강한 평가는 정성적 리뷰와 함께 precision, recall, calibration, known outcome 비교 같은 정량 평가를 포함해야 한다.
여기에 teacher-only 판단과의 비교, confidence rating, inter-rater agreement, student group별 fairness 평가도 있으면 좋다.
특히 교사가 실제로 더 나은 intervention을 하게 되는지를 보는 scenario-based study가 매우 유용할 것이다.

---

## Q3.2 20-minute in-person presentation

### 문제

Imagine you are required to give an in-person presentation of 20 minutes to discuss the most significant aspects of your project.
Outline, with concrete detail, what this presentation would consist of, and which parts of the project you would highlight.
Justify your choices of what you would include. Also comment on what aspects would be most challenging to present.

### 답안

20분 발표라면 먼저 2분 정도 문제 맥락과 왜 정확한 식별이 실제 업무에서 중요한지를 설명하겠다.
그다음 3분은 손상된 라벨, 지워진 글자, 비슷한 외형 때문에 OCR-only 접근이 왜 부족했는지 설명하겠다.
이후 5분은 웹 인터페이스, FastAPI 백엔드, 재사용 가능한 모델 패키지, hybrid retrieval 파이프라인으로 이어지는 핵심 구조를 다루겠다.
다음 4분은 OCR, VLM, body-image, metadata 신호를 어떻게 결합했는지가 가장 큰 기술적 난제였음을 설명하겠다.
그리고 3분은 평가 결과와 대표적인 실패 사례를 보여 주며 성과와 한계를 함께 설명하겠다.
마지막 3분은 catalog retrieval, agent flow 같은 후속 확장과 앞으로의 개선 방향을 정리하겠다.
이 부분들을 강조하는 이유는 문제 정의, 핵심 엔지니어링 판단, 최종 시스템의 가치가 가장 잘 드러나기 때문이다.
가장 설명하기 어려운 부분은 weighting과 fusion trade-off인데, 성능에는 중요하지만 자세히 말하면 너무 복잡해지기 때문이다.

---

## Q3.3 Further work

### 문제

Based on the significant aspects of your project identified in part (b) of this question, suggest further work, with justification,
that another student might do to continue or further advance the outcome in some way. Be as explicit as possible.

### 답안

가장 좋은 후속 연구 중 하나는 손상되거나 일부만 보이는 라벨을 위한 specialist module과 hard-case benchmark를 만드는 것이다.
이것이 중요한 이유는 현재 남은 핵심 약점이 기능 부족이 아니라, 실제 현장에서 가장 중요한 failure case에서 분별력이 약하다는 점이기 때문이다.
두 번째는 현재의 weighted-sum fusion을 signal quality와 confidence를 반영하는 learned 혹은 calibrated fusion으로 바꾸는 것이다.
고정 가중치는 튜닝이 어렵고, 어떤 신호가 불안정할 때 오히려 잘못된 evidence를 과대반영할 수 있기 때문이다.
세 번째는 failure taxonomy, latency profiling, reviewer workflow, safe writeback을 포함한 더 엄격한 evaluation과 feedback pipeline을 만드는 것이다.
이렇게 하면 이 프로젝트는 좋은 프로토타입을 넘어 더 신뢰 가능하고 배포 가능한 identification assistant로 발전할 수 있다.

---

## Q4.1 Computer science project vs IT deployment

### 문제

Explain what makes a project a computer science project rather than an IT deployment. Discuss, with justification, where your project fits within this range and what implications this has for future work on the same project.

### 답안

컴퓨터 과학 프로젝트는 기존 도구를 단순 설치하거나 설정하는 수준을 넘어서야 한다.
즉 문제를 어떻게 정의할지, 어떤 계산적 방법을 선택할지, 어떤 trade-off가 있는지, 그리고 그것을 어떻게 평가할지를 다룬다.
반면 IT deployment는 주로 이미 알려진 기술을 실제 운영 환경에 적용하는 데 초점이 있다.
내 프로젝트는 두 성격을 모두 가지지만, 전체적으로는 computer science project에 더 가깝다.
나는 OCR이나 retrieval 도구를 그대로 붙인 것이 아니라, 문제를 retrieval-first, human-in-the-loop 식별 문제로 다시 정의했다.
또한 OCR, VLM, body-image, metadata 신호를 결합하는 hybrid pipeline을 설계하고 failure mode도 분석했다.
물론 웹 앱, 백엔드, 로컬 운영 구조 같은 deployment 성격도 있다.
따라서 앞으로의 후속 작업은 한편으로는 better fusion이나 hard-case benchmark 같은 연구 방향이 가능하고,
다른 한편으로는 usability, monitoring, operational robustness 같은 배포 방향도 가능하다.
이 점에서 이 프로젝트는 단순 IT rollout보다는 계속 확장 가능한 computer science project라고 볼 수 있다.

---

## Q4.2 Radical inclusion vs born-accessible

### 문제

In terms of inclusive design, explain the core similarities and differences between the radical inclusion approach and the born-accessible approach to inclusive software design. Include examples to illustrate the distinctions.

### 답안

radical inclusion과 born-accessible은 모두 accessibility를 마지막에 덧붙이는 것이 아니라 처음부터 고려해야 한다는 점에서 공통점이 있다.
둘 다 배제되는 사용자를 줄이기 위해 초기 설계 단계에서 inclusion을 반영하려는 접근이다.
하지만 핵심 차이는 강조점에 있다.
born-accessible은 처음부터 keyboard support, screen-reader label, caption, colour contrast 같은 접근성을 기본 내장하는 데 초점을 둔다.
반면 radical inclusion은 가장 배제되기 쉬운 사용자와 어려운 상황을 출발점으로 삼아 전체 설계를 바꾸려는 접근이다.
예를 들어 내 프로젝트에서 born-accessible은 인터페이스가 처음부터 읽기 쉽고 탐색 가능하도록 만드는 것이다.
radical inclusion은 여기에 더해 손상된 라벨, uncertain output, 낮은 전문성을 가진 사용자까지 고려해 workflow 자체를 다시 설계하는 것이다.
즉 born-accessible이 accessibility-by-default라면, radical inclusion은 margin-first design이라고 볼 수 있다.

---

## Q4.3 One significant theoretical construct

### 문제

Identify ONE significant theoretical construct used in your project and explain its role in the project. Discuss why the construct was significant, and whether it was the best or the only option you could have chosen. If it was the best or only option, describe why this was the case; and if it wasn't then discuss what other options were available and why you did not choose them.

### 답안

내 프로젝트에서 중요한 theoretical construct 하나는 **human-in-the-loop decision-support model**이었다.
이 개념의 역할은 시스템을 완전 자동 분류기가 아니라, evidence와 함께 후보를 검색하고 순위를 매기는 assistant로 규정하는 데 있었다.
이것이 중요했던 이유는 내 문제는 open-world 성격이 강하고, 라벨이 훼손될 수 있으며, 잘못된 식별이 실제 사용자에게 피해를 줄 수 있기 때문이다.
물론 이것이 유일한 선택지는 아니었다.
closed-set classifier나 end-to-end 자동 분류 모델을 선택할 수도 있었다.
하지만 그런 방식은 missing label이나 유사한 외형 상황에서 지나치게 자신감 있는 단일 답을 내기 쉽고 더 brittle하다고 판단했다.
그래서 내 맥락에서는 uncertainty를 더 잘 반영하고 실제 사용에도 적합한 human-in-the-loop 구성이 최선의 선택이었다.

---

## Q4.4 One algorithm or method used

### 문제

Describe one algorithm or method you used in the app, program, or system that you designed and developed, and explain your choice process, justifying the decisions you made about using that algorithm or method.

### 답안

내가 사용한 방법 중 하나는 여러 신호를 결합하는 **weighted fusion ranking**이다.
OCR text similarity, VLM 또는 image similarity, body-image feature, metadata는 각각 부분적인 evidence를 제공했고,
나는 이를 가중합 형태로 결합해 후보 순위를 만들었다.
이 방법을 선택한 이유는 구조가 투명하고 결과를 점검하기 쉬우며, labelled data가 많지 않아도 사용할 수 있었기 때문이다.
또한 전체 시스템을 다시 짜지 않고도 신호를 추가하거나 제거할 수 있다는 장점이 있었다.
learned fusion이나 end-to-end multimodal model도 고려했지만, 더 많은 데이터와 튜닝이 필요하고 디버깅이 어려웠다.
그래서 weighted fusion은 완벽하지는 않아도 현실적이고 explainable한 선택이었다.

---

## Q4.5 Reproducibility

### 문제

Reproducibility refers to the extent to which an algorithm, method, or tool can produce the same result when used again under the same, or similar, conditions. Discuss how this concept applies to the algorithm or method you described in Part(d) above.

### 답안

reproducibility는 weighted fusion ranking에 직접 적용된다. 원칙적으로는 같은 입력, 같은 전처리, 같은 모델 버전, 같은 가중치를 쓰면 같은 순위 결과가 나와야 한다.
하지만 모델 버전이 바뀌거나 index가 갱신되거나 OCR 동작이 달라지거나 일부 모델이 nondeterministic하면 reproducibility는 약해질 수 있다.
즉 재현성은 단순히 공식 자체의 문제가 아니라, 그 공식을 둘러싼 전체 pipeline을 얼마나 고정하느냐의 문제이기도 하다.
내 프로젝트에서 좋은 reproducibility를 확보하려면 benchmark data 고정, weight 기록, model version 저장, configuration 고정, preprocessing 문서화가 필요하다.
이 방법은 비형식적인 사람 판단보다는 더 재현 가능하지만, 기반 모델이 계속 바뀐다면 완전한 rule-based system보다는 덜 안정적일 수 있다.
따라서 reproducibility는 가능하지만, 전체 실험 맥락을 함께 통제하고 기록해야만 충분히 확보될 수 있다.
