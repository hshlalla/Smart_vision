# Video Script (3-Minute Version)

## Korean

안녕하세요. 제 프로젝트는 **Smart Image Part Identifier for Secondhand Platforms**이며, CM3020의 **Orchestrating AI models to achieve a goal** 템플릿을 사용했습니다.

이 프로젝트의 목표는 사용자가 업로드한 사진만으로 산업용 또는 전자 부품을 더 쉽게 식별하도록 돕는 것입니다. 이 문제는 단순 분류보다 retrieval 문제에 더 가깝습니다. 중고 부품은 open-world이고, 외형이 비슷한 경우가 많으며, 실제 구분 단서는 작은 라벨이나 part number인 경우가 많기 때문입니다.

이 시스템은 **retrieval-first, human-in-the-loop identification assistant**로 설계했습니다. 즉, 하나의 정답을 단정하기보다 이미지, OCR, 텍스트, 메타데이터를 함께 사용해 Top-K 후보를 보여주고, 사용자가 근거를 보고 판단할 수 있도록 합니다.

먼저 검색 기능을 보겠습니다. 사용자는 텍스트만 입력할 수도 있고, 이미지를 함께 업로드할 수도 있습니다. 검색이 실행되면 시스템은 이미지 임베딩, OCR 텍스트, 메타데이터 텍스트, lexical matching 신호를 결합해 후보를 찾고, 각 결과에 대해 점수와 설명을 반환합니다. 이 방식의 장점은 단순히 비슷한 이미지를 찾는 것이 아니라, maker나 part number 같은 exact evidence도 순위에 반영된다는 점입니다.

다음은 인덱싱 기능입니다. 사용자가 이미지 한 장 또는 여러 장을 업로드하면, 시스템은 먼저 GPT 기반 metadata preview를 생성합니다. 여기서 maker, part number, category, description 초안을 제안하고, 사용자는 이를 수정할 수 있습니다. confirm을 눌렀을 때만 실제 저장이 수행됩니다. 이 구조는 자동 저장보다 안전하고, 실제 listing workflow에 더 적합합니다.

이 프로젝트는 검색 외에도 catalog retrieval과 agent 기능을 포함합니다. catalog 기능은 PDF 문서를 인덱싱하고 내부 문서를 검색할 수 있게 하며, agent는 hybrid search, catalog search, web search를 조합해 추가 근거를 제공합니다. 따라서 이 시스템은 단순 검색 데모가 아니라 실제 의사결정을 돕는 workflow 수준의 프로토타입입니다.

기술적으로 중요한 점은 이 시스템이 여러 불확실한 신호를 함께 다룬다는 것입니다. OCR은 유용하지만 실패할 수 있고, 이미지 유사도만으로는 세부 부품 변형을 구분하기 어렵습니다. 그래서 저는 multimodal retrieval, metadata-aware ranking, reranking, 그리고 human confirmation을 함께 사용하는 방향으로 설계했습니다.

현재 프로토타입은 end-to-end 검색과 인덱싱 흐름을 갖추고 있습니다. 또한 통합 데이터셋과 평가 입력도 준비되었습니다. 다만 OCR CER benchmark, hybrid ablation, latency percentile, usability pilot은 아직 진행 중입니다. 따라서 이 프로젝트를 완전 자동 식별 시스템이라기보다, retrieval-first identification assistant prototype이라고 설명하는 것이 가장 정확합니다.

정리하면, 이 프로젝트의 핵심은 하나의 모델 데모가 아니라 OCR, multimodal embeddings, vector retrieval, catalog search, 그리고 user confirmation을 실제 workflow로 통합했다는 점입니다. 감사합니다.

## English

Hello. My project is **Smart Image Part Identifier for Secondhand Platforms**, and it follows the CM3020 template **Orchestrating AI models to achieve a goal**.

The aim of this project is to help users identify industrial and electronic parts from uploaded photos. This is better framed as a retrieval problem than a simple classification problem, because secondhand parts are open-world, many items look visually similar, and the decisive evidence is often a small label or part number.

The system is designed as a **retrieval-first, human-in-the-loop identification assistant**. Instead of making one opaque prediction, it combines image evidence, OCR text, metadata, and lexical signals to produce a Top-K shortlist that the user can inspect and verify.

First, the search function. A user can enter text only, or upload an image together with text. The system combines image embeddings, OCR-derived text, metadata text, and lexical matching signals to retrieve candidate parts. It then returns ranked results with scores and descriptive evidence. The key point is that ranking is not based on visual similarity alone. Exact evidence such as maker names or part numbers can also affect the result order.

Next, the indexing function. When the user uploads one or more images, the system first generates a GPT-based metadata preview. It suggests maker, part number, category, and description fields, and the user can edit them before saving. Actual indexing happens only after confirmation. This is safer than automatic write-back and fits listing workflows more naturally.

The project also includes catalog retrieval and an agent layer. The catalog feature indexes PDF documents and allows internal document search. The agent can combine hybrid search, catalog search, and web search to provide additional supporting evidence. This makes the system more than a simple search demo. It is a workflow-oriented decision-support prototype.

The technical challenge of this project comes from combining multiple uncertain signals. OCR is useful but noisy, and image similarity alone is not enough for fine-grained part identification. For that reason, the system uses multimodal retrieval, metadata-aware ranking, reranking, and human confirmation together.

At the current stage, the prototype already supports end-to-end search and indexing, and the unified dataset and evaluation inputs have also been prepared. However, some parts are still in progress, including the OCR CER benchmark, hybrid ablation, latency percentile reporting, and the usability pilot. So the fairest description is not a fully autonomous identification system, but a retrieval-first identification assistant prototype.

In summary, the main contribution of this project is not a single-model demo. It integrates OCR, multimodal embeddings, vector retrieval, catalog search, and user confirmation into one practical workflow. Thank you.
