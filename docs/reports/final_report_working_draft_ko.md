# Smart Image Part Identifier for Secondhand Platforms (한국어 작업본)

University of London  
Bachelor in Computer Science

Final Project  
CM3020 Artificial Intelligence, "Orchestrating AI models to achieve a goal"

Name: SuHun Hong  
Email: hshlalla@naver.com

## Draft Status

이 문서는 `docs/reports/final_report_working_draft.md`의 한국어 대응 작업본이다.  
엄밀한 line-by-line 번역본이라기보다는, 동일한 장 구조와 핵심 주장을 한국어로 빠르게 검토할 수 있도록 정리한 companion draft다.

원본 영문 working draft는 다음 자료를 바탕으로 작성되었다.

- 현재 코드베이스
- preliminary / draft feedback
- final report guide
- `submission/evidence/report_support_2026-03-10/`의 검증 artifact

## 1. Introduction

### 1.1 Project Context and Motivation

본 프로젝트는 중고 플랫폼에서 산업용·전자 부품을 사진으로 식별하는 문제를 다룬다. 일반 소비재와 달리, 이 도메인에서는 작은 모델 코드, 제조사 표기, 부품 번호가 결정적인 구분 단서가 되는 경우가 많다. 사용자는 비전문가인 경우가 많고, 업로드 이미지는 glare, blur, low resolution, occlusion 등으로 인해 품질이 일정하지 않다. 따라서 문제는 단순한 비주얼 유사도 검색보다 훨씬 어렵다.

### 1.2 Problem Statement

핵심 문제는 사용자가 올린 이미지 한 장 또는 소량의 텍스트 질의만으로, 실제 거래에 도움이 되는 수준의 candidate shortlist와 evidence를 제공할 수 있는가이다. 이 프로젝트는 이를 closed-set classification이 아니라 retrieval-first assistant 문제로 재정의한다.

### 1.3 Domain and User Requirements

#### Domain Requirements

- inventory는 open-world이고 long-tail 특성을 가진다.
- visually similar but semantically different한 part가 많다.
- OCR text는 매우 중요하지만 noisy하다.
- 결과는 listing workflow에 바로 쓸 수 있도록 structure와 evidence를 가져야 한다.

#### User Requirements

- 사용자는 긴 텍스트 설명보다 사진 기반 입력을 선호한다.
- 사용자는 Top-K shortlist를 통해 후보를 비교하길 원한다.
- 사용자는 AI 출력을 무조건 신뢰하지 않고, 근거와 수정 가능성을 원한다.

### 1.4 Implications for System Design

이 요구사항들은 retrieval-first, evidence-aware, human-in-the-loop architecture를 정당화한다. 시스템은 OCR을 활용하되 실패 가능성을 전제로 해야 하고, visual signal과 text signal을 함께 사용해야 하며, confidence가 낮은 결과를 자동 확정해서는 안 된다.

### 1.5 Aim and Objectives

프로젝트의 aim은 사용자 사진을 입력으로 받아 Top-K shortlist와 listing-oriented evidence를 제공하는 end-to-end prototype을 구현하는 것이다. 주요 objective는 다음과 같다.

- end-to-end workflow 구현
- retrieval effectiveness 검토
- OCR robustness 분석
- latency instrumentation 추가
- engineering reliability 강화
- user assistance 관점에서의 실제 활용 가능성 검토

### 1.6 Report Structure

이 보고서는 introduction, literature review, design, implementation, evaluation, conclusion 순으로 전개되며, 특히 design 장 안에 evaluation strategy를 명시적으로 포함한다.

## 2. Literature Review

### 2.1 Vision-Based Product and Part Retrieval

consumer visual search literature는 이미지 기반 검색이 실용적임을 보여주지만, 부품 검색처럼 fine-grained하고 open-world한 문제에는 그대로 적용하기 어렵다.

### 2.2 Object Detection and Region Focus

일부 문제는 object detection이나 region cropping이 도움이 될 수 있다. 다만 현재 프로젝트는 단일 부품 이미지가 많기 때문에, detection은 필수 기본 구성이라기보다 future experiment에 가깝다.

### 2.3 OCR as a Complement to Visual Retrieval

OCR은 model code, serial, maker text를 잡아낼 수 있어 매우 중요하지만, blur와 glare에 취약하다. 따라서 OCR은 핵심 보조 증거이지 ground truth는 아니다.

### 2.4 Multimodal Embeddings and Text Retrieval

image-text embedding과 multilingual text retrieval 문헌은 hybrid retrieval 설계를 정당화한다. 특히 BGE-M3 계열은 다국어 metadata와 OCR 텍스트 검색에 적합한 선택지다.

### 2.5 Vector Databases and Hybrid Search

Milvus와 같은 vector DB는 modality별 collection 관리와 staged retrieval에 적합하다. 이 프로젝트는 image, text, caption, metadata를 분리 저장하고 조합한다.

### 2.6 Interactive Feedback and Continuous Improvement

interactive AI literature는 evidence 노출과 user correction 가능성이 신뢰 형성에 중요함을 보여준다. 이는 safer writeback과 human review boundary 설계로 이어진다.

### 2.7 Summary of Gaps and Design Implications

선행연구는 각 요소의 가능성을 보여주지만, 중고 부품 식별이라는 실제 workflow 맥락에서 image, OCR, metadata, catalog evidence를 함께 오케스트레이션하는 end-to-end 시스템은 여전히 구현 가치가 크다.

## 3. Design

### 3.1 Design Goals and Constraints

설계 목표는 open-world 지원, fine-grained ambiguity 대응, OCR uncertainty 처리, listing-oriented output, evaluation traceability 확보였다.

### 3.2 System Overview

전체 시스템은 React web UI, FastAPI API layer, hybrid-search model package, Milvus collections, catalog retrieval, agent orchestration으로 구성된다.

### 3.3 Data Model and Retrieval Schema

데이터는 modality별로 분리 저장된다. image vector, OCR/text vector, caption vector, model-level text, metadata field가 서로 다른 retrieval 채널을 형성한다.

### 3.4 Core Workflows

#### 3.4.1 Indexing Workflow

인덱싱 시 이미지를 받은 뒤 바로 저장하지 않는다. 먼저 metadata preview를 생성하고, 이미 등록된 부품과 겹칠 가능성이 있으면 duplicate candidate를 보여준 뒤, 사용자가 새 모델 유지 또는 기존 모델 append를 선택하게 한다. 그 후에 OCR 및 embedding을 생성하고, 관련 collection에 upsert한다.

#### 3.4.2 Search Workflow

검색 시에는 이미지와 optional text query를 받아 OCR, embedding, vector retrieval, candidate fusion, evidence formatting을 수행한다.

#### 3.4.3 Ranking and Evidence Fusion

순위는 dense similarity만으로 정해지지 않는다. OCR, lexical exact match, metadata field match, caption signal을 함께 반영한다.

#### 3.4.4 Catalog Retrieval and Agent-Oriented Reasoning

catalog path는 내부 문서 근거를 제공하고, agent path는 hybrid search와 catalog search를 orchestration하여 source-backed answer를 생성한다.

### 3.5 Human-in-the-Loop Boundary

현재 시스템은 fully autonomous identifier가 아니다. shortlist, evidence exposure, writeback opt-in 정책을 통해 human-in-the-loop 특성을 일부 구현했으며, full accept/edit/reject workflow는 future work다.

여기에 더해, 현재 인덱싱 경로는 “중복처럼 보이는 입력을 자동 폐기”하지 않는다. 실제 현장에서는 같은 부품이 나중에 더 좋은 라벨 사진, 더 풍부한 description, 추가 각도 이미지와 함께 다시 들어올 수 있기 때문이다. 그래서 현재 구현은 duplicate-looking record를 `review + merge` 문제로 다루고, interactive indexing에서는 사용자가 기존 모델 append 여부를 직접 결정하게 한다.

### 3.6 Evaluation-Oriented Design

evaluation strategy는 design 안에 포함된다. retrieval effectiveness, OCR robustness, latency, engineering reliability가 핵심 축이다.

### 3.7 Design Limitations

현재 설계는 strong baseline을 제공하지만, OCR fragility, incomplete review flow, incomplete benchmark automation이라는 한계를 가진다.

## 4. Implementation

### 4.1 Repository Structure and Runtime Setup

구조는 `apps/web`, `apps/api`, `packages/model`, `docs`, `submission`로 나뉘며, UI/contract/model/evidence를 분리한다.

### 4.2 Frontend and API Layer

frontend는 search/index/chat/catalog 화면을 제공하고, API는 auth, hybrid, catalog, agent route를 제공한다.

index 화면은 preview-confirm 흐름과 duplicate candidate 배너를 포함한다. 사용자는 메타 초안을 수정할 수 있고, 기존 부품으로 합칠지 새 모델로 유지할지 명시적으로 선택할 수 있다.

### 4.3 Hybrid Search Core

hybrid search core는 indexing과 retrieval의 단일 진입점 역할을 하며, vector retrieval과 ranking fusion을 관리한다.

### 4.4 OCR, Embeddings, and Optional Captioning

OCR과 embedding은 이미지 기반 단서와 텍스트 기반 단서를 동시에 확보하기 위해 사용된다. captioning은 semantic recall을 보강한다.

### 4.5 Catalog RAG and Tool-Oriented Agent Integration

catalog RAG는 PDF 문서를 기반으로 evidence retrieval을 수행하고, agent는 여러 tool을 호출해 응답을 구성한다.

### 4.6 Reliability, Safety, and Observability Improvements

최근 구현 개선 사항은 다음과 같다.

- agent writeback 기본값을 `false`로 변경
- UI에 explicit opt-in toggle 추가
- hybrid path latency instrumentation 추가
- lazy import 정리로 lightweight pytest 가능

### 4.7 Implementation Stability Evidence

artifact bundle에는 API/model pytest 결과가 기록되어 있다. 이는 최근 변경이 실제로 검증되었다는 최소한의 회귀 근거가 된다.

### 4.8 What Is Implemented Versus Partial

현재 구현된 것과 partial 상태인 것을 구분하는 것이 중요하다. end-to-end prototype, hybrid retrieval, catalog path, safer writeback, regression test는 구현되었고, full human review flow와 일부 정량 benchmark는 partial 또는 planned 상태다.

## 5. Evaluation

### 5.1 Evaluation Strategy

평가는 retrieval, OCR, latency, engineering reliability 네 축으로 구성된다. 이 프로젝트는 classifier accuracy만 보는 것이 아니라, shortlist utility와 evidence quality를 함께 본다.

### 5.2 Evaluation Inputs and Evidence Types

입력 근거는 baseline retrieval result, qualitative failure case, test artifact, instrumentation log, implementation change summary 등이다.

### 5.3 Retrieval Results: Image-Only Baseline

image-only baseline은 retrieval-first framing의 타당성을 보여주지만, 동시에 OCR/text fusion의 필요성도 드러낸다.

### 5.4 Observed Failure Patterns

대표적인 failure는 glare, blur, tiny label, confusingly similar variant, missing metadata다.

### 5.5 OCR Robustness: Current Evidence and Missing Quantitative Benchmark

OCR 관련 qualitative evidence는 있으나, aggregate CER benchmark는 아직 최종 완료되지 않았다.

### 5.6 Latency Evaluation: Instrumented but Not Yet Fully Summarised

latency instrumentation은 구현되어 있으나, p50/p90/p95 요약은 아직 future work로 남아 있다.

### 5.6.1 Planned Retrieval Ablation for Final Design Choice

Qwen-centred pipeline과 mixed OCR-plus-text pipeline을 비교하는 controlled ablation이 계획되어 있다.

### 5.7 Engineering Validation Results

artifact bundle의 pytest 결과는 safer writeback과 import/testability 개선이 실제로 검증되었음을 보여준다.

### 5.8 Objective-by-Objective Assessment

objective별로 implemented, partial, future work를 나눠 솔직하게 평가해야 한다.

### 5.9 Critical Discussion

이 프로젝트는 architecture와 problem framing 측면에서 강하지만, full quantitative evidence 측면에서는 아직 확장 여지가 있다. 따라서 강점과 한계를 균형 있게 함께 서술해야 한다.

## 6. Conclusion and Future Work

### 6.1 Summary of Contributions

가장 큰 기여는 OCR, multimodal retrieval, vector storage, catalog retrieval, agent orchestration을 하나의 usable workflow로 통합했다는 점이다.

### 6.2 Final Positioning of the Prototype

가장 정확한 위치 규정은 **retrieval-first, human-in-the-loop identification assistant**다.

### 6.3 Limitations

- OCR fragility
- incomplete human review workflow
- incomplete benchmark automation
- 아직 제한적인 quantitative evidence

### 6.4 Future Work

- evaluation automation 완성
- audited accept/edit/reject workflow
- Qwen-centred vs mixed pipeline ablation
- hard case region focus
- user-centred validation

## References and Appendices Note

최종 제출 시에는 영문 working draft를 기준으로 bibliography와 figure/table reference를 정리하고, 현재 한국어 작업본은 내부 이해와 검토용으로 사용한다.
