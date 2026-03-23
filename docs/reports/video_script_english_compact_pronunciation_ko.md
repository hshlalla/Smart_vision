# Final Video Script (English, Korean Pronunciation and Translation Guide)

This file is a Korean pronunciation and translation guide for:
- `docs/reports/video_script_english_compact.md`

Use it together with the English script during recording.

Format:
- English original
- Korean pronunciation
- Korean meaning

## Final Script Guide

### Opening

Hello. My project is **Smart Image Part Identifier for Secondhand Platforms**.  
헬로. 마이 프로젝트 이즈 **스마트 이미지 파트 아이덴티파이어 포 세컨핸드 플랫폼즈**.  
안녕하세요. 제 프로젝트는 **중고 플랫폼을 위한 스마트 이미지 부품 식별 시스템**입니다.

This is a web-based system designed to help users identify parts and prepare listing metadata from uploaded images, especially in secondhand marketplace workflows.  
디스 이즈 어 웹-베이스트 시스템 디자인드 투 헬프 유저즈 아이덴티파이 파츠 앤드 프리페어 리스팅 메타데이터 프럼 업로디드 이미지즈, 이스페셜리 인 세컨핸드 마켓플레이스 워크플로우즈.  
이 시스템은 특히 중고 마켓플레이스 워크플로우에서, 사용자가 업로드한 이미지로 부품을 식별하고 판매 등록용 메타데이터를 준비할 수 있도록 돕는 웹 기반 시스템입니다.

### Login

I will start from the login page.  
아이 윌 스타트 프럼 더 로그인 페이지.  
로그인 페이지부터 시작하겠습니다.

This page is the entry point to the system.  
디스 페이지 이즈 디 엔트리 포인트 투 더 시스템.  
이 페이지는 시스템의 시작점입니다.

The project is not just a single model demo. It is implemented as a real web application with a structured user flow.  
더 프로젝트 이즈 낫 저스트 어 싱글 모델 데모. 잇 이즈 임플리멘티드 애즈 어 리얼 웹 애플리케이션 위드 어 스트럭처드 유저 플로우.  
이 프로젝트는 단순한 단일 모델 데모가 아니라, 구조화된 사용자 흐름을 가진 실제 웹 애플리케이션으로 구현되었습니다.

The interface also supports both Korean and English through a language toggle. After login, users can move to the search, indexing, catalog, and agent features.  
디 인터페이스 올소 서포츠 보쓰 코리안 앤드 잉글리시 쓰루 어 랭귀지 토글. 애프터 로그인, 유저즈 캔 무브 투 더 서치, 인덱싱, 카탈로그, 앤드 에이전트 피처즈.  
이 인터페이스는 언어 토글을 통해 한국어와 영어를 모두 지원합니다. 로그인 후에는 검색, 인덱싱, 카탈로그, 에이전트 기능으로 이동할 수 있습니다.

### Indexing

Next is the indexing page.  
넥스트 이즈 디 인덱싱 페이지.  
다음은 인덱싱 페이지입니다.

This page is used to register a new item.  
디스 페이지 이즈 유즈드 투 레지스터 어 뉴 아이템.  
이 페이지는 새로운 아이템을 등록하는 데 사용됩니다.

The user can upload one image or multiple images for the same item. Different views, such as the front, side, and label surface, help the system gather stronger evidence.  
더 유저 캔 업로드 원 이미지 오어 멀티플 이미지즈 포 더 세임 아이템. 디퍼런트 뷰즈, 서치 애즈 더 프런트, 사이드, 앤드 레이블 서피스, 헬프 더 시스템 개더 스트롱거 에비던스.  
사용자는 같은 아이템에 대해 한 장 또는 여러 장의 이미지를 업로드할 수 있습니다. 앞면, 옆면, 라벨 표면 같은 다양한 시점은 시스템이 더 강한 증거를 수집하는 데 도움이 됩니다.

Instead of requiring the user to write all metadata manually, the system first generates a metadata draft.  
인스테드 오브 리콰이어링 더 유저 투 라이트 올 메타데이터 매뉴얼리, 더 시스템 퍼스트 제너레이츠 어 메타데이터 드래프트.  
사용자가 모든 메타데이터를 수동으로 작성하도록 요구하는 대신, 시스템은 먼저 메타데이터 초안을 생성합니다.

When I press the extraction button, the system uses a VLM-based path with Qwen or GPT support to suggest fields such as maker, category, part number, and description.  
웬 아이 프레스 디 익스트랙션 버튼, 더 시스템 유지즈 어 브이엘엠-베이스트 패스 위드 큐웬 오어 지피티 서포트 투 서제스트 필즈 서치 애즈 메이커, 카테고리, 파트 넘버, 앤드 디스크립션.  
추출 버튼을 누르면, 시스템은 Qwen 또는 GPT를 지원하는 VLM 기반 경로를 사용해 제조사, 카테고리, 부품 번호, 설명 같은 항목을 제안합니다.

The user can review and edit these suggestions, and only when the user presses **confirm** does the actual indexing and saving happen.  
더 유저 캔 리뷰 앤드 에딧 디즈 서제스천즈, 앤드 온리 웬 더 유저 프레시즈 **컨펌** 더즈 디 액추얼 인덱싱 앤드 세이빙 해픈.  
사용자는 이 제안들을 검토하고 수정할 수 있으며, 사용자가 **confirm**을 눌렀을 때에만 실제 인덱싱과 저장이 이루어집니다.

This is safer than fully automatic write-back and fits a real listing workflow more naturally.  
디스 이즈 세이퍼 댄 풀리 오토매틱 라이트-백 앤드 핏츠 어 리얼 리스팅 워크플로우 모어 내추럴리.  
이 방식은 완전 자동 저장보다 더 안전하고, 실제 판매 등록 워크플로우에도 더 자연스럽게 맞습니다.

The system can also use OCR as an additional support path, especially when a label or identifier is visible and text evidence is important.  
더 시스템 캔 올소 유즈 오씨알 애즈 언 어디셔널 서포트 패스, 이스페셜리 웬 어 레이블 오어 아이덴티파이어 이즈 비저블 앤드 텍스트 에비던스 이즈 임포턴트.  
또한 라벨이나 식별자가 잘 보이고 텍스트 증거가 중요한 경우, 시스템은 OCR을 추가 보조 경로로 사용할 수 있습니다.

### Search

Now I will show the search page.  
나우 아이 윌 쇼 더 서치 페이지.  
이제 검색 페이지를 보여드리겠습니다.

This page allows the user to search using text only, or using both an image and text together.  
디스 페이지 얼라우즈 더 유저 투 서치 유징 텍스트 온리, 오어 유징 보쓰 언 이미지 앤드 텍스트 투게더.  
이 페이지에서는 텍스트만으로 검색할 수도 있고, 이미지와 텍스트를 함께 사용해서 검색할 수도 있습니다.

The important point here is that the system follows a **retrieval-first approach**.  
디 임포턴트 포인트 히어 이즈 댓 더 시스템 팔로우즈 어 **리트리벌-퍼스트 어프로치**.  
여기서 중요한 점은 이 시스템이 **retrieval-first 접근법**을 따른다는 것입니다.

Instead of forcing one final answer, it returns a shortlist of plausible candidates and lets the user judge them.  
인스테드 오브 포싱 원 파이널 앤서, 잇 리턴즈 어 쇼트리스트 오브 플로저블 캔디데이트츠 앤드 렛츠 더 유저 저지 뎀.  
하나의 최종 답을 강제로 제시하는 대신, 그럴듯한 후보들의 shortlist를 반환하고 사용자가 직접 판단할 수 있게 합니다.

When the search button is pressed, the system combines image embeddings, text signals, metadata signals, and lexical matching to retrieve candidates.  
웬 더 서치 버튼 이즈 프레스트, 더 시스템 컴바인즈 이미지 임베딩즈, 텍스트 시그널즈, 메타데이터 시그널즈, 앤드 렉시컬 매칭 투 리트리브 캔디데이트츠.  
검색 버튼이 눌리면 시스템은 이미지 임베딩, 텍스트 신호, 메타데이터 신호, 그리고 lexical matching을 결합해서 후보를 검색합니다.

Because of this, the result is not based only on visual similarity. It can also reflect clues such as maker names or part numbers.  
비커즈 오브 디스, 더 리절트 이즈 낫 베이스트 온리 온 비주얼 시밀래리티. 잇 캔 올소 리플렉트 클루즈 서치 애즈 메이커 네임즈 오어 파트 넘버즈.  
그래서 결과는 단지 시각적 유사성만으로 결정되지 않습니다. 제조사 이름이나 부품 번호 같은 단서도 반영할 수 있습니다.

When we look at the results, an important strength is that the system tries to provide **evidence-backed output**.  
웬 위 룩 앳 더 리절츠, 언 임포턴트 스트렝쓰 이즈 댓 더 시스템 트라이즈 투 프로바이드 **에비던스-백트 아웃풋**.  
결과를 보면, 중요한 강점 중 하나는 시스템이 **근거 기반 출력**을 제공하려고 한다는 점입니다.

It does not only show a ranking. It is designed so that the user can understand why a candidate is relevant.  
잇 더즈 낫 온리 쇼 어 랭킹. 잇 이즈 디자인드 소우 댓 더 유저 캔 언더스탠드 와이 어 캔디데이트 이즈 렐러번트.  
단순히 순위만 보여주는 것이 아니라, 왜 어떤 후보가 관련 있는지 사용자가 이해할 수 있도록 설계되었습니다.

This is one of the key strengths of the project.  
디스 이즈 원 오브 더 키 스트렝쓰 오브 더 프로젝트.  
이 점이 이 프로젝트의 핵심 강점 중 하나입니다.

### Catalog

Next is the Catalog page.  
넥스트 이즈 더 카탈로그 페이지.  
다음은 Catalog 페이지입니다.

This feature allows the user to upload PDF documents and turn them into searchable knowledge sources.  
디스 피처 얼라우즈 더 유저 투 업로드 피디에프 다큐먼츠 앤드 턴 뎀 인투 서처블 날리지 소시즈.  
이 기능은 사용자가 PDF 문서를 업로드해서 검색 가능한 지식 소스로 바꿀 수 있게 해줍니다.

For example, a parts catalogue or manual can later be used as supporting evidence during retrieval.  
포 이그잼플, 어 파츠 카탈로그 오어 매뉴얼 캔 레이터 비 유즈드 애즈 서포팅 에비던스 듀어링 리트리벌.  
예를 들어, 부품 카탈로그나 매뉴얼은 나중에 검색 과정에서 보조 근거로 활용될 수 있습니다.

### Agent

Next is the Agent page.  
넥스트 이즈 디 에이전트 페이지.  
다음은 Agent 페이지입니다.

This feature combines hybrid search, catalog search, and, when needed, external evidence to provide richer support.  
디스 피처 컴바인즈 하이브리드 서치, 카탈로그 서치, 앤드, 웬 니디드, 익스터널 에비던스 투 프로바이드 리처 서포트.  
이 기능은 hybrid search, catalog search, 그리고 필요할 경우 외부 근거를 결합해 더 풍부한 지원을 제공합니다.

In addition, if a product is not already registered and useful information is found through the web path, I added an upsert path so that it can later be reflected into Milvus.  
인 어디션, 이프 어 프로덕트 이즈 낫 올레디 레지스터드 앤드 유스풀 인포메이션 이즈 파운드 쓰루 더 웹 패스, 아이 애디드 언 업서트 패스 소우 댓 잇 캔 레이터 비 리플렉티드 인투 밀버스.  
또한 제품이 아직 등록되어 있지 않더라도, 웹 경로를 통해 유용한 정보가 발견되면 나중에 Milvus에 반영될 수 있도록 upsert 경로도 추가했습니다.

### Closing

To summarize, the main strength of this project is not only the performance of one model.  
투 서머라이즈, 더 메인 스트렝쓰 오브 디스 프로젝트 이즈 낫 온리 더 퍼포먼스 오브 원 모델.  
정리하자면, 이 프로젝트의 주요 강점은 단지 하나의 모델 성능만이 아닙니다.

Its real contribution is that it integrates **search, metadata assistance, catalog retrieval, agent orchestration, and user confirmation into one practical workflow**.  
잇츠 리얼 컨트리뷰션 이즈 댓 잇 인티그레이츠 **서치, 메타데이터 어시스턴스, 카탈로그 리트리벌, 에이전트 오케스트레이션, 앤드 유저 컨펌메이션 인투 원 프랙티컬 워크플로우**.  
진짜 기여는 **검색, 메타데이터 보조, 카탈로그 검색, 에이전트 오케스트레이션, 그리고 사용자 확인 절차를 하나의 실용적인 워크플로우로 통합했다는 점**입니다.

In that sense, it is best described as a **retrieval-first, human-in-the-loop identification assistant prototype**.  
인 댓 센스, 잇 이즈 베스트 디스크라이브드 애즈 어 **리트리벌-퍼스트, 휴먼-인-더-루프 아이덴티피케이션 어시스턴트 프로토타입**.  
그런 의미에서 이 시스템은 **retrieval-first, human-in-the-loop 식별 보조 프로토타입**이라고 설명하는 것이 가장 적절합니다.

At the current stage, the system already supports an end-to-end search and indexing workflow, and the experiments and evaluation have also been completed.  
앳 더 커런트 스테이지, 더 시스템 얼레디 서포츠 언 엔드-투-엔드 서치 앤드 인덱싱 워크플로우, 앤드 디 익스페리먼츠 앤드 이밸류에이션 해브 올소 빈 컴플리티드.  
현재 단계에서 이 시스템은 이미 end-to-end 검색 및 인덱싱 워크플로우를 지원하며, 실험과 평가도 완료된 상태입니다.

The most accurate description of the project is therefore not a fully autonomous identification system, but a practical identification assistant that helps users narrow down candidates, inspect evidence, and prepare listing information more efficiently.  
더 모스트 애큐릿 디스크립션 오브 더 프로젝트 이즈 데어포어 낫 어 풀리 오토너머스 아이덴티피케이션 시스템, 벗 어 프랙티컬 아이덴티피케이션 어시스턴트 댓 헬프스 유저즈 내로우 다운 캔디데이트츠, 인스펙트 에비던스, 앤드 프리페어 리스팅 인포메이션 모어 이피션틀리.  
따라서 이 프로젝트를 가장 정확하게 설명하는 방식은 완전 자율 식별 시스템이 아니라, 사용자가 후보를 좁히고 근거를 확인하며 판매 정보를 더 효율적으로 준비할 수 있도록 돕는 실용적인 식별 보조 시스템이라는 것입니다.

Thank you.  
땡큐.  
감사합니다.

## Short Notes

- Read slowly and keep pauses between screens.  
  천천히 읽고 화면 전환 사이에 잠깐씩 멈추면 됩니다.
- For difficult words, prioritise clarity over native-like speed.  
  어려운 단어는 원어민처럼 빠르게 말하는 것보다 또렷하게 말하는 것이 더 중요합니다.
- If needed, shorten the Catalog and Agent paragraphs first.  
  시간이 부족하면 Catalog와 Agent 부분부터 먼저 줄이면 됩니다.
