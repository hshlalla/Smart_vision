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

This web-based system helps users identify industrial parts and prepare listing metadata from uploaded images.  
디스 웹-베이스트 시스템 헬프스 유저즈 아이덴티파이 인더스트리얼 파츠 앤드 프리페어 리스팅 메타데이터 프럼 업로디드 이미지즈.  
이 웹 기반 시스템은 사용자가 업로드한 이미지로 산업용 부품을 식별하고 판매 등록용 메타데이터를 준비할 수 있도록 도와줍니다.

### Login

I will begin at the login page.  
아이 윌 비긴 앳 더 로그인 페이지.  
로그인 페이지부터 시작하겠습니다.

From here, users can access the search, indexing, catalog, and agent functions, and they can also switch between Korean and English.  
프럼 히어, 유저즈 캔 액세스 더 서치, 인덱싱, 카탈로그, 앤드 에이전트 펑션즈, 앤드 데이 캔 올소 스위치 비트윈 코리안 앤드 잉글리시.  
여기서 사용자는 검색, 인덱싱, 카탈로그, 에이전트 기능으로 이동할 수 있고, 한국어와 영어도 전환할 수 있습니다.

### Indexing

Next is the indexing page.  
넥스트 이즈 디 인덱싱 페이지.  
다음은 인덱싱 페이지입니다.

Users can upload one image or multiple images for the same item.  
유저즈 캔 업로드 원 이미지 오어 멀티플 이미지즈 포 더 세임 아이템.  
사용자는 같은 아이템에 대해 한 장 또는 여러 장의 이미지를 업로드할 수 있습니다.

The system first generates a metadata draft using a VLM-based path and suggests fields such as maker, category, part number, and description.  
더 시스템 퍼스트 제너레이츠 어 메타데이터 드래프트 유징 어 브이엘엠-베이스트 패스 앤드 서제스츠 필즈 서치 애즈 메이커, 카테고리, 파트 넘버, 앤드 디스크립션.  
시스템은 먼저 VLM 기반 경로를 사용해 메타데이터 초안을 만들고, 제조사, 카테고리, 부품 번호, 설명 같은 항목을 제안합니다.

The user reviews and edits the draft, and only after pressing **confirm** does the system save the final record.  
더 유저 리뷰즈 앤드 에딧츠 더 드래프트, 앤드 온리 애프터 프레싱 **컨펌** 더즈 더 시스템 세이브 더 파이널 레코드.  
사용자는 초안을 검토하고 수정하며, **confirm**을 누른 후에만 시스템이 최종 기록을 저장합니다.

This preview-before-confirm flow is safer and better suited to a real listing workflow.  
디스 프리뷰-비포-컨펌 플로우 이즈 세이퍼 앤드 베터 수티드 투 어 리얼 리스팅 워크플로우.  
이 preview-before-confirm 흐름은 더 안전하고 실제 판매 등록 워크플로우에 더 적합합니다.

### Search

Now I will show the search page.  
나우 아이 윌 쇼 더 서치 페이지.  
이제 검색 페이지를 보여드리겠습니다.

Users can search with text only, or with both image and text together.  
유저즈 캔 서치 위드 텍스트 온리, 오어 위드 보쓰 이미지 앤드 텍스트 투게더.  
사용자는 텍스트만으로 검색할 수도 있고, 이미지와 텍스트를 함께 사용해서 검색할 수도 있습니다.

The key idea is a **retrieval-first approach**.  
더 키 아이디어 이즈 어 **리트리벌-퍼스트 어프로치**.  
핵심 아이디어는 **retrieval-first 접근법**입니다.

Instead of forcing one final answer, the system returns a shortlist of likely candidates and lets the user inspect them.  
인스테드 오브 포싱 원 파이널 앤서, 더 시스템 리턴즈 어 쇼트리스트 오브 라이클리 캔디데이트츠 앤드 렛츠 더 유저 인스펙트 뎀.  
하나의 최종 답을 강제로 제시하는 대신, 시스템은 가능성 높은 후보들의 shortlist를 반환하고 사용자가 직접 확인할 수 있게 합니다.

The system combines image embeddings, text signals, metadata signals, and lexical matching.  
더 시스템 컴바인즈 이미지 임베딩즈, 텍스트 시그널즈, 메타데이터 시그널즈, 앤드 렉시컬 매칭.  
시스템은 이미지 임베딩, 텍스트 신호, 메타데이터 신호, 그리고 lexical matching을 결합합니다.

Because of this, the results are based not only on visual similarity, but also on identifier clues such as maker names or part numbers.  
비커즈 오브 디스, 더 리절츠 아 베이스트 낫 온리 온 비주얼 시밀래리티, 벗 올소 온 아이덴티파이어 클루즈 서치 애즈 메이커 네임즈 오어 파트 넘버즈.  
그래서 결과는 시각적 유사성뿐 아니라 제조사 이름이나 부품 번호 같은 식별 단서도 함께 반영합니다.

The interface also shows evidence for why a result is relevant, which improves trust and supports human decision making.  
디 인터페이스 올소 쇼즈 에비던스 포 와이 어 리절트 이즈 렐러번트, 위치 임프루브즈 트러스트 앤드 서포츠 휴먼 디시전 메이킹.  
또한 인터페이스는 왜 어떤 결과가 관련 있는지에 대한 근거도 보여주기 때문에, 신뢰를 높이고 사용자 판단을 지원합니다.

### Catalog

Next is the Catalog page.  
넥스트 이즈 더 카탈로그 페이지.  
다음은 Catalog 페이지입니다.

Here, users can upload PDF manuals or catalogues and turn them into searchable evidence sources.  
히어, 유저즈 캔 업로드 피디에프 매뉴얼즈 오어 카탈로그즈 앤드 턴 뎀 인투 서처블 에비던스 소시즈.  
여기서 사용자는 PDF 매뉴얼이나 카탈로그를 업로드해서 검색 가능한 근거 자료로 만들 수 있습니다.

### Agent

Next is the Agent page.  
넥스트 이즈 디 에이전트 페이지.  
다음은 Agent 페이지입니다.

This feature combines hybrid search, catalog search, and optional external evidence to provide richer support in one workflow.  
디스 피처 컴바인즈 하이브리드 서치, 카탈로그 서치, 앤드 옵셔널 익스터널 에비던스 투 프로바이드 리처 서포트 인 원 워크플로우.  
이 기능은 hybrid search, catalog search, 그리고 선택적 외부 근거를 결합해서 하나의 워크플로우 안에서 더 풍부한 지원을 제공합니다.

### Closing

To summarize, the main contribution of this project is not just one model.  
투 서머라이즈, 더 메인 컨트리뷰션 오브 디스 프로젝트 이즈 낫 저스트 원 모델.  
정리하면, 이 프로젝트의 핵심 기여는 단지 하나의 모델이 아닙니다.

It is the integration of search, metadata assistance, catalog retrieval, agent support, and user confirmation into one practical workflow.  
잇 이즈 디 인티그레이션 오브 서치, 메타데이터 어시스턴스, 카탈로그 리트리벌, 에이전트 서포트, 앤드 유저 컨펌메이션 인투 원 프랙티컬 워크플로우.  
핵심은 검색, 메타데이터 보조, 카탈로그 검색, 에이전트 지원, 사용자 확인 절차를 하나의 실용적인 워크플로우로 통합했다는 점입니다.

The system is best described as a **retrieval-first, human-in-the-loop identification assistant prototype**.  
더 시스템 이즈 베스트 디스크라이브드 애즈 어 **리트리벌-퍼스트, 휴먼-인-더-루프 아이덴티피케이션 어시스턴트 프로토타입**.  
이 시스템은 **retrieval-first, human-in-the-loop 식별 보조 프로토타입**이라고 설명하는 것이 가장 적절합니다.

Thank you.  
땡큐.  
감사합니다.

## Short Notes

- Read slowly and keep pauses between screens.  
  천천히 읽고 화면 전환 사이에 잠깐씩 멈추면 됩니다.
- For difficult words, prioritise clarity over native-like speed.  
  어려운 단어는 원어민처럼 빠르게 말하는 것보다 또렷하게 말하는 것이 더 중요합니다.
- If needed, shorten the Catalog and Agent sections to one sentence each.  
  시간이 더 부족하면 Catalog와 Agent 부분을 한 문장씩으로 더 줄이면 됩니다.
