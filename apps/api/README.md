# Smart Vision API

`apps/api` 디렉터리는 Smart Vision 하이브리드 검색 파이프라인을 REST API 형태로 제공하는 FastAPI 서비스입니다.  
PaddleOCR-VL, Qwen3-VL-Embedding-2B, BGE-M3, Qwen3-VL-Reranker-2B, Qwen3-VL-2B-Instruct, Milvus를 활용해 장비 이미지와 텍스트를 동시에 검색하거나 신규 데이터를 색인할 수 있습니다.  
또한 LangChain 기반 tool-calling 에이전트(`/api/v1/agent/chat`)를 통해 “이미지 업로드 → 제품 추정 → 웹에서 정보/가격 보강 → (옵션) Milvus 업데이트” 흐름을 제공합니다.

---

## 📁 디렉터리 구조

```
apps/api/
├── smart_vision_api/
│   ├── main.py              # FastAPI 애플리케이션 엔트리포인트
│   ├── api/
│   │   └── v1/
│   │       ├── hybrid.py    # 하이브리드 검색 REST 엔드포인트
│   │       ├── agent.py     # 에이전트 챗 엔드포인트
│   │       └── auth.py      # 로그인/토큰 엔드포인트
│   ├── core/
│   │   ├── config.py        # 설정/환경변수 관리
│   │   ├── auth.py          # 간단 토큰 인증(옵션)
│   │   └── logger.py        # 공통 로거
│   ├── schemas/
│   │   ├── payload.py       # 하이브리드 요청/응답
│   │   ├── agent.py         # 에이전트 요청/응답
│   │   └── auth.py          # 로그인 요청/응답
│   └── services/
│       ├── hybrid.py        # HybridSearchOrchestrator 서비스 래퍼
│       ├── agent.py         # tool-calling agent
│       ├── web_search.py    # open-world 검색(DDG HTML)
│       └── gparts.py        # 예제 가격 소스(옵션)
├── docs/                    # 릴리스 노트 등 문서
├── logs/                    # 실행 로그 출력 디렉터리
├── requirements.txt         # API 의존성 목록
├── pyproject.toml           # 패키징 설정
├── Dockerfile               # 컨테이너 빌드 설정
├── docker-compose.yml       # Milvus + API 로컬 실행 예시
├── .env                     # 환경변수 템플릿
└── scripts/                 # 실행 스크립트 (run_dev.sh 등)
```

---

## 🚀 제공 기능

- **Auth (옵션)**  
  - `GET /api/v1/auth/status`
  - `POST /api/v1/auth/login`
  - `GET /api/v1/auth/me`

- **Hybrid Search**
  - `POST /api/v1/hybrid/index` : 이미지 + 메타데이터를 전처리 후 Milvus 저장 (`model_id` 필수)
  - `POST /api/v1/hybrid/search` : 텍스트/이미지 하이브리드 검색

- **Agent Bot (open-world + Milvus enrichment)**
  - `POST /api/v1/agent/chat` : 이미지/질문 → (smart vision 검색 tool) → 웹 검색/가격 보강 → (옵션) Milvus 업데이트
  - 에이전트의 “기존 모델 재사용” 기준은 `score >= 0.75` 입니다.

---

## 🛠️ 실행 방법

1. **모델 패키지 설치**
   ```bash
   pip install -e ../../packages/model
   ```

2. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```
   - PaddleOCR-VL을 쓰려면 Paddle 계열 버전을 함께 맞춰야 합니다.
   - 현재 저장소 기준 권장 조합은 `paddleocr 3.4.x`, `paddlex 3.4.x` 입니다.
   - `paddlepaddle`은 플랫폼별로 다릅니다: Linux/Windows/Apple Silicon은 `3.1.x`, Intel Mac은 `3.0.x`를 사용합니다.
   - Intel Mac에서는 `PaddleOCRVL`이 요구하는 fused op가 없어 표준 `PaddleOCR` 폴백 경로로 동작합니다.
   - 기존에 다른 major/minor 버전이 깔려 있었다면 가상환경을 새로 만들거나 관련 패키지를 제거 후 재설치하세요.

3. **Milvus 연결**
   - 기본 URI는 `tcp://standalone:19530` 입니다(docker network 내부 기준).
   - 로컬 호스트에서 실행 중인 Milvus에 붙을 때는 `tcp://localhost:19530` 를 사용하세요.
   - 로컬에서 빠르게 테스트하려면 `docker-compose up -d milvus` 를 사용할 수 있습니다.

4. **API 실행 (권장: 스크립트)**
   ```bash
   ./scripts/run_dev.sh
   ```
   - `run_dev.sh`는 `apps/api/.env`를 읽어 환경변수를 자동 로드합니다.
   - `.env`에 `OPENAI_API_KEY` 등이 있으면 별도 `export`가 필요 없습니다.
   - 기본 포트는 `8001`입니다.

5. **직접 uvicorn 실행(선택)**
   ```bash
   uvicorn smart_vision_api.main:app --reload --host 0.0.0.0 --port 8000 --env-file .env
   ```

6. **확인**
   - `run_dev.sh` 실행 시: [http://localhost:8001/api/docs](http://localhost:8001/api/docs)
   - 직접 `uvicorn --port 8000` 실행 시: [http://localhost:8000/api/docs](http://localhost:8000/api/docs)

---

## 📡 API 사용 예시

### 0. 로그인(옵션)

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
```

### 1. 자산 색인
```bash
curl -X POST "http://localhost:8000/api/v1/hybrid/index" \
  -F "image=@sample.jpg" \
  -F "model_id=a000001" \
  -F "maker=SmartVision" \
  -F "part_number=PN-001" \
  -F "category=ETCH" \
  -F "description=example"
```

응답:
```json
{"status":"indexed"}
```

### 2. 멀티모달 검색
```bash
BASE64_IMG=$(base64 -w0 query.jpg)
curl -X POST "http://localhost:8000/api/v1/hybrid/search" \
  -H "Content-Type: application/json" \
  -d "{
        \"query_text\": \"etch chamber\",
        \"image_base64\": \"${BASE64_IMG}\",
        \"top_k\": 5
      }"
```

### 3. 에이전트 챗(이미지 + 질문 → 답변 + sources)

```bash
BASE64_IMG=$(base64 -w0 query.jpg)
curl -X POST "http://localhost:8000/api/v1/agent/chat" \
  -H "Content-Type: application/json" \
  -d "{
        \"message\": \"이 제품 뭐야? 가격도 찾아줘\",
        \"image_base64\": \"${BASE64_IMG}\",
        \"update_milvus\": true
      }"
```

---

## 📦 참고

- PaddleOCR-VL/Qwen3-VL-Embedding-2B/BGE-M3/Qwen3-VL-Reranker-2B/Qwen3-VL-2B-Instruct 모델은 최초 실행 시 자동으로 가중치를 다운로드합니다.
- Milvus 컬렉션(`qwen3_vl_image_parts`, `bge_m3_text_parts`, `attrs_parts_v2`, `bge_m3_model_texts`, `bge_m3_catalog_chunks`)은 API 구동 시 자동 생성됩니다.
- 프론트(`apps/web/`)에서 접근하려면 CORS 설정(`CORS_ORIGINS`)이 필요할 수 있습니다.
- 운영 배포 시에는 `scripts/run_prod.sh` 또는 Dockerfile을 활용해 주세요.

---

## 🤝 문의

- 문의: suhun.hong
