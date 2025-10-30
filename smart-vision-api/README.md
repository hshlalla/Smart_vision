# smart-vision-api

`smart-vision-api` 디렉터리는 Smart Vision 하이브리드 검색 파이프라인을 REST API 형태로 제공하는 FastAPI 서비스입니다.  
PaddleOCR-VL, BGE-VL, BGE-M3, Milvus를 활용해 장비 이미지와 텍스트를 동시에 검색하거나 신규 데이터를 색인할 수 있습니다.

---

## 📁 디렉터리 구조

```
smart-vision-api/
├── smart_vision_api/
│   ├── main.py              # FastAPI 애플리케이션 엔트리포인트
│   ├── api/
│   │   └── v1/
│   │       └── hybrid.py    # 하이브리드 검색 REST 엔드포인트
│   ├── core/
│   │   ├── config.py        # 설정/환경변수 관리
│   │   └── logger.py        # 공통 로거
│   ├── schemas/
│   │   └── payload.py       # Pydantic 요청/응답 모델
│   └── services/
│       └── hybrid.py        # HybridSearchOrchestrator 서비스 래퍼
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

- **POST `/api/v1/hybrid/index`**  
  - 이미지 + 메타데이터를 업로드하면 전처리(ocr/text/image embedding)를 수행하고 Milvus에 저장합니다.

- **POST `/api/v1/hybrid/search`**  
  - 텍스트와/또는 이미지로 하이브리드 검색을 실행합니다.
  - 결과는 결합 점수, part number 매칭 여부 등의 필드를 포함합니다.

엔드포인트는 `smart_match.HybridSearchOrchestrator`를 재사용하여 API와 데모가 동일한 파이프라인 위에서 동작하도록 설계되었습니다.

---

## 🛠️ 실행 방법

1. **모델 패키지 설치**
   ```bash
   pip install -e ../smart-vision-model
   ```

2. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

3. **Milvus 연결**
   - 기본 URI는 `http://localhost:19530` 입니다.
   - 필요 시 `.env` 또는 환경변수 `MILVUS_URI`로 수정하세요.
   - 로컬에서 빠르게 테스트하려면 `docker-compose up -d milvus` 를 사용할 수 있습니다.

4. **API 실행**
   ```bash
   uvicorn smart_vision_api.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **확인**
   - [http://localhost:8000/api/docs](http://localhost:8000/api/docs) 에서 OpenAPI 문서를 확인할 수 있습니다.

---

## 📡 API 사용 예시

### 1. 자산 색인
```bash
curl -X POST "http://localhost:8000/api/v1/hybrid/index" \
  -F "image=@sample.jpg" \
  -F "maker=SurplusGLOBAL" \
  -F "part_number=PN-001" \
  -F "category=ETCH"
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
        \"part_number\": \"PN-001\",
        \"top_k\": 5
      }"
```

응답 예시:
```json
{
  "results": [
    {
      "id": 429128221007828993,
      "source": "image",
      "distance": 0.21,
      "maker": "SurplusGLOBAL",
      "part_number": "PN-001",
      "category": "ETCH",
      "ocr_text": "etching chamber",
      "fusion_score": 0.84,
      "verified": true
    }
  ]
}
```

---

## 📦 참고

- PaddleOCR-VL/BGE-VL/BGE-M3 모델은 최초 실행 시 자동으로 가중치를 다운로드합니다.
- Milvus 컬렉션(`image_parts`, `text_parts`, `attrs_parts`)은 API 구동 시 자동 생성됩니다.
- 운영 배포 시에는 `scripts/run_prod.sh` 또는 Dockerfile을 활용해 주세요.

---

## 🤝 문의

- 문의: Smart Vision AI Team
