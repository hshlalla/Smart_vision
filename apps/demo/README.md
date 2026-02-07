# Smart Vision Demo

`apps/demo` 디렉터리는 하이브리드 검색 파이프라인을 빠르게 확인하기 위한 **Gradio 기반 디버그 UI**입니다.

프로덕트/사용자용 UI는 `apps/web/`가 기본이며, Gradio는 내부 개발/디버깅 목적의 선택 사항입니다.

---

## 📁 디렉터리 구성

```
apps/demo/
├── app.py           # Gradio 진입점 – 인덱싱/검색 탭 UI
├── run_demo.sh      # 데모 실행 스크립트
├── config.py        # 데모 환경 설정(포트 등)
├── requirements.txt # 데모 의존성 목록
└── README.md
```

---

## 🚀 제공 기능

- **Index Asset 탭**  
  장비 이미지를 업로드하고 Maker/Part Number/Category 메타데이터를 입력하면
  자동으로 전처리(OCR+임베딩) 후 Milvus에 색인합니다.

- **Search 탭**  
  텍스트, 이미지, 파트넘버 필터를 조합해 하이브리드 검색을 실행하고,
  결합 점수와 검증 여부를 JSON 형태로 확인할 수 있습니다.

---

## 🛠️ 실행 방법

1. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```
   동시에 `packages/model` 패키지까지 설치하거나 editable 모드로 추가해 주세요.

2. **Milvus 준비**
   - 로컬 또는 원격 Milvus가 실행 중이어야 합니다.
   - 기본 URI는 `tcp://localhost:19530`이며 필요하면 환경변수 `MILVUS_URI`로 수정 가능합니다.

3. **데모 실행**
   ```bash
   bash run_demo.sh
   ```
   혹은 직접:
   ```bash
   python3 app.py
   ```

4. **브라우저 접속**
   - 기본 주소: [http://localhost:7860](http://localhost:7860)

---

## 🧭 사용 방법

1. `bash run_demo.sh` 또는 `python app.py`로 데모를 실행합니다.
2. **Index Asset** 탭에서 이미지를 업로드하고 메타데이터를 입력한 뒤 `인덱싱 실행` 버튼을 클릭합니다.  
   – 성공 시 “✅ 인덱싱이 완료되었습니다.” 메시지가 표시됩니다.
3. **Search** 탭에서 텍스트, 이미지, Part Number 필터를 조합하여 `검색` 버튼을 누릅니다.  
   – 결과는 JSON 형태로 출력되며, `fusion_score`, `verified` 등을 바로 확인할 수 있습니다.
4. 추가 실험을 위해 API(`apps/api`)와 동일한 Milvus 인스턴스를 사용하면 동일 데이터로 검증이 가능합니다.

---

## 📦 추가 참고

- 데모는 `smart_match.HybridSearchOrchestrator`를 직접 활용합니다.
- 새로 색인한 데이터는 Milvus 컬렉션(`image_parts`, `text_parts`, `attrs_parts`)에 저장됩니다.
- PaddleOCR-VL 및 BGE 계열 모델은 최초 실행 시 가중치를 자동 다운로드합니다.

Tip: 실제 사용자 흐름(로그인/모바일/에이전트 챗)을 확인하려면 `apps/web/` + `apps/api/`를 사용하세요.
