### scripts 폴더

- **`index_tracker_incremental.py`**
  - `data/tracker_subset.csv`와 `data/images/<MODEL_ID>/` 구조를 스캔하여 새 모델/새 이미지만 Milvus에 추가합니다.
  - 저장된 이미지 사본(`media/`)을 SHA-256으로 해시해 기존 업로드와 중복 여부를 판단합니다.
  - `--dry-run`으로 변경 예정만 확인할 수 있고, `--verbose`로 상세 로그를 확인할 수 있습니다.
  - 기본 접속 URI는 `tcp://standalone:19530`이며, `--milvus-uri`로 조정 가능합니다.

#### 실행 예시
```bash
python -m smart_match.scripts.index_tracker_incremental \
    --images-root data/images \
    --dataset data/tracker_subset.csv \
    --milvus-uri tcp://standalone:19530 \
    --dry-run
```
`--dry-run`을 제거하면 실제 인덱싱이 수행됩니다.
