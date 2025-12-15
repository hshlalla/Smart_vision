import sys, pathlib, csv, torch
from types import SimpleNamespace
root = pathlib.Path("/workspace/SG-project/Smart_vision/smart-vision-model")
sys.path.insert(0, str(root))
from smart_match.hybrid_search_pipeline.hybrid_pipeline_runner import HybridSearchOrchestrator, FusionWeights

# --- 더미 컴포넌트 정의: OCR/캡션/텍스트 인코더 모두 skip ---
class NoOCR:
    def extract(self, path):
        return SimpleNamespace(tokens=[], markdown_pages=[])

class ZeroText:
    embedding_dim = 1024  # BGEM3 dim 고정값
    def encode_document(self, text): return torch.zeros(self.embedding_dim)
    def encode_query(self, text): return torch.zeros(self.embedding_dim)

orc = HybridSearchOrchestrator(
    fusion_weights=FusionWeights(alpha=1.0, beta=0.0, gamma=0.0)  # 이미지 점수만 사용
)
# bypass 설정
orc.preprocessing._ocr_engine = NoOCR()
orc.preprocessing._captioner = None
orc.text_encoder = ZeroText()

csv_path = "dataset/test_dataset/category_sampled_model_index.csv"

start_idx = 1

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, 1):  # i=1부터 시작
        if i < start_idx:
            continue  # 앞부분 스킵
        # 이하 기존 처리
        mid = row["model_id"]
        cat = row["category"]
        img = row["image_path"]
        metadata = {"model_id": mid, "category": cat}
        try:
            orc.preprocess_and_index(img, metadata)
        except Exception as e:
            print(f"[skip] {mid} / {img}: {e}")
