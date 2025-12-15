# test_search.py (예시)
import sys, pathlib, csv
from pathlib import Path
import pandas as pd
root = pathlib.Path("/workspace/SG-project/Smart_vision/smart-vision-model")
sys.path.insert(0, str(root))
from smart_match.hybrid_search_pipeline.hybrid_pipeline_runner import HybridSearchOrchestrator, FusionWeights

orc = HybridSearchOrchestrator(fusion_weights=FusionWeights(alpha=1.0, beta=0.0, gamma=0.0))

csv_path = "dataset/test_dataset/random_1000_model_test.csv"
rows = []
hits = 0
total = 0
print("1111")
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, 1):
        img = row["image_path"]
        stem = Path(img).stem
        img_seq = stem.split("_")[-1] if "_" in stem else ""
        gt = row["model_id"]
        try:
            results = orc.search(query_image=img, top_k=5)
            top_ids = [r["model_id"] for r in results[:5]]
            top_image_ids = [
                (r.get("images") or [{}])[0].get("image_id", "") if r else "" for r in results[:5]
            ]
            hit_rank = next((idx + 1 for idx, mid in enumerate(top_ids) if mid == gt), None)
            hit = hit_rank is not None
            hits += int(hit)
            total += 1
            rows.append({
                "idx": i,
                "query_image": img,
                "query_image_no": img_seq,
                "gt_model_id": gt,
                "hit": hit,
                "hit_rank": hit_rank or "",
                "top5_model_ids": ";".join(top_ids),
                "top5_image_ids": ";".join(top_image_ids),
                "top1_score": results[0]["score"] if results else "",
            })
            print(f"[{i}] hit={hit} gt={gt} img_no={img_seq} top_ids={top_ids} top_image_ids={top_image_ids}", flush=True)
        except Exception as e:
            rows.append({"idx": i, "query_image": img, "gt_model_id": gt,
                         "query_image_no": img_seq, "hit": False, "hit_rank": "",
                         "top5_model_ids": "", "top5_image_ids": "", "top1_score": "", "error": str(e)})
            total += 1
            print(f"[skip] {img}: {e}", flush=True)

recall_at_5 = hits / total if total else 0.0
print(f"Recall@5: {recall_at_5:.4f} ({hits}/{total})")

df = pd.DataFrame(rows)
df.to_excel("search_results.xlsx", index=False)  # index.csv와 동일 위치에 생성
print("Saved: search_results.xlsx")
