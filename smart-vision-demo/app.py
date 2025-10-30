"""
Hybrid Search Demo Application

Provides a Gradio interface that showcases the hybrid search pipeline:
    - Index tab: upload an image + metadata to store in Milvus
    - Search tab: run multimodal queries using text and/or image inputs
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List

import gradio as gr
from gradio import networking as gradio_networking
from gradio_client import utils as gradio_client_utils
from PIL import Image

from smart_match import HybridSearchOrchestrator
from smart_match.hybrid_search_pipeline.hybrid_pipeline_runner import (
    FusionWeights,
    MilvusConnectionConfig,
)


def _create_orchestrator() -> HybridSearchOrchestrator:
    milvus_uri = os.getenv("MILVUS_URI", "tcp://standalone:19530")
    return HybridSearchOrchestrator(
        milvus=MilvusConnectionConfig(uri=milvus_uri),
        fusion_weights=FusionWeights(alpha=0.6, beta=0.4),
    )


ORCHESTRATOR = _create_orchestrator()


def _dump_temp_image(image: Image.Image) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmp.name)
    return Path(tmp.name)


def index_asset(
    images: List[object],
    model_id: str,
    maker: str,
    part_number: str,
    category: str,
) -> str:
    if not images:
        return "이미지를 업로드해주세요."

    model_id = (model_id or "").strip()
    if not model_id:
        return "Model ID를 입력해주세요."

    results: List[str] = []
    metadata = {
        "model_id": model_id,
        "maker": maker,
        "part_number": part_number,
        "category": category,
    }

    for idx, file_obj in enumerate(images, start=1):
        tmp_path = None
        file_path = None
        try:
            if isinstance(file_obj, dict):
                file_path = file_obj.get("name") or file_obj.get("path")
            elif isinstance(file_obj, (str, Path)):
                file_path = str(file_obj)
            else:
                file_path = getattr(file_obj, "name", None)

            if not file_path:
                results.append("❌ 알 수 없는 파일 형식입니다.")
                continue

            with Image.open(file_path) as img:
                tmp_path = _dump_temp_image(img)
            item_metadata = dict(metadata)
            item_metadata["pk"] = f"{model_id}::img_{idx:03d}"
            ORCHESTRATOR.preprocess_and_index(tmp_path, item_metadata)
            results.append(f"✅ {Path(file_path).name} 인덱싱 완료")
        except Exception as exc:  # pragma: no cover - demo utility
            file_name = Path(file_path).name if file_path else "알 수 없는 파일"
            results.append(f"❌ {file_name} 인덱싱 실패: {exc}")
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    return "\n".join(results)


def run_search(
    query_image: Image.Image,
    query_text: str,
    part_number: str,
    top_k: int,
) -> List[Dict[str, object]]:
    image_path = None
    if query_image is not None:
        image_path = _dump_temp_image(query_image)

    try:
        results = ORCHESTRATOR.search(
            query_image=image_path,
            query_text=query_text,
            top_k=top_k,
            part_number=part_number or None,
        )
        return results
    except Exception as exc:  # pragma: no cover - demo utility
        return [{"error": str(exc)}]
    finally:
        if image_path:
            image_path.unlink(missing_ok=True)


def list_collections() -> Dict[str, Dict[str, object]]:
    try:
        return ORCHESTRATOR.index.describe()
    except Exception as exc:  # pragma: no cover - demo utility
        return {"error": str(exc)}


def drop_collection(name: str) -> str:
    try:
        cleaned = (name or "").strip()
        if not cleaned:
            return "Drop할 컬렉션 이름을 입력해주세요."
        removed = ORCHESTRATOR.index.drop_collection(cleaned)
        return f"✅ 드롭 완료: {removed}"
    except Exception as exc:  # pragma: no cover - demo utility
        return f"❌ 드롭 실패: {exc}"


with gr.Blocks(title="Smart Vision Hybrid Demo") as demo:
    gr.Markdown(
        """
        # Smart Vision Hybrid Search Demo
        BGE-VL, PaddleOCR-VL, BGE-M3로 구성된 하이브리드 파이프라인을 체험해보세요.
        """
    )

    with gr.Tab("single_asset_indexing"):
        with gr.Row():
            with gr.Column():
                upload_images = gr.File(
                    label="장비 이미지 목록",
                    file_types=["image"],
                    file_count="multiple",
                )
                model_id = gr.Textbox(label="Model ID", value="")
                maker = gr.Textbox(label="Maker", value="")
                part_number = gr.Textbox(label="Part Number", value="")
                category = gr.Textbox(label="Category", value="")
                index_button = gr.Button("인덱싱 실행")
            with gr.Column():
                index_output = gr.Markdown()
        index_button.click(
            index_asset,
            inputs=[upload_images, model_id, maker, part_number, category],
            outputs=index_output,
        )

    with gr.Tab("Search"):
        with gr.Row():
            with gr.Column():
                search_image = gr.Image(type="pil", label="쿼리 이미지 (선택)")
                query_text = gr.Textbox(label="쿼리 텍스트 (선택)")
                search_part_number = gr.Textbox(label="Part Number 필터", value="")
                top_k = gr.Slider(label="Top-K", minimum=1, maximum=20, step=1, value=5)
                search_button = gr.Button("검색")
            with gr.Column():
                search_results = gr.JSON(label="검색 결과")
        search_button.click(
            run_search,
            inputs=[search_image, query_text, search_part_number, top_k],
            outputs=search_results,
        )

    with gr.Tab("Milvus Status"):
        with gr.Row():
            with gr.Column():
                refresh_button = gr.Button("컬렉션 목록 새로고침")
                status_output = gr.JSON(label="Milvus 컬렉션 상태")
            with gr.Column():
                drop_name = gr.Textbox(label="드롭할 컬렉션 이름", placeholder="예: image_parts")
                drop_button = gr.Button("컬렉션 드롭")
                drop_result = gr.Markdown()

        refresh_button.click(
            list_collections,
            inputs=[],
            outputs=status_output,
        )
        drop_button.click(
            drop_collection,
            inputs=drop_name,
            outputs=drop_result,
        )


gradio_networking.url_ok = lambda _url: True  # bypass localhost accessibility check in containerized envs

_original_get_type = gradio_client_utils.get_type
_original_json_schema_to_python_type = gradio_client_utils._json_schema_to_python_type


def _safe_get_type(schema):
    if isinstance(schema, bool):
        return {}
    if not isinstance(schema, dict):
        return _original_get_type(schema)
    return _original_get_type(schema)


def _safe_json_schema_to_python_type(schema, defs):
    if isinstance(schema, bool):
        # True => allow anything, False => disallow everything (treated as Any for demo)
        return "Any" if schema else "None"
    return _original_json_schema_to_python_type(schema, defs)


gradio_client_utils.get_type = _safe_get_type  # monkeypatch json schema parser bug
gradio_client_utils._json_schema_to_python_type = _safe_json_schema_to_python_type


if __name__ == "__main__":
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
        show_api=False,
        inbrowser=False,
        prevent_thread_lock=False,
    )
