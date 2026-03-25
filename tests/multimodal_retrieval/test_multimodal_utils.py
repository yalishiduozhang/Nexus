from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import Nexus
from Nexus.modules.multimodal import (
    build_media_base_dir,
    build_prefixed_multimodal_group,
    MultimodalProcessorAdapter,
    normalize_multimodal_item,
)


def test_import_nexus_without_optional_inference_backends():
    assert hasattr(Nexus, "MultimodalEmbedder")


def test_normalize_multimodal_item_supports_separate_media_roots(tmp_path):
    image_root = tmp_path / "images"
    video_root = tmp_path / "videos"
    image_root.mkdir()
    video_root.mkdir()

    base_dir = build_media_base_dir(
        base_dir=str(tmp_path / "metadata"),
        image_root=str(image_root),
        video_root=str(video_root),
    )

    item = normalize_multimodal_item(
        {
            "text": "find the clip",
            "image_path": "page_1.png",
            "video_path": "sample.mp4",
        },
        base_dir=base_dir,
    )

    assert item["text"] == "find the clip"
    assert item["images"][0]["path"] == str(image_root / "page_1.png")
    assert item["videos"][0]["path"] == str(video_root / "sample.mp4")


def test_build_prefixed_multimodal_group_supports_video_fields(tmp_path):
    video_root = tmp_path / "videos"
    video_root.mkdir()

    base_dir = build_media_base_dir(base_dir=str(tmp_path), video_root=str(video_root))
    row = {
        "query_text": "retrieve the matching video",
        "query_video_path": "clips/a.mp4",
        "pos_text": "positive target",
        "pos_video_frames": ["frames/0001.jpg", "frames/0002.jpg"],
    }

    query_group = build_prefixed_multimodal_group(row, "query", base_dir=base_dir)
    pos_group = build_prefixed_multimodal_group(row, "pos", base_dir=base_dir)

    assert query_group[0]["videos"][0]["path"] == str(video_root / "clips/a.mp4")
    assert pos_group[0]["videos"][0]["paths"] == [
        str(tmp_path / "frames/0001.jpg"),
        str(tmp_path / "frames/0002.jpg"),
    ]


def test_normalize_multimodal_item_avoids_duplicate_media_root_join(tmp_path):
    media_root = tmp_path / "data" / "media"
    media_root.mkdir(parents=True)
    image_path = media_root / "query.ppm"
    image_path.write_bytes(b"P6\n1 1\n255\n\xff\x00\x00")

    base_dir = build_media_base_dir(
        base_dir=str(tmp_path / "data"),
        media_root=str(media_root),
    )

    item = normalize_multimodal_item(
        {
            "text": "find the matching image",
            "image_path": "media/query.ppm",
        },
        base_dir=base_dir,
    )

    assert item["images"][0]["path"] == str(image_path)


def test_build_media_base_dir_prefers_existing_cwd_relative_root(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    base_dir = repo_root / "examples" / "multimodal_retrieval" / "data"
    media_root = base_dir / "media"
    media_root.mkdir(parents=True)

    monkeypatch.chdir(repo_root)
    resolved = build_media_base_dir(
        base_dir=str(base_dir),
        media_root="examples/multimodal_retrieval/data/media",
    )

    assert resolved["media"] == str(media_root)
    assert resolved["image"] == str(media_root)
    assert resolved["video"] == str(media_root)


def test_processor_adapter_batches_visual_tensors_without_processor_pad():
    class FakeTokenizer:
        def pad(self, features, padding=True, return_tensors="pt"):
            max_len = max(len(feature["input_ids"]) for feature in features)
            input_ids = []
            attention_mask = []
            for feature in features:
                ids = list(feature["input_ids"])
                mask = list(feature.get("attention_mask", [1] * len(ids)))
                pad_len = max_len - len(ids)
                input_ids.append(ids + [0] * pad_len)
                attention_mask.append(mask + [0] * pad_len)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = FakeTokenizer()

    adapter = MultimodalProcessorAdapter(FakeProcessor(), model_type="qwen2_vl", use_chat_template=False)
    adapter._encode_single = lambda item, max_length=None: {
        "input_ids": [[11, 12, 13]],
        "attention_mask": [[1, 1, 1]],
        "pixel_values": torch.ones((1, 4, 8)),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }

    batch = adapter.encode_batch([{"text": "a"}, {"text": "b"}], max_length=8)

    assert tuple(batch["input_ids"].shape) == (2, 3)
    assert tuple(batch["pixel_values"].shape) == (2, 4, 8)
    assert tuple(batch["image_grid_thw"].shape) == (2, 3)
