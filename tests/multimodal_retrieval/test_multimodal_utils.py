from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import Nexus
from Nexus.modules.multimodal import (
    build_media_base_dir,
    build_prefixed_multimodal_group,
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
