import importlib.util
from pathlib import Path
import types

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module_from_path(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


convert_train = load_module_from_path(
    "convert_vlm2vec_train_to_nexus",
    "tools/multimodal_retrieval/convert_vlm2vec_train_to_nexus.py",
)
inventory_tool = load_module_from_path(
    "export_mmeb_v2_inventory",
    "tools/multimodal_retrieval/export_mmeb_v2_inventory.py",
)


def test_convert_pair_style_row_to_video_item():
    args = types.SimpleNamespace(
        sequence_mode="video",
        dataset_name="MSR-VTT",
        global_dataset_name="video/msrvtt",
        media_root=None,
        image_root=None,
        video_root=None,
    )
    row = {
        "query_text": "find the matching clip",
        "query_image": {"paths": ["frames/1.jpg", "frames/2.jpg"], "bytes": [None, None]},
        "pos_text": "positive caption",
        "pos_image": {"paths": ["frames/3.jpg", "frames/4.jpg"], "bytes": [None, None]},
        "neg_text": [],
        "neg_image": [],
    }

    converted = convert_train.convert_pair_style_row(row, args)

    assert converted["query"]["video_frames"] == ["frames/1.jpg", "frames/2.jpg"]
    assert converted["pos"][0]["video_frames"] == ["frames/3.jpg", "frames/4.jpg"]
    assert converted["metadata"]["dataset_name"] == "MSR-VTT"


def test_convert_llavahound_row_prefers_raw_video():
    args = types.SimpleNamespace(
        source_format="llavahound_caption",
        dataset_name="video_caption_300k",
        global_dataset_name="video_caption_300k",
        media_root=None,
        image_root=None,
        video_root="/data/videos",
        raw_video=True,
        frame_basedir=None,
        llavahound_mode="caption_retrieval",
    )
    row = {
        "id": "sample-1",
        "video": "abc.mp4",
        "conversations": [
            {"from": "human", "value": "<video> Describe this clip"},
            {"from": "gpt", "value": "A person is cooking in the kitchen."},
        ],
    }

    converted = convert_train.convert_llavahound_row(row, args)

    assert converted["query"]["video_path"] == "abc.mp4"
    assert converted["pos"][0]["text"] == "A person is cooking in the kitchen."


def test_load_python_variable_via_runpy_supports_expressions(tmp_path):
    source_file = tmp_path / "mapping.py"
    source_file.write_text(
        "import os\nBASE='root'\nMAPPING={'toy': (os.path.join(BASE, 'images'), None, 'test')}\n",
        encoding="utf-8",
    )

    mapping = inventory_tool.load_python_variable_via_runpy(str(source_file), "MAPPING")

    assert mapping["toy"] == ("root/images", None, "test")
