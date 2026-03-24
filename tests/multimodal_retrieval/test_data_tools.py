import importlib.util
import json
from pathlib import Path
import types
import pytest

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
manifest_lib = load_module_from_path(
    "vlm2vec_manifest_lib",
    "tools/multimodal_retrieval/vlm2vec_manifest_lib.py",
)
hf_tool = load_module_from_path(
    "hf_dataset_manager",
    "tools/multimodal_retrieval/hf_dataset_manager.py",
)
prepare_public_tool = load_module_from_path(
    "prepare_public_data",
    "tools/multimodal_retrieval/prepare_public_data.py",
)
prepare_train_tool = load_module_from_path(
    "prepare_mmeb_v2_train_data",
    "tools/multimodal_retrieval/prepare_mmeb_v2_train_data.py",
)
gpu_tool = load_module_from_path(
    "check_idle_gpus",
    "tools/multimodal_retrieval/check_idle_gpus.py",
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

    mapping = manifest_lib.load_python_variable_via_runpy(str(source_file), "MAPPING")

    assert mapping["toy"] == ("root/images", None, "test")


def test_infer_eval_metadata_source_for_image_tasks():
    metadata_repo, metadata_subset, metadata_split = manifest_lib.infer_eval_metadata_source(
        dataset_name="ImageNet-1K",
        dataset_parser="image_cls",
        hf_value=("", "", ""),
    )

    assert metadata_repo == manifest_lib.MMEB_TEST_INSTRUCT_REPO
    assert metadata_subset == "ImageNet-1K"
    assert metadata_split == "test"


def test_augment_train_source_for_mmeb_adds_download_patterns():
    source = manifest_lib.augment_train_source(
        modality="image",
        source_name="VOC2007",
        source_config={
            "dataset_parser": "mmeb",
            "dataset_name": "TIGER-Lab/MMEB-train",
            "subset_name": "VOC2007",
            "dataset_split": "original",
        },
    )

    assert source["metadata_hf_repo"] == manifest_lib.MMEB_TRAIN_REPO
    assert source["download_patterns"] == ["VOC2007/original-*", "images_zip/VOC2007.zip"]
    assert source["image_root_candidates"] == ["images", "image"]


def test_select_entries_respects_filters_and_byte_cap():
    entries = [
        {"path": "a.jsonl", "size": 10},
        {"path": "b.parquet", "size": 20},
        {"path": "images_zip/VOC2007.zip", "size": 30},
    ]

    selected = hf_tool.select_entries(
        entries,
        include_patterns=["*.jsonl", "*.zip"],
        exclude_patterns=["images_zip/*"],
        max_files=None,
        max_bytes=100,
    )

    assert [entry["path"] for entry in selected] == ["a.jsonl"]


def test_parse_next_link_extracts_cursor_url():
    headers = (
        "HTTP/2 200\n"
        "link: <https://huggingface.co/api/datasets/demo/tree/main?cursor=abc>; rel=\"next\"\n"
        "content-type: application/json\n"
    )

    next_url = hf_tool.parse_next_link(headers)

    assert next_url == "https://huggingface.co/api/datasets/demo/tree/main?cursor=abc"


def test_load_any_dataset_supports_nested_local_directory(tmp_path):
    shard_dir = tmp_path / "repo" / "data"
    shard_dir.mkdir(parents=True)
    (shard_dir / "train.jsonl").write_text(
        json.dumps(
            {
                "qry": "q",
                "qry_image_path": "images/example.png",
                "pos_text": "yes",
                "pos_image_path": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = convert_train.load_any_dataset(str(tmp_path / "repo"), subset=None, split="train", cache_dir=str(tmp_path / "cache"))

    assert len(dataset) == 1


def test_load_any_dataset_rejects_git_lfs_pointers(tmp_path):
    pointer_dir = tmp_path / "repo"
    pointer_dir.mkdir()
    (pointer_dir / "train.parquet").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:deadbeef\n"
        "size 123\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Git LFS pointers"):
        convert_train.load_any_dataset(str(pointer_dir), subset=None, split="train", cache_dir=str(tmp_path / "cache"))


def test_prepare_train_source_resolves_local_mmeb_subset(tmp_path):
    raw_root = tmp_path / "raw"
    subset_root = raw_root / "vlm2vec_train" / "MMEB-train" / "HatefulMemes"
    subset_root.mkdir(parents=True)
    (subset_root / "original.jsonl").write_text(
        json.dumps(
            {
                "qry": "classify meme",
                "qry_image_path": "images/HatefulMemes/example.jpg",
                "pos_text": "Yes",
                "pos_image_path": "",
                "neg_text": "No",
                "neg_image_path": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    image_root = raw_root / "vlm2vec_train" / "MMEB-train" / "image"
    image_root.mkdir(parents=True)

    entry = {
        "modality": "image",
        "source_name": "HatefulMemes",
        "dataset_parser": "mmeb",
        "dataset_name": "TIGER-Lab/MMEB-train",
        "subset_name": "HatefulMemes",
        "dataset_split": "original",
        "image_dir": "vlm2vec_train/MMEB-train/image",
    }

    resolved = prepare_train_tool.resolve_train_source(entry, raw_root)

    assert resolved["input"] == str(subset_root)
    assert resolved["split"] == "original"
    assert resolved["image_root"] == str(image_root)
    assert resolved["is_local"] is True


def test_prepare_public_data_augment_source_for_legacy_manifest():
    manifest = {
        "train": {
            "image": [
                {
                    "source_name": "VOC2007",
                    "dataset_parser": "mmeb",
                    "dataset_name": "TIGER-Lab/MMEB-train",
                    "subset_name": "VOC2007",
                    "dataset_split": "original",
                }
            ]
        }
    }

    selected = prepare_public_tool.select_train_sources(
        manifest,
        modalities=["image"],
        source_names=[],
    )

    assert selected[0]["metadata_hf_repo"] == manifest_lib.MMEB_TRAIN_REPO
    assert selected[0]["download_patterns"] == ["VOC2007/original-*", "images_zip/VOC2007.zip"]


def test_check_idle_gpus_parses_csv_dump():
    gpus = gpu_tool.parse_gpu_lines(
        [
            "index, memory.used, utilization.gpu",
            "0, 15 MiB, 0 %",
            "1, 2048 MiB, 35 %",
        ]
    )

    assert gpus == [
        {"index": 0, "memory_used": 15, "utilization": 0},
        {"index": 1, "memory_used": 2048, "utilization": 35},
    ]
