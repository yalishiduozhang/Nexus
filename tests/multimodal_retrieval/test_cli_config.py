import json
import inspect
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Nexus.evaluation.multimodal_retrieval.__main__ import parse_cli_args as parse_eval_cli_args
from Nexus.training.embedder.multimodal_retrieval.__main__ import parse_cli_args as parse_train_cli_args


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_train_cli_accepts_split_json_configs(tmp_path):
    model_config = tmp_path / "model.json"
    data_config = tmp_path / "data.json"
    training_config = tmp_path / "training.json"

    write_json(
        model_config,
        {
            "model_name_or_path": "Qwen/Qwen2-VL-2B-Instruct",
            "processor_name_or_path": "Qwen/Qwen2-VL-2B-Instruct",
        },
    )
    write_json(
        data_config,
        {
            "train_data": ["examples/multimodal_retrieval/data/train.jsonl"],
        },
    )
    write_json(
        training_config,
        {
            "output_dir": str(tmp_path / "output"),
            "overwrite_output_dir": True,
        },
    )

    model_args, data_args, training_args = parse_train_cli_args(
        [
            "--model_config",
            str(model_config),
            "--data_config",
            str(data_config),
            "--training_config",
            str(training_config),
        ]
    )

    assert model_args.model_name_or_path == "Qwen/Qwen2-VL-2B-Instruct"
    assert data_args.train_data == ["examples/multimodal_retrieval/data/train.jsonl"]
    assert training_args.output_dir == str(tmp_path / "output")


def test_eval_cli_accepts_json_configs_without_onnx(tmp_path):
    eval_config = tmp_path / "eval.json"
    model_config = tmp_path / "model.json"

    write_json(
        eval_config,
        {
            "eval_name": "toy_eval",
            "dataset_dir": "examples/multimodal_retrieval/data/eval",
        },
    )
    write_json(
        model_config,
        {
            "embedder_name_or_path": "Qwen/Qwen2-VL-2B-Instruct",
        },
    )

    eval_args, model_args = parse_eval_cli_args(
        [
            "--eval_config",
            str(eval_config),
            "--model_config",
            str(model_config),
        ]
    )

    assert eval_args.eval_name == "toy_eval"
    assert model_args.embedder_name_or_path == "Qwen/Qwen2-VL-2B-Instruct"


def test_training_import_patches_accelerate_unwrap_model_signature():
    from accelerate import Accelerator

    signature = inspect.signature(Accelerator.unwrap_model)

    assert "keep_torch_compile" in signature.parameters
