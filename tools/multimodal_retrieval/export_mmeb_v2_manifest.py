#!/usr/bin/env python3
"""Export a machine-readable MMEB v2 manifest from local VLM2Vec references."""

import argparse
import ast
import json
import os
import runpy
from typing import Any, Dict


def require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "The manifest exporter requires `PyYAML`. Create an isolated environment and install Nexus dependencies there."
        ) from exc
    return yaml


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm2vec-root", required=True, help="Path to the local VLM2Vec repository.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    yaml = require_yaml()
    with open(path, "r", encoding="utf-8") as input_file:
        return yaml.safe_load(input_file)


def load_python_assignment(path: str, variable_name: str):
    with open(path, "r", encoding="utf-8") as input_file:
        tree = ast.parse(input_file.read(), filename=path)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    return ast.literal_eval(node.value)
    raise ValueError(f"Variable `{variable_name}` not found in {path}")


def load_python_variable_via_runpy(path: str, variable_name: str):
    namespace = runpy.run_path(path, init_globals={"__name__": "__nexus_manifest_loader__"})
    if variable_name not in namespace:
        raise ValueError(f"Variable `{variable_name}` not found in {path}")
    return namespace[variable_name]


def normalize_tuple(value):
    if not isinstance(value, (tuple, list)):
        return [value]
    return list(value)


def main():
    args = parse_args()
    vlm2vec_root = os.path.abspath(args.vlm2vec_root)

    modality_to_datasets = load_python_assignment(
        os.path.join(vlm2vec_root, "experiments", "report_score_v2.py"),
        "modality2dataset",
    )
    hf_mapping = load_python_assignment(
        os.path.join(vlm2vec_root, "src", "constant", "dataset_hf_path.py"),
        "EVAL_DATASET_HF_PATH",
    )
    local_mapping = load_python_variable_via_runpy(
        os.path.join(vlm2vec_root, "src", "constant", "dataset_hflocal_path.py"),
        "EVAL_DATASET_HF_PATH",
    )

    train_configs = {
        "image": load_yaml(os.path.join(vlm2vec_root, "experiments", "public", "train", "train_image.yaml")),
        "video": load_yaml(os.path.join(vlm2vec_root, "experiments", "public", "train", "train_video.yaml")),
        "visdoc": load_yaml(os.path.join(vlm2vec_root, "experiments", "public", "train", "train_visdoc.yaml")),
        "alltasks": load_yaml(os.path.join(vlm2vec_root, "experiments", "public", "train", "train_alltasks.yaml")),
    }

    manifest = {
        "source": "local_vlm2vec",
        "vlm2vec_root": vlm2vec_root,
        "eval": [],
        "train": {},
    }

    for modality, dataset_names in modality_to_datasets.items():
        for dataset_name in dataset_names:
            hf_value = normalize_tuple(hf_mapping.get(dataset_name, ("", "", "")))
            local_value = normalize_tuple(local_mapping.get(dataset_name, ("", "", "")))
            while len(hf_value) < 3:
                hf_value.append(None)
            while len(local_value) < 3:
                local_value.append(None)

            manifest["eval"].append(
                {
                    "modality": modality,
                    "dataset_name": dataset_name,
                    "hf_repo": hf_value[0],
                    "hf_subset": hf_value[1],
                    "hf_split": hf_value[2],
                    "local_repo": local_value[0],
                    "local_subset": local_value[1],
                    "local_split": local_value[2],
                }
            )

    for modality, config in train_configs.items():
        manifest["train"][modality] = []
        for source_name, source_config in config.items():
            row = {"source_name": source_name}
            row.update(source_config)
            manifest["train"][modality].append(row)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2, ensure_ascii=False)

    print(f"Wrote manifest JSON to {args.output}")


if __name__ == "__main__":
    main()
