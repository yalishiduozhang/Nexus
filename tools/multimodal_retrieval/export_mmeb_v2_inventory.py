#!/usr/bin/env python3
"""Export an MMEB v2 inventory markdown report from the local VLM2Vec references."""

import argparse
import ast
import os
import runpy
from typing import Any, Dict


def require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "The inventory exporter requires `PyYAML`. Create an isolated environment and install Nexus dependencies there."
        ) from exc
    return yaml


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm2vec-root", required=True, help="Path to the local VLM2Vec repository.")
    parser.add_argument("--output", required=True, help="Output markdown path.")
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
    namespace = runpy.run_path(path, init_globals={"__name__": "__nexus_inventory_loader__"})
    if variable_name not in namespace:
        raise ValueError(f"Variable `{variable_name}` not found in {path}")
    return namespace[variable_name]


def format_mapping_table(mapping: Dict[str, Any], value_headers):
    lines = []
    header = "| Dataset | " + " | ".join(value_headers) + " |"
    separator = "| --- | " + " | ".join(["---"] * len(value_headers)) + " |"
    lines.extend([header, separator])
    for dataset_name, values in mapping.items():
        if isinstance(values, dict):
            normalized_values = [str(values.get(column, "")) if values.get(column, "") is not None else "" for column in value_headers]
        else:
            if not isinstance(values, (list, tuple)):
                values = [values]
            normalized_values = [str(value) if value is not None else "" for value in values]
        lines.append("| " + dataset_name + " | " + " | ".join(normalized_values) + " |")
    return lines


def main():
    args = parse_args()
    vlm2vec_root = os.path.abspath(args.vlm2vec_root)

    report_score_path = os.path.join(vlm2vec_root, "experiments", "report_score_v2.py")
    modality_to_datasets = load_python_assignment(report_score_path, "modality2dataset")

    hf_mapping = load_python_assignment(
        os.path.join(vlm2vec_root, "src", "constant", "dataset_hf_path.py"),
        "EVAL_DATASET_HF_PATH",
    )
    local_mapping = load_python_variable_via_runpy(
        os.path.join(vlm2vec_root, "src", "constant", "dataset_hflocal_path.py"),
        "EVAL_DATASET_HF_PATH",
    )

    train_image = load_yaml(os.path.join(vlm2vec_root, "experiments", "public", "train", "train_image.yaml"))
    train_video = load_yaml(os.path.join(vlm2vec_root, "experiments", "public", "train", "train_video.yaml"))
    train_visdoc = load_yaml(os.path.join(vlm2vec_root, "experiments", "public", "train", "train_visdoc.yaml"))

    lines = []
    lines.append("# MMEB v2 Inventory")
    lines.append("")
    lines.append("Generated from local VLM2Vec references.")
    lines.append("")
    lines.append("## Evaluation Coverage")
    lines.append("")
    for modality in ["image", "video", "visdoc"]:
        lines.append(f"### {modality}")
        lines.append("")
        lines.append(f"- Task count: {len(modality_to_datasets.get(modality, []))}")
        lines.append("")
        for dataset_name in modality_to_datasets.get(modality, []):
            hf_values = hf_mapping.get(dataset_name, ("", "", ""))
            local_values = local_mapping.get(dataset_name, ("", "", ""))
            lines.append(
                f"- `{dataset_name}`: HF={hf_values[0]} subset={hf_values[1]} split={hf_values[2]} "
                f"| local={local_values[0]} subset={local_values[1]} split={local_values[2]}"
            )
        lines.append("")

    lines.append("## Public Training Sources")
    lines.append("")
    lines.append("### Image")
    lines.append("")
    lines.extend(format_mapping_table(train_image, ["parser", "dataset_name", "subset_name", "split"]))
    lines.append("")
    lines.append("### Video")
    lines.append("")
    lines.extend(format_mapping_table(train_video, ["parser", "dataset_name", "dataset_path", "data_mode"]))
    lines.append("")
    lines.append("### Visdoc")
    lines.append("")
    lines.extend(format_mapping_table(train_visdoc, ["parser", "dataset_name", "dataset_path", "weight"]))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Image public training uses `TIGER-Lab/MMEB-train` subsets.")
    lines.append("- Video public training uses `ShareGPTVideo/train_video_and_instruction` style sources.")
    lines.append("- Visdoc public training uses `vidore/colpali_train_set` and `openbmb/VisRAG-Ret-Train-In-domain-data`.")
    lines.append("- The scoring split used for the 78-task aggregate is defined in `experiments/report_score_v2.py`.")
    lines.append("")

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")

    print(f"Wrote inventory markdown to {args.output}")


if __name__ == "__main__":
    main()
