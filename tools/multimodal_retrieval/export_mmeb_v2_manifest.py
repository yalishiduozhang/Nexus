#!/usr/bin/env python3
"""Export a machine-readable MMEB v2 manifest from local VLM2Vec references."""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from vlm2vec_manifest_lib import build_eval_manifest_entries, build_train_manifest, load_yaml


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm2vec-root", required=True, help="Path to the local VLM2Vec repository.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    return parser.parse_args()


def main():
    args = parse_args()
    vlm2vec_root = os.path.abspath(args.vlm2vec_root)

    manifest = {
        "source": "local_vlm2vec",
        "vlm2vec_root": vlm2vec_root,
        "eval": build_eval_manifest_entries(vlm2vec_root),
        "train": build_train_manifest(
            {
                "image": load_yaml(os.path.join(vlm2vec_root, "experiments", "public", "train", "train_image.yaml")),
                "video": load_yaml(os.path.join(vlm2vec_root, "experiments", "public", "train", "train_video.yaml")),
                "visdoc": load_yaml(
                    os.path.join(vlm2vec_root, "experiments", "public", "train", "train_visdoc.yaml")
                ),
                "alltasks": load_yaml(
                    os.path.join(vlm2vec_root, "experiments", "public", "train", "train_alltasks.yaml")
                ),
            }
        ),
    }

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2, ensure_ascii=False)

    print(f"Wrote manifest JSON to {args.output}")


if __name__ == "__main__":
    main()
