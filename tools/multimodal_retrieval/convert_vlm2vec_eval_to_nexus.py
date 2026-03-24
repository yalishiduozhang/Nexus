#!/usr/bin/env python3
"""Convert VLM2Vec pair-style evaluation data into a Nexus local retrieval dataset."""

import argparse
import base64
import json
import os
from typing import Any, Dict, Iterable, List, Optional


def require_datasets():
    try:
        from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "The conversion script requires `datasets`. Create an isolated environment and install Nexus dependencies there."
        ) from exc

    return Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="HF dataset name, local dataset path, or a json/jsonl/parquet file.")
    parser.add_argument("--output-dir", required=True, help="Output directory containing corpus/queries/qrels jsonl files.")
    parser.add_argument("--subset", default=None, help="Optional HF subset name.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument("--dataset-name", default=None, help="Optional dataset name stored in metadata.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap.")
    parser.add_argument("--media-root", default=None, help="Default media root written to `dataset_meta.json`.")
    parser.add_argument("--image-root", default=None, help="Image root written to `dataset_meta.json`.")
    parser.add_argument("--video-root", default=None, help="Video root written to `dataset_meta.json`.")
    parser.add_argument(
        "--sequence-mode",
        default="auto",
        choices=["auto", "image", "video"],
        help="How to interpret multi-frame candidate/query media.",
    )
    parser.add_argument("--query-prefix", default="q", help="Prefix used for generated query ids.")
    return parser.parse_args()


def load_any_dataset(path_or_name: str, subset: Optional[str], split: str):
    Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, load_from_disk = require_datasets()

    if os.path.isdir(path_or_name) and os.path.exists(os.path.join(path_or_name, "dataset_info.json")):
        dataset = load_from_disk(path_or_name)
    elif os.path.isfile(path_or_name):
        extension = os.path.splitext(path_or_name)[1].lower()
        if extension in [".json", ".jsonl"]:
            dataset = load_dataset("json", data_files=path_or_name)
        elif extension == ".parquet":
            dataset = load_dataset("parquet", data_files=path_or_name)
        else:
            raise ValueError(f"Unsupported local file type: {extension}")
    else:
        dataset = load_dataset(path_or_name, subset)

    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        if split not in dataset:
            available_splits = list(dataset.keys())
            raise ValueError(f"Split `{split}` is not available. Found: {available_splits}")
        dataset = dataset[split]
    elif subset is not None and not isinstance(dataset, (Dataset, IterableDataset)):
        raise ValueError(f"Unsupported dataset object returned from {path_or_name}: {type(dataset)}")

    return dataset


def iter_rows(dataset, max_rows: Optional[int] = None):
    for index, row in enumerate(dataset):
        if max_rows is not None and index >= max_rows:
            break
        yield index, row


def maybe_b64_encode(value: Any) -> Optional[str]:
    if value in [None, b"", ""]:
        return None
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("utf-8")
    return None


def serialize_entry_from_path_or_bytes(path: Any, image_bytes: Any) -> Optional[Any]:
    if path not in [None, ""] and image_bytes in [None, b""]:
        return path
    if path in [None, ""] and image_bytes in [None, b""]:
        return None

    payload = {}
    if path not in [None, ""]:
        payload["path"] = path
    b64_value = maybe_b64_encode(image_bytes)
    if b64_value is not None:
        payload["b64"] = b64_value
    return payload or None


def extract_media_entries(media_struct: Any) -> List[Any]:
    if media_struct in [None, "", []]:
        return []

    if isinstance(media_struct, list):
        merged_entries = []
        for element in media_struct:
            merged_entries.extend(extract_media_entries(element))
        return merged_entries

    if isinstance(media_struct, str):
        return [media_struct]

    if isinstance(media_struct, dict):
        if "frames" in media_struct and media_struct["frames"] not in [None, "", []]:
            merged_entries = []
            for frame in media_struct["frames"]:
                if isinstance(frame, str):
                    merged_entries.append(frame)
                elif isinstance(frame, dict):
                    payload = {}
                    if frame.get("path") not in [None, ""]:
                        payload["path"] = frame["path"]
                    b64_value = maybe_b64_encode(frame.get("bytes"))
                    if b64_value is not None:
                        payload["b64"] = b64_value
                    if payload:
                        merged_entries.append(payload)
            return merged_entries

        paths = media_struct.get("paths") or []
        bytes_values = media_struct.get("bytes") or []
        if not isinstance(paths, list):
            paths = [paths]
        if not isinstance(bytes_values, list):
            bytes_values = [bytes_values]

        count = max(len(paths), len(bytes_values))
        entries = []
        for index in range(count):
            entry = serialize_entry_from_path_or_bytes(
                paths[index] if index < len(paths) else None,
                bytes_values[index] if index < len(bytes_values) else None,
            )
            if entry is not None:
                entries.append(entry)
        return entries

    pil_image_path = getattr(media_struct, "filename", None)
    if pil_image_path not in [None, ""]:
        return [pil_image_path]
    return []


def resolve_sequence_mode(args, row: Optional[Dict[str, Any]] = None) -> str:
    if args.sequence_mode != "auto":
        return args.sequence_mode

    hints = [args.dataset_name or ""]
    if isinstance(row, dict):
        dataset_infos = row.get("dataset_infos") or {}
        hints.extend(
            [
                dataset_infos.get("dataset_name", ""),
                dataset_infos.get("global_dataset_name", ""),
            ]
        )

    lowered = " ".join(hints).lower()
    if any(token in lowered for token in ["video", "msrvtt", "msvd", "youcook", "vatex", "charades", "moment"]):
        return "video"
    return "image"


def attach_media(item: Dict[str, Any], media_entries: List[Any], sequence_mode: str):
    valid_entries = [entry for entry in media_entries if entry not in [None, "", [], {}]]
    if len(valid_entries) == 0:
        return

    if sequence_mode == "video":
        if all(isinstance(entry, str) for entry in valid_entries):
            item["video_frames"] = valid_entries
        else:
            item["videos"] = [{"frames": valid_entries}]
        return

    if len(valid_entries) == 1:
        entry = valid_entries[0]
        if isinstance(entry, str):
            item["image_path"] = entry
        else:
            item["image"] = entry
        return

    item["images"] = valid_entries


def first_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(text for text in value if isinstance(text, str) and text.strip())
    return value if isinstance(value, str) else ""


def build_eval_item(text_value: Any, media_value: Any, sequence_mode: str) -> Dict[str, Any]:
    item = {}
    text = first_text_value(text_value)
    if text not in [None, ""]:
        item["text"] = text
    attach_media(item, extract_media_entries(media_value), sequence_mode=sequence_mode)
    return item


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    dataset = load_any_dataset(args.input, subset=args.subset, split=args.split)

    os.makedirs(args.output_dir, exist_ok=True)
    corpus: Dict[str, Dict[str, Any]] = {}
    queries: List[Dict[str, Any]] = []
    qrels: List[Dict[str, Any]] = []

    for row_index, row in iter_rows(dataset, max_rows=args.max_rows):
        sequence_mode = resolve_sequence_mode(args, row=row)
        query_item = build_eval_item(row.get("query_text"), row.get("query_image"), sequence_mode=sequence_mode)
        query_id = f"{args.query_prefix}{row_index}"
        query_record = {"_id": query_id}
        query_record.update(query_item)
        queries.append(query_record)

        dataset_infos = row.get("dataset_infos") or {}
        cand_names = dataset_infos.get("cand_names") or []
        cand_texts = row.get("cand_text") or []
        cand_images = row.get("cand_image") or []

        if not isinstance(cand_names, list):
            cand_names = [cand_names]
        if not isinstance(cand_texts, list):
            cand_texts = [cand_texts]
        if not isinstance(cand_images, list):
            cand_images = [cand_images]

        for index, cand_name in enumerate(cand_names):
            if cand_name in [None, ""]:
                continue
            if cand_name not in corpus:
                corpus_item = build_eval_item(
                    cand_texts[index] if index < len(cand_texts) else None,
                    cand_images[index] if index < len(cand_images) else None,
                    sequence_mode=sequence_mode,
                )
                corpus_record = {"_id": cand_name}
                corpus_record.update(corpus_item)
                corpus[cand_name] = corpus_record

        label_names = dataset_infos.get("label_name") or []
        if isinstance(label_names, str):
            label_names = [label_names]
        for label_name in label_names:
            if label_name in [None, ""]:
                continue
            qrels.append({"query-id": query_id, "corpus-id": label_name, "score": 1})

    split_name = args.split
    write_jsonl(os.path.join(args.output_dir, "corpus.jsonl"), corpus.values())
    write_jsonl(os.path.join(args.output_dir, f"{split_name}_queries.jsonl"), queries)
    write_jsonl(os.path.join(args.output_dir, f"{split_name}_qrels.jsonl"), qrels)

    metadata = {
        "dataset_name": args.dataset_name,
        "source_input": args.input,
        "subset": args.subset,
        "split": args.split,
        "media_root": args.media_root,
        "image_root": args.image_root,
        "video_root": args.video_root,
        "num_queries": len(queries),
        "num_corpus": len(corpus),
        "num_qrels": len(qrels),
    }
    with open(os.path.join(args.output_dir, "dataset_meta.json"), "w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2, ensure_ascii=False)

    print(
        f"Wrote {len(queries)} queries, {len(corpus)} corpus rows, and {len(qrels)} qrels to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
