import base64
import io
import os
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor


DEFAULT_MULTIMODAL_MODEL_TYPE = "auto"
CHAT_TEMPLATE_MODEL_TYPES = {"qwen2_vl", "qwen2_5_vl", "llava_next"}
CONDITIONAL_GENERATION_MODEL_TYPES = {
    "qwen2_vl": "Qwen2VLForConditionalGeneration",
    "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
    "llava_next": "LlavaNextForConditionalGeneration",
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff")


def resolve_torch_dtype(dtype_name: Optional[str]):
    if dtype_name is None or dtype_name == "auto":
        return None
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def infer_multimodal_model_type(config) -> str:
    if config is None:
        return DEFAULT_MULTIMODAL_MODEL_TYPE
    return getattr(config, "model_type", DEFAULT_MULTIMODAL_MODEL_TYPE) or DEFAULT_MULTIMODAL_MODEL_TYPE


def load_multimodal_processor(
    model_name_or_path: str,
    processor_name_or_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    token: Optional[str] = None,
):
    processor_path = processor_name_or_path or model_name_or_path
    return AutoProcessor.from_pretrained(
        processor_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        token=token,
    )


def _maybe_load_registered_conditional_generation_model(
    config,
    model_name_or_path: str,
    load_kwargs: Dict[str, Any],
):
    model_type = infer_multimodal_model_type(config)
    class_name = CONDITIONAL_GENERATION_MODEL_TYPES.get(model_type)
    if class_name is None:
        return None

    import transformers

    model_cls = getattr(transformers, class_name, None)
    if model_cls is None:
        return None
    return model_cls.from_pretrained(model_name_or_path, **load_kwargs)


def load_multimodal_backbone(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    token: Optional[str] = None,
    model_type: str = DEFAULT_MULTIMODAL_MODEL_TYPE,
    torch_dtype_name: Optional[str] = None,
    attn_implementation: Optional[str] = None,
):
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    if model_type not in [None, "", DEFAULT_MULTIMODAL_MODEL_TYPE]:
        setattr(config, "model_type", model_type)

    load_kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
        "token": token,
        "config": config,
    }
    torch_dtype = resolve_torch_dtype(torch_dtype_name)
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation

    model = _maybe_load_registered_conditional_generation_model(
        config=config,
        model_name_or_path=model_name_or_path,
        load_kwargs=load_kwargs,
    )
    if model is not None:
        return model, config

    try:
        model = AutoModel.from_pretrained(model_name_or_path, **load_kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    return model, config


def _clone_image(image):
    return image.copy() if hasattr(image, "copy") else image


def resolve_media_path(path: str, base_dir: Optional[str] = None) -> str:
    if path is None:
        return None
    if os.path.isabs(path) or base_dir is None:
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def looks_like_image_reference(path: Optional[str]) -> bool:
    if path in [None, ""]:
        return False
    lowered = path.lower().split("?", 1)[0]
    return lowered.endswith(IMAGE_SUFFIXES)


def _coerce_image_specs(image_value: Any, base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    if image_value in [None, "", []]:
        return []

    if not isinstance(image_value, list):
        image_value = [image_value]

    normalized_specs = []
    for image_spec in image_value:
        if image_spec in [None, ""]:
            continue
        if isinstance(image_spec, str):
            normalized_specs.append({"path": resolve_media_path(image_spec, base_dir=base_dir)})
        elif isinstance(image_spec, bytes):
            normalized_specs.append({"bytes": image_spec})
        elif isinstance(image_spec, dict):
            spec = deepcopy(image_spec)
            if "path" in spec and spec["path"] is not None:
                spec["path"] = resolve_media_path(spec["path"], base_dir=base_dir)
            if "paths" in spec and spec["paths"] is not None:
                paths = []
                for path in spec["paths"]:
                    if path in [None, ""]:
                        paths.append(path)
                    else:
                        paths.append(resolve_media_path(path, base_dir=base_dir))
                spec["paths"] = paths
            normalized_specs.append(spec)
        else:
            normalized_specs.append({"pil": image_spec})
    return normalized_specs


def normalize_multimodal_item(item: Any, base_dir: Optional[str] = None) -> Dict[str, Any]:
    if item is None:
        return {"text": "", "images": [], "metadata": {}}
    if isinstance(item, str):
        if looks_like_image_reference(item):
            return {"text": "", "images": _coerce_image_specs(item, base_dir=base_dir), "metadata": {}}
        return {"text": item, "images": [], "metadata": {}}
    if not isinstance(item, dict):
        raise TypeError(f"Unsupported multimodal item type: {type(item)}")

    metadata = deepcopy(item.get("metadata", {}))
    title = item.get("title", "")
    text = item.get("text")
    if text is None:
        text = item.get("query")
    if text is None:
        text = item.get("caption")
    if text is None:
        text = item.get("body", "")
    text = text or ""
    if title:
        text = f"{title} {text}".strip()

    image_value = None
    for key in ["images", "image_paths", "image_path", "image", "pages", "video_frames"]:
        if key in item and item[key] not in [None, "", []]:
            image_value = item[key]
            break

    return {
        "text": text,
        "images": _coerce_image_specs(image_value, base_dir=base_dir),
        "metadata": metadata,
    }


def normalize_multimodal_group(items: Any, base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    if items in [None, "", []]:
        return []
    if isinstance(items, list):
        return [normalize_multimodal_item(item, base_dir=base_dir) for item in items if item not in [None, ""]]
    return [normalize_multimodal_item(items, base_dir=base_dir)]


def build_prefixed_multimodal_group(row: Dict[str, Any], prefix: str, base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    if prefix in row:
        return normalize_multimodal_group(row[prefix], base_dir=base_dir)

    text_value = row.get(f"{prefix}_text")
    title_value = row.get(f"{prefix}_title")
    multi_image_value = None
    for key in [f"{prefix}_images", f"{prefix}_image_paths", f"{prefix}_pages", f"{prefix}_video_frames"]:
        if key in row and row[key] not in [None, "", []]:
            multi_image_value = row[key]
            break

    single_image_value = row.get(f"{prefix}_image_path", row.get(f"{prefix}_image"))
    if multi_image_value is not None:
        return [normalize_multimodal_item({"title": title_value, "text": text_value, "images": multi_image_value}, base_dir=base_dir)]

    if isinstance(text_value, list) or isinstance(single_image_value, list):
        if isinstance(text_value, list):
            texts = text_value
        else:
            texts = [text_value for _ in range(len(single_image_value))]

        if isinstance(single_image_value, list):
            images = single_image_value
        else:
            images = [single_image_value for _ in range(len(texts))]

        group = []
        for text, image in zip(texts, images):
            group.append(
                normalize_multimodal_item(
                    {"title": title_value, "text": text, "image_path": image},
                    base_dir=base_dir,
                )
            )
        return group

    if text_value not in [None, ""] or title_value not in [None, ""] or single_image_value not in [None, ""]:
        return [
            normalize_multimodal_item(
                {"title": title_value, "text": text_value, "image_path": single_image_value},
                base_dir=base_dir,
            )
        ]
    return []


def apply_instruction(text: str, instruction: Optional[str], instruction_format: str) -> str:
    if instruction is None:
        return text
    return instruction_format.format(instruction, text)


def is_empty_multimodal_item(item: Dict[str, Any]) -> bool:
    return (item.get("text", "").strip() == "") and len(item.get("images", [])) == 0


def _require_pillow():
    try:
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise ImportError("Pillow is required for multimodal retrieval. Please install `Pillow`.") from exc


def load_image_from_spec(image_spec: Dict[str, Any]):
    _require_pillow()
    from PIL import Image

    if image_spec is None:
        return None
    if "pil" in image_spec and image_spec["pil"] is not None:
        return _clone_image(image_spec["pil"]).convert("RGB")
    if "bytes" in image_spec and image_spec["bytes"] not in [None, b""]:
        return Image.open(io.BytesIO(image_spec["bytes"])).convert("RGB")
    if "b64" in image_spec and image_spec["b64"] not in [None, ""]:
        return Image.open(io.BytesIO(base64.b64decode(image_spec["b64"]))).convert("RGB")
    if "path" in image_spec and image_spec["path"] not in [None, ""]:
        return Image.open(image_spec["path"]).convert("RGB")
    raise ValueError(f"Unsupported image specification: {image_spec}")


def materialize_images(item: Dict[str, Any]) -> List[Any]:
    images = []
    for image_spec in item.get("images", []):
        if "paths" in image_spec:
            paths = image_spec.get("paths") or []
            bytes_values = image_spec.get("bytes") or [None] * len(paths)
            for path, bytes_value in zip(paths, bytes_values):
                if path in [None, ""] and bytes_value in [None, b""]:
                    continue
                images.append(load_image_from_spec({"path": path, "bytes": bytes_value}))
            continue
        images.append(load_image_from_spec(image_spec))
    return images


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


class MultimodalProcessorAdapter:
    def __init__(
        self,
        processor,
        model_type: str = DEFAULT_MULTIMODAL_MODEL_TYPE,
        use_chat_template: bool = True,
    ):
        self.processor = processor
        self.model_type = model_type
        self.use_chat_template = use_chat_template

    def _build_chat_text(self, item: Dict[str, Any], images: Sequence[Any]) -> str:
        contents = [{"type": "image"} for _ in images]
        if item.get("text", "").strip() or len(contents) == 0:
            contents.append({"type": "text", "text": item.get("text", "") or " "})
        messages = [{"role": "user", "content": contents}]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def _encode_single(
        self,
        item: Dict[str, Any],
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        images = materialize_images(item)
        use_chat_template = (
            self.use_chat_template
            and self.model_type in CHAT_TEMPLATE_MODEL_TYPES
            and hasattr(self.processor, "apply_chat_template")
        )

        if use_chat_template:
            text = self._build_chat_text(item=item, images=images)
        else:
            text = item.get("text", "") or " "

        call_kwargs = {
            "text": text,
            "padding": False,
            "truncation": max_length is not None,
            "return_tensors": None,
        }
        if max_length is not None:
            call_kwargs["max_length"] = max_length
        if len(images) > 0:
            call_kwargs["images"] = images
        return self.processor(**call_kwargs)

    def encode_batch(
        self,
        items: Iterable[Dict[str, Any]],
        max_length: Optional[int] = None,
        instruction: Optional[str] = None,
        instruction_format: str = "{}{}",
    ) -> Dict[str, Any]:
        normalized_items = []
        for item in items:
            normalized_item = normalize_multimodal_item(item)
            normalized_item["text"] = apply_instruction(
                normalized_item.get("text", ""),
                instruction=instruction,
                instruction_format=instruction_format,
            )
            normalized_items.append(normalized_item)

        encoded_items = [self._encode_single(item, max_length=max_length) for item in normalized_items]

        pad_fn = getattr(self.processor, "pad", None)
        if callable(pad_fn):
            return pad_fn(encoded_items, padding=True, return_tensors="pt")

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "Current processor does not provide a `pad` method or a tokenizer. "
                "Please use a processor with padding support."
            )
        return tokenizer.pad(encoded_items, padding=True, return_tensors="pt")
