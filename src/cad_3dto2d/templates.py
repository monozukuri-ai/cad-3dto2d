from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from .types import BoundingBox2D, Point2D


class TemplateSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    file: str
    file_path: str
    paper_size_mm: Point2D | None = None
    frame_bbox_mm: BoundingBox2D | None = None
    title_block_bbox_mm: BoundingBox2D | None = None
    default_scale: float = 1.0


def _templates_dir(templates_dir: Path | None = None) -> Path:
    return Path(templates_dir) if templates_dir else Path(__file__).parent / "templates"


@lru_cache
def _load_index(index_path: str) -> dict[str, Any]:
    data = yaml.safe_load(Path(index_path).read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid template index format: {index_path}")
    return data


def load_template(template_name: str, templates_dir: Path | None = None) -> TemplateSpec:
    templates_dir = _templates_dir(templates_dir)
    index_path = templates_dir / "index.yaml"
    data = _load_index(str(index_path))
    raw = data.get(template_name)
    if raw is None:
        raise KeyError(f"Template not found: {template_name}")
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid template entry for: {template_name}")

    file_value = raw.get("file")
    if not file_value:
        raise ValueError(f"Template file is required: {template_name}")
    file_path = Path(file_value)
    if not file_path.is_absolute():
        file_path = templates_dir / file_path

    payload = dict(raw)
    payload["name"] = template_name
    payload["file"] = str(file_value)
    payload["file_path"] = str(file_path)
    payload.setdefault("default_scale", 1.0)
    return TemplateSpec.model_validate(payload)
