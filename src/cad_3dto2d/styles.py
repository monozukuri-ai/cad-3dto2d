from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from build123d import LineType
from pydantic import BaseModel, ConfigDict

_LINE_TYPE_MAP: dict[str, LineType] = {
    "CONTINUOUS": LineType.CONTINUOUS,
    "DASHED": LineType.DASHED,
    "DOT": LineType.DOT,
    "DOTTED": LineType.DOT,
    "CHAIN": LineType.CENTER,
    "CENTER": LineType.CENTER,
}


class StyleConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    name: str
    line_types: dict[str, str] = {}
    dimension: dict[str, object] | None = None

    def resolve_line_types(self) -> dict[str, LineType]:
        resolved: dict[str, LineType] = {}
        for layer, value in self.line_types.items():
            key = str(value).upper()
            resolved[layer] = _LINE_TYPE_MAP.get(key, LineType.CONTINUOUS)
        return resolved


def _styles_dir(styles_dir: Path | None = None) -> Path:
    return Path(styles_dir) if styles_dir else Path(__file__).parent / "styles"


@lru_cache
def _load_style_file(style_path: str) -> dict[str, Any]:
    data = yaml.safe_load(Path(style_path).read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid style format: {style_path}")
    return data


def load_style(style_name: str, styles_dir: Path | None = None) -> StyleConfig:
    styles_dir = _styles_dir(styles_dir)
    style_path = styles_dir / f"{style_name}.yaml"
    if not style_path.exists():
        raise FileNotFoundError(f"Style not found: {style_name}")
    raw = _load_style_file(str(style_path))
    payload = dict(raw)
    payload["name"] = style_name
    return StyleConfig.model_validate(payload)
