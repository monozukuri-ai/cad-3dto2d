from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from .types import BoundingBox2D, Point2D


class TemplateBaseSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    kind: Literal["svg", "parametric"]
    paper_size_mm: Point2D | None = None
    frame_bbox_mm: BoundingBox2D | None = None
    title_block_bbox_mm: BoundingBox2D | None = None
    reserved_bbox_mm: list[BoundingBox2D] = Field(default_factory=list)
    default_scale: float = 1.0


class SvgTemplateSpec(TemplateBaseSpec):
    kind: Literal["svg"] = "svg"
    file: str
    file_path: str


class MarginSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    left: float = 0.0
    right: float = 0.0
    top: float = 0.0
    bottom: float = 0.0

    @classmethod
    def from_value(cls, value: Any) -> "MarginSpec":
        if isinstance(value, MarginSpec):
            return value
        if isinstance(value, (int, float)):
            return cls(left=float(value), right=float(value), top=float(value), bottom=float(value))
        if isinstance(value, dict):
            return cls(**value)
        raise ValueError("margin_mm must be a number or mapping")


class FrameSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    margin_mm: MarginSpec

    @field_validator("margin_mm", mode="before")
    @classmethod
    def _coerce_margin(cls, value: Any) -> MarginSpec:
        return MarginSpec.from_value(value)


class TitleBlockGridSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    rows_mm: list[float]
    cols_mm: list[float]

    @field_validator("rows_mm", "cols_mm")
    @classmethod
    def _validate_sizes(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("grid rows/cols must not be empty")
        for entry in value:
            if entry <= 0:
                raise ValueError("grid rows/cols must be positive")
        return value


class TitleBlockCellSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str | None = None
    cell: tuple[int, int]
    span: tuple[int, int] = (1, 1)
    text: str | None = None
    key: str | None = None
    default: str | None = None
    prefix: str = ""
    suffix: str = ""
    align: Literal["left", "center", "right"] = "left"
    valign: Literal["top", "middle", "bottom"] = "middle"
    text_height_mm: float | None = None
    offset_mm: Point2D = (0.0, 0.0)
    rotate_deg: float = 0.0

    @field_validator("cell")
    @classmethod
    def _validate_cell(cls, value: tuple[int, int]) -> tuple[int, int]:
        row, col = value
        if row < 0 or col < 0:
            raise ValueError("cell indices must be non-negative")
        return value

    @field_validator("span")
    @classmethod
    def _validate_span(cls, value: tuple[int, int]) -> tuple[int, int]:
        rows, cols = value
        if rows <= 0 or cols <= 0:
            raise ValueError("span values must be positive")
        return value

    @model_validator(mode="after")
    def _validate_text(self) -> "TitleBlockCellSpec":
        if self.text and self.key:
            raise ValueError("title cell must not define both text and key")
        return self


class TitleBlockSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    pos: Literal["top_right", "bottom_right", "top_left", "bottom_left"] = "bottom_right"
    size_mm: Point2D
    grid: TitleBlockGridSpec | None = None
    cells: list[TitleBlockCellSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_grid_cells(self) -> "TitleBlockSpec":
        if self.cells and not self.grid:
            raise ValueError("cells require title_block.grid definition")
        if not self.grid:
            return self
        rows = self.grid.rows_mm
        cols = self.grid.cols_mm
        width, height = self.size_mm
        if sum(cols) > width + 1e-3:
            raise ValueError("grid column widths exceed title block width")
        if sum(rows) > height + 1e-3:
            raise ValueError("grid row heights exceed title block height")
        max_row = len(rows)
        max_col = len(cols)
        for cell in self.cells:
            row, col = cell.cell
            span_rows, span_cols = cell.span
            if row + span_rows > max_row or col + span_cols > max_col:
                raise ValueError("cell span exceeds grid size")
        return self


class ParametricTemplateSpec(TemplateBaseSpec):
    kind: Literal["parametric"] = "parametric"
    frame: FrameSpec
    title_block: TitleBlockSpec | None = None

    @model_validator(mode="after")
    def _populate_bounds(self) -> "ParametricTemplateSpec":
        if not self.paper_size_mm:
            raise ValueError("paper_size_mm is required for parametric templates")
        frame_bbox = self.frame_bbox_mm or _frame_bbox_from_margin(self.paper_size_mm, self.frame.margin_mm)
        title_bbox = self.title_block_bbox_mm
        if self.title_block and not title_bbox:
            title_bbox = _title_block_bbox_from_spec(frame_bbox, self.title_block)
        _validate_bbox(frame_bbox, "frame_bbox_mm")
        _validate_bbox_inside(frame_bbox, (0.0, 0.0, self.paper_size_mm[0], self.paper_size_mm[1]), "frame_bbox_mm")
        if title_bbox:
            _validate_bbox(title_bbox, "title_block_bbox_mm")
            _validate_bbox_inside(title_bbox, frame_bbox, "title_block_bbox_mm")
        return self.model_copy(
            update={
                "frame_bbox_mm": frame_bbox,
                "title_block_bbox_mm": title_bbox,
            }
        )


TemplateSpec = Annotated[SvgTemplateSpec | ParametricTemplateSpec, Field(discriminator="kind")]
_TEMPLATE_ADAPTER = TypeAdapter(TemplateSpec)


def _frame_bbox_from_margin(paper_size_mm: Point2D, margin: MarginSpec) -> BoundingBox2D:
    paper_w, paper_h = paper_size_mm
    min_x = margin.left
    min_y = margin.bottom
    max_x = paper_w - margin.right
    max_y = paper_h - margin.top
    return (min_x, min_y, max_x, max_y)


def _title_block_bbox_from_spec(frame_bbox: BoundingBox2D, title_block: TitleBlockSpec) -> BoundingBox2D:
    min_x, min_y, max_x, max_y = frame_bbox
    width, height = title_block.size_mm
    if title_block.pos == "bottom_right":
        return (max_x - width, min_y, max_x, min_y + height)
    if title_block.pos == "top_right":
        return (max_x - width, max_y - height, max_x, max_y)
    if title_block.pos == "top_left":
        return (min_x, max_y - height, min_x + width, max_y)
    return (min_x, min_y, min_x + width, min_y + height)


def _validate_bbox(bbox: BoundingBox2D, label: str) -> None:
    min_x, min_y, max_x, max_y = bbox
    if max_x <= min_x or max_y <= min_y:
        raise ValueError(f"{label} must have positive width/height")


def _validate_bbox_inside(inner: BoundingBox2D, outer: BoundingBox2D, label: str) -> None:
    if inner[0] < outer[0] or inner[1] < outer[1] or inner[2] > outer[2] or inner[3] > outer[3]:
        raise ValueError(f"{label} must be inside frame_bbox_mm")


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

    payload = dict(raw)
    payload["name"] = template_name
    payload.setdefault("default_scale", 1.0)
    if not payload.get("kind"):
        payload["kind"] = "svg"
    if payload.get("reserved_bbox_mm") is None:
        payload.pop("reserved_bbox_mm", None)

    if payload["kind"] == "svg":
        file_value = payload.get("file")
        if not file_value:
            raise ValueError(f"Template file is required: {template_name}")
        file_path = Path(file_value)
        if not file_path.is_absolute():
            file_path = templates_dir / file_path
        payload["file"] = str(file_value)
        payload["file_path"] = str(file_path)

    return _TEMPLATE_ADAPTER.validate_python(payload)
