from __future__ import annotations

import math
from typing import Iterable, Literal

from build123d import Compound, Line
from pydantic import BaseModel, ConfigDict

from ..types import Point2D, Point3D, Shape


class DimensionSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    offset: float
    extension_gap: float
    arrow_size: float
    text_height: float = 3.5
    text_gap: float = 1.0
    decimal_places: int = 1
    diameter_symbol: str = "D"
    pitch_prefix: str = "P"

    @classmethod
    def default(cls, size_x: float, size_y: float) -> DimensionSettings:
        size_ref = max(size_x, size_y, 1.0)
        return cls(
            offset=max(5.0, size_ref * 0.1),
            extension_gap=max(0.5, size_ref * 0.02),
            arrow_size=max(2.0, size_ref * 0.03),
            text_height=max(2.5, size_ref * 0.04),
            text_gap=max(1.0, size_ref * 0.02),
        )


def _maybe_line(p1: Point3D, p2: Point3D) -> Line | None:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    if (dx * dx + dy * dy + dz * dz) <= 1e-12:
        return None
    return Line(p1, p2)


def default_settings_from_size(size_x: float, size_y: float) -> DimensionSettings:
    return DimensionSettings.default(size_x, size_y)


class DimensionText(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    text: str
    height: float
    anchor: str = "middle"


class DimensionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    lines: list[Line]
    texts: list[DimensionText]

DimensionOrientation = Literal["horizontal", "vertical"]
DimensionSide = Literal["top", "bottom", "left", "right"]


def format_length(value: float, decimal_places: int) -> str:
    if decimal_places <= 0:
        return str(int(round(value)))
    text = f"{value:.{decimal_places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _arrow_head_horizontal(x: float, y: float, arrow_size: float, direction: float) -> list[Line]:
    half = arrow_size / 2
    return [
        Line((x, y, 0), (x + direction * arrow_size, y + half, 0)),
        Line((x, y, 0), (x + direction * arrow_size, y - half, 0)),
    ]


def _arrow_head_vertical(x: float, y: float, arrow_size: float, direction: float) -> list[Line]:
    half = arrow_size / 2
    return [
        Line((x, y, 0), (x + half, y + direction * arrow_size, 0)),
        Line((x, y, 0), (x - half, y + direction * arrow_size, 0)),
    ]


def _arrow_head_angle(
    x: float,
    y: float,
    direction: Point2D,
    arrow_size: float,
) -> list[Line]:
    dx, dy = direction
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return []
    dx /= length
    dy /= length
    base_x = x + dx * arrow_size
    base_y = y + dy * arrow_size
    perp_x, perp_y = -dy, dx
    half = arrow_size / 2
    return [
        Line((x, y, 0), (base_x + perp_x * half, base_y + perp_y * half, 0)),
        Line((x, y, 0), (base_x - perp_x * half, base_y - perp_y * half, 0)),
    ]


def generate_diameter_dimension(
    center: Point2D,
    radius: float,
    leader_angle_deg: float,
    settings: DimensionSettings | None = None,
    leader_length: float | None = None,
    label: str | None = None,
) -> DimensionResult:
    settings = settings or DimensionSettings.default(radius * 2, radius * 2)
    angle_rad = math.radians(leader_angle_deg)
    dir_x = math.cos(angle_rad)
    dir_y = math.sin(angle_rad)
    arrow_tip = (center[0] + dir_x * radius, center[1] + dir_y * radius)
    if leader_length is None:
        leader_length = settings.arrow_size * 4 + settings.text_gap * 2 + settings.text_height
    leader_end = (arrow_tip[0] + dir_x * leader_length, arrow_tip[1] + dir_y * leader_length)
    lines: list[Line] = []
    leader_line = _maybe_line((arrow_tip[0], arrow_tip[1], 0), (leader_end[0], leader_end[1], 0))
    if leader_line:
        lines.append(leader_line)
    lines.extend(_arrow_head_angle(arrow_tip[0], arrow_tip[1], (dir_x, dir_y), settings.arrow_size))

    diameter_label = label or f"{settings.diameter_symbol}{format_length(radius * 2, settings.decimal_places)}"
    text_pos = (
        leader_end[0] + dir_x * settings.text_gap,
        leader_end[1] + dir_y * settings.text_gap,
    )
    anchor = "start" if dir_x >= 0 else "end"
    texts = [
        DimensionText(
            x=text_pos[0],
            y=text_pos[1],
            text=diameter_label,
            height=settings.text_height,
            anchor=anchor,
        )
    ]
    return DimensionResult(lines=lines, texts=texts)


def _horizontal_dimension_from_points(
    p1: Point2D,
    p2: Point2D,
    side: DimensionSide,
    offset: float,
    settings: DimensionSettings,
) -> list[Line]:
    (x1, y1), (x2, y2) = p1, p2
    if x2 < x1:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    if side not in ("top", "bottom"):
        raise ValueError(f"Invalid horizontal side: {side}")
    sign = 1.0 if side == "top" else -1.0
    base_y = max(y1, y2) if side == "top" else min(y1, y2)
    dim_y = base_y + sign * abs(offset)
    lines: list[Line] = []
    for x, y in ((x1, y1), (x2, y2)):
        gap = min(settings.extension_gap, abs(dim_y - y))
        ext_start = y + sign * gap
        line = _maybe_line((x, ext_start, 0), (x, dim_y, 0))
        if line:
            lines.append(line)
    dim_line = _maybe_line((x1, dim_y, 0), (x2, dim_y, 0))
    if dim_line:
        lines.append(dim_line)
    lines.extend(_arrow_head_horizontal(x1, dim_y, settings.arrow_size, direction=1.0))
    lines.extend(_arrow_head_horizontal(x2, dim_y, settings.arrow_size, direction=-1.0))
    return lines


def _vertical_dimension_from_points(
    p1: Point2D,
    p2: Point2D,
    side: DimensionSide,
    offset: float,
    settings: DimensionSettings,
) -> list[Line]:
    (x1, y1), (x2, y2) = p1, p2
    if y2 < y1:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    if side not in ("left", "right"):
        raise ValueError(f"Invalid vertical side: {side}")
    sign = 1.0 if side == "right" else -1.0
    base_x = max(x1, x2) if side == "right" else min(x1, x2)
    dim_x = base_x + sign * abs(offset)
    lines: list[Line] = []
    for x, y in ((x1, y1), (x2, y2)):
        gap = min(settings.extension_gap, abs(dim_x - x))
        ext_start = x + sign * gap
        line = _maybe_line((ext_start, y, 0), (dim_x, y, 0))
        if line:
            lines.append(line)
    dim_line = _maybe_line((dim_x, y1, 0), (dim_x, y2, 0))
    if dim_line:
        lines.append(dim_line)
    lines.extend(_arrow_head_vertical(dim_x, y1, settings.arrow_size, direction=1.0))
    lines.extend(_arrow_head_vertical(dim_x, y2, settings.arrow_size, direction=-1.0))
    return lines


def generate_linear_dimension(
    p1: Point2D,
    p2: Point2D,
    orientation: DimensionOrientation | None = None,
    side: DimensionSide | None = None,
    offset: float | None = None,
    settings: DimensionSettings | None = None,
    label: str | None = None,
) -> DimensionResult:
    settings = settings or DimensionSettings.default(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    if orientation is None:
        orientation = "horizontal" if dx >= dy else "vertical"
    if orientation == "horizontal":
        side = side or "top"
        offset = abs(offset) if offset is not None else settings.offset
        lines = _horizontal_dimension_from_points(p1, p2, side, offset, settings)
        sign = 1 if side == "top" else -1
        dim_y = (max(p1[1], p2[1]) if side == "top" else min(p1[1], p2[1])) + sign * offset
        text = label or format_length(dx, settings.decimal_places)
        texts = [
            DimensionText(
                x=(p1[0] + p2[0]) / 2,
                y=dim_y + sign * settings.text_gap,
                text=text,
                height=settings.text_height,
                anchor="middle",
            )
        ]
        return DimensionResult(lines=lines, texts=texts)
    if orientation == "vertical":
        side = side or "right"
        offset = abs(offset) if offset is not None else settings.offset
        lines = _vertical_dimension_from_points(p1, p2, side, offset, settings)
        sign = 1 if side == "right" else -1
        dim_x = (max(p1[0], p2[0]) if side == "right" else min(p1[0], p2[0])) + sign * offset
        text = label or format_length(dy, settings.decimal_places)
        texts = [
            DimensionText(
                x=dim_x + sign * settings.text_gap,
                y=(p1[1] + p2[1]) / 2,
                text=text,
                height=settings.text_height,
                anchor="start" if side == "right" else "end",
            )
        ]
        return DimensionResult(lines=lines, texts=texts)
    raise ValueError(f"Unsupported orientation: {orientation}")


def generate_basic_dimensions(
    shapes: Iterable[Shape],
    settings: DimensionSettings | None = None,
    horizontal_dir: int = 1,
    vertical_dir: int = 1,
    horizontal_offset: float | None = None,
    vertical_offset: float | None = None,
) -> DimensionResult:
    shapes = list(shapes)
    if not shapes:
        return DimensionResult(lines=[], texts=[])
    bounds = Compound(children=shapes).bounding_box()
    size = bounds.size
    settings = settings or DimensionSettings.default(size.X, size.Y)
    xmin, ymin = bounds.min.X, bounds.min.Y
    xmax, ymax = bounds.max.X, bounds.max.Y
    horizontal_side: DimensionSide = "top" if horizontal_dir >= 0 else "bottom"
    vertical_side: DimensionSide = "right" if vertical_dir >= 0 else "left"
    if horizontal_offset is not None and horizontal_offset < 0:
        horizontal_side = "bottom"
        horizontal_offset = abs(horizontal_offset)
    if vertical_offset is not None and vertical_offset < 0:
        vertical_side = "left"
        vertical_offset = abs(vertical_offset)

    horizontal_offset = horizontal_offset if horizontal_offset is not None else settings.offset
    vertical_offset = vertical_offset if vertical_offset is not None else settings.offset

    horizontal = generate_linear_dimension(
        p1=(xmin, ymax if horizontal_side == "top" else ymin),
        p2=(xmax, ymax if horizontal_side == "top" else ymin),
        orientation="horizontal",
        side=horizontal_side,
        offset=horizontal_offset,
        settings=settings,
    )
    vertical = generate_linear_dimension(
        p1=(xmax if vertical_side == "right" else xmin, ymin),
        p2=(xmax if vertical_side == "right" else xmin, ymax),
        orientation="vertical",
        side=vertical_side,
        offset=vertical_offset,
        settings=settings,
    )
    return DimensionResult(lines=horizontal.lines + vertical.lines, texts=horizontal.texts + vertical.texts)
