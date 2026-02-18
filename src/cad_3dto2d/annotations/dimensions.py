from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..types import Point2D


class DimensionSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    offset: float
    extension_gap: float
    extension_offset: float | None = None
    extension_extension: float | None = None
    arrow_size: float
    arrow_block: str | None = None
    arrow_block1: str | None = None
    arrow_block2: str | None = None
    text_style: str | None = None
    text_font: str | None = None
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


class DimensionText(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    text: str
    height: float
    anchor: str = "middle"


class LinearDimensionSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    p1: Point2D
    p2: Point2D
    base: Point2D
    angle: float
    label: str | None
    settings: DimensionSettings


class DiameterDimensionSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    center: Point2D
    radius: float
    angle: float
    label: str | None
    settings: DimensionSettings


class LeaderNoteSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    target: Point2D  # 矢先（狙う点）
    angle: float  # leader の向き（deg）
    text: str
    settings: DimensionSettings


DimensionOrientation = Literal["horizontal", "vertical"]
DimensionSide = Literal["top", "bottom", "left", "right"]


def format_length(value: float, decimal_places: int) -> str:
    if decimal_places <= 0:
        return str(int(round(value)))
    text = f"{value:.{decimal_places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text
