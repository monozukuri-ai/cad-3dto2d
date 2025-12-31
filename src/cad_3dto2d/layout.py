from __future__ import annotations

from typing import Iterable

from build123d import Axis, Compound
from pydantic import BaseModel, ConfigDict

from .types import BoundingBox2D, Point2D, Point3D, Shape
from .views import ViewProjection, bounding_size


class LayeredShapes(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    visible: list[Shape]
    hidden: list[Shape]


class ThreeViewLayout(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    front: LayeredShapes
    side_x: LayeredShapes
    side_y: LayeredShapes
    combined: LayeredShapes


def _transform(
    shapes: Iterable[Shape],
    rotate_deg: float = 0.0,
    scale: float = 1.0,
    translate: Point3D = (0.0, 0.0, 0.0),
) -> list[Shape]:
    translated = []
    for shape in shapes:
        if rotate_deg:
            shape = shape.rotate(Axis.Z, rotate_deg)
        if scale != 1.0:
            shape = shape.scale(scale)
        if translate != (0.0, 0.0, 0.0):
            shape = shape.translate(translate)
        translated.append(shape)
    return translated


def _transform_layered(
    layered: LayeredShapes,
    rotate_deg: float = 0.0,
    scale: float = 1.0,
    translate: Point3D = (0.0, 0.0, 0.0),
) -> LayeredShapes:
    return LayeredShapes(
        visible=_transform(layered.visible, rotate_deg=rotate_deg, scale=scale, translate=translate),
        hidden=_transform(layered.hidden, rotate_deg=rotate_deg, scale=scale, translate=translate),
    )


def _layered_bounds(layered: LayeredShapes):
    shapes = layered.visible + layered.hidden
    if not shapes:
        return None
    return Compound(children=shapes).bounding_box()


def _combine_views(front: LayeredShapes, side_x: LayeredShapes, side_y: LayeredShapes) -> LayeredShapes:
    return LayeredShapes(
        visible=front.visible + side_x.visible + side_y.visible,
        hidden=front.hidden + side_x.hidden + side_y.hidden,
    )


def fit_three_view_layout(
    layout: ThreeViewLayout,
    frame_bbox_mm: BoundingBox2D,
    paper_size_mm: Point2D | None,
    frame_margin_ratio: float = 1.0,
    scale: float | None = None,
) -> ThreeViewLayout:
    bounds = _layered_bounds(layout.combined)
    if bounds is None:
        return layout

    frame_min_x, frame_min_y, frame_max_x, frame_max_y = frame_bbox_mm
    frame_width = frame_max_x - frame_min_x
    frame_height = frame_max_y - frame_min_y
    if frame_width <= 0 or frame_height <= 0:
        return layout

    size = bounds.size
    if size.X <= 0 or size.Y <= 0:
        return layout

    fit_scale = min(frame_width / size.X, frame_height / size.Y) * frame_margin_ratio
    if scale:
        fit_scale *= scale

    center = bounds.center()
    translate_to_origin = (-center.X, -center.Y, -center.Z)

    frame_center_x = (frame_min_x + frame_max_x) / 2
    frame_center_y = (frame_min_y + frame_max_y) / 2
    if paper_size_mm:
        target_center = (
            frame_center_x - paper_size_mm[0] / 2,
            frame_center_y - paper_size_mm[1] / 2,
            0.0,
        )
    else:
        target_center = (0.0, 0.0, 0.0)

    def apply(layered: LayeredShapes) -> LayeredShapes:
        updated = _transform_layered(layered, translate=translate_to_origin)
        if fit_scale != 1.0:
            updated = _transform_layered(updated, scale=fit_scale)
        if target_center != (0.0, 0.0, 0.0):
            updated = _transform_layered(updated, translate=target_center)
        return updated

    front = apply(layout.front)
    side_x = apply(layout.side_x)
    side_y = apply(layout.side_y)
    combined = _combine_views(front, side_x, side_y)
    return ThreeViewLayout(front=front, side_x=side_x, side_y=side_y, combined=combined)


def layout_three_views(
    front: ViewProjection,
    side_x: ViewProjection,
    side_y: ViewProjection,
    margin_ratio: float = 1.1,
    frame_bbox_mm: BoundingBox2D | None = None,
    paper_size_mm: Point2D | None = None,
    frame_margin_ratio: float = 1.0,
    scale: float | None = None,
) -> ThreeViewLayout:
    front_size = front.bounding_size()
    front_size_with_margin = (front_size[0] * margin_ratio, front_size[1] * margin_ratio)

    front_layer = LayeredShapes(visible=list(front.visible), hidden=list(front.hidden))
    side_x_layer = LayeredShapes(
        visible=_transform(side_x.visible, rotate_deg=90, translate=(front_size_with_margin[0], 0.0, 0.0)),
        hidden=_transform(side_x.hidden, rotate_deg=90, translate=(front_size_with_margin[0], 0.0, 0.0)),
    )
    side_y_layer = LayeredShapes(
        visible=_transform(side_y.visible, translate=(0.0, -front_size_with_margin[1], 0.0)),
        hidden=_transform(side_y.hidden, translate=(0.0, -front_size_with_margin[1], 0.0)),
    )

    combined = _combine_views(front_layer, side_x_layer, side_y_layer)
    three_view_size = bounding_size(combined.visible + combined.hidden)
    center_shift = (-(three_view_size[0] - front_size[0]) / 2, (three_view_size[1] - front_size[1]) / 2, 0.0)
    front_layer = _transform_layered(front_layer, translate=center_shift)
    side_x_layer = _transform_layered(side_x_layer, translate=center_shift)
    side_y_layer = _transform_layered(side_y_layer, translate=center_shift)
    combined = _combine_views(front_layer, side_x_layer, side_y_layer)

    layout = ThreeViewLayout(front=front_layer, side_x=side_x_layer, side_y=side_y_layer, combined=combined)
    if frame_bbox_mm:
        return fit_three_view_layout(
            layout,
            frame_bbox_mm=frame_bbox_mm,
            paper_size_mm=paper_size_mm,
            frame_margin_ratio=frame_margin_ratio,
            scale=scale,
        )
    if scale:
        front_layer = _transform_layered(front_layer, scale=scale)
        side_x_layer = _transform_layered(side_x_layer, scale=scale)
        side_y_layer = _transform_layered(side_y_layer, scale=scale)
        combined = _combine_views(front_layer, side_x_layer, side_y_layer)
        return ThreeViewLayout(front=front_layer, side_x=side_x_layer, side_y=side_y_layer, combined=combined)
    return layout
