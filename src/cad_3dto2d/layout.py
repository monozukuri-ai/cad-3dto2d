from __future__ import annotations

from typing import Iterable, Literal

from build123d import Axis, Compound
from pydantic import BaseModel, ConfigDict

from .types import BoundingBox2D, Point2D, Point3D, Shape
from .views import ViewProjection


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
        visible=_transform(
            layered.visible, rotate_deg=rotate_deg, scale=scale, translate=translate
        ),
        hidden=_transform(
            layered.hidden, rotate_deg=rotate_deg, scale=scale, translate=translate
        ),
    )


def _layered_bounds(layered: LayeredShapes):
    shapes = layered.visible + layered.hidden
    if not shapes:
        return None
    return Compound(children=shapes).bounding_box()


def _layered_bbox(layered: LayeredShapes) -> BoundingBox2D | None:
    bounds = _layered_bounds(layered)
    if not bounds:
        return None
    return (bounds.min.X, bounds.min.Y, bounds.max.X, bounds.max.Y)


def _bbox_center(bbox: BoundingBox2D) -> Point2D:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _shift_bbox(bbox: BoundingBox2D, dx: float, dy: float) -> BoundingBox2D:
    return (bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy)


def _merge_bboxes(bboxes: Iterable[BoundingBox2D | None]) -> BoundingBox2D | None:
    valid = [bbox for bbox in bboxes if bbox is not None]
    if not valid:
        return None
    return (
        min(bbox[0] for bbox in valid),
        min(bbox[1] for bbox in valid),
        max(bbox[2] for bbox in valid),
        max(bbox[3] for bbox in valid),
    )


def _side_offset(
    front_bbox: BoundingBox2D,
    side_bbox: BoundingBox2D,
    gap_x_mm: float,
    side_position: str,
) -> float:
    if side_position == "right":
        return front_bbox[2] - side_bbox[0] + gap_x_mm
    return front_bbox[0] - side_bbox[2] - gap_x_mm


def _top_offset(
    front_bbox: BoundingBox2D,
    top_bbox: BoundingBox2D,
    gap_y_mm: float,
    top_position: str,
) -> float:
    if top_position == "up":
        return front_bbox[3] - top_bbox[1] + gap_y_mm
    return front_bbox[1] - top_bbox[3] - gap_y_mm


def _combine_views(
    front: LayeredShapes, side_x: LayeredShapes, side_y: LayeredShapes
) -> LayeredShapes:
    return LayeredShapes(
        visible=front.visible + side_x.visible + side_y.visible,
        hidden=front.hidden + side_x.hidden + side_y.hidden,
    )


def _apply_layout_offset(layout: ThreeViewLayout, offset: Point2D) -> ThreeViewLayout:
    if offset == (0.0, 0.0):
        return layout
    translate = (offset[0], offset[1], 0.0)
    front = _transform_layered(layout.front, translate=translate)
    side_x = _transform_layered(layout.side_x, translate=translate)
    side_y = _transform_layered(layout.side_y, translate=translate)
    combined = _combine_views(front, side_x, side_y)
    return ThreeViewLayout(front=front, side_x=side_x, side_y=side_y, combined=combined)


def _scale_layered_around_center(layered: LayeredShapes, scale: float) -> LayeredShapes:
    if scale == 1.0:
        return layered
    bounds = _layered_bounds(layered)
    if bounds is None:
        return layered
    center = bounds.center()
    updated = _transform_layered(layered, translate=(-center.X, -center.Y, -center.Z))
    updated = _transform_layered(updated, scale=scale)
    return _transform_layered(updated, translate=(center.X, center.Y, center.Z))


def align_three_view_layout(
    layout: ThreeViewLayout,
    frame_bbox_mm: BoundingBox2D,
    paper_size_mm: Point2D | None,
) -> ThreeViewLayout:
    bounds = _layered_bounds(layout.combined)
    if bounds is None:
        return layout

    center = bounds.center()
    translate_to_origin = (-center.X, -center.Y, -center.Z)

    frame_min_x, frame_min_y, frame_max_x, frame_max_y = frame_bbox_mm
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
    gap_x_mm: float = 30.0,
    gap_y_mm: float = 30.0,
    side_position: Literal["left", "right"] = "right",
    top_position: Literal["up", "down"] = "down",
    layout_offset_x: float = 0.0,
    layout_offset_y: float = 0.0,
    frame_bbox_mm: BoundingBox2D | None = None,
    paper_size_mm: Point2D | None = None,
    scale: float | None = None,
) -> ThreeViewLayout:
    front_layer = LayeredShapes(visible=list(front.visible), hidden=list(front.hidden))
    side_x_layer = LayeredShapes(
        visible=_transform(side_x.visible, rotate_deg=90),
        hidden=_transform(side_x.hidden, rotate_deg=90),
    )
    side_y_layer = LayeredShapes(
        visible=list(side_y.visible),
        hidden=list(side_y.hidden),
    )
    if scale and scale != 1.0:
        front_layer = _scale_layered_around_center(front_layer, scale)
        side_x_layer = _scale_layered_around_center(side_x_layer, scale)
        side_y_layer = _scale_layered_around_center(side_y_layer, scale)

    front_bbox = _layered_bbox(front_layer)
    side_x_bbox = _layered_bbox(side_x_layer)
    side_y_bbox = _layered_bbox(side_y_layer)
    side_dx = 0.0
    top_dy = 0.0
    if front_bbox and side_x_bbox:
        side_dx = _side_offset(front_bbox, side_x_bbox, gap_x_mm, side_position)
    if front_bbox and side_y_bbox:
        top_dy = _top_offset(front_bbox, side_y_bbox, gap_y_mm, top_position)

    combined_bbox = _merge_bboxes(
        [
            front_bbox,
            _shift_bbox(side_x_bbox, side_dx, 0.0) if side_x_bbox else None,
            _shift_bbox(side_y_bbox, 0.0, top_dy) if side_y_bbox else None,
        ]
    )
    center_shift_x = 0.0
    center_shift_y = 0.0
    if combined_bbox and front_bbox:
        front_center = _bbox_center(front_bbox)
        combined_center = _bbox_center(combined_bbox)
        center_shift_x = front_center[0] - combined_center[0]
        center_shift_y = front_center[1] - combined_center[1]

    align_shift_x = 0.0
    align_shift_y = 0.0
    if frame_bbox_mm and combined_bbox:
        centered_combined = _shift_bbox(combined_bbox, center_shift_x, center_shift_y)
        combined_center = _bbox_center(centered_combined)
        frame_min_x, frame_min_y, frame_max_x, frame_max_y = frame_bbox_mm
        frame_center_x = (frame_min_x + frame_max_x) / 2
        frame_center_y = (frame_min_y + frame_max_y) / 2
        if paper_size_mm:
            target_center_x = frame_center_x - paper_size_mm[0] / 2
            target_center_y = frame_center_y - paper_size_mm[1] / 2
        else:
            target_center_x = 0.0
            target_center_y = 0.0
        align_shift_x = target_center_x - combined_center[0]
        align_shift_y = target_center_y - combined_center[1]

    common_x = center_shift_x + align_shift_x + layout_offset_x
    common_y = center_shift_y + align_shift_y + layout_offset_y
    front_translate = (common_x, common_y, 0.0)
    side_x_translate = (common_x + side_dx, common_y, 0.0)
    side_y_translate = (common_x, common_y + top_dy, 0.0)

    if front_translate != (0.0, 0.0, 0.0):
        front_layer = _transform_layered(front_layer, translate=front_translate)
    if side_x_translate != (0.0, 0.0, 0.0):
        side_x_layer = _transform_layered(side_x_layer, translate=side_x_translate)
    if side_y_translate != (0.0, 0.0, 0.0):
        side_y_layer = _transform_layered(side_y_layer, translate=side_y_translate)

    combined = _combine_views(front_layer, side_x_layer, side_y_layer)
    layout = ThreeViewLayout(
        front=front_layer, side_x=side_x_layer, side_y=side_y_layer, combined=combined
    )
    return layout
