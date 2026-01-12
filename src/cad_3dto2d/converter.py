import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Literal, NamedTuple

import ezdxf
from build123d import Compound, ShapeList, import_step, import_svg
from pydantic import BaseModel, ConfigDict

from .annotations.dimensions import (
    DimensionSettings,
    DimensionSide,
    DimensionText,
    DiameterDimensionSpec,
    LinearDimensionSpec,
    format_length,
)
from .annotations.features import FeatureCoordinates, extract_feature_coordinates, extract_primitives
from .annotations.planner import (
    PlannedDiameterDimension,
    PlannedDimension,
    PlanningRules,
    apply_planning_rules,
    group_circles_by_radius,
    plan_hole_dimensions,
    plan_internal_dimensions,
)
from .rendering.dxf_backend import (
    add_ezdxf_dimensions,
    apply_template_to_doc,
    export_dxf_layers,
    render_dxf_to_svg,
)
from .rendering.svg_tools import rasterize_svg
from .layout import LayeredShapes, layout_three_views
from .styles import load_style
from .templates import SvgTemplateSpec, TemplateSpec, load_template
from .types import BoundingBox2D, Point2D, Shape
from .views import project_three_views

RASTER_IMAGE_TYPES = {"png", "jpg", "jpeg"}
_TEXT_WIDTH_FACTOR = 0.6


class ViewDimensionConfig(BaseModel):
    """Configuration for dimension generation on a single view."""

    model_config = ConfigDict(frozen=True)

    horizontal_dir: Literal[1, -1]
    vertical_dir: Literal[1, -1]

    @property
    def horizontal_side(self) -> DimensionSide:
        return "top" if self.horizontal_dir >= 0 else "bottom"

    @property
    def vertical_side(self) -> DimensionSide:
        return "right" if self.vertical_dir >= 0 else "left"


class DimensionPlanOutput(NamedTuple):
    """Planned dimension entities."""

    linear: list[LinearDimensionSpec]
    diameter: list[DiameterDimensionSpec]


def _load_svg_template(template_spec: SvgTemplateSpec) -> ShapeList[Shape]:
    return import_svg(template_spec.file_path)


def _centered_bbox(template_spec: TemplateSpec | None, bbox: BoundingBox2D | None) -> BoundingBox2D | None:
    if not template_spec or not bbox:
        return None
    min_x, min_y, max_x, max_y = bbox
    if template_spec.paper_size_mm:
        paper_w, paper_h = template_spec.paper_size_mm
        return (
            min_x - paper_w / 2,
            min_y - paper_h / 2,
            max_x - paper_w / 2,
            max_y - paper_h / 2,
        )
    return (min_x, min_y, max_x, max_y)


def _centered_frame_bounds(template_spec: TemplateSpec | None) -> BoundingBox2D | None:
    if not template_spec:
        return None
    return _centered_bbox(template_spec, template_spec.frame_bbox_mm)


def _centered_title_block_bounds(template_spec: TemplateSpec | None) -> BoundingBox2D | None:
    if not template_spec:
        return None
    return _centered_bbox(template_spec, template_spec.title_block_bbox_mm)


def _centered_reserved_bounds(template_spec: TemplateSpec | None) -> list[BoundingBox2D]:
    if not template_spec or not template_spec.reserved_bbox_mm:
        return []
    centered: list[BoundingBox2D] = []
    for bbox in template_spec.reserved_bbox_mm:
        centered_bbox = _centered_bbox(template_spec, bbox)
        if centered_bbox:
            centered.append(centered_bbox)
    return centered


def _layered_bbox_2d(layered: LayeredShapes) -> BoundingBox2D | None:
    shapes = layered.visible + layered.hidden
    if not shapes:
        return None
    bounds = Compound(children=shapes).bounding_box()
    return (bounds.min.X, bounds.min.Y, bounds.max.X, bounds.max.Y)


def _clamp_offset(
    base: float,
    direction: int,
    frame_min: float,
    frame_max: float,
    offset: float,
    padding: float = 0.0,
) -> float:
    if direction >= 0:
        available = frame_max - base - padding
        return min(offset, max(0.0, available))
    available = base - frame_min - padding
    return -min(offset, max(0.0, available))


def _estimate_text_width(text: str, height: float, width_factor: float = _TEXT_WIDTH_FACTOR) -> float:
    if not text:
        return 0.0
    return max(height * 0.4, len(text) * height * width_factor)


def _text_bounds(text: DimensionText, width_factor: float = _TEXT_WIDTH_FACTOR) -> BoundingBox2D:
    width = _estimate_text_width(text.text, text.height, width_factor=width_factor)
    if text.anchor == "start":
        min_x = text.x
        max_x = text.x + width
    elif text.anchor == "end":
        min_x = text.x - width
        max_x = text.x
    else:
        half = width / 2
        min_x = text.x - half
        max_x = text.x + half
    half_h = text.height / 2
    min_y = text.y - half_h
    max_y = text.y + half_h
    return (min_x, min_y, max_x, max_y)


def _bbox_intersects(a: BoundingBox2D, b: BoundingBox2D) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _bbox_within_frame(bbox: BoundingBox2D, frame_bounds: BoundingBox2D, padding: float) -> bool:
    frame_min_x, frame_min_y, frame_max_x, frame_max_y = frame_bounds
    return (
        bbox[0] >= frame_min_x + padding
        and bbox[2] <= frame_max_x - padding
        and bbox[1] >= frame_min_y + padding
        and bbox[3] <= frame_max_y - padding
    )


def _resolve_dimension_settings(
    shapes: list[Shape],
    dimension_settings: DimensionSettings | None,
    dimension_overrides: dict[str, object] | None,
) -> DimensionSettings:
    """Resolve dimension settings from shapes size, user settings, and overrides."""
    bounds = Compound(children=shapes).bounding_box()
    size = bounds.size
    settings = dimension_settings or DimensionSettings.default(size.X, size.Y)
    if dimension_settings is None and dimension_overrides:
        settings = DimensionSettings.model_validate(
            {**settings.model_dump(), **dimension_overrides}
        )
    return settings


def _dimension_base_for_plan(
    plan: PlannedDimension,
    side: DimensionSide,
    offset: float,
) -> tuple[Point2D, float]:
    if plan.orientation == "horizontal":
        sign = 1 if side == "top" else -1
        base_y = max(plan.p1[1], plan.p2[1]) if side == "top" else min(plan.p1[1], plan.p2[1])
        dim_y = base_y + sign * offset
        return (plan.p1[0], dim_y), 0.0
    sign = 1 if side == "right" else -1
    base_x = max(plan.p1[0], plan.p2[0]) if side == "right" else min(plan.p1[0], plan.p2[0])
    dim_x = base_x + sign * offset
    return (dim_x, plan.p1[1]), 90.0


def _linear_dimension_spec(
    plan: PlannedDimension,
    side: DimensionSide,
    offset: float,
    label: str | None,
    settings: DimensionSettings,
) -> LinearDimensionSpec:
    base, angle = _dimension_base_for_plan(plan, side, offset)
    return LinearDimensionSpec(
        p1=plan.p1,
        p2=plan.p2,
        base=base,
        angle=angle,
        label=label,
        settings=settings,
    )


def _basic_plan_for_side(
    bounds,
    orientation: Literal["horizontal", "vertical"],
    side: DimensionSide,
) -> PlannedDimension:
    xmin, ymin = bounds.min.X, bounds.min.Y
    xmax, ymax = bounds.max.X, bounds.max.Y
    if orientation == "horizontal":
        y_ref = ymax if side == "top" else ymin
        return PlannedDimension(
            p1=(xmin, y_ref),
            p2=(xmax, y_ref),
            orientation="horizontal",
            side=side,
        )
    x_ref = xmax if side == "right" else xmin
    return PlannedDimension(
        p1=(x_ref, ymin),
        p2=(x_ref, ymax),
        orientation="vertical",
        side=side,
    )


def _basic_offset_for_side(
    bounds,
    orientation: Literal["horizontal", "vertical"],
    side: DimensionSide,
    settings: DimensionSettings,
    frame_bounds: BoundingBox2D | None,
) -> float:
    if not frame_bounds:
        return settings.offset
    frame_min_x, frame_min_y, frame_max_x, frame_max_y = frame_bounds
    padding = settings.text_gap + settings.text_height
    if orientation == "horizontal":
        base = bounds.max.Y if side == "top" else bounds.min.Y
        direction = 1 if side == "top" else -1
        return abs(_clamp_offset(base, direction, frame_min_y, frame_max_y, settings.offset, padding=padding))
    base = bounds.max.X if side == "right" else bounds.min.X
    direction = 1 if side == "right" else -1
    return abs(_clamp_offset(base, direction, frame_min_x, frame_max_x, settings.offset, padding=padding))


def _resolve_basic_dimension_specs(
    shapes: list[Shape],
    settings: DimensionSettings,
    config: ViewDimensionConfig,
    frame_bounds: BoundingBox2D | None,
    avoid_bounds: list[BoundingBox2D] | None,
) -> list[LinearDimensionSpec]:
    bounds = Compound(children=shapes).bounding_box()
    padding = settings.text_gap + settings.text_height * 0.5
    specs: list[LinearDimensionSpec] = []
    for orientation, side in (
        ("horizontal", config.horizontal_side),
        ("vertical", config.vertical_side),
    ):
        primary_plan = _basic_plan_for_side(bounds, orientation, side)
        primary_offset = _basic_offset_for_side(bounds, orientation, side, settings, frame_bounds)
        label = format_length(
            abs(primary_plan.p2[0] - primary_plan.p1[0])
            if orientation == "horizontal"
            else abs(primary_plan.p2[1] - primary_plan.p1[1]),
            settings.decimal_places,
        )
        candidate_sides = [side]
        if frame_bounds or avoid_bounds:
            flipped = "bottom" if side == "top" else "top" if side in ("top", "bottom") else (
                "left" if side == "right" else "right"
            )
            if flipped not in candidate_sides:
                candidate_sides.append(flipped)

        selected_side = side
        selected_offset = primary_offset
        if frame_bounds or avoid_bounds:
            for candidate in candidate_sides:
                candidate_plan = _basic_plan_for_side(bounds, orientation, candidate)
                candidate_offset = _basic_offset_for_side(bounds, orientation, candidate, settings, frame_bounds)
                text = _dimension_text_for_plan(candidate_plan, candidate, candidate_offset, label, settings)
                if _text_fits_bounds(text, frame_bounds, padding=padding, avoid_bounds=avoid_bounds):
                    selected_side = candidate
                    selected_offset = candidate_offset
                    primary_plan = candidate_plan
                    break

        specs.append(_linear_dimension_spec(primary_plan, selected_side, selected_offset, label, settings))
    return specs


def _plan_feature_dimensions(
    features: FeatureCoordinates,
    config: ViewDimensionConfig,
    settings: DimensionSettings,
    rules: PlanningRules,
) -> tuple[list[PlannedDimension], list[PlannedDiameterDimension], dict[str, bool]]:
    """Plan dimensions for internal features and holes."""
    internal_dims = plan_internal_dimensions(
        features,
        horizontal_side=config.horizontal_side,
        vertical_side=config.vertical_side,
    )
    hole_positions, hole_diameters, hole_pitches, pitch_axes = plan_hole_dimensions(
        features,
        horizontal_side=config.horizontal_side,
        vertical_side=config.vertical_side,
    )

    # Convert pitch dimensions to labeled line dimensions
    pitch_line_dims: list[PlannedDimension] = []
    for plan in hole_pitches:
        label = f"{plan.count}x{settings.pitch_prefix}{format_length(plan.pitch, settings.decimal_places)}"
        pitch_line_dims.append(
            PlannedDimension(
                p1=plan.p1,
                p2=plan.p2,
                orientation=plan.orientation,
                side=plan.side,
                label=label,
            )
        )

    line_dims, diameter_dims = apply_planning_rules(
        hole_positions, pitch_line_dims, internal_dims, hole_diameters, rules=rules,
    )

    # Collapse diameter dimensions when pitch patterns exist
    if rules.collapse_diameter_with_pitch and (pitch_axes["horizontal"] or pitch_axes["vertical"]):
        groups = group_circles_by_radius(features.circles)
        collapsed: list[PlannedDiameterDimension] = []
        angle_candidates = [45, 135] if config.horizontal_dir >= 0 else [-45, -135]
        for idx, group in enumerate(groups[: rules.max_diameter_groups]):
            if not group:
                continue
            circle = group[0]
            angle = angle_candidates[idx % len(angle_candidates)]
            label = (
                f"{len(group)}x{settings.diameter_symbol}"
                f"{format_length(circle.radius * 2, settings.decimal_places)}"
            )
            collapsed.append(
                PlannedDiameterDimension(
                    center=circle.center,
                    radius=circle.radius,
                    leader_angle_deg=angle,
                    label=label,
                )
            )
        diameter_dims = collapsed

    return line_dims, diameter_dims, pitch_axes


def _dimension_text_for_plan(
    plan: PlannedDimension,
    side: DimensionSide,
    offset: float,
    label: str,
    settings: DimensionSettings,
) -> DimensionText:
    if plan.orientation == "horizontal":
        sign = 1 if side == "top" else -1
        dim_y = (max(plan.p1[1], plan.p2[1]) if side == "top" else min(plan.p1[1], plan.p2[1])) + sign * offset
        return DimensionText(
            x=(plan.p1[0] + plan.p2[0]) / 2,
            y=dim_y + sign * settings.text_gap,
            text=label,
            height=settings.text_height,
            anchor="middle",
        )
    sign = 1 if side == "right" else -1
    dim_x = (max(plan.p1[0], plan.p2[0]) if side == "right" else min(plan.p1[0], plan.p2[0])) + sign * offset
    return DimensionText(
        x=dim_x + sign * settings.text_gap,
        y=(plan.p1[1] + plan.p2[1]) / 2,
        text=label,
        height=settings.text_height,
        anchor="start" if side == "right" else "end",
    )


def _text_fits_bounds(
    text: DimensionText,
    frame_bounds: BoundingBox2D | None,
    padding: float,
    avoid_bounds: list[BoundingBox2D] | None = None,
) -> bool:
    bbox = _text_bounds(text)
    if frame_bounds and not _bbox_within_frame(bbox, frame_bounds, padding=padding):
        return False
    if avoid_bounds:
        for block in avoid_bounds:
            if _bbox_intersects(bbox, block):
                return False
    return True


def _diameter_text_for_angle(
    center: Point2D,
    radius: float,
    angle_deg: float,
    label: str,
    settings: DimensionSettings,
) -> DimensionText:
    angle_rad = math.radians(angle_deg)
    dir_x = math.cos(angle_rad)
    dir_y = math.sin(angle_rad)
    leader_length = settings.arrow_size * 4 + settings.text_gap * 2 + settings.text_height
    arrow_tip = (center[0] + dir_x * radius, center[1] + dir_y * radius)
    leader_end = (arrow_tip[0] + dir_x * leader_length, arrow_tip[1] + dir_y * leader_length)
    text_pos = (
        leader_end[0] + dir_x * settings.text_gap,
        leader_end[1] + dir_y * settings.text_gap,
    )
    anchor = "start" if dir_x >= 0 else "end"
    return DimensionText(
        x=text_pos[0],
        y=text_pos[1],
        text=label,
        height=settings.text_height,
        anchor=anchor,
    )


def _resolve_line_dimension_specs(
    line_dims: list[PlannedDimension],
    settings: DimensionSettings,
    frame_bounds: BoundingBox2D | None,
    avoid_bounds: list[BoundingBox2D] | None = None,
) -> list[LinearDimensionSpec]:
    """Resolve planned dimensions into DXF-ready specs with frame-aware placement."""
    specs: list[LinearDimensionSpec] = []
    lane_step = settings.text_height + settings.text_gap + settings.arrow_size
    lane_index = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    base_offset = settings.offset + lane_step

    for plan in line_dims:
        if plan.label:
            label = plan.label
        elif plan.orientation == "horizontal":
            label = format_length(abs(plan.p2[0] - plan.p1[0]), settings.decimal_places)
        else:
            label = format_length(abs(plan.p2[1] - plan.p1[1]), settings.decimal_places)

        text_width = _estimate_text_width(label, settings.text_height)
        padding = settings.text_gap + settings.text_height * 0.5

        def resolve_offset(side: DimensionSide) -> float:
            offset_value = base_offset + lane_index[side] * lane_step
            if frame_bounds:
                frame_min_x, frame_min_y, frame_max_x, frame_max_y = frame_bounds
                if plan.orientation == "horizontal":
                    base = max(plan.p1[1], plan.p2[1]) if side == "top" else min(plan.p1[1], plan.p2[1])
                    direction = 1 if side == "top" else -1
                    return abs(
                        _clamp_offset(
                            base,
                            direction,
                            frame_min_y,
                            frame_max_y,
                            offset_value,
                            padding=settings.text_gap + settings.text_height,
                        )
                    )
                base = max(plan.p1[0], plan.p2[0]) if side == "right" else min(plan.p1[0], plan.p2[0])
                direction = 1 if side == "right" else -1
                return abs(
                    _clamp_offset(
                        base,
                        direction,
                        frame_min_x,
                        frame_max_x,
                        offset_value,
                        padding=settings.text_gap + text_width,
                    )
                )
            return offset_value

        candidate_sides = [plan.side]
        if frame_bounds or avoid_bounds:
            flipped = "bottom" if plan.side == "top" else "top" if plan.side in ("top", "bottom") else (
                "left" if plan.side == "right" else "right"
            )
            if flipped not in candidate_sides:
                candidate_sides.append(flipped)

        selected_side = plan.side
        selected_offset = resolve_offset(plan.side)
        if frame_bounds or avoid_bounds:
            for side in candidate_sides:
                offset = resolve_offset(side)
                text = _dimension_text_for_plan(plan, side, offset, label, settings)
                if _text_fits_bounds(text, frame_bounds, padding=padding, avoid_bounds=avoid_bounds):
                    selected_side = side
                    selected_offset = offset
                    break

        lane_index[selected_side] += 1
        specs.append(_linear_dimension_spec(plan, selected_side, selected_offset, label, settings))

    return specs


def _resolve_diameter_dimension_specs(
    diameter_dims: list[PlannedDiameterDimension],
    settings: DimensionSettings,
    frame_bounds: BoundingBox2D | None,
    avoid_bounds: list[BoundingBox2D] | None = None,
) -> list[DiameterDimensionSpec]:
    """Resolve diameter dimensions into DXF-ready specs with angle adjustment."""
    specs: list[DiameterDimensionSpec] = []
    padding = settings.text_gap + settings.text_height * 0.5

    for plan in diameter_dims:
        label = plan.label
        if label is None:
            label = f"{settings.diameter_symbol}{format_length(plan.radius * 2, settings.decimal_places)}"

        angle_candidates = [plan.leader_angle_deg, plan.leader_angle_deg + 180.0]
        selected_angle = plan.leader_angle_deg
        if frame_bounds or avoid_bounds:
            for angle in angle_candidates:
                text = _diameter_text_for_angle(plan.center, plan.radius, angle, label, settings)
                if _text_fits_bounds(text, frame_bounds, padding=padding, avoid_bounds=avoid_bounds):
                    selected_angle = angle
                    break

        specs.append(
            DiameterDimensionSpec(
                center=plan.center,
                radius=plan.radius,
                angle=selected_angle,
                label=label,
                settings=settings,
            )
        )

    return specs


def _generate_view_dimensions(
    view: LayeredShapes,
    config: ViewDimensionConfig,
    dimension_settings: DimensionSettings | None,
    dimension_overrides: dict[str, object] | None,
    frame_bounds: BoundingBox2D | None,
    avoid_bounds: list[BoundingBox2D] | None = None,
) -> DimensionPlanOutput:
    """Plan all dimensions for a single view."""
    shapes = view.visible + view.hidden
    if not shapes:
        return DimensionPlanOutput([], [])

    settings = _resolve_dimension_settings(shapes, dimension_settings, dimension_overrides)
    rules = PlanningRules()

    # Generate basic bounding box dimension specs
    basic_specs = _resolve_basic_dimension_specs(shapes, settings, config, frame_bounds, avoid_bounds)

    # Extract and plan feature dimensions
    primitives = extract_primitives(shapes)
    features = extract_feature_coordinates(primitives)
    line_dims, diameter_dims, _ = _plan_feature_dimensions(features, config, settings, rules)

    # Resolve planned dimensions into DXF-ready specs
    line_specs = _resolve_line_dimension_specs(line_dims, settings, frame_bounds, avoid_bounds=avoid_bounds)
    diameter_specs = _resolve_diameter_dimension_specs(diameter_dims, settings, frame_bounds, avoid_bounds=avoid_bounds)

    return DimensionPlanOutput(basic_specs + line_specs, diameter_specs)


def _view_configs_for_layout(
    side_position: Literal["left", "right"],
    top_position: Literal["up", "down"],
) -> list[ViewDimensionConfig]:
    side_sign = 1 if side_position == "right" else -1
    top_sign = 1 if top_position == "up" else -1
    return [
        ViewDimensionConfig(horizontal_dir=-top_sign, vertical_dir=-side_sign),  # front
        ViewDimensionConfig(horizontal_dir=-top_sign, vertical_dir=side_sign),   # side_x
        ViewDimensionConfig(horizontal_dir=top_sign, vertical_dir=-side_sign),   # side_y
    ]


def _resolve_layout_positions(
    template_spec: TemplateSpec | None,
    side_position: Literal["left", "right"] | None,
    top_position: Literal["up", "down"] | None,
) -> tuple[Literal["left", "right"], Literal["up", "down"]]:
    resolved_side = side_position
    if resolved_side is None and template_spec:
        resolved_side = template_spec.side_position
    if resolved_side is None:
        resolved_side = "right"

    resolved_top = top_position
    if resolved_top is None and template_spec:
        resolved_top = template_spec.top_position
    if resolved_top is None:
        resolved_top = "down"

    return resolved_side, resolved_top


def _build_layers(
    model,
    add_template: bool,
    template_spec: TemplateSpec | None,
    x_offset: float,
    y_offset: float,
    side_position: Literal["left", "right"] | None,
    top_position: Literal["up", "down"] | None,
    layout_offset_x: float,
    layout_offset_y: float,
    add_dimensions: bool,
    dimension_settings: DimensionSettings | None,
    dimension_overrides: dict[str, object] | None,
) -> tuple[dict[str, list[Shape]], DimensionPlanOutput]:
    layers: dict[str, list[Shape]] = {}
    linear_dims: list[LinearDimensionSpec] = []
    diameter_dims: list[DiameterDimensionSpec] = []

    # Project and layout the three views
    template_offset_x = template_spec.layout_offset_mm[0] if template_spec else 0.0
    template_offset_y = template_spec.layout_offset_mm[1] if template_spec else 0.0
    combined_offset_x = template_offset_x + layout_offset_x
    combined_offset_y = template_offset_y + layout_offset_y
    resolved_side_position, resolved_top_position = _resolve_layout_positions(
        template_spec,
        side_position,
        top_position,
    )
    views = project_three_views(model)
    layout = layout_three_views(
        views.front,
        views.side_x,
        views.side_y,
        side_position=resolved_side_position,
        top_position=resolved_top_position,
        layout_offset_x=combined_offset_x,
        layout_offset_y=combined_offset_y,
        frame_bbox_mm=template_spec.frame_bbox_mm if template_spec else None,
        paper_size_mm=template_spec.paper_size_mm if template_spec else None,
        scale=template_spec.default_scale if template_spec else None,
    )
    layers["visible"] = layout.combined.visible
    layers["hidden"] = layout.combined.hidden

    # Generate dimensions for each view
    if add_dimensions:
        frame_bounds = _centered_frame_bounds(template_spec)
        title_block_bounds = _centered_title_block_bounds(template_spec)
        avoid_bounds = _centered_reserved_bounds(template_spec)
        if title_block_bounds:
            avoid_bounds.append(title_block_bounds)
        if not avoid_bounds:
            avoid_bounds = None

        layout_views = [layout.front, layout.side_x, layout.side_y]
        view_bounds = [_layered_bbox_2d(view) for view in layout_views]
        view_configs = _view_configs_for_layout(resolved_side_position, resolved_top_position)
        for index, (view, config) in enumerate(zip(layout_views, view_configs)):
            view_avoid = list(avoid_bounds) if avoid_bounds else []
            for other_index, bounds in enumerate(view_bounds):
                if other_index == index or not bounds:
                    continue
                view_avoid.append(bounds)
            if not view_avoid:
                view_avoid = None
            output = _generate_view_dimensions(
                view, config, dimension_settings, dimension_overrides, frame_bounds, avoid_bounds=view_avoid,
            )
            linear_dims.extend(output.linear)
            diameter_dims.extend(output.diameter)

    # Add template layer
    if add_template and isinstance(template_spec, SvgTemplateSpec):
        template = _load_svg_template(template_spec)
        tmp_size = Compound(children=template).bounding_box().size
        layers["template"] = [
            shape.translate((-tmp_size.X / 2 + x_offset, -tmp_size.Y / 2 + y_offset, 0))
            for shape in template
        ]

    return layers, DimensionPlanOutput(linear_dims, diameter_dims)


def _page_size_from_layers(
    layers: dict[str, list[Shape]],
    template_spec: TemplateSpec | None,
    add_template: bool,
) -> tuple[float, float]:
    if add_template and template_spec and template_spec.paper_size_mm:
        return template_spec.paper_size_mm
    shapes: list[Shape] = []
    for items in layers.values():
        shapes.extend(items)
    if shapes:
        bounds = Compound(children=shapes).bounding_box()
        size = bounds.size
        return (max(size.X, 1.0), max(size.Y, 1.0))
    if template_spec and template_spec.paper_size_mm:
        return template_spec.paper_size_mm
    return (297.0, 210.0)


def _export_outputs(
    layers: dict[str, list[Shape]],
    dimension_plans: DimensionPlanOutput,
    output_files: list[str],
    line_weight: float,
    line_types: dict[str, "LineType"] | None,
    template_spec: TemplateSpec | None,
    add_template: bool,
    x_offset: float,
    y_offset: float,
) -> None:
    with tempfile.NamedTemporaryFile(dir=os.getcwd(), suffix=".dxf", delete=False) as tmp_dxf:
        base_dxf = tmp_dxf.name
    try:
        export_dxf_layers(layers, base_dxf, line_weight, line_types=line_types)
        doc = ezdxf.readfile(base_dxf)
    finally:
        if os.path.exists(base_dxf):
            os.remove(base_dxf)

    if add_template and template_spec:
        apply_template_to_doc(doc, template_spec, x_offset=x_offset, y_offset=y_offset)

    if dimension_plans.linear or dimension_plans.diameter:
        add_ezdxf_dimensions(doc, dimension_plans.linear, dimension_plans.diameter)

    needs_svg = any(os.path.splitext(path)[1].lower() in {".svg", ".png", ".jpg", ".jpeg"} for path in output_files)
    svg_payload = None
    if needs_svg:
        page_size = _page_size_from_layers(layers, template_spec, add_template=add_template)
        svg_payload = render_dxf_to_svg(doc, page_size_mm=page_size)

    for output_file in output_files:
        _, ext = os.path.splitext(output_file)
        ext = ext.lower()
        if ext == ".dxf":
            doc.saveas(output_file)
            continue
        if ext == ".svg":
            if svg_payload is None:
                page_size = _page_size_from_layers(layers, template_spec, add_template=add_template)
                svg_payload = render_dxf_to_svg(doc, page_size_mm=page_size)
            Path(output_file).write_text(svg_payload, encoding="utf-8")
            continue
        if ext[1:] in RASTER_IMAGE_TYPES:
            if svg_payload is None:
                page_size = _page_size_from_layers(layers, template_spec, add_template=add_template)
                svg_payload = render_dxf_to_svg(doc, page_size_mm=page_size)
            with tempfile.NamedTemporaryFile(dir=os.getcwd(), suffix=".svg", delete=False) as tmp_svg:
                tmp_svg.write(svg_payload.encode("utf-8"))
                svg_path = tmp_svg.name
            try:
                rasterize_svg(svg_path, output_file)
            finally:
                if os.path.exists(svg_path):
                    os.remove(svg_path)
            continue
        raise ValueError(f"Invalid export file type: {ext}")


def convert_2d_drawing(
    step_file: str,
    output_files: Iterable[str],
    line_weight: float = 0.5,
    add_template: bool = True,
    template_name: str = "A4_LandscapeTD",
    x_offset: float = 0,
    y_offset: float = 0,
    side_position: Literal["left", "right"] | None = None,
    top_position: Literal["up", "down"] | None = None,
    layout_offset_x: float = 0.0,
    layout_offset_y: float = 0.0,
    style_name: str | None = "iso",
    add_dimensions: bool = False,
    dimension_settings: DimensionSettings | None = None,
) -> None:
    model = import_step(step_file)
    style = load_style(style_name) if style_name else None
    line_types = style.resolve_line_types() if style else None
    dimension_overrides = style.dimension if style and style.dimension else None
    template_spec = load_template(template_name)
    layers, dimension_plans = _build_layers(
        model,
        add_template,
        template_spec,
        x_offset,
        y_offset,
        side_position,
        top_position,
        layout_offset_x,
        layout_offset_y,
        add_dimensions,
        dimension_settings,
        dimension_overrides,
    )
    targets = list(output_files)
    if not targets:
        return
    _export_outputs(
        layers,
        dimension_plans,
        targets,
        line_weight,
        line_types=line_types,
        template_spec=template_spec,
        add_template=add_template,
        x_offset=x_offset,
        y_offset=y_offset,
    )
