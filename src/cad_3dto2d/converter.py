import math
import os
from typing import Iterable, Literal, NamedTuple

from build123d import Compound, ShapeList, import_step, import_svg
from pydantic import BaseModel, ConfigDict

from .annotations.dimensions import (
    DimensionResult,
    DimensionSettings,
    DimensionSide,
    DimensionText,
    default_settings_from_size,
    format_length,
    generate_basic_dimensions,
    generate_diameter_dimension,
    generate_linear_dimension,
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
from .exporters.dxf import export_dxf_layers
from .exporters.svg import export_svg_layers, inject_svg_text, rasterize_svg
from .layout import LayeredShapes, layout_three_views
from .styles import load_style
from .templates import TemplateSpec, load_template
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


class DimensionOutput(NamedTuple):
    """Output from dimension generation."""

    lines: list[Shape]
    texts: list[DimensionText]


def _load_template(template_spec: TemplateSpec) -> ShapeList[Shape]:
    return import_svg(template_spec.file_path)


def _centered_frame_bounds(template_spec: TemplateSpec | None) -> BoundingBox2D | None:
    if not template_spec or not template_spec.frame_bbox_mm:
        return None
    min_x, min_y, max_x, max_y = template_spec.frame_bbox_mm
    if template_spec.paper_size_mm:
        paper_w, paper_h = template_spec.paper_size_mm
        return (
            min_x - paper_w / 2,
            min_y - paper_h / 2,
            max_x - paper_w / 2,
            max_y - paper_h / 2,
        )
    return (min_x, min_y, max_x, max_y)


def _centered_title_block_bounds(template_spec: TemplateSpec | None) -> BoundingBox2D | None:
    if not template_spec or not template_spec.title_block_bbox_mm:
        return None
    min_x, min_y, max_x, max_y = template_spec.title_block_bbox_mm
    if template_spec.paper_size_mm:
        paper_w, paper_h = template_spec.paper_size_mm
        return (
            min_x - paper_w / 2,
            min_y - paper_h / 2,
            max_x - paper_w / 2,
            max_y - paper_h / 2,
        )
    return (min_x, min_y, max_x, max_y)


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


def _max_length_to_frame(
    point: Point2D,
    direction: Point2D,
    frame_bounds: BoundingBox2D,
) -> float:
    x0, y0 = point
    dx, dy = direction
    frame_min_x, frame_min_y, frame_max_x, frame_max_y = frame_bounds
    candidates: list[float] = []
    if abs(dx) > 1e-9:
        if dx > 0:
            candidates.append((frame_max_x - x0) / dx)
        else:
            candidates.append((frame_min_x - x0) / dx)
    if abs(dy) > 1e-9:
        if dy > 0:
            candidates.append((frame_max_y - y0) / dy)
        else:
            candidates.append((frame_min_y - y0) / dy)
    if not candidates:
        return 0.0
    return max(0.0, min(candidates))


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


def _shift_text_out_of_block(
    text: DimensionText,
    block_bounds: BoundingBox2D,
    frame_bounds: BoundingBox2D,
    padding: float,
    width_factor: float = _TEXT_WIDTH_FACTOR,
) -> DimensionText:
    text_bbox = _text_bounds(text, width_factor=width_factor)
    if not _bbox_intersects(text_bbox, block_bounds):
        return text

    block_min_x, block_min_y, block_max_x, block_max_y = block_bounds
    text_min_x, text_min_y, text_max_x, text_max_y = text_bbox

    candidates: list[tuple[float, float]] = [
        (block_min_x - padding - text_max_x, 0.0),
        (block_max_x + padding - text_min_x, 0.0),
        (0.0, block_min_y - padding - text_max_y),
        (0.0, block_max_y + padding - text_min_y),
    ]
    best: tuple[float, float] | None = None
    best_score = float("inf")
    for dx, dy in candidates:
        if dx == 0.0 and dy == 0.0:
            continue
        shifted = text.model_copy(update={"x": text.x + dx, "y": text.y + dy})
        shifted_bbox = _text_bounds(shifted, width_factor=width_factor)
        if not _bbox_within_frame(shifted_bbox, frame_bounds, padding=padding):
            continue
        score = abs(dx) + abs(dy)
        if score < best_score:
            best_score = score
            best = (dx, dy)

    if best is None:
        return text
    return text.model_copy(update={"x": text.x + best[0], "y": text.y + best[1]})


def _clamp_text_to_frame(
    text: DimensionText,
    frame_bounds: BoundingBox2D,
    padding: float = 0.0,
    width_factor: float = _TEXT_WIDTH_FACTOR,
) -> DimensionText:
    frame_min_x, frame_min_y, frame_max_x, frame_max_y = frame_bounds
    min_x, min_y, max_x, max_y = _text_bounds(text, width_factor=width_factor)
    new_x = text.x
    new_y = text.y

    min_x_limit = frame_min_x + padding
    max_x_limit = frame_max_x - padding
    min_y_limit = frame_min_y + padding
    max_y_limit = frame_max_y - padding

    if min_x < min_x_limit:
        new_x += min_x_limit - min_x
    elif max_x > max_x_limit:
        new_x -= max_x - max_x_limit

    if min_y < min_y_limit:
        new_y += min_y_limit - min_y
    elif max_y > max_y_limit:
        new_y -= max_y - max_y_limit

    if new_x == text.x and new_y == text.y:
        return text
    return text.model_copy(update={"x": new_x, "y": new_y})


def _clamp_texts_to_frame(
    texts: list[DimensionText],
    frame_bounds: BoundingBox2D,
    padding: float,
    avoid_bounds: list[BoundingBox2D] | None = None,
) -> list[DimensionText]:
    adjusted: list[DimensionText] = []
    for text in texts:
        current = _clamp_text_to_frame(text, frame_bounds, padding=padding)
        if avoid_bounds:
            for block in avoid_bounds:
                current = _shift_text_out_of_block(current, block, frame_bounds, padding=padding)
        adjusted.append(current)
    return adjusted


def _resolve_dimension_settings(
    shapes: list[Shape],
    dimension_settings: DimensionSettings | None,
    dimension_overrides: dict[str, object] | None,
) -> DimensionSettings:
    """Resolve dimension settings from shapes size, user settings, and overrides."""
    bounds = Compound(children=shapes).bounding_box()
    size = bounds.size
    settings = dimension_settings or default_settings_from_size(size.X, size.Y)
    if dimension_settings is None and dimension_overrides:
        settings = DimensionSettings.model_validate(
            {**settings.model_dump(), **dimension_overrides}
        )
    return settings


def _generate_basic_view_dimensions(
    shapes: list[Shape],
    settings: DimensionSettings,
    config: ViewDimensionConfig,
    frame_bounds: BoundingBox2D | None,
) -> DimensionResult:
    """Generate basic bounding box dimensions for a view."""
    bounds = Compound(children=shapes).bounding_box()
    padding = settings.text_gap + settings.text_height

    horizontal_offset = None
    vertical_offset = None

    if frame_bounds:
        frame_min_x, frame_min_y, frame_max_x, frame_max_y = frame_bounds
        base_y = bounds.max.Y if config.horizontal_dir >= 0 else bounds.min.Y
        base_x = bounds.max.X if config.vertical_dir >= 0 else bounds.min.X
        horizontal_offset = _clamp_offset(
            base_y, config.horizontal_dir, frame_min_y, frame_max_y,
            settings.offset, padding=padding,
        )
        vertical_offset = _clamp_offset(
            base_x, config.vertical_dir, frame_min_x, frame_max_x,
            settings.offset, padding=padding,
        )

    return generate_basic_dimensions(
        shapes,
        settings=settings,
        horizontal_dir=config.horizontal_dir,
        vertical_dir=config.vertical_dir,
        horizontal_offset=horizontal_offset,
        vertical_offset=vertical_offset,
    )


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
    frame_bounds: BoundingBox2D,
    padding: float,
    avoid_bounds: list[BoundingBox2D] | None = None,
) -> bool:
    bbox = _text_bounds(text)
    if not _bbox_within_frame(bbox, frame_bounds, padding=padding):
        return False
    if avoid_bounds:
        for block in avoid_bounds:
            if _bbox_intersects(bbox, block):
                return False
    return True


def _generate_line_dimensions(
    line_dims: list[PlannedDimension],
    settings: DimensionSettings,
    frame_bounds: BoundingBox2D | None,
    avoid_bounds: list[BoundingBox2D] | None = None,
) -> DimensionOutput:
    """Generate dimension lines from planned dimensions with frame clamping."""
    lines: list[Shape] = []
    texts: list[DimensionText] = []
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
        if frame_bounds:
            flipped = "bottom" if plan.side == "top" else "top" if plan.side in ("top", "bottom") else (
                "left" if plan.side == "right" else "right"
            )
            if flipped not in candidate_sides:
                candidate_sides.append(flipped)

        selected_side = plan.side
        selected_offset = resolve_offset(plan.side)
        if frame_bounds:
            for side in candidate_sides:
                offset = resolve_offset(side)
                text = _dimension_text_for_plan(plan, side, offset, label, settings)
                if _text_fits_bounds(text, frame_bounds, padding=padding, avoid_bounds=avoid_bounds):
                    selected_side = side
                    selected_offset = offset
                    break

        lane_index[selected_side] += 1

        result = generate_linear_dimension(
            plan.p1, plan.p2,
            orientation=plan.orientation,
            side=selected_side,
            offset=selected_offset,
            settings=settings,
            label=label,
        )
        lines.extend(result.lines)
        texts.extend(result.texts)

    return DimensionOutput(lines, texts)


def _generate_diameter_dimensions(
    diameter_dims: list[PlannedDiameterDimension],
    settings: DimensionSettings,
    frame_bounds: BoundingBox2D | None,
) -> DimensionOutput:
    """Generate diameter dimensions with leader lines and frame clamping."""
    lines: list[Shape] = []
    texts: list[DimensionText] = []
    padding = settings.text_gap + settings.text_height

    for plan in diameter_dims:
        label = plan.label
        if label is None:
            label = f"{settings.diameter_symbol}{format_length(plan.radius * 2, settings.decimal_places)}"
        leader_length = settings.arrow_size * 4 + settings.text_gap * 2 + settings.text_height

        if frame_bounds:
            direction = (
                math.cos(math.radians(plan.leader_angle_deg)),
                math.sin(math.radians(plan.leader_angle_deg)),
            )
            arrow_tip = (
                plan.center[0] + plan.radius * direction[0],
                plan.center[1] + plan.radius * direction[1],
            )
            available = _max_length_to_frame(arrow_tip, direction, frame_bounds)
            leader_length = min(leader_length, max(0.0, available - padding))

        result = generate_diameter_dimension(
            plan.center,
            plan.radius,
            leader_angle_deg=plan.leader_angle_deg,
            settings=settings,
            leader_length=leader_length,
            label=label,
        )
        lines.extend(result.lines)
        texts.extend(result.texts)

    return DimensionOutput(lines, texts)


def _generate_view_dimensions(
    view: LayeredShapes,
    config: ViewDimensionConfig,
    dimension_settings: DimensionSettings | None,
    dimension_overrides: dict[str, object] | None,
    frame_bounds: BoundingBox2D | None,
    avoid_bounds: list[BoundingBox2D] | None = None,
) -> DimensionOutput:
    """Generate all dimensions for a single view."""
    shapes = view.visible + view.hidden
    if not shapes:
        return DimensionOutput([], [])

    settings = _resolve_dimension_settings(shapes, dimension_settings, dimension_overrides)
    rules = PlanningRules()

    # Generate basic bounding box dimensions
    basic_result = _generate_basic_view_dimensions(shapes, settings, config, frame_bounds)

    # Extract and plan feature dimensions
    primitives = extract_primitives(shapes)
    features = extract_feature_coordinates(primitives)
    line_dims, diameter_dims, _ = _plan_feature_dimensions(features, config, settings, rules)

    # Generate planned dimensions
    line_output = _generate_line_dimensions(line_dims, settings, frame_bounds, avoid_bounds=avoid_bounds)
    diameter_output = _generate_diameter_dimensions(diameter_dims, settings, frame_bounds)

    # Combine all outputs
    all_lines = list(basic_result.lines) + line_output.lines + diameter_output.lines
    all_texts = list(basic_result.texts) + line_output.texts + diameter_output.texts
    if frame_bounds:
        all_texts = _clamp_texts_to_frame(
            all_texts,
            frame_bounds,
            padding=settings.text_gap + settings.text_height * 0.5,
            avoid_bounds=avoid_bounds,
        )

    return DimensionOutput(all_lines, all_texts)


# View configurations for three-view drawing
VIEW_CONFIGS = [
    ViewDimensionConfig(horizontal_dir=1, vertical_dir=1),   # front
    ViewDimensionConfig(horizontal_dir=1, vertical_dir=1),   # side_x
    ViewDimensionConfig(horizontal_dir=-1, vertical_dir=1),  # side_y
]


def _build_layers(
    model,
    add_template: bool,
    template_spec: TemplateSpec | None,
    x_offset: float,
    y_offset: float,
    add_dimensions: bool,
    dimension_settings: DimensionSettings | None,
    dimension_overrides: dict[str, object] | None,
) -> tuple[dict[str, list[Shape]], list[DimensionText]]:
    layers: dict[str, list[Shape]] = {}
    dim_texts: list[DimensionText] = []

    # Project and layout the three views
    views = project_three_views(model)
    layout = layout_three_views(
        views.front,
        views.side_x,
        views.side_y,
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
        avoid_bounds = [title_block_bounds] if title_block_bounds else None
        dims: list[Shape] = []

        layout_views = [layout.front, layout.side_x, layout.side_y]
        for view, config in zip(layout_views, VIEW_CONFIGS):
            output = _generate_view_dimensions(
                view, config, dimension_settings, dimension_overrides, frame_bounds, avoid_bounds=avoid_bounds,
            )
            dims.extend(output.lines)
            dim_texts.extend(output.texts)

        if dims:
            layers["dims"] = dims

    # Add template layer
    if add_template and template_spec:
        template = _load_template(template_spec)
        tmp_size = Compound(children=template).bounding_box().size
        layers["template"] = [
            shape.translate((-tmp_size.X / 2 + x_offset, -tmp_size.Y / 2 + y_offset, 0))
            for shape in template
        ]

    return layers, dim_texts


def _export_layers(
    layers: dict[str, list[Shape]],
    output_file: str,
    line_weight: float,
    line_types: dict[str, "LineType"] | None,
    text_annotations: list[DimensionText] | None,
) -> None:
    _, ext = os.path.splitext(output_file)
    ext = ext.lower()
    if ext == ".dxf":
        export_dxf_layers(layers, output_file, line_weight, line_types=line_types)
        return
    if ext == ".svg":
        export_svg_layers(layers, output_file, line_weight, line_types=line_types)
        if text_annotations:
            inject_svg_text(output_file, text_annotations)
        return
    if ext[1:] in RASTER_IMAGE_TYPES:
        svg_file = os.path.splitext(output_file)[0] + ".svg"
        export_svg_layers(layers, svg_file, line_weight, line_types=line_types)
        if text_annotations:
            inject_svg_text(svg_file, text_annotations)
        rasterize_svg(svg_file, output_file)
        return
    raise ValueError(f"Invalid export file type: {ext}")


def convert_2d_drawing(
    step_file: str,
    output_files: Iterable[str],
    line_weight: float = 0.5,
    add_template: bool = True,
    template_name: str = "A4_LandscapeTD",
    x_offset: float = 0,
    y_offset: float = 0,
    style_name: str | None = "iso",
    add_dimensions: bool = False,
    dimension_settings: DimensionSettings | None = None,
) -> None:
    model = import_step(step_file)
    style = load_style(style_name) if style_name else None
    line_types = style.resolve_line_types() if style else None
    dimension_overrides = style.dimension if style and style.dimension else None
    template_spec = load_template(template_name)
    layers, text_annotations = _build_layers(
        model,
        add_template,
        template_spec,
        x_offset,
        y_offset,
        add_dimensions,
        dimension_settings,
        dimension_overrides,
    )
    for target in output_files:
        _export_layers(layers, target, line_weight, line_types=line_types, text_annotations=text_annotations)
