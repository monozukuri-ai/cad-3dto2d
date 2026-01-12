from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..types import Point2D
from .dimensions import DimensionOrientation, DimensionSide
from .features import CirclePrimitive, FeatureCoordinates


class PlannedDimension(BaseModel):
    model_config = ConfigDict(frozen=True)

    p1: Point2D
    p2: Point2D
    orientation: DimensionOrientation
    side: DimensionSide
    label: str | None = None


class PlannedDiameterDimension(BaseModel):
    model_config = ConfigDict(frozen=True)

    center: Point2D
    radius: float
    leader_angle_deg: float
    label: str | None = None


class PlannedPitchDimension(BaseModel):
    model_config = ConfigDict(frozen=True)

    p1: Point2D
    p2: Point2D
    orientation: DimensionOrientation
    side: DimensionSide
    count: int
    pitch: float


class PlanningRules(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_line_dims: int = 8
    max_hole_positions: int = 4
    max_pitch_dims: int = 2
    max_internal_dims: int = 4
    max_diameter_dims: int = 2
    keep_min_position_dims: int = 1
    collapse_diameter_with_pitch: bool = True
    max_diameter_groups: int = 1


def _select_coords(coords: list[float], max_count: int) -> list[float]:
    if len(coords) <= max_count:
        return coords
    return [coords[0], coords[-1]]


def _group_circle_indices(
    circles: list[CirclePrimitive], axis: int, tol: float
) -> list[list[int]]:
    if not circles:
        return []
    indexed = list(enumerate(circles))
    indexed.sort(key=lambda item: item[1].center[axis])
    groups: list[list[int]] = [[indexed[0][0]]]
    group_ref = indexed[0][1].center[axis]
    for idx, circle in indexed[1:]:
        if abs(circle.center[axis] - group_ref) <= tol:
            groups[-1].append(idx)
        else:
            groups.append([idx])
            group_ref = circle.center[axis]
    return groups


def _has_close(value: float, values: list[float], tol: float) -> bool:
    return any(abs(value - other) <= tol for other in values)


def _detect_pitch_dimension(
    circles: list[CirclePrimitive],
    group_indices: list[int],
    axis: int,
    side: DimensionSide,
    min_pitch: float,
    pitch_tol_ratio: float,
) -> PlannedPitchDimension | None:
    """Detect a pitch pattern along the given axis (0=x, 1=y)."""
    other_axis = 1 - axis
    orientation: DimensionOrientation = "horizontal" if axis == 0 else "vertical"
    sorted_indices = sorted(group_indices, key=lambda i: circles[i].center[axis])
    group_circles = [circles[i] for i in sorted_indices]
    coords = [c.center[axis] for c in group_circles]
    diffs = [
        coords[i + 1] - coords[i]
        for i in range(len(coords) - 1)
        if coords[i + 1] > coords[i]
    ]
    if not diffs:
        return None
    pitch = sum(diffs) / len(diffs)
    if pitch < min_pitch:
        return None
    if len(diffs) > 1 and (max(diffs) - min(diffs)) > pitch * pitch_tol_ratio:
        return None
    ref = group_circles[0].center[other_axis]
    if axis == 0:
        p1, p2 = (coords[0], ref), (coords[-1], ref)
    else:
        p1, p2 = (ref, coords[0]), (ref, coords[-1])
    return PlannedPitchDimension(
        p1=p1,
        p2=p2,
        orientation=orientation,
        side=side,
        count=len(group_indices),
        pitch=pitch,
    )


def plan_internal_dimensions(
    features: FeatureCoordinates,
    horizontal_side: DimensionSide,
    vertical_side: DimensionSide,
    max_per_axis: int = 2,
) -> list[PlannedDimension]:
    xmin, ymin, xmax, ymax = features.bounds
    dims: list[PlannedDimension] = []

    x_coords = _select_coords(sorted(features.x_coords), max_per_axis)
    y_coords = _select_coords(sorted(features.y_coords), max_per_axis)

    y_ref = ymax if horizontal_side == "top" else ymin
    for x in x_coords:
        dims.append(
            PlannedDimension(
                p1=(xmin, y_ref),
                p2=(x, y_ref),
                orientation="horizontal",
                side=horizontal_side,
            )
        )

    x_ref = xmax if vertical_side == "right" else xmin
    for y in y_coords:
        dims.append(
            PlannedDimension(
                p1=(x_ref, ymin),
                p2=(x_ref, y),
                orientation="vertical",
                side=vertical_side,
            )
        )

    return dims


def plan_hole_dimensions(
    features: FeatureCoordinates,
    horizontal_side: DimensionSide,
    vertical_side: DimensionSide,
    max_circles: int = 2,
    min_pitch_count: int = 3,
    pitch_tol_ratio: float = 0.1,
    min_pitch: float = 0.5,
) -> tuple[
    list[PlannedDimension],
    list[PlannedDiameterDimension],
    list[PlannedPitchDimension],
    dict[str, bool],
]:
    xmin, ymin, xmax, ymax = features.bounds
    position_dims: list[PlannedDimension] = []
    diameter_dims: list[PlannedDiameterDimension] = []
    pitch_dims: list[PlannedPitchDimension] = []
    pitch_axes = {"horizontal": False, "vertical": False}

    circles = features.circles
    if circles:
        angle_candidates = [45, 135] if horizontal_side == "top" else [-45, -135]
        for idx, circle in enumerate(circles[:max_circles]):
            angle = angle_candidates[idx % len(angle_candidates)]
            diameter_dims.append(
                PlannedDiameterDimension(
                    center=circle.center, radius=circle.radius, leader_angle_deg=angle
                )
            )
        tol = 1e-3
        horizontal_groups = _group_circle_indices(circles, axis=1, tol=tol)
        vertical_groups = _group_circle_indices(circles, axis=0, tol=tol)
        skip_horizontal: set[int] = set()
        skip_vertical: set[int] = set()
        for group in horizontal_groups:
            if len(group) < min_pitch_count:
                continue
            pitch_dim = _detect_pitch_dimension(
                circles,
                group,
                axis=0,
                side=horizontal_side,
                min_pitch=min_pitch,
                pitch_tol_ratio=pitch_tol_ratio,
            )
            if pitch_dim:
                skip_horizontal.update(group)
                pitch_dims.append(pitch_dim)
                pitch_axes["horizontal"] = True
        for group in vertical_groups:
            if len(group) < min_pitch_count:
                continue
            pitch_dim = _detect_pitch_dimension(
                circles,
                group,
                axis=1,
                side=vertical_side,
                min_pitch=min_pitch,
                pitch_tol_ratio=pitch_tol_ratio,
            )
            if pitch_dim:
                skip_vertical.update(group)
                pitch_dims.append(pitch_dim)
                pitch_axes["vertical"] = True
        used_x: list[float] = []
        used_y: list[float] = []
        for idx, circle in enumerate(circles):
            cx, cy = circle.center
            if idx not in skip_horizontal and not _has_close(cx, used_x, tol):
                used_x.append(cx)
                position_dims.append(
                    PlannedDimension(
                        p1=(xmin, cy),
                        p2=(cx, cy),
                        orientation="horizontal",
                        side=horizontal_side,
                    )
                )
            if idx not in skip_vertical and not _has_close(cy, used_y, tol):
                used_y.append(cy)
                position_dims.append(
                    PlannedDimension(
                        p1=(cx, ymin),
                        p2=(cx, cy),
                        orientation="vertical",
                        side=vertical_side,
                    )
                )
        if pitch_axes["horizontal"] and pitch_axes["vertical"] and not position_dims:
            first = circles[0]
            position_dims.append(
                PlannedDimension(
                    p1=(xmin, first.center[1]),
                    p2=(first.center[0], first.center[1]),
                    orientation="horizontal",
                    side=horizontal_side,
                )
            )
    return position_dims, diameter_dims, pitch_dims, pitch_axes


def group_circles_by_radius(
    circles: list[CirclePrimitive], tol: float = 1e-3
) -> list[list[CirclePrimitive]]:
    if not circles:
        return []
    circles_sorted = sorted(circles, key=lambda c: c.radius)
    groups: list[list[CirclePrimitive]] = [[circles_sorted[0]]]
    for circle in circles_sorted[1:]:
        if abs(circle.radius - groups[-1][-1].radius) <= tol:
            groups[-1].append(circle)
        else:
            groups.append([circle])
    return groups


def apply_planning_rules(
    hole_positions: list[PlannedDimension],
    hole_pitches: list[PlannedDimension],
    internal_dims: list[PlannedDimension],
    hole_diameters: list[PlannedDiameterDimension],
    rules: PlanningRules | None = None,
) -> tuple[list[PlannedDimension], list[PlannedDiameterDimension]]:
    rules = rules or PlanningRules()
    line_dims: list[PlannedDimension] = []
    line_dims.extend(hole_positions[: rules.max_hole_positions])
    line_dims.extend(hole_pitches[: rules.max_pitch_dims])
    remaining = rules.max_line_dims - len(line_dims)
    if remaining > 0:
        line_dims.extend(internal_dims[: min(remaining, rules.max_internal_dims)])
    diameter_dims = hole_diameters[: rules.max_diameter_dims]
    return line_dims, diameter_dims
