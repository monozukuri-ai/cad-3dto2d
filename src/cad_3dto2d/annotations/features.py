from __future__ import annotations

from typing import Iterable, Literal

from build123d import Compound, Edge, GeomType
from pydantic import BaseModel, ConfigDict

from ..types import BoundingBox2D, Point2D, Shape


class LinePrimitive(BaseModel):
    model_config = ConfigDict(frozen=True)

    p1: Point2D
    p2: Point2D
    orientation: Literal["horizontal", "vertical", "other"]


class CirclePrimitive(BaseModel):
    model_config = ConfigDict(frozen=True)

    center: Point2D
    radius: float


class PrimitiveResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    lines: list[LinePrimitive]
    circles: list[CirclePrimitive]
    bounds: BoundingBox2D


class FeatureCoordinates(BaseModel):
    model_config = ConfigDict(frozen=True)

    x_coords: list[float]
    y_coords: list[float]
    circles: list[CirclePrimitive]
    bounds: BoundingBox2D


def _iter_edges(shapes: Iterable[Shape]) -> Iterable[Edge]:
    for shape in shapes:
        if isinstance(shape, Edge):
            yield shape
            continue
        if hasattr(shape, "edges"):
            yield from shape.edges()


def _cluster_values(values: list[float], tol: float) -> list[float]:
    if not values:
        return []
    values = sorted(values)
    clusters: list[list[float]] = [[values[0]]]
    for value in values[1:]:
        if abs(value - clusters[-1][-1]) <= tol:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(group) / len(group) for group in clusters]


def extract_primitives(shapes: Iterable[Shape], tol: float = 1e-6) -> PrimitiveResult:
    shapes = list(shapes)
    bounds = (0.0, 0.0, 0.0, 0.0)
    if shapes:
        bbox = Compound(children=shapes).bounding_box()
        bounds = (bbox.min.X, bbox.min.Y, bbox.max.X, bbox.max.Y)

    lines: list[LinePrimitive] = []
    circles: list[CirclePrimitive] = []
    for edge in _iter_edges(shapes):
        geom_type = edge.geom_type
        if geom_type == GeomType.LINE:
            p1 = edge.start_point()
            p2 = edge.end_point()
            dx = p2.X - p1.X
            dy = p2.Y - p1.Y
            if abs(dy) <= tol and abs(dx) > tol:
                orientation = "horizontal"
            elif abs(dx) <= tol and abs(dy) > tol:
                orientation = "vertical"
            else:
                orientation = "other"
            lines.append(
                LinePrimitive(p1=(p1.X, p1.Y), p2=(p2.X, p2.Y), orientation=orientation)
            )
        elif geom_type == GeomType.CIRCLE:
            center = edge.arc_center
            radius = edge.radius
            circles.append(CirclePrimitive(center=(center.X, center.Y), radius=radius))

    return PrimitiveResult(lines=lines, circles=circles, bounds=bounds)


def extract_feature_coordinates(
    primitives: PrimitiveResult, tol: float = 1e-3
) -> FeatureCoordinates:
    xmin, ymin, xmax, ymax = primitives.bounds
    vertical_x: list[float] = []
    horizontal_y: list[float] = []
    for line in primitives.lines:
        if line.orientation == "vertical":
            vertical_x.append(line.p1[0])
        elif line.orientation == "horizontal":
            horizontal_y.append(line.p1[1])

    x_coords = [
        x
        for x in _cluster_values(vertical_x, tol)
        if abs(x - xmin) > tol and abs(x - xmax) > tol
    ]
    y_coords = [
        y
        for y in _cluster_values(horizontal_y, tol)
        if abs(y - ymin) > tol and abs(y - ymax) > tol
    ]
    circles = _dedupe_circles(primitives.circles, tol=tol)
    circles = _filter_hole_circles(circles, primitives.bounds)
    return FeatureCoordinates(
        x_coords=x_coords, y_coords=y_coords, circles=circles, bounds=primitives.bounds
    )


def _dedupe_circles(
    circles: list[CirclePrimitive], tol: float
) -> list[CirclePrimitive]:
    if not circles:
        return []
    circles = sorted(circles, key=lambda c: (c.center[0], c.center[1], c.radius))
    deduped: list[CirclePrimitive] = [circles[0]]
    for circle in circles[1:]:
        last = deduped[-1]
        if (
            abs(circle.center[0] - last.center[0]) <= tol
            and abs(circle.center[1] - last.center[1]) <= tol
            and abs(circle.radius - last.radius) <= tol
        ):
            continue
        deduped.append(circle)
    return deduped


def _filter_hole_circles(
    circles: list[CirclePrimitive],
    bounds: BoundingBox2D,
    max_radius_ratio: float = 0.35,
) -> list[CirclePrimitive]:
    if not circles:
        return []
    xmin, ymin, xmax, ymax = bounds
    limit = max(xmax - xmin, ymax - ymin) * max_radius_ratio
    return [circle for circle in circles if circle.radius <= limit]
