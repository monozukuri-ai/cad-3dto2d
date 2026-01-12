from __future__ import annotations

from typing import TypeAlias

from build123d import Edge, Face, Wire

# Geometric primitives
Point2D: TypeAlias = tuple[float, float]
Point3D: TypeAlias = tuple[float, float, float]
BoundingBox2D: TypeAlias = tuple[
    float, float, float, float
]  # (min_x, min_y, max_x, max_y)

# CAD shape types
Shape: TypeAlias = Wire | Face | Edge
