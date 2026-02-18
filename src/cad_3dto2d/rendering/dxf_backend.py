from __future__ import annotations

import math

import ezdxf
from build123d import ExportDXF, LineType, Unit
from ezdxf.addons.drawing import Frontend, RenderContext, layout
from ezdxf.addons.drawing.config import BackgroundPolicy, Configuration
from ezdxf.addons.drawing.svg import SVGBackend
from ezdxf.enums import TextEntityAlignment

from ..annotations.dimensions import (
    DiameterDimensionSpec,
    DimensionSettings,
    LeaderNoteSpec,
    LinearDimensionSpec,
)
from ..templates import (
    ParametricTemplateSpec,
    TemplateSpec,
    TitleBlockCellSpec,
    TitleBlockSpec,
)
from ..types import BoundingBox2D, Shape

_LINE_TYPES: dict[str, LineType] = {
    "visible": LineType.CONTINUOUS,
    "hidden": LineType.DASHED,
    "template": LineType.CONTINUOUS,
    "title": LineType.CONTINUOUS,
}
_TITLE_TEXT_HEIGHT_DEFAULT = 3.5


def export_dxf_layers(
    layers: dict[str, list[Shape]],
    output_file: str,
    line_weight: float,
    line_types: dict[str, LineType] | None = None,
) -> None:
    exporter = ExportDXF(unit=Unit.MM, line_weight=line_weight)
    resolved_types = dict(_LINE_TYPES)
    if line_types:
        resolved_types.update(line_types)
    layer_order = ["visible", "hidden", "template"]
    ordered_layers = [name for name in layer_order if name in layers]
    ordered_layers.extend(name for name in layers if name not in layer_order)
    for layer_name in ordered_layers:
        exporter.add_layer(
            layer_name, line_type=resolved_types.get(layer_name, LineType.CONTINUOUS)
        )
    for layer_name in ordered_layers:
        exporter.add_shape(layers[layer_name], layer=layer_name)
    exporter.write(output_file)


def apply_template_to_doc(
    doc: "ezdxf.document.Drawing",
    template_spec: TemplateSpec,
    title_values: dict[str, str] | None = None,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> None:
    if isinstance(template_spec, ParametricTemplateSpec):
        _apply_parametric_template(
            doc,
            template_spec,
            title_values=title_values,
            x_offset=x_offset,
            y_offset=y_offset,
        )


def _dimension_overrides(settings: DimensionSettings) -> dict[str, float | int | str]:
    override: dict[str, float | int | str] = {
        "dimtxt": settings.text_height,
        "dimasz": settings.arrow_size,
        "dimexo": settings.extension_offset
        if settings.extension_offset is not None
        else settings.extension_gap,
        "dimexe": settings.extension_extension
        if settings.extension_extension is not None
        else settings.extension_gap,
        "dimgap": settings.text_gap,
        "dimdec": settings.decimal_places,
    }
    if settings.arrow_block:
        override["dimblk"] = settings.arrow_block
    if settings.arrow_block1:
        override["dimblk1"] = settings.arrow_block1
    if settings.arrow_block2:
        override["dimblk2"] = settings.arrow_block2
    return override


def _ensure_dimension_text_style(
    doc: "ezdxf.document.Drawing", settings: DimensionSettings
) -> str | None:
    style_name = settings.text_style
    font_name = settings.text_font
    if font_name and not style_name:
        style_name = "DIM_TEXT"
    if not style_name:
        return None
    if not doc.styles.has_entry(style_name):
        doc.styles.new(
            style_name,
            dxfattribs={"font": font_name or "txt"},
        )
    elif font_name:
        doc.styles.get(style_name).dxf.font = font_name
    return style_name


def _apply_parametric_template(
    doc: "ezdxf.document.Drawing",
    template_spec: ParametricTemplateSpec,
    title_values: dict[str, str] | None,
    x_offset: float,
    y_offset: float,
) -> None:
    msp = doc.modelspace()
    _ensure_layer(doc, "template")
    _ensure_layer(doc, "title")
    frame_bbox = _centered_bbox(
        template_spec.frame_bbox_mm, template_spec.paper_size_mm
    )
    if frame_bbox:
        frame_bbox = _offset_bbox(frame_bbox, x_offset, y_offset)
        _add_bbox_rect(msp, frame_bbox, layer="template")
    if template_spec.title_block_bbox_mm:
        title_bbox = _centered_bbox(
            template_spec.title_block_bbox_mm, template_spec.paper_size_mm
        )
        if title_bbox:
            title_bbox = _offset_bbox(title_bbox, x_offset, y_offset)
            _add_bbox_rect(msp, title_bbox, layer="title")
            if template_spec.title_block:
                _draw_title_block(
                    msp,
                    title_bbox,
                    template_spec.title_block,
                    title_values=title_values,
                    layer="title",
                )


def _draw_title_block(
    msp,
    title_bbox: BoundingBox2D,
    title_spec: TitleBlockSpec,
    title_values: dict[str, str] | None,
    layer: str,
) -> None:
    if not title_spec.grid:
        return
    x_edges, y_edges = _grid_edges(
        title_bbox, title_spec.grid.rows_mm, title_spec.grid.cols_mm
    )
    segments = _grid_segments(x_edges, y_edges)
    for cell in title_spec.cells:
        _apply_cell_merge(segments, cell, x_edges, y_edges)
    _draw_segments(msp, segments, layer=layer)
    _draw_cell_texts(
        msp, title_spec, x_edges, y_edges, title_values=title_values, layer=layer
    )


def _grid_edges(
    title_bbox: BoundingBox2D,
    rows_mm: list[float],
    cols_mm: list[float],
) -> tuple[list[float], list[float]]:
    min_x, min_y, max_x, max_y = title_bbox
    x_edges = [min_x]
    for width in cols_mm:
        x_edges.append(x_edges[-1] + width)
    y_edges = [max_y]
    for height in rows_mm:
        y_edges.append(y_edges[-1] - height)
    return x_edges, y_edges


def _grid_segments(
    x_edges: list[float], y_edges: list[float]
) -> set[tuple[tuple[float, float], tuple[float, float]]]:
    segments: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for y in y_edges[1:-1]:
        for i in range(len(x_edges) - 1):
            segments.add(_normalize_segment((x_edges[i], y), (x_edges[i + 1], y)))
    for x in x_edges[1:-1]:
        for j in range(len(y_edges) - 1):
            segments.add(_normalize_segment((x, y_edges[j]), (x, y_edges[j + 1])))
    return segments


def _apply_cell_merge(
    segments: set[tuple[tuple[float, float], tuple[float, float]]],
    cell: TitleBlockCellSpec,
    x_edges: list[float],
    y_edges: list[float],
) -> None:
    span_rows, span_cols = cell.span
    if span_rows <= 1 and span_cols <= 1:
        return
    row, col = cell.cell
    for internal_col in range(col + 1, col + span_cols):
        x = x_edges[internal_col]
        for r in range(row, row + span_rows):
            segments.discard(_normalize_segment((x, y_edges[r]), (x, y_edges[r + 1])))
    for internal_row in range(row + 1, row + span_rows):
        y = y_edges[internal_row]
        for c in range(col, col + span_cols):
            segments.discard(_normalize_segment((x_edges[c], y), (x_edges[c + 1], y)))


def _draw_segments(
    msp,
    segments: set[tuple[tuple[float, float], tuple[float, float]]],
    layer: str,
) -> None:
    for p1, p2 in segments:
        msp.add_line(p1, p2, dxfattribs={"layer": layer})


def _draw_cell_texts(
    msp,
    title_spec: TitleBlockSpec,
    x_edges: list[float],
    y_edges: list[float],
    title_values: dict[str, str] | None,
    layer: str,
) -> None:
    for cell in title_spec.cells:
        text = _resolve_cell_text(cell, title_values)
        if not text:
            continue
        bounds = _cell_bounds(cell, x_edges, y_edges)
        anchor = _cell_anchor(bounds, cell.align, cell.valign)
        anchor = (anchor[0] + cell.offset_mm[0], anchor[1] + cell.offset_mm[1])
        height = (
            cell.text_height_mm if cell.text_height_mm else _TITLE_TEXT_HEIGHT_DEFAULT
        )
        alignment = _text_alignment(cell.align, cell.valign)
        entity = msp.add_text(
            text,
            dxfattribs={"height": height, "layer": layer, "rotation": cell.rotate_deg},
        )
        entity.set_placement(anchor, align=alignment)


def _resolve_cell_text(
    cell: TitleBlockCellSpec, title_values: dict[str, str] | None
) -> str:
    if cell.text is not None:
        value = cell.text
    elif cell.key:
        value = ""
        if title_values and cell.key in title_values:
            value = str(title_values[cell.key])
        elif cell.default is not None:
            value = cell.default
    else:
        value = ""
    if not value:
        return ""
    return f"{cell.prefix}{value}{cell.suffix}"


def _cell_bounds(
    cell: TitleBlockCellSpec,
    x_edges: list[float],
    y_edges: list[float],
) -> BoundingBox2D:
    row, col = cell.cell
    span_rows, span_cols = cell.span
    left = x_edges[col]
    right = x_edges[col + span_cols]
    top = y_edges[row]
    bottom = y_edges[row + span_rows]
    return (left, bottom, right, top)


def _cell_anchor(bounds: BoundingBox2D, align: str, valign: str) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = bounds
    if align == "right":
        x = max_x
    elif align == "center":
        x = (min_x + max_x) / 2
    else:
        x = min_x
    if valign == "top":
        y = max_y
    elif valign == "bottom":
        y = min_y
    else:
        y = (min_y + max_y) / 2
    return x, y


def _text_alignment(align: str, valign: str) -> TextEntityAlignment:
    mapping = {
        ("left", "top"): TextEntityAlignment.TOP_LEFT,
        ("center", "top"): TextEntityAlignment.TOP_CENTER,
        ("right", "top"): TextEntityAlignment.TOP_RIGHT,
        ("left", "middle"): TextEntityAlignment.MIDDLE_LEFT,
        ("center", "middle"): TextEntityAlignment.MIDDLE_CENTER,
        ("right", "middle"): TextEntityAlignment.MIDDLE_RIGHT,
        ("left", "bottom"): TextEntityAlignment.BOTTOM_LEFT,
        ("center", "bottom"): TextEntityAlignment.BOTTOM_CENTER,
        ("right", "bottom"): TextEntityAlignment.BOTTOM_RIGHT,
    }
    return mapping.get((align, valign), TextEntityAlignment.MIDDLE_LEFT)


def _normalize_segment(
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    rp1 = (round(p1[0], 6), round(p1[1], 6))
    rp2 = (round(p2[0], 6), round(p2[1], 6))
    return (rp1, rp2) if rp1 <= rp2 else (rp2, rp1)


def _centered_bbox(
    bbox: BoundingBox2D, paper_size_mm: tuple[float, float] | None
) -> BoundingBox2D:
    min_x, min_y, max_x, max_y = bbox
    if not paper_size_mm:
        return bbox
    paper_w, paper_h = paper_size_mm
    return (
        min_x - paper_w / 2,
        min_y - paper_h / 2,
        max_x - paper_w / 2,
        max_y - paper_h / 2,
    )


def _offset_bbox(
    bbox: BoundingBox2D, x_offset: float, y_offset: float
) -> BoundingBox2D:
    return (
        bbox[0] + x_offset,
        bbox[1] + y_offset,
        bbox[2] + x_offset,
        bbox[3] + y_offset,
    )


def _add_bbox_rect(msp, bbox: BoundingBox2D, layer: str) -> None:
    min_x, min_y, max_x, max_y = bbox
    points = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ]
    msp.add_lwpolyline(points, close=True, dxfattribs={"layer": layer})


def _ensure_layer(doc: "ezdxf.document.Drawing", name: str) -> None:
    if not doc.layers.has_entry(name):
        doc.layers.new(name, dxfattribs={"linetype": "CONTINUOUS"})


def _set_dimension_text(dim_entity, text: str) -> None:
    if hasattr(dim_entity, "set_text"):
        dim_entity.set_text(text)
    else:
        dim_entity.dxf.text = text


def add_ezdxf_dimensions(
    doc: "ezdxf.document.Drawing",
    linear_dims: list[LinearDimensionSpec],
    diameter_dims: list[DiameterDimensionSpec],
    layer_name: str = "dims",
) -> None:
    if not doc.layers.has_entry(layer_name):
        doc.layers.new(layer_name, dxfattribs={"linetype": "CONTINUOUS"})
    msp = doc.modelspace()

    for dim in linear_dims:
        override = _dimension_overrides(dim.settings)
        text_style = _ensure_dimension_text_style(doc, dim.settings)
        if text_style:
            override["dimtxsty"] = text_style
        entity = msp.add_linear_dim(
            base=dim.base,
            p1=dim.p1,
            p2=dim.p2,
            angle=dim.angle,
            override=override,
            dxfattribs={"layer": layer_name},
        )
        if dim.label:
            _set_dimension_text(entity, dim.label)
        entity.render()

    for dim in diameter_dims:
        override = _dimension_overrides(dim.settings)
        text_style = _ensure_dimension_text_style(doc, dim.settings)
        if text_style:
            override["dimtxsty"] = text_style
        entity = msp.add_diameter_dim(
            center=dim.center,
            radius=dim.radius,
            angle=dim.angle,
            override=override,
            dxfattribs={"layer": layer_name},
        )
        if dim.label:
            _set_dimension_text(entity, dim.label)
        entity.render()


def render_dxf_to_svg(
    doc: "ezdxf.document.Drawing",
    page_size_mm: tuple[float, float],
    margin_mm: float = 1.0,
) -> str:
    ctx = RenderContext(doc)
    backend = SVGBackend()
    config = Configuration(background_policy=BackgroundPolicy.WHITE)
    page = layout.Page(
        page_size_mm[0],
        page_size_mm[1],
        layout.Units.mm,
        margins=layout.Margins.all(margin_mm),
    )
    Frontend(ctx, backend, config=config).draw_layout(doc.modelspace(), finalize=True)
    return backend.get_string(page)


def add_ezdxf_leader_notes(
    doc: "ezdxf.document.Drawing",
    notes: list[LeaderNoteSpec],
    layer_name: str = "notes",
) -> None:
    if not notes:
        return
    if not doc.layers.has_entry(layer_name):
        doc.layers.new(layer_name, dxfattribs={"linetype": "CONTINUOUS"})

    msp = doc.modelspace()

    for note in notes:
        s = note.settings
        ang = math.radians(note.angle)
        dx = math.cos(ang)
        dy = math.sin(ang)

        # leaderの長さ（だいたい）
        leader_len = s.arrow_size * 4 + s.text_gap * 2 + s.text_height

        tip = note.target
        end = (tip[0] + dx * leader_len, tip[1] + dy * leader_len)

        # leader線
        msp.add_line(tip, end, dxfattribs={"layer": layer_name})

        # 簡易矢印（V字）
        a1 = ang + math.radians(150)
        a2 = ang - math.radians(150)
        p1 = (
            tip[0] + math.cos(a1) * s.arrow_size,
            tip[1] + math.sin(a1) * s.arrow_size,
        )
        p2 = (
            tip[0] + math.cos(a2) * s.arrow_size,
            tip[1] + math.sin(a2) * s.arrow_size,
        )
        msp.add_line(tip, p1, dxfattribs={"layer": layer_name})
        msp.add_line(tip, p2, dxfattribs={"layer": layer_name})

        # テキスト位置
        text_pos = (end[0] + dx * s.text_gap, end[1] + dy * s.text_gap)
        align = (
            TextEntityAlignment.MIDDLE_LEFT
            if dx >= 0
            else TextEntityAlignment.MIDDLE_RIGHT
        )
        text_style = _ensure_dimension_text_style(doc, s)

        text_attribs = {"height": s.text_height, "layer": layer_name}
        if text_style:
            text_attribs["style"] = text_style
        t = msp.add_text(
            note.text,
            dxfattribs=text_attribs,
        )
        t.set_placement(text_pos, align=align)
