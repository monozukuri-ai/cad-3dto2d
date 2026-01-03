from __future__ import annotations

from build123d import ExportDXF, LineType, Unit
import ezdxf
from ezdxf.addons.drawing import Frontend, RenderContext, layout
from ezdxf.addons.drawing.config import BackgroundPolicy, Configuration
from ezdxf.addons.drawing.svg import SVGBackend

from ..annotations.dimensions import DiameterDimensionSpec, DimensionSettings, LinearDimensionSpec
from ..types import Shape

_LINE_TYPES: dict[str, LineType] = {
    "visible": LineType.CONTINUOUS,
    "hidden": LineType.DASHED,
    "template": LineType.CONTINUOUS,
}


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
        exporter.add_layer(layer_name, line_type=resolved_types.get(layer_name, LineType.CONTINUOUS))
    for layer_name in ordered_layers:
        exporter.add_shape(layers[layer_name], layer=layer_name)
    exporter.write(output_file)


def _dimension_overrides(settings: DimensionSettings) -> dict[str, float | int]:
    return {
        "dimtxt": settings.text_height,
        "dimasz": settings.arrow_size,
        "dimexo": settings.extension_gap,
        "dimexe": settings.extension_gap,
        "dimgap": settings.text_gap,
        "dimdec": settings.decimal_places,
    }


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
    margin_mm: float = 0.0,
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
