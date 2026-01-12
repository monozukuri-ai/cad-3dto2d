import argparse
import os
from cad_3dto2d.converter import convert_2d_drawing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_file", type=str, required=True)
    parser.add_argument(
        "--formats",
        type=str,
        default="png,svg",
        help="Comma-separated list: png,svg,jpg,jpeg,dxf",
    )
    parser.add_argument("--template", type=str, default="A4_LandscapeTD")
    parser.add_argument("--style", type=str, default="iso")
    parser.add_argument("--x_offset", type=float, default=0.0)
    parser.add_argument("--y_offset", type=float, default=0.0)
    parser.add_argument(
        "--side_position",
        type=str,
        choices=("left", "right"),
        default=None,
        help="Place the side view on the left or right of the front view (defaults to template).",
    )
    parser.add_argument(
        "--top_position",
        type=str,
        choices=("up", "down"),
        default=None,
        help="Place the top view above or below the front view (defaults to template).",
    )
    parser.add_argument("--layout_offset_x", type=float, default=0.0)
    parser.add_argument("--layout_offset_y", type=float, default=0.0)
    parser.add_argument(
        "--layout_scale",
        type=float,
        default=None,
        help="Scale factor for the three-view layout (views only; gaps unchanged).",
    )
    parser.add_argument("--add_dimensions", action="store_true")
    args = parser.parse_args()
    formats = [fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()]
    if not formats:
        parser.error("No output formats specified.")
    valid = {"png", "svg", "jpg", "jpeg", "dxf"}
    invalid = sorted(set(formats) - valid)
    if invalid:
        parser.error(f"Unsupported format(s): {', '.join(invalid)}")
    if args.layout_scale is not None and args.layout_scale <= 0:
        parser.error("--layout_scale must be greater than 0.")

    base, _ = os.path.splitext(args.step_file)
    outputs = set([f"{base}.{fmt}" for fmt in formats])
    convert_2d_drawing(
        args.step_file,
        outputs,
        add_template=True,
        template_name=args.template,
        style_name=args.style,
        x_offset=args.x_offset,
        y_offset=args.y_offset,
        side_position=args.side_position,
        top_position=args.top_position,
        layout_offset_x=args.layout_offset_x,
        layout_offset_y=args.layout_offset_y,
        layout_scale=args.layout_scale,
        add_dimensions=args.add_dimensions,
    )


if __name__ == "__main__":
    main()
