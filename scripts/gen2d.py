import argparse
import os
from cad_3dto2d.converter import convert_2d_drawing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_file", type=str, required=True)
    parser.add_argument(
        "--formats",
        type=str,
        default="png",
        help="Comma-separated list: png,svg,jpg,jpeg,dxf",
    )
    parser.add_argument("--template", type=str, default="A4_LandscapeTD")
    parser.add_argument("--style", type=str, default="iso")
    parser.add_argument("--x_offset", type=float, default=0.0)
    parser.add_argument("--y_offset", type=float, default=0.0)
    parser.add_argument("--add_dimensions", action="store_true")
    args = parser.parse_args()
    formats = [fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()]
    if not formats:
        parser.error("No output formats specified.")
    valid = {"png", "svg", "jpg", "jpeg", "dxf"}
    invalid = sorted(set(formats) - valid)
    if invalid:
        parser.error(f"Unsupported format(s): {', '.join(invalid)}")

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
        add_dimensions=args.add_dimensions,
    )


if __name__ == "__main__":
    main()
