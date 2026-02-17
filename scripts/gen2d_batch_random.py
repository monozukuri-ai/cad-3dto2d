import argparse
import concurrent.futures
import csv
import multiprocessing as mp
import os
import random
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

from tqdm.auto import tqdm

from cad_3dto2d.converter import convert_2d_drawing

SUPPORTED_IMAGE_FORMATS = ("png", "svg", "jpg", "jpeg")
STEP_SUFFIXES = {".step", ".stp"}


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    side_position = job["side_position"] or None
    top_position = job["top_position"] or None
    try:
        convert_2d_drawing(
            str(job["step_file"]),
            [str(job["image_file"]), str(job["dxf_file"])],
            add_template=True,
            template_name=str(job["template"]),
            style_name=str(job["style"]),
            layout_offset_x=float(job["layout_offset_x"]),
            layout_offset_y=float(job["layout_offset_y"]),
            layout_scale=float(job["layout_scale"]),
            side_position=side_position,
            top_position=top_position,
            add_dimensions=bool(job["add_dimensions"]),
        )
    except Exception as exc:
        return {
            "job_id": int(job["job_id"]),
            "status": "error",
            "error": str(exc),
        }
    return {"job_id": int(job["job_id"]), "status": "ok", "error": ""}

def _repo_asset_dir(asset_name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "cad_3dto2d" / asset_name


def _load_index_entries(asset_dir: Path, key: str) -> list[str]:
    index_path = asset_dir / "index.yaml"
    if yaml is None or not index_path.exists():
        return []
    data = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    entries = data.get(key, data)
    if not isinstance(entries, dict):
        return []
    return sorted(str(name) for name, value in entries.items() if isinstance(value, dict))


def _yaml_stems(asset_dir: Path) -> list[str]:
    return sorted(
        path.stem for path in asset_dir.glob("*.yaml") if path.name != "index.yaml"
    )


def _available_templates() -> list[str]:
    templates_dir = _repo_asset_dir("templates")
    names = set(_yaml_stems(templates_dir))
    names.update(_load_index_entries(templates_dir, "templates"))
    return sorted(names)


def _available_styles() -> list[str]:
    styles_dir = _repo_asset_dir("styles")
    names = set(_yaml_stems(styles_dir))
    names.update(_load_index_entries(styles_dir, "styles"))
    return sorted(names)


def _parse_csv_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_pool(
    requested: str | None,
    available: list[str],
    label: str,
    parser: argparse.ArgumentParser,
) -> list[str]:
    if not available:
        parser.error(f"No {label} entries found.")
    if not requested:
        return available
    names = _parse_csv_names(requested)
    if not names:
        parser.error(f"Empty {label} list is not allowed.")
    invalid = sorted(set(names) - set(available))
    if invalid:
        parser.error(f"Unknown {label}(s): {', '.join(invalid)}")
    return names


def _discover_step_files(input_dir: Path, recursive: bool) -> list[Path]:
    walker = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        path for path in walker if path.is_file() and path.suffix.lower() in STEP_SUFFIXES
    )


def _print_available(label: str, values: list[str]) -> None:
    if not values:
        print(f"No {label} found.")
        return
    print(f"Available {label}:")
    for value in values:
        print(f"- {value}")


def _validate_ranges(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.variants_per_step <= 0:
        parser.error("--variants-per-step must be greater than 0.")
    if args.layout_scale_min <= 0 or args.layout_scale_max <= 0:
        parser.error("--layout-scale-min/max must be greater than 0.")
    if args.layout_scale_min > args.layout_scale_max:
        parser.error("--layout-scale-min must be <= --layout-scale-max.")
    if args.layout_offset_x_min > args.layout_offset_x_max:
        parser.error("--layout-offset-x-min must be <= --layout-offset-x-max.")
    if args.layout_offset_y_min > args.layout_offset_y_max:
        parser.error("--layout-offset-y-min must be <= --layout-offset-y-max.")


def _default_style_pool(styles: list[str]) -> list[str]:
    arrow_styles = [name for name in styles if name.startswith("iso_arrow_")]
    return arrow_styles if arrow_styles else styles


def _manifest_path(output_dir: Path, manifest_arg: str | None) -> Path:
    if manifest_arg:
        return Path(manifest_arg)
    return output_dir / "manifest.csv"


def _create_progress(total_jobs: int):
    return tqdm(
        total=total_jobs,
        unit="job",
        dynamic_ncols=True,
        desc="gen2d",
        bar_format=(
            "{desc}: {percentage:6.2f}%|{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        ),
    )


def _resolve_workers(value: int | None, parser: argparse.ArgumentParser) -> int:
    if value is not None and value <= 0:
        parser.error("--workers must be greater than 0.")
    if value is not None:
        return value
    cpu = os.cpu_count() or 1
    return min(cpu, 4)


def _build_jobs(
    step_files: list[Path],
    input_dir: Path,
    output_dir: Path,
    variants_per_step: int,
    image_format: str,
    template_pool: list[str],
    style_pool: list[str],
    rng: random.Random,
    layout_scale_min: float,
    layout_scale_max: float,
    layout_offset_x_min: float,
    layout_offset_x_max: float,
    layout_offset_y_min: float,
    layout_offset_y_max: float,
    random_side_position: bool,
    random_top_position: bool,
    add_dimensions: bool,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    job_id = 0

    for step_path in step_files:
        relative_step = step_path.relative_to(input_dir)
        output_parent = output_dir / relative_step.parent
        output_parent.mkdir(parents=True, exist_ok=True)
        stem = relative_step.stem

        for variant in range(1, variants_per_step + 1):
            job_id += 1
            suffix = f"__v{variant:03d}" if variants_per_step > 1 else ""
            image_out = output_parent / f"{stem}{suffix}.{image_format}"
            dxf_out = output_parent / f"{stem}{suffix}.dxf"
            side_position = (
                rng.choice(("left", "right")) if random_side_position else ""
            )
            top_position = rng.choice(("up", "down")) if random_top_position else ""
            jobs.append(
                {
                    "job_id": job_id,
                    "job_label": f"{relative_step} [{variant}/{variants_per_step}]",
                    "step_file": str(step_path),
                    "relative_step": str(relative_step),
                    "variant": variant,
                    "image_file": str(image_out),
                    "dxf_file": str(dxf_out),
                    "template": rng.choice(template_pool),
                    "style": rng.choice(style_pool),
                    "layout_scale": rng.uniform(layout_scale_min, layout_scale_max),
                    "layout_offset_x": rng.uniform(
                        layout_offset_x_min, layout_offset_x_max
                    ),
                    "layout_offset_y": rng.uniform(
                        layout_offset_y_min, layout_offset_y_max
                    ),
                    "side_position": side_position,
                    "top_position": top_position,
                    "add_dimensions": add_dimensions,
                }
            )
    return jobs


def _progress_postfix(inflight_jobs: list[dict[str, Any]]) -> str:
    if not inflight_jobs:
        return "idle"
    labels = [str(job["job_label"]) for job in inflight_jobs[:3]]
    if len(inflight_jobs) > 3:
        labels.append(f"+{len(inflight_jobs) - 3} more")
    return f"inflight={len(inflight_jobs)} {' | '.join(labels)}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate randomized 2D drawings (image + DXF) for all STEP files in a "
            "directory."
        )
    )
    parser.add_argument("--input_dir", type=Path, default=Path("."))
    parser.add_argument("--output_dir", type=Path, default=Path("batch_outputs"))
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for STEP/STP files under input_dir.",
    )
    parser.add_argument("--variants-per-step", type=int, default=1)
    parser.add_argument(
        "--image-format",
        type=str,
        choices=SUPPORTED_IMAGE_FORMATS,
        default="png",
        help="Image format in addition to DXF.",
    )
    parser.add_argument(
        "--templates",
        type=str,
        default=None,
        help="Comma-separated template names. Default: all templates.",
    )
    parser.add_argument(
        "--styles",
        type=str,
        default=None,
        help=(
            "Comma-separated style names. Default: styles prefixed with "
            "'iso_arrow_' (or all styles if unavailable)."
        ),
    )
    parser.add_argument("--layout-scale-min", type=float, default=0.8)
    parser.add_argument("--layout-scale-max", type=float, default=1.2)
    parser.add_argument("--layout-offset-x-min", type=float, default=-10.0)
    parser.add_argument("--layout-offset-x-max", type=float, default=10.0)
    parser.add_argument("--layout-offset-y-min", type=float, default=-10.0)
    parser.add_argument("--layout-offset-y-max", type=float, default=10.0)
    parser.add_argument(
        "--random-side-position",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Randomly choose side_position (left/right) for each drawing.",
    )
    parser.add_argument(
        "--random-top-position",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Randomly choose top_position (up/down) for each drawing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: min(cpu_count, 4)).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="CSV output path. Default: <output_dir>/manifest.csv",
    )
    parser.add_argument(
        "--add-dimensions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable dimensions (default: enabled).",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue processing even if one file fails (default: enabled).",
    )
    parser.add_argument("--list-templates", action="store_true")
    parser.add_argument("--list-styles", action="store_true")
    args = parser.parse_args()

    _validate_ranges(args, parser)

    all_templates = _available_templates()
    all_styles = _available_styles()
    if args.list_templates:
        _print_available("templates", all_templates)
    if args.list_styles:
        _print_available("styles", all_styles)
    if args.list_templates or args.list_styles:
        return 0

    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    template_pool = _resolve_pool(args.templates, all_templates, "template", parser)
    if args.styles:
        style_pool = _resolve_pool(args.styles, all_styles, "style", parser)
    else:
        style_pool = _default_style_pool(all_styles)
        if not style_pool:
            parser.error("No styles found.")

    step_files = _discover_step_files(args.input_dir, recursive=args.recursive)
    if not step_files:
        print(f"No STEP files found under: {args.input_dir}")
        return 0

    workers = _resolve_workers(args.workers, parser)
    manifest_path = _manifest_path(args.output_dir, args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    jobs = _build_jobs(
        step_files=step_files,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        variants_per_step=args.variants_per_step,
        image_format=args.image_format,
        template_pool=template_pool,
        style_pool=style_pool,
        rng=rng,
        layout_scale_min=args.layout_scale_min,
        layout_scale_max=args.layout_scale_max,
        layout_offset_x_min=args.layout_offset_x_min,
        layout_offset_x_max=args.layout_offset_x_max,
        layout_offset_y_min=args.layout_offset_y_min,
        layout_offset_y_max=args.layout_offset_y_max,
        random_side_position=args.random_side_position,
        random_top_position=args.random_top_position,
        add_dimensions=args.add_dimensions,
    )
    total_jobs = len(jobs)
    success = 0
    failed = 0
    rows: list[dict[str, str | int | float]] = []
    stop_requested = False

    print(
        "Batch generation start:",
        f"step_files={len(step_files)}",
        f"variants_per_step={args.variants_per_step}",
        f"total_jobs={total_jobs}",
        f"seed={args.seed}",
        f"workers={workers}",
    )
    print(f"Templates: {', '.join(template_pool)}")
    print(f"Styles: {', '.join(style_pool)}")
    progress = _create_progress(total_jobs)

    try:
        mp_context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_context,
        ) as executor:
            inflight: dict[concurrent.futures.Future, dict[str, Any]] = {}
            job_cursor = 0

            while job_cursor < len(jobs) and len(inflight) < workers:
                job = jobs[job_cursor]
                job_cursor += 1
                inflight[executor.submit(_run_job, job)] = job

            while inflight:
                progress.set_postfix_str(_progress_postfix(list(inflight.values())))
                done, _ = concurrent.futures.wait(
                    inflight.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for future in done:
                    job = inflight.pop(future)
                    progress.set_description_str(str(job["job_label"]))
                    try:
                        result = future.result()
                        status = str(result["status"])
                        error = str(result["error"])
                    except Exception as exc:
                        status = "error"
                        error = str(exc)

                    if status == "ok":
                        success += 1
                    else:
                        failed += 1
                        progress.write(
                            f"ERROR [{job['job_id']}/{total_jobs}] {job['job_label']}: {error}"
                        )
                        if not args.continue_on_error:
                            stop_requested = True

                    rows.append(
                        {
                            "job_id": int(job["job_id"]),
                            "step_file": str(job["step_file"]),
                            "variant": int(job["variant"]),
                            "image_file": str(job["image_file"]),
                            "dxf_file": str(job["dxf_file"]),
                            "template": str(job["template"]),
                            "style": str(job["style"]),
                            "layout_scale": round(float(job["layout_scale"]), 6),
                            "layout_offset_x": round(float(job["layout_offset_x"]), 6),
                            "layout_offset_y": round(float(job["layout_offset_y"]), 6),
                            "side_position": str(job["side_position"]),
                            "top_position": str(job["top_position"]),
                            "seed": args.seed,
                            "add_dimensions": int(args.add_dimensions),
                            "status": status,
                            "error": error,
                        }
                    )
                    progress.update(1)

                if stop_requested:
                    continue

                while job_cursor < len(jobs) and len(inflight) < workers:
                    job = jobs[job_cursor]
                    job_cursor += 1
                    inflight[executor.submit(_run_job, job)] = job
    finally:
        progress.close()

    rows.sort(key=lambda row: int(row["job_id"]))

    fieldnames = [
        "job_id",
        "step_file",
        "variant",
        "image_file",
        "dxf_file",
        "template",
        "style",
        "layout_scale",
        "layout_offset_x",
        "layout_offset_y",
        "side_position",
        "top_position",
        "seed",
        "add_dimensions",
        "status",
        "error",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        "Batch generation done:",
        f"success={success}",
        f"failed={failed}",
        f"manifest={manifest_path}",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
