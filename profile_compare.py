#!/usr/bin/env python3
"""Compare repeated S3-direct and STAC+stackstac profiling runs.

Example::

    python profile_compare.py config/config_azufre.json \
        --date 2015-08-18 --fresh --repeats 1 \
        --output-dir profiling/azufre_compare_20150818

The comparison invokes ``profile_workflow.py`` once per mode and repetition,
then writes per-run metrics, averaged metrics, and a grouped timing plot.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


PHASES = (
    "download",
    "prepare_bands",
    "auxiliary_information",
    "snowflakes",
    "total",
)


def _mode_config(config_path, sensor, mode, root):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["satellite"] = sensor
    if sensor == "Landsat":
        config["DOWNLOAD_LANDSAT"] = mode
        config["DOWNLOAD_SENTINEL"] = False
    else:
        config["DOWNLOAD_SENTINEL"] = mode
        config["DOWNLOAD_LANDSAT"] = False
    path = root / f"config_{mode.lower().replace('-', '_')}.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def _summary(metrics, mode):
    totals = metrics.get("phase_totals_seconds", {})
    events = metrics.get("events", [])
    event_totals = {}
    for event in events:
        event_totals[event["phase"]] = (
            event_totals.get(event["phase"], 0.0)
            + float(event["duration_seconds"])
        )
    transfer_totals = {}
    for event in events:
        if "bytes_transferred" in event:
            transfer_totals[event["phase"]] = (
                transfer_totals.get(event["phase"], 0)
                + int(event["bytes_transferred"])
            )
    if mode == "STAC-API":
        download = 0.0
    else:
        download = event_totals.get("download", 0.0)
    prepare_bands = (
        event_totals.get("prepare_merged_raw", 0.0)
        + event_totals.get("prepare_merged_stac", 0.0)
    )
    if mode == "STAC-API":
        prepare_bands = max(
            prepare_bands - event_totals.get("stac_item_search", 0.0),
            0.0,
        )
    download_bytes = transfer_totals.get("download", 0)
    prepare_bytes = transfer_totals.get("prepare_merged_stac", 0)
    if mode == "STAC-API":
        prepare_bytes = max(
            prepare_bytes - transfer_totals.get("stac_item_search", 0),
            0,
        )
    auxiliary_information = float(
        totals.get("auxiliary_information", 0.0)
    )
    snowflakes = float(totals.get("snowflakes", 0.0))
    summed_total = download + prepare_bands + auxiliary_information + snowflakes
    return {
        "download": download,
        "prepare_bands": prepare_bands,
        "auxiliary_information": auxiliary_information,
        "snowflakes": snowflakes,
        "total": summed_total,
        "workflow_wall_seconds": float(
            totals.get("workflow_total", 0.0)
        ),
        "download_bytes": download_bytes,
        "prepare_bands_bytes": prepare_bytes,
        "transfer_bytes": download_bytes + prepare_bytes,
    }


def _average(values):
    return {
        key: sum(item[key] for item in values) / len(values)
        for key in values[0]
    }


def _plot(output_dir, averages):
    import matplotlib.pyplot as plt

    labels = (
        "download_preprocess",
        "auxiliary_information",
        "snowflakes",
        "total",
    )
    positions = list(range(len(labels)))
    width = 0.36
    figure, (axis, volume_axis) = plt.subplots(
        1, 2, figsize=(14, 5.5), width_ratios=(1.35, 1)
    )
    display_names = {
        "S3": "S3 direct access",
        "USGS-M2M": "USGS M2M",
        "STAC-API": "STAC+stackstac",
    }
    colors = ("#2c7fb8", "#d95f0e")
    for offset, (mode, color) in enumerate(zip(averages, colors)):
        label = display_names.get(mode, mode)
        values = [
            averages[mode]["download"] + averages[mode]["prepare_bands"],
            averages[mode]["auxiliary_information"],
            averages[mode]["snowflakes"],
            averages[mode]["total"],
        ]
        axis.bar(
            [position + (offset - 0.5) * width for position in positions],
            values,
            width,
            label=label,
            color=color,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            "Download and preprocess",
            "Auxiliary information",
            "SnowFLAKES",
            "Total",
        ]
    )
    axis.set_ylabel("Seconds")
    axis.set_title(
        f"{display_names.get(next(iter(averages)), next(iter(averages)))} "
        "vs STAC+stackstac"
    )
    axis.legend()

    display_names = {
        "S3": "S3 direct access",
        "USGS-M2M": "USGS M2M",
        "STAC-API": "STAC+stackstac",
    }
    colors = ("#2c7fb8", "#d95f0e")
    for offset, (mode, color) in enumerate(zip(averages, colors)):
        label = display_names.get(mode, mode)
        volume_axis.bar(
            [(offset - 0.5) * width],
            [averages[mode]["transfer_bytes"] / 1e6],
            width,
            label=label,
            color=color,
        )
    volume_axis.set_xticks([0])
    volume_axis.set_xticklabels(["Total transferred"])
    volume_axis.set_ylabel("MB")
    volume_axis.set_title("Data transfer volume")
    volume_axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "comparison_phase_durations.png", dpi=160)
    plt.close(figure)


def _plot_transfer(output_dir, averages):
    import matplotlib.pyplot as plt

    labels = ("Download", "Prepare bands", "Total")
    positions = list(range(len(labels)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5))
    for offset, (mode, label, color) in enumerate(
        (
            ("S3", "S3 direct access", "#2c7fb8"),
            ("STAC-API", "STAC+stackstac", "#d95f0e"),
        )
    ):
        values = [
            averages[mode]["download_bytes"],
            averages[mode]["prepare_bands_bytes"],
            averages[mode]["transfer_bytes"],
        ]
        axis.bar(
            [position + (offset - 0.5) * width for position in positions],
            [value / 1e6 for value in values],
            width,
            label=label,
            color=color,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("MB transferred")
    axis.set_title("Data transfer volume")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "comparison_transfer_volume.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--date", required=True)
    parser.add_argument(
        "--sensor",
        choices=("Sentinel-2", "Landsat"),
        default="Sentinel-2",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--no-query", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Reuse existing run metrics and regenerate plots only",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = (
        ("USGS-M2M", "STAC-API")
        if args.sensor == "Landsat"
        else ("S3", "STAC-API")
    )
    runs = []
    averages = {}
    with tempfile.TemporaryDirectory(prefix="snowflakes_profile_compare_") as temp:
        temp_root = Path(temp)
        for mode in modes:
            mode_values = []
            for repetition in range(1, args.repeats + 1):
                run_dir = args.output_dir / mode.lower().replace("-", "_") / f"run_{repetition:02d}"
                run_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = run_dir / "metrics.json"
                if not args.plot_only:
                    mode_config = _mode_config(
                        args.config.resolve(), args.sensor, mode, temp_root
                    )
                    command = [
                        sys.executable,
                        str(Path(__file__).with_name("profile_workflow.py")),
                        str(mode_config),
                        "--date",
                        args.date,
                        "--output-dir",
                        str(run_dir),
                    ]
                    if args.fresh:
                        command.append("--fresh")
                    if args.no_query:
                        command.append("--no-query")
                    print(f"[{mode}] repetition {repetition}/{args.repeats}")
                    subprocess.run(command, check=True)
                elif not metrics_path.is_file():
                    raise FileNotFoundError(
                        f"Missing metrics for plot-only mode: {metrics_path}"
                    )
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                summary = _summary(metrics, mode)
                mode_values.append(summary)
                runs.append({"mode": mode, "repetition": repetition, **summary})
            averages[mode] = _average(mode_values)

    result = {"repeats": args.repeats, "runs": runs, "averages": averages}
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _plot(args.output_dir, averages)
    print(f"Comparison results written to {args.output_dir}")


if __name__ == "__main__":
    main()
