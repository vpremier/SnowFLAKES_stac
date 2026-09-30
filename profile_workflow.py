#!/usr/bin/env python3
"""Profile the SnowFLAKES query, download, loading and processing workflow.

Examples
--------
python profile_workflow.py config/config_mendoza_new.json --date 2020-06-27
python profile_workflow.py config/config_mendoza_new.json --no-query \
    --date 2020-06-27 --output-dir profiling/mendoza_s3

Use ``--fresh`` to reuse cached query CSVs while redownloading and processing
the date in an isolated temporary workspace:

python profile_workflow.py config/config_mendoza_new.json --date 2020-06-27 \
    --fresh --output-dir profiling/mendoza_fresh

The script writes ``metrics.json``, ``metrics.csv``, a phase-duration plot and
``workflow.prof`` (usable with ``python -m pstats`` or SnakeViz). The original
workflow modules are imported and called directly; no production code path is
changed.
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import json
import pstats
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _output_dir(config, requested):
    if requested:
        return Path(requested)
    configured = config.get("output_directory")
    if configured:
        return Path(configured) / "profiling"
    return ROOT / "profiling"


def _timed(records, phase, function):
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            records.append({
                "phase": phase,
                "duration_seconds": time.perf_counter() - started,
                "started_utc": datetime.now(timezone.utc).isoformat(),
            })
    wrapper.__name__ = getattr(function, "__name__", phase)
    wrapper.__doc__ = getattr(function, "__doc__", None)
    return wrapper


def _install_timers(records):
    import main as workflow

    targets = {
        "download": "_download_raw",
        "prepare_merged_stac": "_load_stac_date",
        "prepare_merged_raw": "_load_raw_date",
        "save_bands": "_save_bands",
        "save_composites": "_save_composites",
    }
    originals = {}
    for phase, name in targets.items():
        function = getattr(workflow, name)
        originals[name] = function
        setattr(workflow, name, _timed(records, phase, function))

    try:
        import SnowFLAKES.main_SnowFLAKES as snowflakes
        originals["run_snowflakes"] = snowflakes.run_snowflakes
        snowflakes.run_snowflakes = _timed(
            records, "snowflakes", snowflakes.run_snowflakes
        )
        # These functions are imported as aliases by main_SnowFLAKES, so wrap
        # the aliases there to obtain a stage-by-stage SnowFLAKES breakdown.
        snowflakes_stages = {
            "auxiliary_preparation_total": "create_auxiliary_information",
            "training_collection": "collect_trainings",
            "model_training": "model_training",
            "scf_prediction": "SCF_dist_SV",
            "scf_postprocessing": "remove_low_scf",
            "uncertainty": "get_uncertainty",
            "glacier_check": "snow_around_glacier",
        }
        for phase, name in snowflakes_stages.items():
            key = f"snowflakes::{name}"
            originals[key] = getattr(snowflakes, name)
            setattr(
                snowflakes,
                name,
                _timed(records, phase, getattr(snowflakes, name)),
            )
        # These products are shared by all scenes and are prepared only once.
        # Keep their timings as separate events so they can be reported and
        # removed from both per-scene auxiliary and workflow totals.
        from SnowFLAKES import auxiliary_folder_population as auxiliary
        auxiliary_stages = {
            "dem_preparation": ("load_cdse_collection", "calc_slope_aspect"),
            "water_mask_preparation": ("water_identifier",),
            "glacier_mask_preparation": ("glacier_mask_cutting",),
        }
        for phase, names in auxiliary_stages.items():
            for name in names:
                key = f"auxiliary::{name}"
                originals[key] = getattr(auxiliary, name)
                setattr(
                    auxiliary,
                    name,
                    _timed(records, phase, getattr(auxiliary, name)),
                )
    except ImportError:
        pass
    try:
        from loading import load_stac
        originals["save_stac_bands"] = load_stac.save_stac_bands
        load_stac.save_stac_bands = _timed(
            records, "save_bands", load_stac.save_stac_bands
        )
    except ImportError:
        pass
    return workflow, originals


def _restore_timers(workflow, originals):
    for name, function in originals.items():
        if name == "run_snowflakes":
            import SnowFLAKES.main_SnowFLAKES as snowflakes
            snowflakes.run_snowflakes = function
        elif name.startswith("snowflakes::"):
            import SnowFLAKES.main_SnowFLAKES as snowflakes
            setattr(snowflakes, name.split("::", 1)[1], function)
        elif name.startswith("auxiliary::"):
            from SnowFLAKES import auxiliary_folder_population as auxiliary
            setattr(auxiliary, name.split("::", 1)[1], function)
        elif name == "save_stac_bands":
            from loading import load_stac
            load_stac.save_stac_bands = function
        else:
            setattr(workflow, name, function)


def _write_results(output_dir, metadata, records, profile_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    by_phase = {}
    for record in records:
        by_phase.setdefault(record["phase"], 0.0)
        by_phase[record["phase"]] += record["duration_seconds"]
    grouped_snowflakes = {
        "training_collection",
        "model_training",
        "scf_prediction",
        "scf_postprocessing",
        "glacier_check",
    }
    one_time_phases = {
        "dem_preparation",
        "water_mask_preparation",
        "glacier_mask_preparation",
    }
    one_time_total = sum(by_phase.get(phase, 0.0) for phase in one_time_phases)
    snowflakes_total = sum(by_phase.get(phase, 0.0) for phase in grouped_snowflakes)
    display_totals = {
        phase: value
        for phase, value in by_phase.items()
        if phase not in grouped_snowflakes
        and phase != "auxiliary_preparation_total"
        and phase != "workflow_total_including_one_time"
        and phase != "save_composites"
        and phase != "save_bands"
        and phase != "snowflakes"
        and phase not in one_time_phases
    }
    auxiliary_total = by_phase.get("auxiliary_preparation_total", 0.0)
    auxiliary_information = max(auxiliary_total - one_time_total, 0.0)
    workflow_total = by_phase.get("workflow_total", 0.0)
    workflow_including_one_time = by_phase.get(
        "workflow_total_including_one_time", 0.0
    )
    if workflow_total or workflow_including_one_time:
        metadata["workflow_total_including_one_time_seconds"] = (
            workflow_including_one_time or workflow_total + one_time_total
        )
        display_totals["workflow_total"] = workflow_total
    if snowflakes_total:
        display_totals["snowflakes"] = snowflakes_total
    # Keep the plot readable and deterministic: scene-specific auxiliary work
    # immediately precedes the aggregated SnowFLAKES stages, with workflow
    # total shown afterward.
    ordered_totals = {
        phase: value
        for phase, value in display_totals.items()
        if phase not in {"auxiliary_information", "snowflakes", "workflow_total"}
    }
    ordered_totals["auxiliary_information"] = auxiliary_information
    if snowflakes_total:
        ordered_totals["snowflakes"] = snowflakes_total
    if workflow_total or workflow_including_one_time:
        ordered_totals["workflow_total"] = workflow_total
    display_totals = ordered_totals
    metadata["phase_totals_seconds"] = display_totals
    metadata["snowflakes_breakdown_seconds"] = {
        phase: by_phase.get(phase, 0.0) for phase in sorted(grouped_snowflakes)
    }
    metadata["one_time_preparation_seconds"] = {
        phase: by_phase.get(phase, 0.0) for phase in sorted(one_time_phases)
    }
    metadata["one_time_preparation_total_seconds"] = one_time_total
    metadata["events"] = records
    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["phase", "duration_seconds", "started_utc"])
        writer.writeheader()
        writer.writerows(records)

    try:
        import matplotlib.pyplot as plt
        phases = list(display_totals)
        values = [display_totals[p] for p in phases]
        fig, axis = plt.subplots(figsize=(10, 5))
        axis.bar(phases, values)
        axis.set_ylabel("Seconds")
        axis.set_title("SnowFLAKES workflow profiling")
        axis.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(output_dir / "phase_durations.png", dpi=160)
        plt.close(fig)
    except ImportError:
        metadata["plot"] = "matplotlib is not installed"

    stats = pstats.Stats(str(profile_path))
    stats.sort_stats("cumulative").print_stats(40)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="SnowFLAKES JSON configuration")
    parser.add_argument("--date", required=True, help="Single acquisition date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", help="Directory for profiling results")
    parser.add_argument("--no-query", action="store_true", help="Use existing query CSVs")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Profile from an isolated empty workspace, reusing existing query CSVs",
    )
    parser.add_argument("--no-processing", action="store_true", help="Only profile the query phase")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    try:
        profile_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    except ValueError as error:
        raise SystemExit("--date must use YYYY-MM-DD format") from error
    config["date_start"] = profile_date.isoformat()
    config["date_end"] = (profile_date + timedelta(days=1)).isoformat()
    # The normal CLI loads both environment files before querying/loading.
    # Do the same here because the profiler calls the Python entry points
    # directly rather than spawning ``main.sh``.
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        load_dotenv(config_path.parent / ".env")
    except ImportError:
        pass
    output_dir = _output_dir(config, args.output_dir)
    if not args.output_dir:
        output_dir = output_dir / profile_date.isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_workspace = None
    if args.fresh:
        # Keep the original query cache, but isolate all RAW/MERGED/TILES and
        # SnowFLAKES outputs so the profiling run can never reuse or modify
        # the user's existing products.
        study_name = (
            config.get("study_area")
            or config.get("study_area_name")
            or Path(config["shapefile"]).stem
        )
        original_study = Path(config.get("output_directory", ""))
        if not original_study.is_dir():
            original_study = Path(config["working_directory"]) / study_name
        query_source = original_study / "QUERY"
        temporary_workspace = Path(tempfile.mkdtemp(prefix="snowflakes_profile_"))
        fresh_study = temporary_workspace / study_name
        if query_source.is_dir():
            shutil.copytree(query_source, fresh_study / "QUERY")
        # Reuse expensive scene-independent auxiliary products (DEM, water,
        # glacier and terrain masks). Scene-specific auxiliary outputs are not
        # copied, so the selected date is still fully reprocessed.
        original_auxiliary = original_study / "01_TEST_auxiliary_folder"
        if original_auxiliary.is_dir():
            shutil.copytree(
                original_auxiliary,
                fresh_study / "01_TEST_auxiliary_folder",
            )
        config["working_directory"] = str(temporary_workspace)
        config["output_directory"] = str(fresh_study)
        # Fresh profiling must include a complete SnowFLAKES pass, regardless
        # of whether the production configuration is normally download-only.
        config["run_snowflakes"] = True
        # Reuse cached queries when available; otherwise create them in the
        # temporary workspace. Query time is not included in the profile.
        args.no_query = query_source.is_dir()
    records = []
    metadata = {
        "config": str(config_path),
        "date": profile_date.isoformat(),
        "download_sentinel": config.get("DOWNLOAD_SENTINEL", config.get("download_sentinel")),
        "download_landsat": config.get("DOWNLOAD_LANDSAT", config.get("download_landsat_mode")),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Use a temporary one-date configuration so both query and processing
    # consume exactly the requested acquisition date.
    temporary_config = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="snowflakes_profile_", delete=False
    )
    json.dump(config, temporary_config, indent=2)
    temporary_config.close()
    run_config_path = Path(temporary_config.name)
    try:
      if not args.no_query:
        from data_download import query_available
        started = time.perf_counter()
        query_available._run_from_config(str(run_config_path))
        if not args.fresh:
            records.append({"phase": "query", "duration_seconds": time.perf_counter() - started,
                            "started_utc": datetime.now(timezone.utc).isoformat()})

      if not args.no_processing:
        import main as workflow
        workflow_module, originals = _install_timers(records)
        profiler = cProfile.Profile()
        started = time.perf_counter()
        try:
            profiler.enable()
            workflow.run(str(run_config_path))
        finally:
            profiler.disable()
            _restore_timers(workflow_module, originals)
        workflow_elapsed = time.perf_counter() - started
        one_time_phases = {
            "dem_preparation",
            "water_mask_preparation",
            "glacier_mask_preparation",
        }
        one_time_elapsed = sum(
            record["duration_seconds"]
            for record in records
            if record["phase"] in one_time_phases
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        records.append({
            "phase": "workflow_total_including_one_time",
            "duration_seconds": workflow_elapsed,
            "started_utc": timestamp,
        })
        records.append({
            "phase": "workflow_total",
            "duration_seconds": max(workflow_elapsed - one_time_elapsed, 0.0),
            "started_utc": timestamp,
        })
        profile_path = output_dir / "workflow.prof"
        profiler.dump_stats(str(profile_path))
      else:
          profile_path = output_dir / "workflow.prof"
          cProfile.Profile().dump_stats(str(profile_path))
    finally:
        run_config_path.unlink(missing_ok=True)
        if temporary_workspace is not None:
            shutil.rmtree(temporary_workspace, ignore_errors=True)

    _write_results(output_dir, metadata, records, profile_path)
    print(f"Profiling results written to {output_dir}")


if __name__ == "__main__":
    main()
