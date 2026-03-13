from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from society.analysis import build_experiment_report, render_experiment_report
from society.config import AutoCivConfig, load_config
from society.constants import EXPERIMENT_MODES
from society.generation import GenerationRunner
from society.providers import build_provider
from society.storage import StorageManager
from society.utils import write_text


def _resolve_storage_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _resolve_output_prefix(repo_root: Path, root_dir: Path, output_prefix: str | None, generation_ids: list[int]) -> Path:
    if output_prefix is not None:
        path = Path(output_prefix)
        return path if path.is_absolute() else repo_root / path
    return root_dir / "exports" / f"experiment_{generation_ids[0]}_{generation_ids[-1]}"


def _config_with_mode_and_storage(
    config: AutoCivConfig,
    *,
    mode: str,
    root_dir: Path,
    db_path: Path,
) -> AutoCivConfig:
    payload = config.model_dump(mode="json")
    payload["experiment"] = {**payload.get("experiment", {}), "mode": mode}
    payload["storage"] = {
        **payload.get("storage", {}),
        "root_dir": str(root_dir),
        "db_path": str(db_path),
    }
    return AutoCivConfig.model_validate(payload)


def _run_experiment(
    *,
    config: AutoCivConfig,
    repo_root: Path,
    generations: int,
    start_generation_id: int | None,
    seed: int | None,
    output_prefix: str | Path | None,
) -> dict[str, Any]:
    root_dir = _resolve_storage_path(repo_root, config.storage.root_dir)
    db_path = _resolve_storage_path(repo_root, config.storage.db_path)
    storage = StorageManager(root_dir=root_dir, db_path=db_path)
    provider = build_provider(config.provider.name, config.provider.model)
    try:
        runner = GenerationRunner(config=config, storage=storage, provider=provider, repo_root=repo_root)
        start_id = start_generation_id or storage.next_generation_id()
        base_seed = config.generation.seed if seed is None else seed
        generation_ids: list[int] = []
        for offset in range(generations):
            generation_id = start_id + offset
            runner.run(generation_id=generation_id, seed=base_seed + offset)
            generation_ids.append(generation_id)

        report = build_experiment_report(storage, generation_ids)
        output_base = _resolve_output_prefix(repo_root, root_dir, None if output_prefix is None else str(output_prefix), generation_ids)
        json_path = write_text(output_base.with_suffix(".json"), json.dumps(report, indent=2, sort_keys=True))
        md_path = write_text(output_base.with_suffix(".md"), render_experiment_report(report))
        return {
            **report,
            "exports": {
                "json_path": str(json_path),
                "markdown_path": str(md_path),
            },
        }
    finally:
        storage.close()


def _suite_summary(reports_by_mode: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = reports_by_mode.get("isolated_baseline")
    baseline_final = None if baseline is None else baseline["generation_metrics"][-1]
    rows: list[dict[str, Any]] = []
    for mode, report in reports_by_mode.items():
        metrics = report["generation_metrics"]
        final_metric = metrics[-1]
        average_transfer = round(
            statistics.fmean(metric["memorial_transfer_score"] for metric in metrics),
            4,
        )
        total_hidden_failures = sum(
            metric["diffusion_alerts"] + metric["anti_corruption"] + metric["taboo_recurrence"]
            for metric in metrics
        )
        row = {
            "mode": mode,
            "generation_ids": report["generation_ids"],
            "final_public_eval_average": final_metric["public_eval_average"],
            "final_memorial_transfer_score": final_metric["memorial_transfer_score"],
            "average_memorial_transfer_score": average_transfer,
            "final_warned_lineages": final_metric["warned_lineages"],
            "final_strategy_drift_rate": final_metric["strategy_drift_rate"],
            "final_propagation_blocked": final_metric["propagation_blocked"],
            "final_review_only": final_metric["review_only"],
            "total_hidden_failures": total_hidden_failures,
            "exports": report.get("exports", {}),
        }
        if baseline_final is not None:
            row["delta_vs_isolated_public_eval_average"] = round(
                final_metric["public_eval_average"] - baseline_final["public_eval_average"],
                4,
            )
            row["delta_vs_isolated_memorial_transfer_score"] = round(
                final_metric["memorial_transfer_score"] - baseline_final["memorial_transfer_score"],
                4,
            )
            row["delta_vs_isolated_strategy_drift_rate"] = round(
                final_metric["strategy_drift_rate"] - baseline_final["strategy_drift_rate"],
                4,
            )
        rows.append(row)
    return rows


def render_hypothesis_suite_report(report: dict[str, Any]) -> str:
    lines = [
        f"# Hypothesis Suite {report['generation_span'][0]}-{report['generation_span'][1]}",
        "",
        "## Modes",
        "",
    ]
    for item in report["mode_summaries"]:
        lines.append(
            f"- {item['mode']}: final_public_eval_average={item['final_public_eval_average']} "
            f"final_memorial_transfer_score={item['final_memorial_transfer_score']} "
            f"average_memorial_transfer_score={item['average_memorial_transfer_score']} "
            f"final_warned_lineages={item['final_warned_lineages']} "
            f"final_strategy_drift_rate={item['final_strategy_drift_rate']} "
            f"final_propagation_blocked={item['final_propagation_blocked']} "
            f"total_hidden_failures={item['total_hidden_failures']}"
        )
        if "delta_vs_isolated_public_eval_average" in item:
            lines.append(
                f"  delta_vs_isolated: public_eval_average={item['delta_vs_isolated_public_eval_average']} "
                f"memorial_transfer_score={item['delta_vs_isolated_memorial_transfer_score']} "
                f"strategy_drift_rate={item['delta_vs_isolated_strategy_drift_rate']}"
            )
    return "\n".join(lines) + "\n"


def run_experiment_from_config(
    *,
    config_path: str | Path,
    repo_root: str | Path = ".",
    generations: int,
    start_generation_id: int | None = None,
    seed: int | None = None,
    output_prefix: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    if generations <= 0:
        raise ValueError("generations must be greater than zero")

    repo_root = Path(repo_root)
    resolved_config_path = repo_root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    config = load_config(resolved_config_path)
    if mode is not None:
        root_dir = _resolve_storage_path(repo_root, config.storage.root_dir)
        db_path = _resolve_storage_path(repo_root, config.storage.db_path)
        config = _config_with_mode_and_storage(config, mode=mode, root_dir=root_dir, db_path=db_path)
    return _run_experiment(
        config=config,
        repo_root=repo_root,
        generations=generations,
        start_generation_id=start_generation_id,
        seed=seed,
        output_prefix=output_prefix,
    )


def run_hypothesis_suite_from_config(
    *,
    config_path: str | Path,
    repo_root: str | Path = ".",
    generations: int,
    modes: list[str] | None = None,
    start_generation_id: int | None = None,
    seed: int | None = None,
    output_prefix: str | None = None,
) -> dict[str, Any]:
    if generations <= 0:
        raise ValueError("generations must be greater than zero")

    repo_root = Path(repo_root)
    resolved_config_path = repo_root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    base_config = load_config(resolved_config_path)
    selected_modes = list(modes or EXPERIMENT_MODES)
    base_root_dir = _resolve_storage_path(repo_root, base_config.storage.root_dir)
    suite_output_base = (
        (Path(output_prefix) if output_prefix is not None else base_root_dir / "exports" / "hypothesis_suite")
        if output_prefix is None or Path(output_prefix).is_absolute()
        else repo_root / output_prefix
    )
    reports_by_mode: dict[str, dict[str, Any]] = {}
    for mode in selected_modes:
        mode_root_dir = base_root_dir / "hypothesis_modes" / mode
        mode_config = _config_with_mode_and_storage(
            base_config,
            mode=mode,
            root_dir=mode_root_dir,
            db_path=mode_root_dir / "db.sqlite",
        )
        mode_report = _run_experiment(
            config=mode_config,
            repo_root=repo_root,
            generations=generations,
            start_generation_id=start_generation_id,
            seed=seed,
            output_prefix=suite_output_base / mode / "experiment",
        )
        reports_by_mode[mode] = mode_report

    generation_span = (
        start_generation_id or 1,
        (start_generation_id or 1) + generations - 1,
    )
    suite_report = {
        "modes": selected_modes,
        "generation_span": generation_span,
        "mode_summaries": _suite_summary(reports_by_mode),
        "reports_by_mode": reports_by_mode,
    }
    json_path = write_text(
        suite_output_base.with_suffix(".json"),
        json.dumps(suite_report, indent=2, sort_keys=True),
    )
    md_path = write_text(
        suite_output_base.with_suffix(".md"),
        render_hypothesis_suite_report(suite_report),
    )
    return {
        **suite_report,
        "exports": {
            "json_path": str(json_path),
            "markdown_path": str(md_path),
        },
    }
