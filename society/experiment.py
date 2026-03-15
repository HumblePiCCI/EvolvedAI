from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, cast

from society.analysis import build_experiment_report, render_experiment_report
from society.config import AutoCivConfig, load_config
from society.constants import EXPERIMENT_MODES
from society.generation import GenerationRunner
from society.providers import build_provider
from society.storage import StorageManager
from society.utils import write_text

COMPARATIVE_BATCH_KEYS = (
    "failure_recurrence_rate",
    "cooperative_truthfulness_score",
    "correction_acceptance_average",
    "hidden_eval_failure_rate",
    "strategy_drift_rate",
    "lineage_diffusion_index",
)

T_CRITICAL_95_BY_DF = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.16,
    14: 2.145,
    15: 2.131,
    16: 2.12,
    17: 2.11,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.08,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.06,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


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


def _config_with_provider_overrides(
    config: AutoCivConfig,
    *,
    provider_name: str | None = None,
    provider_model: str | None = None,
    provider_base_url: str | None = None,
    provider_timeout_seconds: float | None = None,
    provider_reasoning_effort: str | None = None,
    provider_max_output_tokens: int | None = None,
) -> AutoCivConfig:
    if all(
        value is None
        for value in (
            provider_name,
            provider_model,
            provider_base_url,
            provider_timeout_seconds,
            provider_reasoning_effort,
            provider_max_output_tokens,
        )
    ):
        return config

    payload = config.model_dump(mode="json")
    provider_payload = payload.get("provider", {})
    if provider_name is not None:
        provider_payload["name"] = provider_name
    if provider_model is not None:
        provider_payload["model"] = provider_model
    if provider_base_url is not None:
        provider_payload["base_url"] = provider_base_url
    if provider_timeout_seconds is not None:
        provider_payload["timeout_seconds"] = provider_timeout_seconds
    if provider_reasoning_effort is not None:
        provider_payload["reasoning_effort"] = provider_reasoning_effort
    if provider_max_output_tokens is not None:
        provider_payload["max_output_tokens"] = provider_max_output_tokens
    payload["provider"] = provider_payload
    return AutoCivConfig.model_validate(payload)


def _run_experiment(
    *,
    config: AutoCivConfig,
    repo_root: Path,
    generations: int,
    start_generation_id: int | None,
    seed: int | None,
    output_prefix: str | Path | None,
    seed_sensitive_provider: bool = False,
) -> dict[str, Any]:
    root_dir = _resolve_storage_path(repo_root, config.storage.root_dir)
    db_path = _resolve_storage_path(repo_root, config.storage.db_path)
    storage = StorageManager(root_dir=root_dir, db_path=db_path)
    provider = build_provider(
        config.provider.name,
        config.provider.model,
        seed_sensitive=seed_sensitive_provider,
        api_key_env=config.provider.api_key_env,
        base_url=config.provider.base_url,
        timeout_seconds=config.provider.timeout_seconds,
        reasoning_effort=config.provider.reasoning_effort,
        max_output_tokens=config.provider.max_output_tokens,
    )
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


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def _confidence_multiplier_95(sample_size: int) -> float:
    if sample_size <= 1:
        return 0.0
    degrees_of_freedom = sample_size - 1
    return T_CRITICAL_95_BY_DF.get(degrees_of_freedom, 1.96)


def _metric_band(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "sample_variance": 0.0,
            "sample_stddev": 0.0,
            "standard_error": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "ci95_margin": 0.0,
        }

    mean_value = statistics.fmean(values)
    median_value = statistics.median(values)
    minimum = min(values)
    maximum = max(values)
    sample_variance = statistics.variance(values) if len(values) > 1 else 0.0
    sample_stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    standard_error = sample_stddev / math.sqrt(len(values)) if len(values) > 1 else 0.0
    ci95_margin = _confidence_multiplier_95(len(values)) * standard_error if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": round(mean_value, 4),
        "median": round(float(median_value), 4),
        "min": round(minimum, 4),
        "max": round(maximum, 4),
        "sample_variance": round(sample_variance, 6),
        "sample_stddev": round(sample_stddev, 6),
        "standard_error": round(standard_error, 6),
        "ci95_low": round(mean_value - ci95_margin, 4),
        "ci95_high": round(mean_value + ci95_margin, 4),
        "ci95_margin": round(ci95_margin, 4),
    }


def _metric_bands(metric_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not metric_rows:
        return {}
    keys = [key for key, value in metric_rows[0].items() if isinstance(value, (int, float))]
    return {
        key: _metric_band([float(item[key]) for item in metric_rows])
        for key in keys
    }


def _delta_bands(
    seed_metrics: list[dict[str, Any]],
    baseline_seed_metrics: list[dict[str, Any]],
    delta_fields: dict[str, tuple[str, str]],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    baseline_by_seed = {int(item["seed"]): item for item in baseline_seed_metrics}
    series: dict[str, list[float]] = {field: [] for field in delta_fields}
    for item in seed_metrics:
        seed = int(item["seed"])
        baseline = baseline_by_seed[seed]
        for field, (direction, metric_key) in delta_fields.items():
            value = float(item[metric_key])
            baseline_value = float(baseline[metric_key])
            delta = value - baseline_value if direction == "mode_minus_baseline" else baseline_value - value
            series[field].append(delta)

    means = {field: _metric_band(values)["mean"] for field, values in series.items()}
    bands = {field: _metric_band(values) for field, values in series.items()}
    return means, bands


def _provider_snapshot(config: AutoCivConfig, *, seed_sensitive_provider: bool) -> dict[str, Any]:
    return {
        "name": config.provider.name,
        "model": config.provider.model,
        "base_url": config.provider.base_url,
        "timeout_seconds": config.provider.timeout_seconds,
        "reasoning_effort": config.provider.reasoning_effort,
        "max_output_tokens": config.provider.max_output_tokens,
        "seed_sensitive_provider": seed_sensitive_provider,
    }


def _report_metric_snapshot(report: dict[str, Any]) -> dict[str, float]:
    metrics = report["generation_metrics"]
    final_metric = metrics[-1]
    recurrence_values = [
        metric["failure_recurrence_rate"]
        for metric in metrics
        if metric.get("failure_recurrence_opportunities", 0) > 0
    ]
    return {
        "public_eval_average": _mean([metric["public_eval_average"] for metric in metrics]),
        "cooperative_truthfulness_score": _mean(
            [metric["cooperative_truthfulness_score"] for metric in metrics]
        ),
        "correction_acceptance_average": _mean(
            [metric["correction_acceptance_average"] for metric in metrics]
        ),
        "hidden_eval_failure_rate": _mean([metric["hidden_eval_failure_rate"] for metric in metrics]),
        "failure_recurrence_rate": _mean(recurrence_values),
        "strategy_drift_rate": _mean([metric["strategy_drift_rate"] for metric in metrics]),
        "lineage_diffusion_index": _mean([metric["lineage_diffusion_index"] for metric in metrics]),
        "final_public_eval_average": final_metric["public_eval_average"],
        "final_cooperative_truthfulness_score": final_metric["cooperative_truthfulness_score"],
        "final_correction_acceptance_average": final_metric["correction_acceptance_average"],
        "final_hidden_eval_failure_rate": final_metric["hidden_eval_failure_rate"],
        "final_failure_recurrence_rate": final_metric["failure_recurrence_rate"],
        "final_strategy_drift_rate": final_metric["strategy_drift_rate"],
        "final_lineage_diffusion_index": final_metric["lineage_diffusion_index"],
    }


def _resolve_output_base(repo_root: Path, root_dir: Path, output_prefix: str | None, default_name: str) -> Path:
    if output_prefix is None:
        return root_dir / "exports" / default_name
    output_path = Path(output_prefix)
    return output_path if output_path.is_absolute() else repo_root / output_path


def _comparative_batch_summary(
    reports_by_seed: dict[int, dict[str, dict[str, Any]]],
    selected_modes: list[str],
) -> list[dict[str, Any]]:
    seed_metrics_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in selected_modes}
    for seed, mode_reports in sorted(reports_by_seed.items()):
        for mode in selected_modes:
            report = mode_reports[mode]
            seed_metrics_by_mode[mode].append(
                {
                    "seed": seed,
                    **_report_metric_snapshot(report),
                    "generation_ids": report["generation_ids"],
                    "exports": report.get("exports", {}),
                }
            )

    summaries: list[dict[str, Any]] = []
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in selected_modes:
        seed_metrics = seed_metrics_by_mode[mode]
        batch_means = {
            key: _mean([item[key] for item in seed_metrics])
            for key in (
                "public_eval_average",
                "cooperative_truthfulness_score",
                "correction_acceptance_average",
                "hidden_eval_failure_rate",
                "failure_recurrence_rate",
                "strategy_drift_rate",
                "lineage_diffusion_index",
                "final_public_eval_average",
                "final_cooperative_truthfulness_score",
                "final_correction_acceptance_average",
                "final_hidden_eval_failure_rate",
                "final_failure_recurrence_rate",
                "final_strategy_drift_rate",
                "final_lineage_diffusion_index",
            )
        }
        batch_bands = _metric_bands(seed_metrics)
        summary = {
            "mode": mode,
            "seed_count": len(seed_metrics),
            "seed_metrics": seed_metrics,
            "batch_means": batch_means,
            "batch_bands": batch_bands,
        }
        summaries.append(summary)
        by_mode[mode] = summary

    isolated_baseline_summary = by_mode.get("isolated_baseline")
    inheritance_off_summary = by_mode.get("inheritance_off")
    for summary in summaries:
        if isolated_baseline_summary is not None:
            deltas, bands = _delta_bands(
                cast(list[dict[str, Any]], summary["seed_metrics"]),
                cast(list[dict[str, Any]], isolated_baseline_summary["seed_metrics"]),
                {
                    "cooperative_truthfulness_score_delta": ("mode_minus_baseline", "cooperative_truthfulness_score"),
                    "correction_acceptance_average_delta": ("mode_minus_baseline", "correction_acceptance_average"),
                    "hidden_eval_failure_rate_delta": ("mode_minus_baseline", "hidden_eval_failure_rate"),
                    "hidden_eval_failure_rate_reduction": ("baseline_minus_mode", "hidden_eval_failure_rate"),
                    "strategy_drift_rate_delta": ("mode_minus_baseline", "strategy_drift_rate"),
                    "lineage_diffusion_index_delta": ("mode_minus_baseline", "lineage_diffusion_index"),
                },
            )
            summary["deltas_vs_isolated_baseline"] = deltas
            summary["delta_bands_vs_isolated_baseline"] = bands
        if inheritance_off_summary is not None:
            deltas, bands = _delta_bands(
                cast(list[dict[str, Any]], summary["seed_metrics"]),
                cast(list[dict[str, Any]], inheritance_off_summary["seed_metrics"]),
                {
                    "failure_recurrence_rate_delta": ("mode_minus_baseline", "failure_recurrence_rate"),
                    "recurrence_reduction": ("baseline_minus_mode", "failure_recurrence_rate"),
                    "cooperative_truthfulness_score_delta": ("mode_minus_baseline", "cooperative_truthfulness_score"),
                    "correction_acceptance_average_delta": ("mode_minus_baseline", "correction_acceptance_average"),
                    "hidden_eval_failure_rate_reduction": ("baseline_minus_mode", "hidden_eval_failure_rate"),
                    "strategy_drift_rate_delta": ("mode_minus_baseline", "strategy_drift_rate"),
                    "lineage_diffusion_index_delta": ("mode_minus_baseline", "lineage_diffusion_index"),
                },
            )
            summary["deltas_vs_inheritance_off"] = deltas
            summary["delta_bands_vs_inheritance_off"] = bands
    return summaries


def _suite_summary(reports_by_mode: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = reports_by_mode.get("isolated_baseline")
    baseline_final = None if baseline is None else baseline["generation_metrics"][-1]
    rows: list[dict[str, Any]] = []
    for mode, report in reports_by_mode.items():
        metrics = report["generation_metrics"]
        final_metric = metrics[-1]
        metric_snapshot = _report_metric_snapshot(report)
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
            "final_cooperative_truthfulness_score": final_metric["cooperative_truthfulness_score"],
            "final_correction_acceptance_average": final_metric["correction_acceptance_average"],
            "final_hidden_eval_failure_rate": final_metric["hidden_eval_failure_rate"],
            "final_failure_recurrence_rate": final_metric["failure_recurrence_rate"],
            "final_lineage_diffusion_index": final_metric["lineage_diffusion_index"],
            "average_cooperative_truthfulness_score": metric_snapshot["cooperative_truthfulness_score"],
            "average_correction_acceptance_average": metric_snapshot["correction_acceptance_average"],
            "average_hidden_eval_failure_rate": metric_snapshot["hidden_eval_failure_rate"],
            "average_failure_recurrence_rate": metric_snapshot["failure_recurrence_rate"],
            "average_lineage_diffusion_index": metric_snapshot["lineage_diffusion_index"],
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


def render_comparative_batches_report(report: dict[str, Any]) -> str:
    lines = [
        f"# Comparative Batches {report['generation_span'][0]}-{report['generation_span'][1]}",
        "",
        f"- seeds: {', '.join(str(seed) for seed in report['seeds'])}",
        f"- provider: {report['provider']['name']}:{report['provider']['model']}",
        f"- interval_method: {report['statistical_method']}",
        "",
        "## Metric Definitions",
        "",
        "- failure_recurrence_rate: share of child lineages repeating a parent failure when parent failures existed.",
        "- cooperative_truthfulness_score: mean of honesty, calibration, and citation_fidelity public eval scores.",
        "- correction_acceptance_average: mean correction_acceptance public eval score.",
        "- hidden_eval_failure_rate: failed hidden evals divided by total hidden eval records.",
        "- strategy_drift_rate and lineage_diffusion_index: raw drift/diffusion signals, reported as deltas rather than signed improvements.",
        "",
        "## Modes",
        "",
    ]
    for item in report["mode_summaries"]:
        batch = item["batch_means"]
        batch_bands = item["batch_bands"]
        lines.append(
            f"- {item['mode']}: recurrence_rate={batch['failure_recurrence_rate']} "
            f"cooperative_truthfulness={batch['cooperative_truthfulness_score']} "
            f"correction_acceptance={batch['correction_acceptance_average']} "
            f"hidden_eval_failure_rate={batch['hidden_eval_failure_rate']} "
            f"strategy_drift_rate={batch['strategy_drift_rate']} "
            f"lineage_diffusion_index={batch['lineage_diffusion_index']}"
        )
        lines.append(
            "  95%_bands: "
            f"recurrence=[{batch_bands['failure_recurrence_rate']['ci95_low']}, {batch_bands['failure_recurrence_rate']['ci95_high']}] "
            f"truthfulness=[{batch_bands['cooperative_truthfulness_score']['ci95_low']}, {batch_bands['cooperative_truthfulness_score']['ci95_high']}] "
            f"correction=[{batch_bands['correction_acceptance_average']['ci95_low']}, {batch_bands['correction_acceptance_average']['ci95_high']}] "
            f"hidden_failures=[{batch_bands['hidden_eval_failure_rate']['ci95_low']}, {batch_bands['hidden_eval_failure_rate']['ci95_high']}] "
            f"drift=[{batch_bands['strategy_drift_rate']['ci95_low']}, {batch_bands['strategy_drift_rate']['ci95_high']}] "
            f"diffusion=[{batch_bands['lineage_diffusion_index']['ci95_low']}, {batch_bands['lineage_diffusion_index']['ci95_high']}]"
        )
        isolated = item.get("deltas_vs_isolated_baseline")
        isolated_bands = item.get("delta_bands_vs_isolated_baseline")
        if isolated is not None:
            lines.append(
                "  vs_isolated_baseline: "
                f"cooperative_truthfulness={isolated['cooperative_truthfulness_score_delta']} "
                f"correction_acceptance={isolated['correction_acceptance_average_delta']} "
                f"hidden_eval_failure_rate_delta={isolated['hidden_eval_failure_rate_delta']} "
                f"hidden_eval_failure_rate_reduction={isolated['hidden_eval_failure_rate_reduction']} "
                f"strategy_drift_rate={isolated['strategy_drift_rate_delta']} "
                f"lineage_diffusion_index={isolated['lineage_diffusion_index_delta']}"
            )
        if isolated_bands is not None:
            lines.append(
                "  vs_isolated_95%_bands: "
                f"truthfulness=[{isolated_bands['cooperative_truthfulness_score_delta']['ci95_low']}, {isolated_bands['cooperative_truthfulness_score_delta']['ci95_high']}] "
                f"correction=[{isolated_bands['correction_acceptance_average_delta']['ci95_low']}, {isolated_bands['correction_acceptance_average_delta']['ci95_high']}] "
                f"hidden_reduction=[{isolated_bands['hidden_eval_failure_rate_reduction']['ci95_low']}, {isolated_bands['hidden_eval_failure_rate_reduction']['ci95_high']}] "
                f"drift=[{isolated_bands['strategy_drift_rate_delta']['ci95_low']}, {isolated_bands['strategy_drift_rate_delta']['ci95_high']}] "
                f"diffusion=[{isolated_bands['lineage_diffusion_index_delta']['ci95_low']}, {isolated_bands['lineage_diffusion_index_delta']['ci95_high']}]"
            )
        inheritance_off = item.get("deltas_vs_inheritance_off")
        inheritance_off_bands = item.get("delta_bands_vs_inheritance_off")
        if inheritance_off is not None:
            lines.append(
                "  vs_inheritance_off: "
                f"failure_recurrence_rate={inheritance_off['failure_recurrence_rate_delta']} "
                f"recurrence_reduction={inheritance_off['recurrence_reduction']} "
                f"cooperative_truthfulness={inheritance_off['cooperative_truthfulness_score_delta']} "
                f"correction_acceptance={inheritance_off['correction_acceptance_average_delta']} "
                f"hidden_eval_failure_rate_reduction={inheritance_off['hidden_eval_failure_rate_reduction']} "
                f"strategy_drift_rate={inheritance_off['strategy_drift_rate_delta']} "
                f"lineage_diffusion_index={inheritance_off['lineage_diffusion_index_delta']}"
            )
        if inheritance_off_bands is not None:
            lines.append(
                "  vs_inheritance_off_95%_bands: "
                f"recurrence_reduction=[{inheritance_off_bands['recurrence_reduction']['ci95_low']}, {inheritance_off_bands['recurrence_reduction']['ci95_high']}] "
                f"truthfulness=[{inheritance_off_bands['cooperative_truthfulness_score_delta']['ci95_low']}, {inheritance_off_bands['cooperative_truthfulness_score_delta']['ci95_high']}] "
                f"correction=[{inheritance_off_bands['correction_acceptance_average_delta']['ci95_low']}, {inheritance_off_bands['correction_acceptance_average_delta']['ci95_high']}] "
                f"hidden_reduction=[{inheritance_off_bands['hidden_eval_failure_rate_reduction']['ci95_low']}, {inheritance_off_bands['hidden_eval_failure_rate_reduction']['ci95_high']}] "
                f"drift=[{inheritance_off_bands['strategy_drift_rate_delta']['ci95_low']}, {inheritance_off_bands['strategy_drift_rate_delta']['ci95_high']}] "
                f"diffusion=[{inheritance_off_bands['lineage_diffusion_index_delta']['ci95_low']}, {inheritance_off_bands['lineage_diffusion_index_delta']['ci95_high']}]"
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
    provider_name: str | None = None,
    provider_model: str | None = None,
    provider_base_url: str | None = None,
    provider_timeout_seconds: float | None = None,
    provider_reasoning_effort: str | None = None,
    provider_max_output_tokens: int | None = None,
) -> dict[str, Any]:
    if generations <= 0:
        raise ValueError("generations must be greater than zero")

    repo_root = Path(repo_root)
    resolved_config_path = repo_root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    config = load_config(resolved_config_path)
    config = _config_with_provider_overrides(
        config,
        provider_name=provider_name,
        provider_model=provider_model,
        provider_base_url=provider_base_url,
        provider_timeout_seconds=provider_timeout_seconds,
        provider_reasoning_effort=provider_reasoning_effort,
        provider_max_output_tokens=provider_max_output_tokens,
    )
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
    provider_name: str | None = None,
    provider_model: str | None = None,
    provider_base_url: str | None = None,
    provider_timeout_seconds: float | None = None,
    provider_reasoning_effort: str | None = None,
    provider_max_output_tokens: int | None = None,
) -> dict[str, Any]:
    if generations <= 0:
        raise ValueError("generations must be greater than zero")

    repo_root = Path(repo_root)
    resolved_config_path = repo_root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    base_config = load_config(resolved_config_path)
    base_config = _config_with_provider_overrides(
        base_config,
        provider_name=provider_name,
        provider_model=provider_model,
        provider_base_url=provider_base_url,
        provider_timeout_seconds=provider_timeout_seconds,
        provider_reasoning_effort=provider_reasoning_effort,
        provider_max_output_tokens=provider_max_output_tokens,
    )
    selected_modes = list(modes or EXPERIMENT_MODES)
    base_root_dir = _resolve_storage_path(repo_root, base_config.storage.root_dir)
    suite_output_base = _resolve_output_base(repo_root, base_root_dir, output_prefix, "hypothesis_suite")
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
        "provider": _provider_snapshot(base_config, seed_sensitive_provider=False),
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


def run_comparative_batches_from_config(
    *,
    config_path: str | Path,
    repo_root: str | Path = ".",
    generations: int,
    seeds: list[int],
    modes: list[str] | None = None,
    start_generation_id: int | None = None,
    output_prefix: str | None = None,
    provider_name: str | None = None,
    provider_model: str | None = None,
    provider_base_url: str | None = None,
    provider_timeout_seconds: float | None = None,
    provider_reasoning_effort: str | None = None,
    provider_max_output_tokens: int | None = None,
) -> dict[str, Any]:
    if generations <= 0:
        raise ValueError("generations must be greater than zero")
    if not seeds:
        raise ValueError("seeds must contain at least one seed")

    repo_root = Path(repo_root)
    resolved_config_path = repo_root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    base_config = load_config(resolved_config_path)
    base_config = _config_with_provider_overrides(
        base_config,
        provider_name=provider_name,
        provider_model=provider_model,
        provider_base_url=provider_base_url,
        provider_timeout_seconds=provider_timeout_seconds,
        provider_reasoning_effort=provider_reasoning_effort,
        provider_max_output_tokens=provider_max_output_tokens,
    )
    selected_modes = list(modes or EXPERIMENT_MODES)
    base_root_dir = _resolve_storage_path(repo_root, base_config.storage.root_dir)
    comparative_output_base = _resolve_output_base(
        repo_root,
        base_root_dir,
        output_prefix,
        "comparative_batches",
    )

    reports_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    seed_runs: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        mode_reports: dict[str, dict[str, Any]] = {}
        seed_snapshot: dict[str, Any] = {}
        for mode in selected_modes:
            mode_root_dir = base_root_dir / "comparative_batches" / f"seed_{seed}" / mode
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
                output_prefix=comparative_output_base / f"seed_{seed}" / mode / "experiment",
                seed_sensitive_provider=True,
            )
            mode_reports[mode] = mode_report
            seed_snapshot[mode] = {
                "generation_ids": mode_report["generation_ids"],
                "metric_snapshot": _report_metric_snapshot(mode_report),
                "exports": mode_report.get("exports", {}),
            }
        reports_by_seed[seed] = mode_reports
        seed_runs[str(seed)] = seed_snapshot

    generation_span = (
        start_generation_id or 1,
        (start_generation_id or 1) + generations - 1,
    )
    comparative_report = {
        "seeds": seeds,
        "modes": selected_modes,
        "generation_span": generation_span,
        "provider": _provider_snapshot(base_config, seed_sensitive_provider=True),
        "statistical_method": "seed-level mean with two-sided 95% t-intervals",
        "mode_summaries": _comparative_batch_summary(reports_by_seed, selected_modes),
        "seed_runs": seed_runs,
    }
    json_path = write_text(
        comparative_output_base.with_suffix(".json"),
        json.dumps(comparative_report, indent=2, sort_keys=True),
    )
    md_path = write_text(
        comparative_output_base.with_suffix(".md"),
        render_comparative_batches_report(comparative_report),
    )
    return {
        **comparative_report,
        "exports": {
            "json_path": str(json_path),
            "markdown_path": str(md_path),
        },
    }
