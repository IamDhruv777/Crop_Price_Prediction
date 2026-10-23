"""Independently verify the numbers and behaviour claimed by the dashboard.

Run this from the repository root:

    python scripts/verify_dashboard_claims.py

It recomputes the deployed metrics directly from
``data/deployed_test_predictions.csv``, compares them with
``data/metrics_deployed.json``, checks conformal coverage, prints artifact
hashes, and exercises the 1/7/30-day date-planner path. No numbers are trusted
from the dashboard or README.

This script verifies a historical backtest. It does not prove that the system
can forecast live prices, and it does not turn interpolated planning days into
validated daily forecasts.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)

from src.inference.forecast_path import build_forecast_path  # noqa: E402

PREDS_PATH = ROOT / "data" / "deployed_test_predictions.csv"
METRICS_PATH = ROOT / "data" / "metrics_deployed.json"
REGISTRY_PATH = ROOT / "models" / "trained" / "deployment_registry.json"
HORIZONS = ((1, "1-Day"), (7, "7-Day"), (30, "30-Day"))
ARTIFACTS = (
    PREDS_PATH,
    METRICS_PATH,
    REGISTRY_PATH,
    ROOT / "data" / "mondrian_conformal_quantiles.json",
    ROOT / "models" / "trained" / "xgboost_1d.joblib",
    ROOT / "models" / "trained" / "xgboost_7d.joblib",
    ROOT / "data" / "processed" / "inference_feature_store.csv",
)

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    if not condition:
        FAILURES.append(label)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def recompute(df: pd.DataFrame, horizon: int) -> dict[str, float]:
    suffix = f"{horizon}d"
    target = f"target_price_{suffix}"
    predicted = f"predicted_price_{suffix}"
    lower = f"lower_bound_{suffix}"
    upper = f"upper_bound_{suffix}"

    required = [target, predicted, lower, upper]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for {horizon}-day metrics: {missing}")

    usable = df.dropna(subset=required).copy()
    y_true = usable[target].to_numpy(dtype=float)
    y_pred = usable[predicted].to_numpy(dtype=float)

    residual = y_true - y_pred
    covered = (y_true >= usable[lower].to_numpy(dtype=float)) & (
        y_true <= usable[upper].to_numpy(dtype=float)
    )
    usable["_covered"] = covered

    return {
        "n": len(usable),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual**2) / np.sum((y_true - y_true.mean()) ** 2)),
        "coverage": float(covered.mean() * 100.0),
        "macro_coverage": float(usable.groupby("commodity")["_covered"].mean().mean() * 100.0),
    }


def print_header() -> None:
    print("=" * 78)
    print("DASHBOARD CLAIM VERIFIER - historical Agmarknet holdout")
    print("=" * 78)
    print(f"Repository : {ROOT}")
    print(f"Commit     : {git('rev-parse', '--short', 'HEAD')}")
    status = git("status", "--short").splitlines()
    print(f"Worktree   : {len(status)} changed/untracked path(s), not a clean release")
    print()


def print_artifacts() -> None:
    print("[1] ARTIFACT INTEGRITY (SHA-256)")
    for path in ARTIFACTS:
        if not path.exists():
            check(path.relative_to(ROOT).as_posix(), False, "missing")
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path.relative_to(ROOT).as_posix():<58} {size_mb:7.2f} MB  {sha256(path)}")
    print()


def verify_metrics(df: pd.DataFrame, metrics: dict) -> None:
    print("[2] RECOMPUTED ACCURACY FROM data/deployed_test_predictions.csv")
    print(
        f"  {'Horizon':<8} {'N':>9} {'RMSE artifact':>14} {'RMSE recomputed':>15} "
        f"{'MAE artifact':>13} {'MAE recomputed':>14} {'R2 artifact':>11} {'R2 recomputed':>13}"
    )

    for horizon, label in HORIZONS:
        calc = recompute(df, horizon)
        artifact = metrics["by_horizon"][label]
        print(
            f"  {label:<8} {calc['n']:>9,} {artifact['RMSE']:>14.2f} {calc['rmse']:>15.2f} "
            f"{artifact['MAE']:>13.2f} {calc['mae']:>14.2f} {artifact['R2']:>11.4f} {calc['r2']:>13.4f}"
        )
        check(f"{label} sample count matches artifact", calc["n"] == int(artifact["n_samples"]),
              f"{calc['n']:,} rows")
        check(f"{label} RMSE matches artifact", abs(calc["rmse"] - artifact["RMSE"]) <= 0.011,
              f"diff {calc['rmse'] - artifact['RMSE']:+.4f}")
        check(f"{label} MAE matches artifact", abs(calc["mae"] - artifact["MAE"]) <= 0.011,
              f"diff {calc['mae'] - artifact['MAE']:+.4f}")
        check(f"{label} R2 matches artifact", abs(calc["r2"] - artifact["R2"]) <= 0.00011,
              f"diff {calc['r2'] - artifact['R2']:+.6f}")
    print()


def verify_coverage(df: pd.DataFrame, metrics: dict) -> None:
    print("[3] RECOMPUTED CONFORMAL 95% COVERAGE")
    print(
        f"  {'Horizon':<8} {'pooled artifact':>16} {'pooled recomputed':>18} "
        f"{'macro artifact':>15} {'macro recomputed':>17}"
    )
    for horizon, label in HORIZONS:
        calc = recompute(df, horizon)
        pooled = metrics["by_horizon"][label]["Conformal_Coverage_95"]
        macro = metrics["mondrian_averages"][label]["macro_coverage"]
        print(
            f"  {label:<8} {pooled:>16.2f} {calc['coverage']:>18.2f} "
            f"{macro:>15.2f} {calc['macro_coverage']:>17.2f}"
        )
        check(f"{label} pooled coverage matches artifact", abs(calc["coverage"] - pooled) <= 0.011,
              f"diff {calc['coverage'] - pooled:+.4f}")
        check(f"{label} macro coverage matches artifact", abs(calc["macro_coverage"] - macro) <= 0.011,
              f"diff {calc['macro_coverage'] - macro:+.4f}")
    print()


def verify_registry() -> None:
    print("[4] DEPLOYED MODEL REGISTRY")
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    for label in ("1-Day", "7-Day", "30-Day"):
        item = registry[label]
        model_path = ROOT / item["model_file"]
        print(
            f"  {label:<8} {item['model_name']:<30} version={item['model_version']:<18} "
            f"anchor={item['anchor_formula']}"
        )
        check(f"{label} model artifact exists", model_path.exists(),
              item["model_file"])
    print()


def verify_date_planner(df: pd.DataFrame) -> None:
    print("[5] DATE PLANNER: DIRECT ANCHORS VS INTERPOLATION")
    sample = df[(df["commodity"] == "Wheat") & (df["market"] == "Devala")].sort_values("date")
    if sample.empty:
        sample = df.dropna(subset=["predicted_price_30d"]).sort_values("date")
    latest = sample.iloc[-1]
    path = build_forecast_path(latest, max_days=30)

    print(
        f"  Sample: {latest['commodity']} / {latest['market']} / "
        f"latest available day {pd.to_datetime(latest['date']).date()}"
    )
    print(f"  {'Day':>4} {'Date':<12} {'Price':>10} {'Low':>10} {'High':>10}  Kind")
    for day in (1, 7, 13, 14, 19, 30):
        row = path.loc[path["day"] == day]
        if row.empty:
            print(f"  {day:>4} {'-':<12} {'not available':>10}")
            continue
        item = row.iloc[0]
        print(
            f"  {day:>4} {pd.Timestamp(item['date']).date().isoformat():<12} "
            f"{item['price']:>10.2f} {item['low']:>10.2f} {item['high']:>10.2f}  {item['kind']}"
        )

    check("day 1 is a direct model anchor", path.loc[path["day"] == 1, "kind"].item() == "validated")
    check("day 7 is a direct model anchor", path.loc[path["day"] == 7, "kind"].item() == "validated")
    check("day 30 is a direct model anchor", path.loc[path["day"] == 30, "kind"].item() == "validated")
    check("day 13 is explicitly interpolated", path.loc[path["day"] == 13, "kind"].item() == "interpolated")
    check("day 14 is explicitly interpolated", path.loc[path["day"] == 14, "kind"].item() == "interpolated")
    check("day 19 is explicitly interpolated", path.loc[path["day"] == 19, "kind"].item() == "interpolated")
    check("interpolated prices stay inside their bands",
          bool(((path["low"] <= path["price"]) & (path["price"] <= path["high"])).all()))
    print(
        "  Note: only days 1, 7 and 30 are model outputs. Days 2-6 and 8-29 are a straight\n"
        "        line between those anchors, shown as a planning aid, not a daily model.\n"
    )


def verify_map_component() -> None:
    print("[6] MAP COMPONENT GUARDS")
    from dashboard.components.unified_zoom_map import build_unified_map_html

    frame = pd.DataFrame(
        [
            {
                "state": "Punjab",
                "district": "Ludhiana",
                "market_name": "Ludhiana",
                "modal_price": 1800.0,
                "predicted_price_7d": 1830.0,
                "latitude": 30.901,
                "longitude": 75.857,
            }
        ]
    )
    html = build_unified_map_html("Wheat", frame, is_hi=False)
    checks = {
        "street-level style function": "getStreetLevelDistrictStyle" in html,
        "no-data district stays light": "fillOpacity: noData ? 0.12 : 0.28" in html,
        "high-zoom hover stays light": "fillOpacity: zoomedIn ? 0.38 : 0.92" in html,
        "permanent mandi price labels": "mandi-price-badge" in html,
        "labels refresh on zoom": "refreshMandiLabels" in html,
        "new level-3 badge text": "District Price Bands + Local Mandi Pins" in html,
    }
    for label, passed in checks.items():
        check(label, passed)
    check("empty map has an honest no-data message",
          "No mandi data available" in build_unified_map_html("Wheat", pd.DataFrame(), is_hi=False))
    print()


def main() -> int:
    print_header()
    for required in (PREDS_PATH, METRICS_PATH, REGISTRY_PATH):
        if not required.exists():
            print(f"ERROR: required file missing: {required}")
            return 1

    print_artifacts()
    predictions = pd.read_csv(PREDS_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(predictions):,} prediction rows from {PREDS_PATH.name}\n")

    verify_metrics(predictions, metrics)
    verify_coverage(predictions, metrics)
    verify_registry()
    verify_date_planner(predictions)
    verify_map_component()

    print("=" * 78)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        print("=" * 78)
        return 1

    print("RESULT: ALL CHECKS PASSED")
    print("What this proves : the dashboard metrics can be recomputed from the shipped CSV,")
    print("                   the conformal bands match, and the map/date-planner code is present.")
    print("What this does NOT prove: live-price accuracy, a guaranteed 95% band for an individual")
    print("                   farmer, or genuine daily models for days 2-6 and 8-29.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
