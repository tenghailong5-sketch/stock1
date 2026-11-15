"""Runtime loader for certified M4 point-in-time factor inputs.

The loader accepts only explicitly supplied PIT archives. It never upgrades
retrieval-time snapshots or current industry mappings into historical evidence.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.model.m4_factors import build_m4_pit_factor_snapshot


_INPUT_FILES = {
    "stock_bars": "stock_bars.parquet",
    "stock_returns": "stock_returns.parquet",
    "industry_returns": "industry_returns.parquet",
    "market_returns": "market_returns.parquet",
}


def _read_frame(root: Path, stem: str) -> pd.DataFrame:
    parquet = root / f"{stem}.parquet"
    csv = root / f"{stem}.csv"
    if parquet.is_file():
        return pd.read_parquet(parquet)
    if csv.is_file():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"missing_input:{stem}.parquet|{stem}.csv")


def build_runtime_m4_pit_snapshot(
    *, decision_time, input_dir: str | Path | None = None,
    expected_universe_size: int | None = None,
    coverage_floor: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load certified PIT inputs and build the M4 cross-section, fail-closed."""
    root_value = input_dir or os.getenv("FHM_M4_PIT_INPUT_DIR", "").strip()
    if not root_value:
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "NOT_CONFIGURED",
            "builder_invoked": False,
            "historical_pit_eligible": False,
            "missing_inputs": list(_INPUT_FILES.values()),
            "reason": "FHM_M4_PIT_INPUT_DIR_not_configured",
        }
    root = Path(root_value).expanduser()
    expected = expected_universe_size
    if expected is None:
        raw_expected = os.getenv("FHM_M4_EXPECTED_UNIVERSE_SIZE", "").strip()
        try:
            expected = int(raw_expected)
        except (TypeError, ValueError):
            expected = None
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "INVALID_CONFIGURATION",
            "builder_invoked": False,
            "historical_pit_eligible": False,
            "reason": "FHM_M4_EXPECTED_UNIVERSE_SIZE_must_be_positive_integer",
            "input_dir": str(root),
        }

    frames = {}
    missing = []
    try:
        for key, filename in _INPUT_FILES.items():
            stem = Path(filename).stem
            try:
                frames[key] = _read_frame(root, stem)
            except FileNotFoundError:
                missing.append(filename)
        hard_gate_path = root / "hard_gate_snapshot.parquet"
        hard_gate_csv = root / "hard_gate_snapshot.csv"
        hard_gate = None
        if hard_gate_path.is_file():
            hard_gate = pd.read_parquet(hard_gate_path)
        elif hard_gate_csv.is_file():
            hard_gate = pd.read_csv(hard_gate_csv)
        if missing:
            return pd.DataFrame(), pd.DataFrame(), {
                "status": "MISSING_CERTIFIED_INPUTS",
                "builder_invoked": False,
                "historical_pit_eligible": False,
                "missing_inputs": missing,
                "input_dir": str(root),
            }
        snapshot, audit = build_m4_pit_factor_snapshot(
            frames["stock_bars"], frames["stock_returns"],
            frames["industry_returns"], frames["market_returns"],
            decision_time=decision_time,
            expected_universe_size=expected,
            coverage_floor=coverage_floor,
            hard_gate_snapshot=hard_gate,
        )
        return snapshot, audit, {
            "status": "BUILT",
            "builder_invoked": True,
            "historical_pit_eligible": bool(
                not snapshot.empty
                and snapshot.get("historical_pit_eligible", pd.Series(dtype=bool)).eq(True).all()
            ),
            "input_dir": str(root),
            "expected_universe_size": expected,
            "coverage_floor": float(coverage_floor),
            "snapshot_rows": int(len(snapshot)),
            "factor_audit_rows": int(len(audit)),
            "hard_gate_snapshot_supplied": hard_gate is not None,
        }
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "INVALID_CERTIFIED_INPUTS",
            "builder_invoked": False,
            "historical_pit_eligible": False,
            "input_dir": str(root),
            "reason": f"{type(exc).__name__}:{str(exc)[:300]}",
        }
