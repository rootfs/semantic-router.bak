"""Config Applicator — write parameter fixes back to config.yaml.

Handles both threshold adjustments (modifying band boundaries in projection
mappings) and weight adjustments (modifying projection score input weights).
Preserves YAML structure and comments where possible.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from .fix_calculator import ParameterFix


def _deep_get(d: dict, keys: list[str]) -> Any:
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k)
        elif isinstance(d, list):
            try:
                d = d[int(k)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return d


def apply_parameter_fixes(
    config_path: str,
    fixes: list[ParameterFix],
    backup: bool = True,
) -> list[str]:
    """Apply parameter fixes to config.yaml.

    Returns list of applied fix descriptions.
    """
    import yaml

    if backup:
        ts = int(time.time())
        backup_path = f"{config_path}.v{ts}.bak"
        shutil.copy2(config_path, backup_path)

    config = yaml.safe_load(Path(config_path).read_text())
    applied: list[str] = []

    for fix in fixes:
        if _apply_single_fix(config, fix):
            applied.append(
                f"[{fix.param_type}] {fix.param_path}: "
                f"{fix.current_value} → {fix.proposed_value} "
                f"(+{fix.probes_fixed}/-{fix.probes_broken} probes)"
            )

    if applied:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False,
                      sort_keys=False, allow_unicode=True)

    return applied


def _apply_single_fix(config: dict, fix: ParameterFix) -> bool:
    """Apply one parameter fix to the config dict."""
    projections = config.get("routing", {}).get("projections", {})

    if fix.param_type == "threshold":
        return _apply_threshold_fix(projections, fix)
    elif fix.param_type == "weight":
        return _apply_weight_fix(projections, fix)
    return False


def _apply_threshold_fix(projections: dict, fix: ParameterFix) -> bool:
    """Modify a band boundary in a projection mapping."""
    path = fix.param_path
    if "mappings[" not in path or "outputs[" not in path:
        return False

    mapping_name = path.split("mappings[")[1].split("]")[0]
    output_name = path.split("outputs[")[1].split("]")[0]

    field_name = "gte"
    if path.endswith(".gt"):
        field_name = "gt"
    elif path.endswith(".lt"):
        field_name = "lt"
    elif path.endswith(".lte"):
        field_name = "lte"

    for mapping in projections.get("mappings", []):
        if mapping.get("name") != mapping_name:
            continue
        for output in mapping.get("outputs", []):
            if output.get("name") != output_name:
                continue
            if field_name in output:
                output[field_name] = fix.proposed_value
                return True
    return False


def _apply_weight_fix(projections: dict, fix: ParameterFix) -> bool:
    """Modify a projection score input weight."""
    path = fix.param_path
    if "scores[" not in path or "inputs[" not in path:
        return False

    score_name = path.split("scores[")[1].split("]")[0]
    signal_key = path.split("inputs[")[1].split("]")[0]

    for score in projections.get("scores", []):
        if score.get("name") != score_name:
            continue
        for inp in score.get("inputs", []):
            inp_key = f"{inp.get('type', '')}:{inp.get('name', '')}"
            if inp_key != signal_key:
                continue
            inp["weight"] = fix.proposed_value
            return True
    return False
