"""Fix Selector — choose the best fix from candidates using severity-weighted loss.

Ranks candidate fixes by net improvement weighted by misrouting severity,
then applies fixes greedily (smallest perturbation first among those with
positive net improvement).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .fix_calculator import ParameterFix, StructuralFix


SEVERITY_MATRIX: dict[tuple[str, str], float] = {
    ("local_security_containment", "local_standard"): 10.0,
    ("local_security_containment", "cloud_frontier_reasoning"): 8.0,
    ("local_security_containment", "local_privacy_policy"): 5.0,
    ("local_privacy_policy", "local_standard"): 4.0,
    ("local_privacy_policy", "cloud_frontier_reasoning"): 3.0,
    ("cloud_frontier_reasoning", "local_standard"): 2.0,
    ("local_standard", "local_security_containment"): 0.5,
    ("local_standard", "local_privacy_policy"): 0.5,
    ("local_standard", "cloud_frontier_reasoning"): 1.0,
}

DEFAULT_SEVERITY = 2.0


def misrouting_severity(expected: str, actual: str) -> float:
    if expected == actual:
        return 0.0
    return SEVERITY_MATRIX.get((expected, actual), DEFAULT_SEVERITY)


@dataclass
class RankedFix:
    """A fix with its computed score."""

    fix: ParameterFix | StructuralFix
    score: float
    fix_category: str  # "parameter" or "structural"


def rank_fixes(
    fixes: list[ParameterFix | StructuralFix],
    probe_severity: dict[str, float] | None = None,
) -> list[RankedFix]:
    """Rank fixes by effectiveness.

    Parameter fixes are scored by: net_improvement / perturbation.
    Structural fixes get a flat score based on target probe count.
    All are weighted by the severity of the probes they fix.
    """
    ranked = []

    for fix in fixes:
        if isinstance(fix, ParameterFix):
            if fix.net_improvement <= 0 and fix.probes_fixed == 0:
                continue

            severity_weight = 1.0
            if probe_severity:
                severity_weight = max(
                    probe_severity.get(pid, DEFAULT_SEVERITY)
                    for pid in fix.target_probes
                ) if fix.target_probes else DEFAULT_SEVERITY

            perturbation = max(fix.perturbation, 0.001)
            score = (fix.net_improvement * severity_weight) / perturbation

            ranked.append(RankedFix(
                fix=fix,
                score=score,
                fix_category="parameter",
            ))
        elif isinstance(fix, StructuralFix):
            severity_weight = 1.0
            if probe_severity:
                severity_weight = max(
                    probe_severity.get(pid, DEFAULT_SEVERITY)
                    for pid in fix.target_probes
                ) if fix.target_probes else DEFAULT_SEVERITY

            score = len(fix.target_probes) * severity_weight * 0.5
            ranked.append(RankedFix(
                fix=fix,
                score=score,
                fix_category="structural",
            ))

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


def select_fixes(
    ranked: list[RankedFix],
    max_param_fixes: int = 3,
    max_structural_fixes: int = 5,
) -> tuple[list[ParameterFix], list[StructuralFix]]:
    """Select fixes to apply, respecting limits per category.

    Parameter fixes are preferred over structural fixes when they have
    a positive net improvement, because they are smaller perturbations.
    """
    param_fixes: list[ParameterFix] = []
    structural_fixes: list[StructuralFix] = []

    modified_params: set[str] = set()

    for r in ranked:
        if isinstance(r.fix, ParameterFix):
            if len(param_fixes) >= max_param_fixes:
                continue
            if r.fix.param_path in modified_params:
                continue
            if r.fix.net_improvement < 0:
                continue
            param_fixes.append(r.fix)
            modified_params.add(r.fix.param_path)
        elif isinstance(r.fix, StructuralFix):
            if len(structural_fixes) >= max_structural_fixes:
                continue
            structural_fixes.append(r.fix)

    return param_fixes, structural_fixes
