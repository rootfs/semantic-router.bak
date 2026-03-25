"""Fix Calculator — compute minimal parameter adjustments from causal reports.

Given a CausalReport identifying a threshold gap or weight issue, computes
the exact parameter change needed to fix the probe. Then uses the
ProjectionModel to check whether that change would break any other probe
(regression checking), all without querying the router.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .causal_tracer import CausalReport, ThresholdGap, ScoreDecomposition
from .projection_model import ProjectionModel, BandOutput
from .trace_collector import ProbeTrace, TraceCache


@dataclass
class ParameterFix:
    """A single proposed parameter adjustment."""

    param_type: str  # "threshold", "weight", "structural"
    param_path: str  # YAML path: e.g. "projections.mappings[security_policy_band].outputs[policy_security_local_only].gte"
    current_value: float
    proposed_value: float
    perturbation: float  # abs(proposed - current)
    reason: str

    # Which probes this fix targets
    target_probes: list[str] = field(default_factory=list)

    # Regression analysis (filled by regression_check)
    probes_fixed: int = 0
    probes_broken: int = 0
    net_improvement: int = 0


@dataclass
class StructuralFix:
    """A structural change (add/remove condition, not numeric)."""

    fix_type: str  # "add_tier", "remove_projection_or"
    decision_name: str
    description: str
    target_probes: list[str] = field(default_factory=list)


def compute_threshold_fixes(
    report: CausalReport,
    model: ProjectionModel,
) -> list[ParameterFix]:
    """Compute threshold adjustment fixes from a causal report."""
    fixes = []

    for gap in report.threshold_gaps:
        if gap.gap >= 0:
            continue

        margin = 0.01
        new_threshold = gap.score_value - margin

        info = model.get_band_for_output(gap.output_name)
        if not info:
            continue
        mapping, band = info

        param_path = (
            f"projections.mappings[{mapping.name}]"
            f".outputs[{gap.output_name}]"
        )
        if band.gte is not None:
            param_path += ".gte"
            current = band.gte
        elif band.gt is not None:
            param_path += ".gt"
            current = band.gt
        else:
            continue

        fixes.append(ParameterFix(
            param_type="threshold",
            param_path=param_path,
            current_value=current,
            proposed_value=round(new_threshold, 4),
            perturbation=abs(current - new_threshold),
            reason=(
                f"Probe {report.probe_id}: {gap.score_name}={gap.score_value:.4f} "
                f"misses threshold {current} by {abs(gap.gap):.4f}"
            ),
            target_probes=[report.probe_id],
        ))

    return fixes


def compute_weight_fixes(
    report: CausalReport,
    model: ProjectionModel,
) -> list[ParameterFix]:
    """Compute projection weight adjustment fixes."""
    fixes = []

    for gap in report.threshold_gaps:
        if gap.gap >= 0:
            continue

        needed_increase = abs(gap.gap) + 0.02

        for decomp in report.score_decompositions:
            if decomp.score_name != gap.score_name:
                continue

            score_def = model._score_by_name.get(decomp.score_name)
            if not score_def:
                continue

            for i, contrib in enumerate(decomp.contributions):
                raw_val = contrib["raw_value"]
                if raw_val <= 0:
                    continue

                current_weight = score_def.inputs[i].weight
                delta_w = needed_increase / raw_val
                new_weight = current_weight + delta_w

                if new_weight > 2.0 or delta_w > current_weight:
                    continue

                fixes.append(ParameterFix(
                    param_type="weight",
                    param_path=(
                        f"projections.scores[{decomp.score_name}]"
                        f".inputs[{contrib['signal']}].weight"
                    ),
                    current_value=current_weight,
                    proposed_value=round(new_weight, 4),
                    perturbation=round(delta_w, 4),
                    reason=(
                        f"Probe {report.probe_id}: increase weight of "
                        f"{contrib['signal']} (raw={raw_val:.3f}) from "
                        f"{current_weight:.3f} to {new_weight:.3f} to push "
                        f"{decomp.score_name} above threshold"
                    ),
                    target_probes=[report.probe_id],
                ))

    return fixes


def compute_fixes_for_report(
    report: CausalReport,
    model: ProjectionModel,
) -> list[ParameterFix | StructuralFix]:
    """Compute all candidate fixes for a single causal report.

    Returns fixes ranked by preference: threshold > weight > structural.
    """
    all_fixes: list[ParameterFix | StructuralFix] = []

    if report.fix_type == "threshold_adjust":
        all_fixes.extend(compute_threshold_fixes(report, model))
        all_fixes.extend(compute_weight_fixes(report, model))

    if report.fix_type in ("structural_add_tier", "unknown"):
        all_fixes.append(StructuralFix(
            fix_type="add_tier",
            decision_name=report.expected_decision,
            description=report.fix_detail,
            target_probes=[report.probe_id],
        ))

    if report.fix_type == "threshold_adjust" and not all_fixes:
        all_fixes.append(StructuralFix(
            fix_type="add_tier",
            decision_name=report.expected_decision,
            description=f"No small parameter fix available: {report.fix_detail}",
            target_probes=[report.probe_id],
        ))

    return all_fixes


def regression_check_threshold(
    fix: ParameterFix,
    cache: TraceCache,
    model: ProjectionModel,
) -> ParameterFix:
    """Check how a threshold fix affects ALL probes using local recomputation.

    Does NOT require querying the router — recomputes routing decisions
    analytically from cached trace data.
    """
    mapping_name = fix.param_path.split("mappings[")[1].split("]")[0] if "mappings[" in fix.param_path else ""
    output_name = fix.param_path.split("outputs[")[1].split("]")[0] if "outputs[" in fix.param_path else ""

    if not mapping_name or not output_name:
        return fix

    mapping = model._mapping_by_source.get(
        next((m.source for m in model.mappings if m.name == mapping_name), "")
    )
    if not mapping:
        return fix

    score_name = mapping.source
    fixed_count = 0
    broken_count = 0

    for probe in cache.probes:
        score_val = probe.projection_scores.get(score_name)
        if score_val is None:
            score_val = model.compute_score(
                score_name, probe.signal_confidences, probe.matched_signals
            )

        currently_matches = score_val >= fix.current_value
        would_match = score_val >= fix.proposed_value

        if not currently_matches and would_match:
            if not probe.correct:
                fixed_count += 1
            else:
                broken_count += 1
        elif currently_matches and not would_match:
            if probe.correct:
                broken_count += 1

    fix.probes_fixed = fixed_count
    fix.probes_broken = broken_count
    fix.net_improvement = fixed_count - broken_count
    return fix


def regression_check_all(
    fixes: list[ParameterFix],
    cache: TraceCache,
    model: ProjectionModel,
) -> list[ParameterFix]:
    """Run regression checks on all parameter fixes."""
    checked = []
    for fix in fixes:
        if fix.param_type == "threshold":
            checked.append(regression_check_threshold(fix, cache, model))
        else:
            checked.append(fix)
    return checked
