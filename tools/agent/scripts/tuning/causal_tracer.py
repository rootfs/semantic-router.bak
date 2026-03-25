"""Causal Tracer — walk eval trace trees to identify root causes of misrouting.

For each misrouted probe, traces both the WINNING (wrong) decision and the
EXPECTED (should-have-won) decision to identify:
  - Which specific leaf node caused the expected decision to fail
  - The confidence gap (how far from matching)
  - Which projection score / threshold is responsible
  - Candidate parameter adjustments
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .projection_model import ProjectionModel
from .trace_collector import ProbeTrace


@dataclass
class LeafCause:
    """A specific leaf node that caused a decision to fail or succeed."""

    signal_type: str
    signal_name: str
    matched: bool
    confidence: float
    role: str  # "failing_leaf" | "winning_leaf"
    decision_name: str


@dataclass
class ThresholdGap:
    """The distance between a projection score and its band boundary."""

    score_name: str
    score_value: float
    output_name: str
    threshold: float
    gap: float  # positive = above threshold, negative = below


@dataclass
class ScoreDecomposition:
    """Breakdown of a projection score into its weighted input contributions."""

    score_name: str
    total: float
    contributions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CausalReport:
    """Full causal analysis of a single misrouted probe."""

    probe_id: str
    expected_decision: str
    actual_decision: str
    confidence: float

    # Why the wrong decision won
    winning_leaves: list[LeafCause] = field(default_factory=list)

    # Why the expected decision lost
    failing_leaves: list[LeafCause] = field(default_factory=list)

    # Threshold analysis for projection-type failures
    threshold_gaps: list[ThresholdGap] = field(default_factory=list)

    # Score decompositions for relevant projections
    score_decompositions: list[ScoreDecomposition] = field(default_factory=list)

    # Suggested fix type
    fix_type: str = ""  # "threshold_adjust", "weight_adjust", "structural_add_tier", "structural_remove_or"
    fix_detail: str = ""


def _walk_trace_leaves(
    node: dict[str, Any],
    decision_name: str,
    role: str,
) -> list[LeafCause]:
    """Recursively extract all leaf nodes from a trace tree."""
    if not node:
        return []

    if node.get("node_type") == "leaf":
        return [LeafCause(
            signal_type=node.get("signal_type", ""),
            signal_name=node.get("signal_name", ""),
            matched=node.get("matched", False),
            confidence=node.get("confidence", 0.0),
            role=role,
            decision_name=decision_name,
        )]

    results = []
    for child in node.get("children", []):
        results.extend(_walk_trace_leaves(child, decision_name, role))
    return results


def _find_first_failing_leaf(node: dict[str, Any]) -> dict[str, Any] | None:
    """Find the first leaf that caused an AND chain to fail."""
    if not node:
        return None

    if node.get("node_type") == "leaf":
        if not node.get("matched", False):
            return node
        return None

    if node.get("node_type") == "AND":
        for child in node.get("children", []):
            result = _find_first_failing_leaf(child)
            if result:
                return result

    if node.get("node_type") == "OR":
        all_failed = all(
            not c.get("matched", False) for c in node.get("children", [])
        )
        if all_failed:
            best_conf = -1.0
            best_child = None
            for child in node.get("children", []):
                if child.get("node_type") == "leaf":
                    if child.get("confidence", 0) > best_conf:
                        best_conf = child.get("confidence", 0)
                        best_child = child
                else:
                    result = _find_first_failing_leaf(child)
                    if result:
                        return result
            return best_child

    return None


def analyze_probe(
    probe: ProbeTrace,
    projection_model: ProjectionModel,
) -> CausalReport:
    """Produce a full causal analysis for a misrouted probe."""
    report = CausalReport(
        probe_id=probe.probe_id,
        expected_decision=probe.expected_decision,
        actual_decision=probe.actual_decision,
        confidence=probe.confidence,
    )

    actual_trace = None
    expected_trace = None
    for dt in probe.eval_traces:
        if dt.get("decision_name") == probe.actual_decision:
            actual_trace = dt
        if dt.get("decision_name") == probe.expected_decision:
            expected_trace = dt

    if actual_trace and actual_trace.get("root_trace"):
        report.winning_leaves = _walk_trace_leaves(
            actual_trace["root_trace"], probe.actual_decision, "winning_leaf"
        )

    if expected_trace and expected_trace.get("root_trace"):
        report.failing_leaves = _walk_trace_leaves(
            expected_trace["root_trace"], probe.expected_decision, "failing_leaf"
        )

        failing_node = _find_first_failing_leaf(expected_trace["root_trace"])
        if failing_node:
            _analyze_failing_node(
                failing_node, probe, projection_model, report
            )

    if not report.fix_type:
        _infer_fix_type(report, probe)

    return report


def _analyze_failing_node(
    node: dict[str, Any],
    probe: ProbeTrace,
    model: ProjectionModel,
    report: CausalReport,
) -> None:
    """Analyze why a specific leaf node failed and compute threshold gaps."""
    sig_type = node.get("signal_type", "")
    sig_name = node.get("signal_name", "")

    if sig_type == "projection":
        info = model.get_band_for_output(sig_name)
        if info:
            mapping, band = info
            score_def = model.get_score_for_output(sig_name)
            if score_def:
                score_value = model.compute_score(
                    score_def.name,
                    probe.signal_confidences,
                    probe.matched_signals,
                )

                actual_score = probe.projection_scores.get(score_def.name, score_value)

                threshold = band.lower_bound
                if threshold is not None:
                    gap = actual_score - threshold
                    report.threshold_gaps.append(ThresholdGap(
                        score_name=score_def.name,
                        score_value=actual_score,
                        output_name=sig_name,
                        threshold=threshold,
                        gap=gap,
                    ))

                    decomp = _decompose_score(score_def, probe, model)
                    report.score_decompositions.append(decomp)

                    if abs(gap) < 0.15:
                        report.fix_type = "threshold_adjust"
                        report.fix_detail = (
                            f"{score_def.name}={actual_score:.4f}, "
                            f"threshold={threshold}, gap={gap:.4f}"
                        )
                    else:
                        report.fix_type = "structural_add_tier"
                        report.fix_detail = (
                            f"{score_def.name}={actual_score:.4f} is far from "
                            f"threshold={threshold} (gap={gap:.4f}); "
                            f"projection is unsuitable — use category_kb tier"
                        )

    elif sig_type == "category_kb":
        conf_key = f"category_kb:{sig_name}"
        conf_val = probe.signal_confidences.get(conf_key, 0.0)
        report.fix_type = "structural_add_tier"
        report.fix_detail = (
            f"category_kb(\"{sig_name}\") not matched "
            f"(confidence={conf_val:.3f})"
        )


def _decompose_score(
    score_def: Any,
    probe: ProbeTrace,
    model: ProjectionModel,
) -> ScoreDecomposition:
    """Break down a projection score into per-input contributions."""
    contributions = []
    total = 0.0
    for inp in score_def.inputs:
        raw_val = model._input_value(
            inp, probe.signal_confidences, probe.matched_signals
        )
        weighted = inp.weight * raw_val
        total += weighted
        contributions.append({
            "signal": f"{inp.signal_type}:{inp.signal_name}",
            "weight": inp.weight,
            "raw_value": round(raw_val, 4),
            "contribution": round(weighted, 4),
            "value_source": inp.value_source,
        })

    return ScoreDecomposition(
        score_name=score_def.name,
        total=round(total, 4),
        contributions=contributions,
    )


def _infer_fix_type(report: CausalReport, probe: ProbeTrace) -> None:
    """Fallback fix type inference when trace analysis didn't identify one."""
    winning_types = {l.signal_type for l in report.winning_leaves if l.matched}
    failing_has_no_ckb = not any(
        l.signal_type == "category_kb" for l in report.failing_leaves
    )

    if "projection" in winning_types and failing_has_no_ckb:
        report.fix_type = "structural_add_tier"
        report.fix_detail = (
            f"Wrong decision won via projection; expected decision has no "
            f"category_kb condition"
        )
    elif not report.fix_type:
        report.fix_type = "unknown"
        report.fix_detail = "Could not determine root cause from trace"


def analyze_all(
    failures: list[ProbeTrace],
    projection_model: ProjectionModel,
) -> list[CausalReport]:
    return [analyze_probe(p, projection_model) for p in failures]
