"""Causal Tracing Engine — analytical diagnosis and repair for DSL configs.

Implements the analytical pipeline:
  1. FindFailingLeaves — Algorithm 1 (trace tree walkdown)
  2. FindFalsePositiveLeaves — for priority conflict analysis
  3. Symbolic Score Decomposition — projection formula walkthrough
  4. Analytical Fix Computation — threshold & weight adjustment
  5. Analytical Regression Checking — severity-weighted impact analysis
  6. Fix Selection — constraint-satisfying greedy selection
  7. Config Mutation — apply fixes to YAML config dict

All operations are grounded on real trace data from the semantic router's
eval_trace API and the projection formulas from the YAML config.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TraceLeaf:
    signal_type: str
    signal_name: str
    matched: bool
    confidence: float
    path: str  # human-readable path from root, e.g. "AND/OR[1]"


@dataclass
class FailureClassification:
    kind: str  # "parametric", "structural", "conflict"
    leaf: TraceLeaf
    score_name: str | None = None
    mapping_name: str | None = None
    threshold: float | None = None
    threshold_dir: str | None = None  # "gte" or "lt"
    current_score: float | None = None
    gap: float | None = None
    active_terms: list[dict] = field(default_factory=list)
    silent_terms: list[dict] = field(default_factory=list)


@dataclass
class Fix:
    fix_type: str  # "threshold", "weight"
    target: str  # score or mapping name
    param_path: str
    old_value: float
    new_value: float
    affected_probes: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    net_improvement: int = 0
    explanation: str = ""


@dataclass
class StructuralFix:
    """A structural fix that modifies decision rule trees, not just parameters."""
    decision_name: str
    action: str  # "remove_false_positive_branches", "change_operator"
    description: str
    remove_signals: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DSL Config model — projection formulas extracted from YAML
# ---------------------------------------------------------------------------

@dataclass
class ScoreFormula:
    name: str
    method: str
    inputs: list[dict]  # [{name, type, weight, value_source?}]


@dataclass
class ThresholdMapping:
    name: str
    source_score: str
    outputs: list[dict]  # [{name, gte?, lt?}]


@dataclass
class DSLConfig:
    scores: list[ScoreFormula]
    mappings: list[ThresholdMapping]
    decisions: list[dict]

    def find_score(self, name: str) -> ScoreFormula | None:
        for s in self.scores:
            if s.name == name:
                return s
        return None

    def find_mapping_for_projection(
        self, proj_name: str,
    ) -> tuple[ThresholdMapping | None, dict | None]:
        for m in self.mappings:
            for out in m.outputs:
                if out["name"] == proj_name:
                    return m, out
        return None, None


def load_dsl_config(cfg: dict) -> DSLConfig:
    """Extract projection formulas and decisions from a parsed YAML config."""
    routing = cfg.get("routing", {})
    proj_raw = routing.get("projections", {})

    scores = []
    for s in proj_raw.get("scores", []):
        scores.append(ScoreFormula(
            name=s["name"],
            method=s.get("method", "weighted_sum"),
            inputs=s.get("inputs", []),
        ))

    mappings = []
    for m in proj_raw.get("mappings", []):
        mappings.append(ThresholdMapping(
            name=m["name"],
            source_score=m["source"],
            outputs=m.get("outputs", []),
        ))

    decisions = routing.get("decisions", [])
    return DSLConfig(scores=scores, mappings=mappings, decisions=decisions)


# ---------------------------------------------------------------------------
# Algorithm 1: FindFailingLeaves
# ---------------------------------------------------------------------------

def find_failing_leaves(trace_tree: dict) -> list[TraceLeaf]:
    """Walk the trace tree for the expected decision and return the minimal
    set of failing leaves whose fix would make the decision match.

    For AND nodes: all failing children are critical.
    For OR nodes: only the child closest to matching (fixing one suffices).
    For NOT nodes: the child that unexpectedly matched is explored.
    """
    leaves: list[TraceLeaf] = []
    _walk_failing(trace_tree, leaves, path="root")
    return leaves


def _walk_failing(node: dict, leaves: list[TraceLeaf], path: str) -> None:
    if not node:
        return
    node_type = node.get("node_type", "")

    if node_type == "leaf":
        if not node.get("matched", False):
            leaves.append(TraceLeaf(
                signal_type=node.get("signal_type", ""),
                signal_name=node.get("signal_name", ""),
                matched=False,
                confidence=node.get("confidence", 0.0),
                path=path,
            ))
        return

    children = node.get("children", [])
    op = node_type.upper()

    if op == "AND":
        for i, child in enumerate(children):
            if not child.get("matched", False):
                _walk_failing(child, leaves, f"{path}/AND[{i}]")
    elif op == "OR":
        failing = [(i, c) for i, c in enumerate(children)
                   if not c.get("matched", False)]
        if failing:
            best_idx, best_child = max(
                failing, key=lambda ic: ic[1].get("confidence", 0.0))
            _walk_failing(best_child, leaves, f"{path}/OR[{best_idx}]")
    elif op == "NOT":
        if children:
            _walk_failing(children[0], leaves, f"{path}/NOT[0]")


# ---------------------------------------------------------------------------
# FindFalsePositiveLeaves — for priority conflict analysis
# ---------------------------------------------------------------------------

def find_false_positive_leaves(trace_tree: dict) -> list[TraceLeaf]:
    """For a decision that SHOULD NOT match but does: find all matching leaves.

    For OR: all matching children are collected (removing any subset < all
    won't flip the OR to false).
    For AND: the weakest matching child is the best target.
    """
    leaves: list[TraceLeaf] = []
    _walk_positive(trace_tree, leaves, path="root")
    return leaves


def _walk_positive(node: dict, leaves: list[TraceLeaf], path: str) -> None:
    if not node:
        return
    node_type = node.get("node_type", "")

    if node_type == "leaf":
        if node.get("matched", False):
            leaves.append(TraceLeaf(
                signal_type=node.get("signal_type", ""),
                signal_name=node.get("signal_name", ""),
                matched=True,
                confidence=node.get("confidence", 0.0),
                path=path,
            ))
        return

    children = node.get("children", [])
    op = node_type.upper()

    if op == "OR":
        for i, child in enumerate(children):
            if child.get("matched", False):
                _walk_positive(child, leaves, f"{path}/OR[{i}]")
    elif op == "AND":
        weakest = min(
            ((i, c) for i, c in enumerate(children)
             if c.get("matched", False)),
            key=lambda ic: ic[1].get("confidence", 0.0),
            default=None,
        )
        if weakest:
            _walk_positive(weakest[1], leaves, f"{path}/AND[{weakest[0]}]")
    elif op == "NOT":
        if children:
            _walk_positive(children[0], leaves, f"{path}/NOT[0]")


# ---------------------------------------------------------------------------
# Symbolic Score Decomposition
# ---------------------------------------------------------------------------

def decompose_projection(
    leaf: TraceLeaf,
    signal_confidences: dict[str, float],
    dsl: DSLConfig,
    matched_signals: dict[str, list[str]] | None = None,
) -> FailureClassification:
    """For a failing projection leaf, decompose the score gap symbolically."""
    proj_name = leaf.signal_name
    mapping, output_band = dsl.find_mapping_for_projection(proj_name)
    if mapping is None:
        return FailureClassification(kind="structural", leaf=leaf)

    score_formula = dsl.find_score(mapping.source_score)
    if score_formula is None:
        return FailureClassification(kind="structural", leaf=leaf)

    if "gte" in output_band:
        threshold = output_band["gte"]
        threshold_dir = "gte"
    elif "lt" in output_band:
        threshold = output_band["lt"]
        threshold_dir = "lt"
    else:
        return FailureClassification(kind="structural", leaf=leaf)

    computed_score = 0.0
    active_terms = []
    silent_terms = []

    for inp in score_formula.inputs:
        sig_name = inp["name"]
        sig_type = inp["type"]
        weight = inp.get("weight", 1.0)
        sig_value = _projection_input_value(
            inp, signal_confidences, matched_signals)
        term_contrib = weight * sig_value
        computed_score += term_contrib

        conf_key = _resolve_confidence_key(sig_type, sig_name)
        term_info = {
            "signal": sig_name, "type": sig_type, "weight": weight,
            "confidence": round(sig_value, 6),
            "contribution": round(term_contrib, 6),
            "conf_key": conf_key,
        }
        (active_terms if sig_value > 0 else silent_terms).append(term_info)

    if threshold_dir == "gte":
        gap = threshold - computed_score
    else:
        gap = computed_score - threshold

    kind = "parametric"
    if computed_score < 1e-9:
        kind = "structural"

    return FailureClassification(
        kind=kind, leaf=leaf, score_name=mapping.source_score,
        mapping_name=mapping.name, threshold=threshold,
        threshold_dir=threshold_dir,
        current_score=round(computed_score, 6),
        gap=round(gap, 6),
        active_terms=active_terms, silent_terms=silent_terms,
    )


# ---------------------------------------------------------------------------
# Score computation helpers (mirrors Go router's projectionScoreValue)
# ---------------------------------------------------------------------------

_SIGNAL_TYPE_TO_MATCHED_KEY = {
    "keyword": "keywords", "embedding": "embeddings",
    "category_kb": "category_kb", "jailbreak": "jailbreak",
    "pii": "pii", "structure": "structure", "domain": "domains",
    "projection": "projection", "context": "context",
    "complexity": "complexity", "modality": "modality", "authz": "authz",
}


def _resolve_confidence_key(sig_type: str, sig_name: str) -> str:
    type_prefixes = {
        "keyword": "keyword", "embedding": "embedding",
        "category_kb": "category_kb", "jailbreak": "jailbreak",
        "pii": "pii", "structure": "structure", "projection": "projection",
    }
    prefix = type_prefixes.get(sig_type, sig_type)
    return f"{prefix}:{sig_name}"


def _signal_is_matched(
    sig_type: str, sig_name: str,
    signal_confidences: dict[str, float],
    matched_signals: dict[str, list[str]] | None,
) -> bool:
    if sig_type == "category_kb" and sig_name == "__contrastive__":
        conf_key = _resolve_confidence_key(sig_type, sig_name)
        return signal_confidences.get(conf_key, 0.0) != 0.0
    if matched_signals is not None:
        key = _SIGNAL_TYPE_TO_MATCHED_KEY.get(sig_type, sig_type)
        return sig_name in matched_signals.get(key, [])
    conf_key = _resolve_confidence_key(sig_type, sig_name)
    return signal_confidences.get(conf_key, 0.0) > 0


def _projection_input_value(
    inp: dict,
    signal_confidences: dict[str, float],
    matched_signals: dict[str, list[str]] | None,
) -> float:
    sig_type = inp["type"]
    sig_name = inp["name"]
    value_source = inp.get("value_source", "")
    is_matched = _signal_is_matched(
        sig_type, sig_name, signal_confidences, matched_signals)
    if value_source == "confidence":
        if not is_matched:
            return 0.0
        conf_key = _resolve_confidence_key(sig_type, sig_name)
        conf = signal_confidences.get(conf_key, 0.0)
        return conf if conf > 0 else 1.0
    else:
        match_val = inp.get("match", 0.0)
        miss_val = inp.get("miss", 0.0)
        if match_val == 0:
            match_val = 1.0
        return match_val if is_matched else miss_val


def compute_score(
    formula: ScoreFormula,
    signal_confidences: dict[str, float],
    matched_signals: dict[str, list[str]] | None = None,
) -> float:
    """Recompute a projection score from cached signal data."""
    total = 0.0
    for inp in formula.inputs:
        weight = inp.get("weight", 1.0)
        value = _projection_input_value(inp, signal_confidences, matched_signals)
        total += weight * value
    return total


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

SEVERITY_WEIGHTS = {
    "security": 10, "privacy": 10, "functional": 3, "preference": 1,
}
DEFAULT_SEVERITY = 3
EPSILON = 0.02


def probe_severity(probe: dict) -> int:
    """Severity weight based on routing lane (Equation 5 in the paper)."""
    tags = [t.lower() for t in probe.get("tags", [])]
    expected = probe.get("expected", probe.get("expected_decision", "")).lower()
    if "security" in tags or "jailbreak" in tags:
        return SEVERITY_WEIGHTS["security"]
    if "privacy" in tags or "pii" in tags:
        return SEVERITY_WEIGHTS["privacy"]
    if "security" in expected or "containment" in expected:
        return SEVERITY_WEIGHTS["security"]
    if "privacy" in expected:
        return SEVERITY_WEIGHTS["privacy"]
    if "baseline" in tags:
        return SEVERITY_WEIGHTS["preference"]
    return DEFAULT_SEVERITY


# ---------------------------------------------------------------------------
# Analytical Fix Computation
# ---------------------------------------------------------------------------

def compute_threshold_fix(fc: FailureClassification) -> Fix | None:
    """Threshold adjustment: τ' = score ∓ ε."""
    if fc.kind != "parametric" or fc.current_score is None or fc.threshold is None:
        return None

    if fc.threshold_dir == "gte":
        new_threshold = fc.current_score - EPSILON
        if new_threshold < 0:
            new_threshold = 0.01
        verb = "Lower"
        dir_key = "gte"
    elif fc.threshold_dir == "lt":
        new_threshold = fc.current_score + EPSILON
        verb = "Raise"
        dir_key = "lt"
    else:
        return None

    if abs(new_threshold - fc.threshold) < 1e-6:
        return None

    return Fix(
        fix_type="threshold",
        target=fc.mapping_name or "",
        param_path=f"projections.mappings[{fc.mapping_name}].outputs[{fc.leaf.signal_name}].{dir_key}",
        old_value=fc.threshold,
        new_value=round(new_threshold, 4),
        explanation=(
            f"{verb} threshold for {fc.leaf.signal_name} from {fc.threshold:.4f} to "
            f"{new_threshold:.4f} (score={fc.current_score:.4f}, gap={fc.gap:.4f})"
        ),
    )


def compute_weight_fix(fc: FailureClassification) -> Fix | None:
    """Weight adjustment: w' = w + (Δ + ε) / s_i(x) for strongest active signal."""
    if fc.kind != "parametric" or not fc.active_terms or fc.gap is None:
        return None

    best_term = max(fc.active_terms, key=lambda t: t["confidence"])
    if best_term["confidence"] < 1e-9:
        return None

    if fc.threshold_dir == "lt":
        delta_w = -(fc.gap + EPSILON) / best_term["confidence"]
        verb = "Decrease"
    else:
        delta_w = (fc.gap + EPSILON) / best_term["confidence"]
        verb = "Increase"

    new_weight = best_term["weight"] + delta_w
    if new_weight < 0:
        return None

    return Fix(
        fix_type="weight",
        target=fc.score_name or "",
        param_path=f"projections.scores[{fc.score_name}].inputs[{best_term['signal']}].weight",
        old_value=best_term["weight"],
        new_value=round(new_weight, 4),
        explanation=(
            f"{verb} weight of {best_term['signal']} in {fc.score_name} from "
            f"{best_term['weight']:.4f} to {new_weight:.4f} "
            f"(gap={fc.gap:.4f}, signal_conf={best_term['confidence']:.4f})"
        ),
    )


# ---------------------------------------------------------------------------
# Analytical Regression Checking
# ---------------------------------------------------------------------------

def check_regression_threshold(
    fix: Fix,
    mapping: ThresholdMapping,
    probe_cache: list[dict],
    dsl: DSLConfig,
    severity_fn=None,
) -> Fix:
    """For threshold change τ→τ', only probes with score in [min(τ,τ'), max(τ,τ')]
    can change outcome.  Uses severity-weighted net improvement."""
    if severity_fn is None:
        severity_fn = probe_severity

    old_t, new_t = fix.old_value, fix.new_value
    lo, hi = min(old_t, new_t), max(old_t, new_t)

    score_formula = dsl.find_score(mapping.source_score)
    if score_formula is None:
        return fix

    affected, regressions = [], []
    severity_gain, severity_loss = 0, 0

    for probe in probe_cache:
        probe_id = probe.get("id", "?")
        sc = probe.get("signal_confidences", {})
        ms = probe.get("matched_signals")
        w = severity_fn(probe)
        score = compute_score(score_formula, sc, ms)

        if lo <= score <= hi:
            affected.append(probe_id)
            old_matches_gte = score >= old_t
            new_matches_gte = score >= new_t
            if old_matches_gte != new_matches_gte:
                if probe.get("correct", False):
                    regressions.append(probe_id)
                    severity_loss += w
                else:
                    severity_gain += w

    fix.affected_probes = affected
    fix.regressions = regressions
    fix.net_improvement = severity_gain - severity_loss
    return fix


def check_regression_weight(
    fix: Fix,
    score_formula: ScoreFormula,
    mapping: ThresholdMapping,
    probe_cache: list[dict],
    signal_name: str,
    severity_fn=None,
) -> Fix:
    """For weight change w→w', only probes with s_i(x)>0 can have their score change."""
    if severity_fn is None:
        severity_fn = probe_severity

    delta_w = fix.new_value - fix.old_value
    conf_key = None
    for inp in score_formula.inputs:
        if inp["name"] == signal_name:
            conf_key = _resolve_confidence_key(inp["type"], inp["name"])
            break
    if conf_key is None:
        return fix

    affected, regressions = [], []
    severity_gain, severity_loss = 0, 0

    for probe in probe_cache:
        probe_id = probe.get("id", "?")
        sc = probe.get("signal_confidences", {})
        sig_val = sc.get(conf_key, 0.0)
        w = severity_fn(probe)

        if sig_val <= 0:
            continue
        affected.append(probe_id)

        old_score = compute_score(score_formula, sc, probe.get("matched_signals"))
        new_score = old_score + delta_w * sig_val

        regressed = False
        boundary_changed = False
        for out in mapping.outputs:
            thresh = out.get("gte", out.get("lt"))
            if thresh is None:
                continue
            if "gte" in out:
                old_match = old_score >= thresh
                new_match = new_score >= thresh
            else:
                old_match = old_score < thresh
                new_match = new_score < thresh
            if old_match != new_match:
                boundary_changed = True
                if probe.get("correct", False):
                    regressed = True

        if regressed:
            regressions.append(probe_id)
            severity_loss += w
        elif boundary_changed and not probe.get("correct", False):
            severity_gain += w

    fix.affected_probes = affected
    fix.regressions = regressions
    fix.net_improvement = severity_gain - severity_loss
    return fix


# ---------------------------------------------------------------------------
# Priority conflict analysis
# ---------------------------------------------------------------------------

def analyze_priority_conflict(probe_result: dict, dsl: DSLConfig) -> dict:
    """When expected decision matches but a higher-priority one also matches."""
    expected = probe_result["expected"]
    actual = probe_result["actual"]
    traces = probe_result.get("eval_trace", [])
    sc = probe_result.get("signal_confidences", {})
    ms = probe_result.get("matched_signals", {})

    expected_trace = actual_trace = None
    for t in traces:
        if t.get("decision_name") == expected:
            expected_trace = t
        if t.get("decision_name") == actual:
            actual_trace = t

    if not actual_trace or not actual_trace.get("matched"):
        return {"failure_kind": "unknown", "error": "competing decision trace not found"}

    expected_matched = expected_trace.get("matched", False) if expected_trace else False
    actual_root = actual_trace.get("root_trace", {})
    fp_leaves = find_false_positive_leaves(actual_root)

    structural_leaves, parametric_leaves = [], []

    for leaf in fp_leaves:
        if leaf.signal_type == "projection":
            mapping, output_band = dsl.find_mapping_for_projection(leaf.signal_name)
            if mapping and output_band:
                score_formula = dsl.find_score(mapping.source_score)
                if score_formula:
                    current_score = compute_score(score_formula, sc, ms)
                    if "gte" in output_band:
                        threshold = output_band["gte"]
                        parametric_leaves.append({
                            "leaf": leaf, "score_name": mapping.source_score,
                            "mapping_name": mapping.name,
                            "threshold": threshold, "threshold_dir": "gte",
                            "current_score": current_score,
                            "gap": current_score - threshold,
                            "fix_direction": "raise_threshold",
                        })
                    elif "lt" in output_band:
                        threshold = output_band["lt"]
                        parametric_leaves.append({
                            "leaf": leaf, "score_name": mapping.source_score,
                            "mapping_name": mapping.name,
                            "threshold": threshold, "threshold_dir": "lt",
                            "current_score": current_score,
                            "gap": threshold - current_score,
                            "fix_direction": "lower_threshold",
                        })
                    continue
        structural_leaves.append(leaf)

    kind = "priority_conflict"
    if structural_leaves and not parametric_leaves:
        kind = "priority_conflict_structural"
    elif parametric_leaves:
        kind = "priority_conflict_parametric"

    rule_fix = None
    if structural_leaves:
        rule_fix = {
            "fix_type": "structural_rule_change",
            "decision": actual,
            "action": "remove_false_positive_branches",
            "description": (
                f"Decision '{actual}' has {len(structural_leaves)} "
                f"non-projection signals that false-positive match."
            ),
            "remove_signals": [
                {"type": l.signal_type, "name": l.signal_name}
                for l in structural_leaves
            ],
        }

    return {
        "failure_kind": kind,
        "expected_decision": expected,
        "expected_matched": expected_matched,
        "competing_decision": actual,
        "false_positive_leaves": [_leaf_to_dict(l) for l in fp_leaves],
        "structural_fp": [_leaf_to_dict(l) for l in structural_leaves],
        "parametric_fp": [{
            "signal_type": p["leaf"].signal_type,
            "signal_name": p["leaf"].signal_name,
            "score_name": p["score_name"],
            "mapping_name": p["mapping_name"],
            "threshold": p["threshold"],
            "threshold_dir": p["threshold_dir"],
            "current_score": round(p["current_score"], 6),
            "gap": round(p["gap"], 6),
            "fix_direction": p["fix_direction"],
        } for p in parametric_leaves],
        "structural_rule_fix": rule_fix,
    }


# ---------------------------------------------------------------------------
# Full analytical diagnosis for one probe
# ---------------------------------------------------------------------------

def diagnose_probe(probe_result: dict, dsl: DSLConfig) -> dict:
    """Run the full causal tracing pipeline for a single misrouted probe.

    Handles:
      A. Expected decision doesn't match → FindFailingLeaves
      B. Expected decision matches but is outprioritized → Priority conflict
    """
    expected = probe_result["expected"]
    traces = probe_result.get("eval_trace", [])
    sc = probe_result.get("signal_confidences", {})

    expected_trace = None
    for t in traces:
        if t.get("decision_name") == expected:
            expected_trace = t
            break

    if expected_trace is None:
        return {
            "error": f"No trace found for expected decision {expected}",
            "failing_leaves": [], "decompositions": [],
            "failure_kind": "missing_trace",
        }

    if expected_trace.get("matched", False):
        return analyze_priority_conflict(probe_result, dsl)

    ms = probe_result.get("matched_signals", {})
    root = expected_trace.get("root_trace", {})
    failing_leaves = find_failing_leaves(root)

    decompositions = []
    for leaf in failing_leaves:
        if leaf.signal_type == "projection":
            dec = decompose_projection(leaf, sc, dsl, ms)
        else:
            dec = FailureClassification(kind="structural", leaf=leaf)
        decompositions.append(dec)

    kinds = [d.kind for d in decompositions]
    overall = "parametric" if any(k == "parametric" for k in kinds) else "structural"

    return {
        "expected_decision": expected,
        "trace_matched": False,
        "trace_confidence": expected_trace.get("confidence", 0.0),
        "failing_leaves": [_leaf_to_dict(l) for l in failing_leaves],
        "decompositions": [_fc_to_dict(d) for d in decompositions],
        "failure_kind": overall,
    }


# ---------------------------------------------------------------------------
# Fix selection — severity-weighted constraint-satisfying strategy
# ---------------------------------------------------------------------------

def _batch_structural_fix(
    diagnoses: list[dict],
    probe_cache: list[dict],
    severity_fn=None,
) -> StructuralFix | None:
    """Regression-aware batch structural fix.  Collects all proposed category
    removals, then keeps only those with positive severity-weighted net."""
    if severity_fn is None:
        severity_fn = probe_severity

    removal_candidates: dict[tuple[str, str, str], list[str]] = {}
    decision_name = None

    for diag in diagnoses:
        rule_fix = diag.get("structural_rule_fix")
        if not rule_fix:
            continue
        decision_name = rule_fix["decision"]
        for sig in rule_fix.get("remove_signals", []):
            key = (rule_fix["decision"], sig["type"], sig["name"])
            removal_candidates.setdefault(key, [])

    if not removal_candidates or not decision_name:
        return None

    probe_best_cats: dict[str, str] = {}
    for probe in probe_cache:
        for t in probe.get("eval_trace", []):
            if t.get("decision_name") == decision_name and t.get("matched"):
                root = t.get("root_trace", {})
                for child in root.get("children", []):
                    if (child.get("matched") and child.get("node_type") == "leaf"):
                        sname = child.get("signal_name", "")
                        if sname.startswith("__best__:"):
                            probe_best_cats[probe.get("id", "?")] = sname
                            break

    net_positive = []
    for (dec, sig_type, sig_name), _ in removal_candidates.items():
        gain, loss, fixes, regressions = 0, 0, [], []
        for probe in probe_cache:
            pid = probe.get("id", "?")
            if probe_best_cats.get(pid, "") != sig_name:
                continue
            w = severity_fn(probe)
            if not probe.get("correct", False):
                gain += w
                fixes.append(pid)
            else:
                loss += w
                regressions.append(pid)
        net = gain - loss
        if net > 0:
            net_positive.append({
                "type": sig_type, "name": sig_name,
                "net": net, "fixes": len(fixes),
                "regressions": len(regressions),
            })

    if not net_positive:
        return None

    net_positive.sort(key=lambda x: x["net"], reverse=True)
    return StructuralFix(
        decision_name=decision_name,
        action="remove_false_positive_branches",
        description=(
            f"Batch-remove {len(net_positive)} false-positive categories from "
            f"'{decision_name}' OR condition."
        ),
        remove_signals=[{"type": r["type"], "name": r["name"]}
                        for r in net_positive],
    )


def select_fix(
    diagnoses: list[dict],
    probe_cache: list[dict],
    dsl: DSLConfig,
    severity_fn=None,
) -> Fix | StructuralFix | None:
    """Choose the best analytical fix from all failures.

    Strategy:
      1. Regression-aware batch structural fix (priority conflicts).
      2. Parametric threshold/weight fixes with regression checking.
      3. Accept the fix with the best severity-weighted net improvement.
    """
    has_structural = any(d.get("structural_rule_fix") for d in diagnoses)
    if has_structural:
        batch_fix = _batch_structural_fix(diagnoses, probe_cache, severity_fn)
        if batch_fix:
            return batch_fix

    candidates: list[tuple[Fix, FailureClassification]] = []

    for diag in diagnoses:
        for dec_dict in diag.get("decompositions", []):
            if dec_dict["kind"] != "parametric":
                continue
            fc = _reconstruct_fc(dec_dict, matched=False)
            _add_parametric_candidates(fc, candidates, probe_cache, dsl, severity_fn)

        for pf in diag.get("parametric_fp", []):
            fc = FailureClassification(
                kind="parametric",
                leaf=TraceLeaf(
                    signal_type=pf["signal_type"],
                    signal_name=pf["signal_name"],
                    matched=True, confidence=0.0, path="",
                ),
                score_name=pf.get("score_name"),
                mapping_name=pf.get("mapping_name"),
                threshold=pf.get("threshold"),
                threshold_dir=pf.get("threshold_dir"),
                current_score=pf.get("current_score"),
                gap=pf.get("gap"),
            )
            fix_dir = pf.get("fix_direction", "")
            if (fix_dir == "raise_threshold"
                    and fc.threshold is not None
                    and fc.current_score is not None):
                new_t = fc.current_score + EPSILON
                fix = Fix(
                    fix_type="threshold", target=fc.mapping_name or "",
                    param_path=f"projections.mappings[{fc.mapping_name}].outputs[{fc.leaf.signal_name}].gte",
                    old_value=fc.threshold,
                    new_value=round(new_t, 4),
                    explanation=f"Raise threshold for {fc.leaf.signal_name} to prevent false-positive",
                )
                if abs(fix.new_value - fix.old_value) > 1e-6:
                    mapping, _ = dsl.find_mapping_for_projection(fc.leaf.signal_name)
                    if mapping:
                        fix = check_regression_threshold(
                            fix, mapping, probe_cache, dsl, severity_fn)
                    candidates.append((fix, fc))

    if not candidates:
        return None

    best_fix, best_score = None, -float("inf")
    for fix, fc in candidates:
        score = fix.net_improvement
        if len(fix.regressions) == 0:
            score += 10
        if score > best_score:
            best_score = score
            best_fix = fix

    if best_fix and best_fix.net_improvement <= 0 and best_fix.regressions:
        return None
    return best_fix


def _reconstruct_fc(dec_dict: dict, matched: bool) -> FailureClassification:
    return FailureClassification(
        kind="parametric",
        leaf=TraceLeaf(
            signal_type=dec_dict["signal_type"],
            signal_name=dec_dict["signal_name"],
            matched=matched, confidence=0.0, path="",
        ),
        score_name=dec_dict.get("score_name"),
        mapping_name=dec_dict.get("mapping_name"),
        threshold=dec_dict.get("threshold"),
        threshold_dir=dec_dict.get("threshold_dir"),
        current_score=dec_dict.get("current_score"),
        gap=dec_dict.get("gap"),
        active_terms=dec_dict.get("active_terms", []),
        silent_terms=dec_dict.get("silent_terms", []),
    )


def _add_parametric_candidates(
    fc: FailureClassification,
    candidates: list,
    probe_cache: list[dict],
    dsl: DSLConfig,
    severity_fn=None,
):
    threshold_fix = compute_threshold_fix(fc)
    if threshold_fix:
        mapping, _ = dsl.find_mapping_for_projection(fc.leaf.signal_name)
        if mapping:
            threshold_fix = check_regression_threshold(
                threshold_fix, mapping, probe_cache, dsl, severity_fn)
            candidates.append((threshold_fix, fc))

    weight_fix = compute_weight_fix(fc)
    if weight_fix and fc.active_terms:
        score_formula = dsl.find_score(fc.score_name or "")
        mapping, _ = dsl.find_mapping_for_projection(fc.leaf.signal_name)
        if score_formula and mapping:
            best_term = max(fc.active_terms, key=lambda t: t["confidence"])
            weight_fix = check_regression_weight(
                weight_fix, score_formula, mapping, probe_cache,
                best_term["signal"], severity_fn)
            candidates.append((weight_fix, fc))


# ---------------------------------------------------------------------------
# Apply fixes to YAML config dict
# ---------------------------------------------------------------------------

def apply_structural_fix(cfg: dict, sfix: StructuralFix) -> dict:
    """Remove specified signal leaves from a decision's OR condition."""
    cfg = copy.deepcopy(cfg)
    for dec in cfg["routing"]["decisions"]:
        if dec["name"] != sfix.decision_name:
            continue
        if sfix.action == "remove_false_positive_branches":
            remove_set = {(s["type"], s["name"]) for s in sfix.remove_signals}
            rules = dec.get("rules", {})
            conditions = rules.get("conditions", [])
            new_conditions = [
                c for c in conditions
                if (c.get("type", ""), c.get("name", "")) not in remove_set
            ]
            if new_conditions:
                rules["conditions"] = new_conditions
                if len(new_conditions) == 1:
                    rules["operator"] = "AND"
            break
    return cfg


def apply_fix_to_config(cfg: dict, fix: Fix, dsl: DSLConfig) -> dict:
    """Mutate config dict for a parametric fix, maintaining partition consistency."""
    cfg = copy.deepcopy(cfg)
    proj = cfg["routing"]["projections"]

    if fix.fix_type == "threshold":
        for m in proj.get("mappings", []):
            if m["name"] != fix.target:
                continue
            for out in m.get("outputs", []):
                if "gte" in out and abs(out["gte"] - fix.old_value) < 1e-9:
                    out["gte"] = fix.new_value
                elif "lt" in out and abs(out["lt"] - fix.old_value) < 1e-9:
                    out["lt"] = fix.new_value
            gte_outputs = [o for o in m["outputs"] if "gte" in o]
            lt_outputs = [o for o in m["outputs"] if "lt" in o]
            if gte_outputs and lt_outputs:
                boundary = gte_outputs[0]["gte"]
                for lt_out in lt_outputs:
                    lt_out["lt"] = boundary

    elif fix.fix_type == "weight":
        for s in proj.get("scores", []):
            if s["name"] == fix.target:
                signal_name = fix.param_path.split("inputs[")[1].split("]")[0]
                for inp in s.get("inputs", []):
                    if inp["name"] == signal_name:
                        inp["weight"] = fix.new_value

    return cfg


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _leaf_to_dict(leaf: TraceLeaf) -> dict:
    return {
        "signal_type": leaf.signal_type,
        "signal_name": leaf.signal_name,
        "matched": leaf.matched,
        "confidence": leaf.confidence,
        "path": leaf.path,
    }


def _fc_to_dict(fc: FailureClassification) -> dict:
    d: dict[str, Any] = {
        "kind": fc.kind,
        "signal_type": fc.leaf.signal_type,
        "signal_name": fc.leaf.signal_name,
    }
    if fc.score_name:
        d["score_name"] = fc.score_name
    if fc.mapping_name:
        d["mapping_name"] = fc.mapping_name
    if fc.threshold is not None:
        d["threshold"] = fc.threshold
    if fc.threshold_dir:
        d["threshold_dir"] = fc.threshold_dir
    if fc.current_score is not None:
        d["current_score"] = fc.current_score
    if fc.gap is not None:
        d["gap"] = fc.gap
    if fc.active_terms:
        d["active_terms"] = fc.active_terms
    if fc.silent_terms:
        d["silent_terms"] = fc.silent_terms
    return d
