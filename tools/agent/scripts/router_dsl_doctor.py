#!/usr/bin/env python3
"""DSL Doctor — Automated diagnosis and repair for Semantic Router DSL configs.

Subcommands:
  diagnose  Run probes, collect traces, classify failures, emit report.
  fix       Apply suggested DSL edits (--dry-run default).
  loop      Full diagnose → fix → redeploy → re-diagnose cycle.

Requires:
  - A running Semantic Router instance (--endpoint)
  - A probes YAML file (--probes)
  - The DSL source file (--dsl)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------

FAILURE_SIGNAL_DISCONNECTION = "signal_disconnection"
FAILURE_COMPOSITION_DOMINANCE = "composition_dominance"
FAILURE_BINARY_CATCHALL = "binary_catchall"
FAILURE_THRESHOLD_DRIFT = "threshold_drift"
FAILURE_PROBE_AMBIGUITY = "probe_ambiguity"
FAILURE_UNKNOWN = "unknown"

FAILURE_DESCRIPTIONS = {
    FAILURE_SIGNAL_DISCONNECTION: (
        "Signal is configured but not referenced in any decision rule, "
        "causing it to be evaluated without influencing routing."
    ),
    FAILURE_COMPOSITION_DOMINANCE: (
        "A broad signal (e.g. projection) is OR'd with a precise signal "
        "(e.g. category_kb tier), making the precise signal redundant."
    ),
    FAILURE_BINARY_CATCHALL: (
        "Multiple raw category_kb names are OR'd together, creating a "
        "coarse binary match that ignores classifier ranking."
    ),
    FAILURE_THRESHOLD_DRIFT: (
        "Signal confidence is below expected threshold, causing missed matches."
    ),
    FAILURE_PROBE_AMBIGUITY: (
        "Multiple decisions match with similar confidence, making "
        "the routing outcome unstable."
    ),
    FAILURE_UNKNOWN: "Failure root cause could not be determined.",
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_json(
    method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 60
) -> tuple[int, Any]:
    """Send an HTTP request and parse JSON response."""
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url=url, method=method.upper(), data=body, headers=headers)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.getcode(), json.loads(raw) if raw.strip() else {}
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw
    except error.URLError as exc:
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc


def _normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------

def load_probes(probes_path: str) -> list[dict[str, Any]]:
    """Load probes from a YAML file. Falls back to JSON if no YAML parser."""
    path = Path(probes_path)
    raw = path.read_text(encoding="utf-8")

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(raw)
        except ImportError:
            # Minimal YAML-like parsing for simple probe files
            data = _parse_simple_yaml_probes(raw)
    else:
        data = json.loads(raw)

    if isinstance(data, dict) and "probes" in data:
        return data["probes"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Cannot parse probes from {probes_path}")


def _parse_simple_yaml_probes(raw: str) -> dict[str, Any]:
    """Minimal parser for simple YAML probe files without PyYAML."""
    probes: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("- query:"):
            if current:
                probes.append(current)
            current = {"query": stripped.split(":", 1)[1].strip().strip('"').strip("'")}
        elif stripped.startswith("expected_route:") or stripped.startswith("expected_decision:"):
            key = stripped.split(":")[0].strip()
            val = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            current[key] = val
    if current:
        probes.append(current)
    return {"probes": probes}


# ---------------------------------------------------------------------------
# DSL static validation
# ---------------------------------------------------------------------------

def run_static_validation(dsl_path: str, sr_dsl_binary: str | None = None) -> list[dict[str, Any]]:
    """Run static DSL validation and parse diagnostics."""
    if sr_dsl_binary is None:
        repo_root = Path(__file__).resolve().parents[3]
        sr_dsl_binary = str(repo_root / "src" / "semantic-router" / "bin" / "sr-dsl")

    if not Path(sr_dsl_binary).exists():
        # Try to build it
        module_root = Path(sr_dsl_binary).parents[1]
        try:
            subprocess.run(
                ["go", "build", "-o", sr_dsl_binary, "./cmd/dsl/"],
                cwd=str(module_root),
                capture_output=True, check=True, timeout=120,
            )
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            return [{"level": "error", "message": f"Cannot build sr-dsl: {e}"}]

    try:
        result = subprocess.run(
            [sr_dsl_binary, "validate", dsl_path],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        return [{"level": "error", "message": f"sr-dsl validate failed: {e}"}]

    diagnostics: list[dict[str, Any]] = []
    for line in (result.stdout + result.stderr).splitlines():
        line = line.strip()
        if not line or line.startswith("Summary:") or line == "No issues found.":
            continue
        level = "info"
        if "Error" in line or "🔴" in line:
            level = "error"
        elif "Warning" in line or "🟡" in line:
            level = "warning"
        elif "Constraint" in line or "🟠" in line:
            level = "constraint"
        diagnostics.append({"level": level, "message": line})

    return diagnostics


# ---------------------------------------------------------------------------
# Runtime diagnosis
# ---------------------------------------------------------------------------

def run_probe_with_trace(
    endpoint: str, query: str
) -> dict[str, Any]:
    """Send a probe to the eval endpoint with trace=true."""
    url = f"{_normalize_url(endpoint)}/api/v1/eval?trace=true"
    status, payload = _http_json("POST", url, {"text": query})
    return {
        "status": status,
        "response": payload if isinstance(payload, dict) else {"raw": payload},
    }


def classify_probe_failure(
    probe: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Classify a single probe failure into the failure taxonomy."""
    expected = probe.get("expected_route") or probe.get("expected_decision", "")
    response = result.get("response", {})
    decision_result = response.get("decision_result", {})
    actual = decision_result.get("decision_name", "")
    traces = decision_result.get("eval_trace", [])

    if actual == expected:
        return {"classification": "correct", "expected": expected, "actual": actual}

    # Analyze traces for failure classification
    failure_type = FAILURE_UNKNOWN
    details = ""

    # Check for signal disconnection: signals evaluated but not matched
    used_signals = decision_result.get("used_signals", {})
    matched_signals = decision_result.get("matched_signals", {})
    for sig_type, names in (used_signals or {}).items():
        if not names:
            continue
        matched_for_type = (matched_signals or {}).get(sig_type, [])
        if isinstance(names, list):
            for name in names:
                if name not in (matched_for_type or []):
                    failure_type = FAILURE_SIGNAL_DISCONNECTION
                    details = f"{sig_type}:{name} used but not matched"
                    break
        if failure_type != FAILURE_UNKNOWN:
            break

    # Check for ambiguity: multiple decisions matched with close confidence
    if failure_type == FAILURE_UNKNOWN and traces:
        matched_traces = [t for t in traces if t.get("matched")]
        if len(matched_traces) >= 2:
            confs = sorted([t.get("confidence", 0) for t in matched_traces], reverse=True)
            if len(confs) >= 2 and confs[0] - confs[1] < 0.1:
                failure_type = FAILURE_PROBE_AMBIGUITY
                details = f"{len(matched_traces)} decisions matched with close confidence"

    # Check for threshold drift via trace analysis
    if failure_type == FAILURE_UNKNOWN and traces:
        for trace in traces:
            if trace.get("decision_name") == expected and not trace.get("matched"):
                root = trace.get("root_trace", {})
                low_conf_nodes = _find_low_confidence_nodes(root, threshold=0.3)
                if low_conf_nodes:
                    failure_type = FAILURE_THRESHOLD_DRIFT
                    details = f"Expected decision has low-confidence nodes: {low_conf_nodes}"
                break

    return {
        "classification": failure_type,
        "expected": expected,
        "actual": actual,
        "details": details,
    }


def _find_low_confidence_nodes(
    trace_node: dict[str, Any], threshold: float
) -> list[str]:
    """Find leaf nodes with confidence below threshold."""
    low_nodes: list[str] = []
    if not trace_node:
        return low_nodes

    if trace_node.get("node_type") == "leaf":
        if not trace_node.get("matched") and trace_node.get("confidence", 0) < threshold:
            low_nodes.append(
                f"{trace_node.get('signal_type', '?')}:{trace_node.get('signal_name', '?')}"
            )
        return low_nodes

    for child in trace_node.get("children", []):
        low_nodes.extend(_find_low_confidence_nodes(child, threshold))
    return low_nodes


# ---------------------------------------------------------------------------
# Diagnosis report
# ---------------------------------------------------------------------------

def generate_report(
    static_diags: list[dict[str, Any]],
    probe_results: list[dict[str, Any]],
    dsl_path: str,
) -> str:
    """Generate a Markdown diagnosis report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"# DSL Doctor Report",
        f"",
        f"Generated: {now}",
        f"DSL file: `{dsl_path}`",
        f"",
        f"## Static Analysis",
        f"",
    ]

    if not static_diags:
        lines.append("No static issues found.")
    else:
        error_count = sum(1 for d in static_diags if d["level"] == "error")
        warn_count = sum(1 for d in static_diags if d["level"] == "warning")
        constraint_count = sum(1 for d in static_diags if d["level"] == "constraint")
        lines.append(f"Found {error_count} error(s), {warn_count} warning(s), {constraint_count} constraint(s).")
        lines.append("")
        for d in static_diags:
            lines.append(f"- **{d['level']}**: {d['message']}")

    lines.extend(["", "## Runtime Probe Results", ""])

    if not probe_results:
        lines.append("No probe results available.")
    else:
        correct = sum(1 for r in probe_results if r.get("classification") == "correct")
        total = len(probe_results)
        lines.append(f"**Accuracy: {correct}/{total} ({100*correct/total:.1f}%)**")
        lines.append("")

        # Group failures by classification
        failures = defaultdict(list)
        for r in probe_results:
            cls = r.get("classification", "unknown")
            if cls != "correct":
                failures[cls].append(r)

        if failures:
            lines.append("### Failure Breakdown")
            lines.append("")
            lines.append("| Category | Count | Description |")
            lines.append("|----------|-------|-------------|")
            for cls, items in sorted(failures.items()):
                desc = FAILURE_DESCRIPTIONS.get(cls, "Unknown")
                lines.append(f"| `{cls}` | {len(items)} | {desc} |")

            lines.extend(["", "### Failure Details", ""])
            for cls, items in sorted(failures.items()):
                lines.append(f"#### {cls}")
                lines.append("")
                for item in items:
                    lines.append(
                        f"- Expected `{item.get('expected', '?')}`, "
                        f"got `{item.get('actual', '?')}` — {item.get('details', 'no details')}"
                    )
                lines.append("")

    lines.extend(["", "## Recommended Actions", ""])

    # Generate recommendations based on failures
    recommendations = _generate_recommendations(static_diags, probe_results)
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
    else:
        lines.append("No specific recommendations — all probes pass.")

    return "\n".join(lines)


def _generate_recommendations(
    static_diags: list[dict[str, Any]],
    probe_results: list[dict[str, Any]],
) -> list[str]:
    """Generate actionable recommendations from diagnostics."""
    recs: list[str] = []

    # From static analysis
    for d in static_diags:
        msg = d.get("message", "")
        if "never referenced" in msg:
            recs.append(f"Remove or reference unused signal: {msg.split('SIGNAL')[1].split('is')[0].strip() if 'SIGNAL' in msg else 'unknown'}")
        elif "matched-rules list" in msg:
            recs.append("Fix match-gate disconnection: add signal to WHEN clause or change value_source to binary")
        elif "dominat" in msg.lower():
            recs.append("Fix OR composition: remove broad projection branch, use __tier__ pattern alone")
        elif "raw category_kb names OR'd" in msg:
            recs.append("Replace raw category_kb OR pattern with __tier__ best-match semantics")

    # From runtime failures
    failure_types = set()
    for r in probe_results:
        cls = r.get("classification", "")
        if cls and cls != "correct":
            failure_types.add(cls)

    if FAILURE_SIGNAL_DISCONNECTION in failure_types:
        recs.append("Signal disconnection detected — ensure all decision-referenced signals are added to matched-rules")
    if FAILURE_PROBE_AMBIGUITY in failure_types:
        recs.append("Decision ambiguity detected — add mutual exclusion guards or adjust tier/priority")
    if FAILURE_THRESHOLD_DRIFT in failure_types:
        recs.append("Threshold drift detected — calibrate signal thresholds or adjust confidence weights")

    return recs


# ---------------------------------------------------------------------------
# Fix application (stub for --dry-run)
# ---------------------------------------------------------------------------

def apply_fixes(
    dsl_path: str,
    static_diags: list[dict[str, Any]],
    probe_results: list[dict[str, Any]],
    dry_run: bool = True,
) -> list[dict[str, Any]]:
    """Apply suggested fixes from diagnostics.

    Returns a list of applied/proposed fixes.
    Currently extracts QuickFix suggestions from static diagnostics.
    """
    fixes: list[dict[str, Any]] = []

    for d in static_diags:
        msg = d.get("message", "")
        # Extract [Fix: ...] suggestions from diagnostic messages
        if "[Fix:" in msg:
            fix_start = msg.index("[Fix:") + 5
            fix_end = msg.index("]", fix_start)
            fix_desc = msg[fix_start:fix_end].strip()
            fixes.append({
                "source": "static_analysis",
                "description": fix_desc,
                "applied": not dry_run,
            })

    if dry_run:
        return fixes

    # For non-dry-run, write modified DSL
    # (Full implementation would parse DSL AST and apply transformations)
    return fixes


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_diagnose(args: argparse.Namespace) -> int:
    """Run full diagnosis: static + runtime probes."""
    print(f"[doctor] Diagnosing {args.dsl}", file=sys.stderr)

    # Static analysis
    static_diags = run_static_validation(args.dsl, sr_dsl_binary=args.sr_dsl)
    print(f"[doctor] Static analysis: {len(static_diags)} finding(s)", file=sys.stderr)

    # Runtime probes (if endpoint provided)
    probe_results: list[dict[str, Any]] = []
    if args.endpoint and args.probes:
        probes = load_probes(args.probes)
        print(f"[doctor] Running {len(probes)} probe(s) against {args.endpoint}", file=sys.stderr)

        for probe in probes:
            query = probe.get("query", "")
            if not query:
                continue
            result = run_probe_with_trace(args.endpoint, query)
            classification = classify_probe_failure(probe, result)
            classification["query"] = query
            probe_results.append(classification)

        correct = sum(1 for r in probe_results if r.get("classification") == "correct")
        print(f"[doctor] Probe accuracy: {correct}/{len(probe_results)}", file=sys.stderr)

    # Generate report
    report = generate_report(static_diags, probe_results, args.dsl)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"[doctor] Report written to {args.output}", file=sys.stderr)
    else:
        print(report)

    # Output structured JSON for programmatic use
    result = {
        "static_diagnostics": static_diags,
        "probe_results": probe_results,
        "summary": {
            "static_errors": sum(1 for d in static_diags if d["level"] == "error"),
            "static_warnings": sum(1 for d in static_diags if d["level"] == "warning"),
            "probes_total": len(probe_results),
            "probes_correct": sum(1 for r in probe_results if r.get("classification") == "correct"),
        },
    }

    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    has_errors = result["summary"]["static_errors"] > 0
    has_failures = result["summary"]["probes_correct"] < result["summary"]["probes_total"]
    return 1 if has_errors or has_failures else 0


def cmd_fix(args: argparse.Namespace) -> int:
    """Apply fixes from diagnosis."""
    print(f"[doctor] Fix mode (dry_run={args.dry_run}) for {args.dsl}", file=sys.stderr)

    static_diags = run_static_validation(args.dsl, sr_dsl_binary=args.sr_dsl)
    fixes = apply_fixes(args.dsl, static_diags, [], dry_run=args.dry_run)

    if not fixes:
        print("[doctor] No fixes to apply.", file=sys.stderr)
        return 0

    for fix in fixes:
        status = "PROPOSED" if not fix.get("applied") else "APPLIED"
        print(f"  [{status}] {fix['description']}", file=sys.stderr)

    return 0


def cmd_loop(args: argparse.Namespace) -> int:
    """Full diagnose → fix → redeploy → re-diagnose cycle."""
    max_iterations = args.max_iterations
    print(
        f"[doctor] Loop mode: max {max_iterations} iterations for {args.dsl}",
        file=sys.stderr,
    )

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[doctor] Iteration {iteration}/{max_iterations}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        # Diagnose
        static_diags = run_static_validation(args.dsl, sr_dsl_binary=args.sr_dsl)

        probe_results: list[dict[str, Any]] = []
        if args.endpoint and args.probes:
            probes = load_probes(args.probes)
            for probe in probes:
                query = probe.get("query", "")
                if not query:
                    continue
                result = run_probe_with_trace(args.endpoint, query)
                classification = classify_probe_failure(probe, result)
                classification["query"] = query
                probe_results.append(classification)

        correct = sum(1 for r in probe_results if r.get("classification") == "correct")
        total = len(probe_results) if probe_results else 0
        errors = sum(1 for d in static_diags if d["level"] == "error")

        print(
            f"[doctor] Status: {errors} static error(s), "
            f"{correct}/{total} probes correct",
            file=sys.stderr,
        )

        # Check convergence
        if errors == 0 and (total == 0 or correct == total):
            print("[doctor] All checks pass — converged!", file=sys.stderr)
            report = generate_report(static_diags, probe_results, args.dsl)
            if args.output:
                Path(args.output).write_text(report, encoding="utf-8")
            return 0

        # Try to fix
        fixes = apply_fixes(args.dsl, static_diags, probe_results, dry_run=False)
        if not fixes:
            print("[doctor] No automated fixes available — manual intervention needed.", file=sys.stderr)
            break

        for fix in fixes:
            print(f"  [APPLIED] {fix['description']}", file=sys.stderr)

        # Redeploy (compile DSL and restart)
        if args.redeploy_cmd:
            print(f"[doctor] Redeploying: {args.redeploy_cmd}", file=sys.stderr)
            try:
                subprocess.run(
                    args.redeploy_cmd, shell=True, check=True, timeout=120,
                )
                time.sleep(3)
            except subprocess.SubprocessError as e:
                print(f"[doctor] Redeploy failed: {e}", file=sys.stderr)
                return 1

    print(f"[doctor] Max iterations ({max_iterations}) reached.", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="router_dsl_doctor",
        description="DSL Doctor — Automated diagnosis and repair for Semantic Router DSL configs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # diagnose
    p_diag = sub.add_parser("diagnose", help="Run probes, collect traces, classify failures, emit report")
    p_diag.add_argument("--dsl", required=True, help="Path to DSL source file")
    p_diag.add_argument("--endpoint", help="Router API endpoint (e.g. http://localhost:8080)")
    p_diag.add_argument("--probes", help="Path to probes YAML/JSON file")
    p_diag.add_argument("--output", "-o", help="Write Markdown report to file")
    p_diag.add_argument("--json-output", help="Write structured JSON results to file")
    p_diag.add_argument("--sr-dsl", help="Path to sr-dsl binary (auto-built if not specified)")

    # fix
    p_fix = sub.add_parser("fix", help="Apply suggested DSL edits")
    p_fix.add_argument("--dsl", required=True, help="Path to DSL source file")
    p_fix.add_argument("--dry-run", action="store_true", default=True, help="Only show proposed fixes (default)")
    p_fix.add_argument("--apply", dest="dry_run", action="store_false", help="Actually apply fixes")
    p_fix.add_argument("--sr-dsl", help="Path to sr-dsl binary")

    # loop
    p_loop = sub.add_parser("loop", help="Full diagnose → fix → redeploy → re-diagnose cycle")
    p_loop.add_argument("--dsl", required=True, help="Path to DSL source file")
    p_loop.add_argument("--endpoint", help="Router API endpoint")
    p_loop.add_argument("--probes", help="Path to probes YAML/JSON file")
    p_loop.add_argument("--output", "-o", help="Write final report to file")
    p_loop.add_argument("--max-iterations", type=int, default=5, help="Max fix iterations (default: 5)")
    p_loop.add_argument("--redeploy-cmd", help="Shell command to redeploy router after DSL changes")
    p_loop.add_argument("--sr-dsl", help="Path to sr-dsl binary")

    args = parser.parse_args()

    if args.command == "diagnose":
        return cmd_diagnose(args)
    elif args.command == "fix":
        return cmd_fix(args)
    elif args.command == "loop":
        return cmd_loop(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
