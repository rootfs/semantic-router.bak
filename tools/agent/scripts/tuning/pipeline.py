#!/usr/bin/env python3
"""DSL Tuning Pipeline — two-phase causal optimization.

Phase 1: Structural fixes (existing DSL Doctor — add/remove WHEN conditions)
Phase 2: Causal parameter tuning (new — trace-based analytical optimization)

Usage:
  python -m tuning.pipeline \
    --endpoint http://localhost:8080 \
    --config /path/to/config.yaml \
    --probes /path/to/probes.yaml \
    [--max-iterations 5] \
    [--val-split 0.2]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .trace_collector import TraceCache, ProbeTrace, collect_traces, print_summary
from .projection_model import load_projection_model, ProjectionModel
from .causal_tracer import analyze_all, CausalReport
from .fix_calculator import (
    compute_fixes_for_report, regression_check_all,
    ParameterFix, StructuralFix,
)
from .fix_selector import (
    rank_fixes, select_fixes, misrouting_severity, RankedFix,
)
from .config_applicator import apply_parameter_fixes


def load_probes(path: str) -> list[dict[str, str]]:
    """Load probes from YAML or JSON."""
    raw = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        import yaml
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    if isinstance(data, dict):
        for key in ("probes", "decisions"):
            if key in data:
                result = []
                for group in data[key]:
                    if "probes" in group:
                        for p in group["probes"]:
                            p.setdefault("expected_decision", group.get("expected_decision", group.get("id", "")))
                            result.append(p)
                    elif "query" in group:
                        result.append(group)
                return result if result else data.get("probes", [])
        return data.get("probes", [])
    return data


def split_probes(
    probes: list[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split probes into train and validation sets."""
    rng = random.Random(seed)
    shuffled = list(probes)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def hot_reload_router(config_path: str, endpoint: str, pid: int = 0) -> None:
    """Reload the router after config changes.

    Tries (in order): SIGHUP signal, fsnotify (touch file), or just wait.
    """
    if pid:
        try:
            os.kill(pid, signal.SIGHUP)
            print(f"  Sent SIGHUP to PID {pid}")
        except (ProcessLookupError, PermissionError):
            print(f"  SIGHUP failed for PID {pid}, relying on fsnotify")

    time.sleep(3)

    try:
        from urllib import request as urllib_request
        url = f"{endpoint.rstrip('/')}/config/hash"
        req = urllib_request.Request(url, method="GET")
        with urllib_request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            print(f"  Router config hash: {data.get('hash', 'unknown')[:16]}...")
    except Exception:
        print("  Could not verify reload via /config/hash")


def _print_causal_reports(reports: list[CausalReport]) -> None:
    by_type = defaultdict(list)
    for r in reports:
        by_type[r.fix_type].append(r)

    for ft, items in sorted(by_type.items()):
        print(f"\n  [{ft}] ({len(items)} probe(s))")
        for r in items[:5]:
            print(f"    {r.probe_id}: expected={r.expected_decision}, "
                  f"got={r.actual_decision}")
            if r.threshold_gaps:
                for g in r.threshold_gaps:
                    print(f"      {g.score_name}={g.score_value:.4f}, "
                          f"threshold={g.threshold}, gap={g.gap:+.4f}")
            if r.score_decompositions:
                for d in r.score_decompositions:
                    print(f"      {d.score_name} = {d.total:.4f}")
                    for c in d.contributions:
                        if c["contribution"] != 0:
                            print(f"        {c['signal']}: "
                                  f"w={c['weight']:.3f} × val={c['raw_value']:.3f} "
                                  f"= {c['contribution']:.4f}")
            if r.fix_detail:
                print(f"      → {r.fix_detail}")


def run_phase1_structural(
    endpoint: str,
    config_path: str,
    train_probes: list[dict],
    router_pid: int,
    max_iterations: int = 3,
) -> float:
    """Phase 1: Structural fixes via existing DSL Doctor pipeline.

    Returns the accuracy after structural fixes.
    """
    print("\n" + "=" * 70)
    print("  PHASE 1: Structural Fixes (DSL Doctor)")
    print("=" * 70)

    scripts_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(scripts_dir))

    try:
        from live_fix_pipeline import (
            diagnose_from_runtime, apply_config_fixes,
            run_all_probes, print_accuracy,
        )
    except ImportError:
        print("  WARNING: live_fix_pipeline not available, skipping Phase 1")
        cache = collect_traces(endpoint, train_probes, config_path)
        return cache.accuracy

    for iteration in range(1, max_iterations + 1):
        print(f"\n  --- Phase 1 Iteration {iteration} ---")
        results = run_all_probes(endpoint, train_probes, trace=True)
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        pct = 100 * correct / total if total else 0
        print(f"  Accuracy: {correct}/{total} ({pct:.0f}%)")

        if all(r["correct"] for r in results):
            print("  All probes pass — structural fixes complete")
            return 1.0

        diagnosis = diagnose_from_runtime(results, config_path)
        if not diagnosis.get("fixes"):
            print("  No structural fixes available")
            break

        applied = apply_config_fixes(config_path, diagnosis["fixes"])
        for msg in applied:
            print(f"    APPLIED: {msg}")

        if not applied:
            break

        hot_reload_router(config_path, endpoint, router_pid)

    final = run_all_probes(endpoint, train_probes, trace=False)
    return sum(1 for r in final if r["correct"]) / len(final)


def run_phase2_causal(
    endpoint: str,
    config_path: str,
    train_probes: list[dict],
    val_probes: list[dict],
    router_pid: int,
    max_iterations: int = 5,
) -> float:
    """Phase 2: Causal parameter tuning.

    Uses trace-based analytical optimization to tune projection weights
    and thresholds.
    """
    print("\n" + "=" * 70)
    print("  PHASE 2: Causal Parameter Tuning")
    print("=" * 70)

    iteration_history: list[dict[str, Any]] = []

    for iteration in range(1, max_iterations + 1):
        print(f"\n  --- Phase 2 Iteration {iteration} ---")

        # Step 1: Collect traces
        print("  Collecting traces...")
        cache = collect_traces(endpoint, train_probes, config_path)
        print(f"  Train accuracy: {cache.accuracy:.1%} "
              f"({sum(1 for p in cache.probes if p.correct)}/{len(cache.probes)})")

        if cache.accuracy >= 1.0:
            print("  All train probes pass!")
            break

        # Step 2: Load projection model
        model = load_projection_model(config_path)

        # Step 3: Causal analysis
        print("  Analyzing failures causally...")
        reports = analyze_all(cache.failures, model)
        _print_causal_reports(reports)

        # Step 4: Compute candidate fixes
        all_fixes: list[ParameterFix | StructuralFix] = []
        for report in reports:
            all_fixes.extend(compute_fixes_for_report(report, model))

        param_fixes = [f for f in all_fixes if isinstance(f, ParameterFix)]
        structural_fixes = [f for f in all_fixes if isinstance(f, StructuralFix)]

        print(f"\n  Candidate fixes: {len(param_fixes)} parameter, "
              f"{len(structural_fixes)} structural")

        if not param_fixes and not structural_fixes:
            print("  No fixes available — stopping")
            break

        # Step 5: Regression check parameter fixes
        if param_fixes:
            print("  Running regression checks (analytical, no router needed)...")
            param_fixes = regression_check_all(param_fixes, cache, model)

            for f in param_fixes:
                print(f"    {f.param_path}: {f.current_value}→{f.proposed_value} "
                      f"(+{f.probes_fixed}/-{f.probes_broken} = "
                      f"net {f.net_improvement:+d})")

        # Step 6: Select and rank
        probe_severity = {
            p.probe_id: misrouting_severity(p.expected_decision, p.actual_decision)
            for p in cache.failures
        }
        ranked = rank_fixes(all_fixes, probe_severity)
        selected_params, selected_structural = select_fixes(ranked)

        # Step 7: Apply parameter fixes
        if selected_params:
            print(f"\n  Applying {len(selected_params)} parameter fix(es)...")
            applied = apply_parameter_fixes(config_path, selected_params)
            for msg in applied:
                print(f"    {msg}")
            hot_reload_router(config_path, endpoint, router_pid)

        # Step 8: Validate on validation set
        if val_probes:
            print("\n  Validating on held-out set...")
            val_cache = collect_traces(endpoint, val_probes, config_path)
            print(f"  Validation accuracy: {val_cache.accuracy:.1%}")
        else:
            val_cache = None

        iteration_history.append({
            "iteration": iteration,
            "train_accuracy": cache.accuracy,
            "val_accuracy": val_cache.accuracy if val_cache else None,
            "param_fixes": len(selected_params),
            "structural_fixes": len(selected_structural),
        })

        # Step 9: Check for structural fixes that need the Doctor
        if selected_structural and not selected_params:
            print(f"\n  {len(selected_structural)} structural fix(es) needed — "
                  f"delegate to Doctor pipeline")
            for sf in selected_structural:
                print(f"    [{sf.fix_type}] {sf.decision_name}: {sf.description}")
            break

    # Final evaluation
    final_cache = collect_traces(endpoint, train_probes, config_path)
    print(f"\n  Final train accuracy: {final_cache.accuracy:.1%}")

    if val_probes:
        final_val = collect_traces(endpoint, val_probes, config_path)
        print(f"  Final validation accuracy: {final_val.accuracy:.1%}")

    return final_cache.accuracy


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DSL Tuning Pipeline — two-phase causal optimization"
    )
    parser.add_argument("--endpoint", default="http://localhost:8080")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--probes", required=True, help="Path to probes file")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--router-pid", type=int, default=0)
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip structural fixes, go straight to parameter tuning")
    parser.add_argument("--output", help="Write optimization log to file")

    args = parser.parse_args()

    print("=" * 70)
    print("  DSL Tuning Pipeline")
    print(f"  Endpoint: {args.endpoint}")
    print(f"  Config:   {args.config}")
    print(f"  Probes:   {args.probes}")
    print("=" * 70)

    probes = load_probes(args.probes)
    print(f"\nLoaded {len(probes)} probes")

    train_probes, val_probes = split_probes(probes, args.val_split)
    print(f"Split: {len(train_probes)} train, {len(val_probes)} validation")

    # Phase 1: Structural
    if not args.skip_phase1:
        phase1_acc = run_phase1_structural(
            args.endpoint, args.config, train_probes,
            args.router_pid, max_iterations=3,
        )
        print(f"\nPhase 1 result: {phase1_acc:.1%}")
    else:
        print("\nSkipping Phase 1 (structural fixes)")

    # Phase 2: Causal parameter tuning
    phase2_acc = run_phase2_causal(
        args.endpoint, args.config, train_probes, val_probes,
        args.router_pid, max_iterations=args.max_iterations,
    )

    # Summary
    print("\n" + "=" * 70)
    print("  TUNING COMPLETE")
    print("=" * 70)
    print(f"  Final accuracy: {phase2_acc:.1%}")

    if args.output:
        final_cache = collect_traces(args.endpoint, probes, args.config)
        log = {
            "final_accuracy": final_cache.accuracy,
            "config_hash": final_cache.config_hash,
            "failures": [
                {"id": p.probe_id, "expected": p.expected_decision,
                 "actual": p.actual_decision}
                for p in final_cache.failures
            ],
        }
        Path(args.output).write_text(json.dumps(log, indent=2))
        print(f"  Log written to {args.output}")

    return 0 if phase2_acc >= 0.9 else 1


if __name__ == "__main__":
    sys.exit(main())
