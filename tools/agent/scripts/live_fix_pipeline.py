#!/usr/bin/env python3
"""Live DSL Doctor Pipeline — observe, diagnose, fix, verify.

Sends real probes through a running router, collects traces,
diagnoses failures from runtime data, rewrites config, hot-reloads,
and re-probes to verify the fix.
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib import error, request

ENDPOINT = os.environ.get("ROUTER_ENDPOINT", "http://localhost:8080")
CONFIG_PATH = os.environ.get("ROUTER_CONFIG",
    "/data/semantic-router.bak/deploy/recipes/privacy/config.yaml")
ROUTER_PID = int(os.environ.get("ROUTER_PID", "0"))
PROBES_PATH = os.environ.get("PROBES_PATH",
    "/data/semantic-router.bak/deploy/recipes/privacy/privacy.probes.yaml")

# ── Probe loading ────────────────────────────────────────────────────────────

def load_probes(path: str) -> list[dict[str, str]]:
    """Load probes from the YAML file (simple parser, no PyYAML needed)."""
    raw = Path(path).read_text()
    probes: list[dict[str, str]] = []
    current_decision = ""
    current: dict[str, str] = {}
    in_query_block = False
    query_lines: list[str] = []

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("- id:") and "expected_decision:" in raw.split(stripped)[0].rsplit("decisions:", 1)[-1]:
            pass
        if re.match(r'^  - id:\s+\w+$', line) and not line.startswith("      "):
            if "expected_decision:" in stripped:
                pass
            else:
                current_decision = stripped.split("id:")[1].strip()
                continue
        if stripped.startswith("expected_decision:"):
            current_decision = stripped.split(":")[1].strip()
            continue
        if stripped.startswith("- id:") and line.startswith("      "):
            if current and "query" in current:
                probes.append(current)
            current = {"id": stripped.split("id:")[1].strip(),
                        "expected_decision": current_decision}
            in_query_block = False
            query_lines = []
            continue
        if stripped.startswith("query:"):
            rest = stripped[6:].strip()
            if rest and rest != "|":
                current["query"] = rest
                in_query_block = False
            else:
                in_query_block = True
                query_lines = []
            continue
        if in_query_block:
            if stripped.startswith("tags:") or stripped.startswith("- id:"):
                current["query"] = " ".join(query_lines)
                in_query_block = False
                if stripped.startswith("tags:"):
                    continue
            else:
                query_lines.append(stripped)
                continue

    if current and "query" in current:
        probes.append(current)

    return probes


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def eval_probe(endpoint: str, query: str, trace: bool = True) -> dict[str, Any]:
    url = f"{endpoint}/api/v1/eval" + ("?trace=true" if trace else "")
    body = json.dumps({"text": query}).encode()
    req = request.Request(url, data=body, method="POST",
                          headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ── Probe runner ─────────────────────────────────────────────────────────────

def run_all_probes(endpoint: str, probes: list[dict], trace: bool = True
                   ) -> list[dict[str, Any]]:
    results = []
    for p in probes:
        query = p["query"]
        expected = p["expected_decision"]
        resp = eval_probe(endpoint, query, trace=trace)
        dr = resp.get("decision_result", {})
        actual = dr.get("decision_name", "NONE")
        correct = actual == expected

        result = {
            "id": p.get("id", "?"),
            "query_preview": query[:80],
            "expected": expected,
            "actual": actual,
            "correct": correct,
            "matched_signals": dr.get("matched_signals", {}),
            "unmatched_signals": dr.get("unmatched_signals", {}),
            "used_signals": dr.get("used_signals", {}),
            "signal_confidences": resp.get("signal_confidences", {}),
            "eval_trace": dr.get("eval_trace", []),
        }
        results.append(result)
    return results


def print_accuracy(results: list[dict], label: str = ""):
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    pct = 100 * correct / total if total else 0
    print(f"\n{'='*70}")
    print(f"  {label}Accuracy: {correct}/{total} ({pct:.0f}%)")
    print(f"{'='*70}\n")

    by_decision = defaultdict(lambda: {"correct": 0, "total": 0, "wrong": []})
    for r in results:
        bucket = by_decision[r["expected"]]
        bucket["total"] += 1
        if r["correct"]:
            bucket["correct"] += 1
        else:
            bucket["wrong"].append(r)

    print(f"  {'Decision':<35} {'Correct':>7} {'Total':>5} {'Accuracy':>8}")
    print(f"  {'-'*35} {'-'*7} {'-'*5} {'-'*8}")
    for dec in sorted(by_decision):
        b = by_decision[dec]
        pct = 100 * b["correct"] / b["total"] if b["total"] else 0
        print(f"  {dec:<35} {b['correct']:>7} {b['total']:>5} {pct:>7.0f}%")

    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\n  Misrouted probes:")
        for r in wrong:
            print(f"    {r['id']}: expected={r['expected']}, got={r['actual']}")
            print(f"      query: {r['query_preview']}")


# ── Runtime trace diagnosis ──────────────────────────────────────────────────

def _load_decision_rules(config_path: str) -> dict[str, dict]:
    """Load decision rules from the running config to understand route structure."""
    import yaml
    config = yaml.safe_load(Path(config_path).read_text())
    decisions = {}
    for dec in config.get("routing", {}).get("decisions", []):
        name = dec["name"]
        rules = dec.get("rules", {})
        conditions = rules.get("conditions", [])
        operator = rules.get("operator", "AND")

        flat_conds = []
        for c in conditions:
            if "conditions" in c:
                for sub_c in c["conditions"]:
                    flat_conds.append(sub_c)
            else:
                flat_conds.append(c)

        decisions[name] = {
            "priority": dec.get("priority", 0),
            "operator": operator,
            "conditions": flat_conds,
            "category_kb_conds": [c for c in flat_conds if c.get("type") == "category_kb"],
            "projection_conds": [c for c in flat_conds if c.get("type") == "projection"],
        }
    return decisions


def diagnose_from_runtime(results: list[dict], config_path: str) -> dict[str, Any]:
    """Analyze runtime signal_confidences to discover failure patterns.

    Examines what the category_kb classifier actually returned for each
    probe to understand why it was misrouted. Works entirely from
    observed runtime data — no ground-truth DSL comparison.
    """
    failures = [r for r in results if not r["correct"]]
    if not failures:
        return {"status": "healthy", "failures": [], "fixes": []}

    decision_rules = _load_decision_rules(config_path)
    patterns: dict[str, list] = defaultdict(list)
    fixes: list[dict[str, Any]] = []

    for f in failures:
        actual = f["actual"]
        expected = f["expected"]
        confidences = f.get("signal_confidences", {})
        matched_projs = set(f.get("matched_signals", {}).get("projection", []))

        # Extract category_kb confidences
        ckb_scores = {}
        for k, v in confidences.items():
            if k.startswith("category_kb:") and not k.startswith("category_kb:__"):
                cat_name = k.split(":", 1)[1]
                ckb_scores[cat_name] = v

        # Find the best category and its tier
        best_cat = max(ckb_scores, key=ckb_scores.get) if ckb_scores else ""
        best_score = ckb_scores.get(best_cat, 0)

        # Get the actual decision's rule conditions
        actual_rules = decision_rules.get(actual, {})
        actual_ckb = actual_rules.get("category_kb_conds", [])
        actual_proj = actual_rules.get("projection_conds", [])

        # Check which category_kb conditions in the winning decision fired
        # (score > security_threshold 0.25)
        raw_ckb_in_actual = [c["name"] for c in actual_ckb if not c["name"].startswith("__")]
        fired_raw = [name for name in raw_ckb_in_actual if ckb_scores.get(name, 0) > 0.25]

        tier_ckb_in_actual = [c["name"] for c in actual_ckb if c["name"].startswith("__tier__:")]
        proj_in_actual_matched = [c["name"] for c in actual_proj if c["name"] in matched_projs]

        # Pattern 1: Binary catchall — raw category_kb names fire broadly
        if fired_raw and actual_rules.get("operator") == "OR":
            patterns["binary_catchall"].append({
                "probe": f["id"],
                "decision": actual,
                "raw_matches": fired_raw,
                "best_category": best_cat,
                "best_score": round(best_score, 3),
                "detail": (
                    f"Best category is '{best_cat}' ({best_score:.3f}), but "
                    f"raw conditions {fired_raw} all exceed security_threshold=0.25, "
                    f"so the P{actual_rules.get('priority',0)} OR fires as a catch-all"
                ),
            })
            continue

        # Pattern 2: Composition dominance — projection fires broadly in
        # an OR with __tier__, stealing probes from lower-priority decisions
        if proj_in_actual_matched and tier_ckb_in_actual and actual_rules.get("operator") == "OR":
            patterns["composition_dominance"].append({
                "probe": f["id"],
                "wrong_decision": actual,
                "wrong_via_projection": proj_in_actual_matched,
                "expected": expected,
                "best_category": best_cat,
                "best_score": round(best_score, 3),
            })
            continue

        # Pattern 3: projection fires broadly even without __tier__ OR
        if proj_in_actual_matched and not tier_ckb_in_actual:
            patterns["projection_overreach"].append({
                "probe": f["id"],
                "wrong_decision": actual,
                "wrong_via_projection": proj_in_actual_matched,
                "expected": expected,
                "best_category": best_cat,
            })
            continue

        # Pattern 4: Expected decision has no matching condition at all
        expected_rules = decision_rules.get(expected, {})
        if not expected_rules:
            patterns["missing_decision"].append({
                "probe": f["id"], "expected": expected,
            })
            continue

        patterns["unclassified"].append({
            "probe": f["id"], "expected": expected, "actual": actual,
            "best_category": best_cat, "best_score": round(best_score, 3),
        })

    # ── Generate fix recommendations from discovered patterns ────────────

    if patterns["binary_catchall"]:
        affected_decisions = set(p["decision"] for p in patterns["binary_catchall"])
        for dec in affected_decisions:
            examples = [p for p in patterns["binary_catchall"] if p["decision"] == dec]
            raw_names = set()
            for ex in examples:
                raw_names.update(ex["raw_matches"])
            best_cats = [ex["best_category"] for ex in examples]
            fixes.append({
                "type": "replace_binary_with_tier",
                "decision": dec,
                "description": (
                    f"Decision '{dec}' matches {len(examples)} probes via {len(raw_names)} "
                    f"raw category_kb names ({', '.join(sorted(raw_names))}). "
                    f"These all exceed security_threshold=0.25 even for irrelevant probes — "
                    f"the best actual categories were: {', '.join(set(best_cats))}. "
                    f"Replace raw names with category_kb('__tier__:<tier>') to use "
                    f"best-match ranking instead of binary threshold presence."
                ),
                "action": "replace_raw_category_kb_with_tier",
                "affected_probes": len(examples),
            })

    if patterns["composition_dominance"]:
        affected = defaultdict(list)
        for p in patterns["composition_dominance"]:
            affected[p["wrong_decision"]].append(p)
        for dec, examples in affected.items():
            proj_names = set()
            for ex in examples:
                proj_names.update(ex["wrong_via_projection"])
            fixes.append({
                "type": "remove_broad_projection_or",
                "decision": dec,
                "description": (
                    f"Decision '{dec}' steals {len(examples)} probes via projection "
                    f"({', '.join(sorted(proj_names))}) OR'd with __tier__. "
                    f"The projection fires broadly, defeating the precise tier signal. "
                    f"Remove the projection from the OR or demote to a lower-priority route."
                ),
                "action": "remove_projection_from_or",
                "projections_to_remove": sorted(proj_names),
                "affected_probes": len(examples),
            })

    if patterns["projection_overreach"]:
        affected = defaultdict(list)
        for p in patterns["projection_overreach"]:
            affected[p["wrong_decision"]].append(p)
        for dec, examples in affected.items():
            fixes.append({
                "type": "add_tier_primary",
                "decision": dec,
                "description": (
                    f"Decision '{dec}' matches {len(examples)} probes via projection alone. "
                    f"Add category_kb('__tier__:<tier>') as the primary condition to leverage "
                    f"the classifier's best-match ranking."
                ),
                "action": "add_tier_as_primary",
            })

    # Check all decisions that appear as 'expected' but never won:
    # they may need __tier__ added to actually receive their traffic
    expected_never_won = set(f["expected"] for f in failures) - set(f["actual"] for f in failures)
    for dec in expected_never_won:
        rules = decision_rules.get(dec, {})
        has_tier = any(
            c.get("name", "").startswith("__tier__:")
            for c in rules.get("category_kb_conds", [])
        )
        if not has_tier:
            fixes.append({
                "type": "add_tier_to_starved_decision",
                "decision": dec,
                "description": (
                    f"Decision '{dec}' never won any probe. It has no "
                    f"category_kb('__tier__:...') condition. Add one so the "
                    f"classifier's best-match ranking can route probes to it."
                ),
                "action": "add_tier_as_primary",
            })

    return {
        "status": "unhealthy",
        "total_failures": len(failures),
        "patterns": {k: len(v) for k, v in patterns.items()},
        "pattern_details": dict(patterns),
        "fixes": fixes,
    }


def _extract_leaf_signals(trace_node: dict) -> list[dict]:
    """Recursively extract leaf signal info from a trace tree."""
    if not trace_node:
        return []
    if trace_node.get("node_type") == "leaf":
        return [{
            "type": trace_node.get("signal_type", ""),
            "name": trace_node.get("signal_name", ""),
            "matched": trace_node.get("matched", False),
            "confidence": trace_node.get("confidence", 0),
        }]
    children = trace_node.get("children", [])
    result = []
    for c in children:
        result.extend(_extract_leaf_signals(c))
    return result


# ── Config rewriter ──────────────────────────────────────────────────────────

def apply_config_fixes(config_path: str, fixes: list[dict]) -> list[str]:
    """Apply diagnosed fixes using yaml.safe_load/dump for consistency."""
    import yaml
    config = yaml.safe_load(Path(config_path).read_text())
    decisions = config.get("routing", {}).get("decisions", [])
    applied: list[str] = []

    for fix in fixes:
        action = fix.get("action", "")
        decision_name = fix.get("decision", "")
        dec = next((d for d in decisions if d.get("name") == decision_name), None)
        if not dec:
            continue

        rules = dec.get("rules", {})
        conditions = rules.get("conditions", [])

        if action == "replace_raw_category_kb_with_tier":
            tier_name = _infer_tier(decision_name)
            removed = [c["name"] for c in conditions
                       if c.get("type") == "category_kb" and not c["name"].startswith("__")]
            new_conds = [c for c in conditions
                         if not (c.get("type") == "category_kb" and not c["name"].startswith("__"))]
            has_tier = any(c.get("name", "").startswith("__tier__:")
                          for c in new_conds if c.get("type") == "category_kb")
            if not has_tier:
                new_conds.insert(0, {"name": f"__tier__:{tier_name}", "type": "category_kb"})
            rules["conditions"] = new_conds
            applied.append(
                f"[{decision_name}] Replaced {len(removed)} raw category_kb "
                f"({', '.join(removed)}) with __tier__:{tier_name}"
            )

        elif action == "remove_projection_from_or":
            proj_to_remove = set(fix.get("projections_to_remove", []))
            removed = [c["name"] for c in conditions
                       if c.get("type") == "projection" and c["name"] in proj_to_remove]
            new_conds = [c for c in conditions
                         if not (c.get("type") == "projection" and c["name"] in proj_to_remove)]
            rules["conditions"] = new_conds
            if len(new_conds) == 1:
                rules["operator"] = "AND"
            applied.append(
                f"[{decision_name}] Removed broad projection OR: {', '.join(removed)}"
            )

        elif action == "add_tier_as_primary":
            tier_name = _infer_tier(decision_name)
            has_tier = any(c.get("name", "").startswith("__tier__:")
                          for c in conditions if c.get("type") == "category_kb")
            if not has_tier:
                if rules.get("operator") == "AND" and conditions:
                    rules["conditions"] = [
                        {"name": f"__tier__:{tier_name}", "type": "category_kb"},
                        {"conditions": list(conditions), "operator": "AND"},
                    ]
                    rules["operator"] = "OR"
                else:
                    conditions.insert(0, {"name": f"__tier__:{tier_name}", "type": "category_kb"})
                applied.append(f"[{decision_name}] Added __tier__:{tier_name} as primary signal")

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return applied


def _infer_tier(decision_name: str) -> str:
    for prefix in ("local_", "cloud_", "onprem_", "edge_"):
        if decision_name.startswith(prefix):
            candidate = decision_name[len(prefix):]
            if candidate and candidate != "standard":
                return candidate
    return decision_name


def hot_reload_router(pid: int) -> int:
    """Send SIGHUP to the router to trigger config reload. Restarts if needed.
    Returns the (possibly new) PID."""
    if not pid:
        print("  WARNING: No ROUTER_PID set.")
        return pid

    try:
        os.kill(pid, signal.SIGHUP)
        time.sleep(5)
        os.kill(pid, 0)  # check alive
        print(f"  Sent SIGHUP to PID {pid}, reload complete.")
        return pid
    except ProcessLookupError:
        print(f"  Router (PID {pid}) died on SIGHUP — restarting...")
        import subprocess
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ":".join([
            "/data/semantic-router.bak/candle-binding/target/release",
            "/data/semantic-router.bak/nlp-binding/target/release",
            "/data/semantic-router.bak/ml-binding/target/release",
        ])
        proc = subprocess.Popen(
            ["/data/semantic-router.bak/bin/router",
             "-config=config.yaml", "--enable-system-prompt-api=true"],
            cwd="/data/semantic-router.bak/deploy/recipes/privacy",
            stdout=open("/tmp/router.log", "a"),
            stderr=subprocess.STDOUT,
            env=env,
        )
        new_pid = proc.pid
        print(f"  Restarted router with PID {new_pid}, waiting for init...")
        time.sleep(12)
        return new_pid


# ── Main pipeline ────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 70)
    print("  DSL Doctor Live Pipeline")
    print("  Router: %s  Config: %s" % (ENDPOINT, CONFIG_PATH))
    print("=" * 70)

    probes = load_probes(PROBES_PATH)
    print(f"\nLoaded {len(probes)} probes from {PROBES_PATH}")
    for p in probes:
        print(f"  {p['id']:30s} → expects {p['expected_decision']}")

    max_iterations = 5
    all_applied: list[str] = []
    iteration_history: list[tuple[int, int]] = []
    router_pid = ROUTER_PID

    for iteration in range(1, max_iterations + 1):
        label = f"ITERATION {iteration}"

        # ── Observe ──────────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"  {label}: Probe the router")
        print(f"{'='*70}")

        results = run_all_probes(ENDPOINT, probes, trace=True)
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        iteration_history.append((correct, total))
        print_accuracy(results, f"{label} — ")

        if all(r["correct"] for r in results):
            print(f"  All probes pass after {iteration} iteration(s). Converged!")
            break

        # ── Diagnose ─────────────────────────────────────────────────────
        print(f"  --- Diagnosis ---")
        diagnosis = diagnose_from_runtime(results, CONFIG_PATH)
        print(f"  Status: {diagnosis['status']}, failures: {diagnosis['total_failures']}")
        for pattern, count in diagnosis.get("patterns", {}).items():
            if count > 0:
                print(f"    {pattern}: {count} probe(s)")

        for pattern, details in diagnosis.get("pattern_details", {}).items():
            if not details:
                continue
            print(f"\n  [{pattern}]")
            for d in details[:5]:
                if pattern == "binary_catchall":
                    print(f"    {d['probe']}: '{d['decision']}' swallows via "
                          f"raw category_kb {d['raw_matches']} "
                          f"(best={d['best_category']} {d['best_score']})")
                elif pattern == "composition_dominance":
                    print(f"    {d['probe']}: '{d['wrong_decision']}' steals via "
                          f"projection {d['wrong_via_projection']}, "
                          f"should be '{d['expected']}' "
                          f"(best={d['best_category']} {d['best_score']})")
                elif pattern == "projection_overreach":
                    print(f"    {d['probe']}: '{d['wrong_decision']}' fires via "
                          f"projection {d['wrong_via_projection']}, "
                          f"expected '{d['expected']}'")
                else:
                    print(f"    {json.dumps(d, default=str)}")

        print(f"\n  Fixes ({len(diagnosis['fixes'])}):")
        for i, fix in enumerate(diagnosis["fixes"], 1):
            desc = fix['description']
            print(f"    {i}. [{fix['type']}] {fix['decision']}")
            print(f"       {desc[:150]}{'...' if len(desc) > 150 else ''}")

        if not diagnosis["fixes"]:
            print("  No automated fixes — stopping.")
            break

        # ── Apply ────────────────────────────────────────────────────────
        print(f"\n  --- Applying fixes ---")
        applied = apply_config_fixes(CONFIG_PATH, diagnosis["fixes"])
        for msg in applied:
            print(f"    APPLIED: {msg}")
            all_applied.append(msg)

        if not applied:
            print("    No changes made — stopping.")
            break

        # ── Hot-reload ───────────────────────────────────────────────────
        print(f"\n  --- Hot-reloading router ---")
        router_pid = hot_reload_router(router_pid)

    # ── Final summary ────────────────────────────────────────────────────
    final_results = run_all_probes(ENDPOINT, probes, trace=False)
    final_correct = sum(1 for r in final_results if r["correct"])

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Iteration history:")
    for i, (c, t) in enumerate(iteration_history):
        print(f"    {'Initial' if i == 0 else f'After fix {i}'}: "
              f"{c}/{t} ({100*c/t:.0f}%)")

    initial_correct = iteration_history[0][0] if iteration_history else 0
    total = len(probes)
    print(f"\n  Before: {initial_correct}/{total} ({100*initial_correct/total:.0f}%)")
    print(f"  After:  {final_correct}/{total} ({100*final_correct/total:.0f}%)")
    delta = final_correct - initial_correct
    print(f"  Improvement: +{delta} probes (+{100*delta/total:.0f}pp)")

    print(f"\n  Total fixes applied: {len(all_applied)}")
    for msg in all_applied:
        print(f"    {msg}")

    remaining = [r for r in final_results if not r["correct"]]
    if remaining:
        print(f"\n  Remaining failures ({len(remaining)}):")
        for r in remaining:
            print(f"    {r['id']}: expected={r['expected']}, got={r['actual']}")
    else:
        print(f"\n  All probes pass!")

    return 0 if final_correct > initial_correct else 1


if __name__ == "__main__":
    sys.exit(main())
