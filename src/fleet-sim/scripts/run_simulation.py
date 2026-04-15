#!/usr/bin/env python3
"""Full simulation runner: Stage-aware routing validation for coding agents.

Runs all 6 routing policies × 2 datasets × 2 fleet configs, plus burst
and stress tests. Collects fleet metrics, phase telemetry, and prefix-cache
risk analysis.

Section 9 of the validation plan (Experimental Procedure).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fleet_sim.core.fleet import Fleet, FleetConfig, PoolConfig
from fleet_sim.gpu_profiles.profiles import A100_80GB, H100_80GB
from fleet_sim.routing.stage_policies import (
    stage_classify,
    stage_decode_volume_classify,
    stage_interference_classify_factory,
)
from fleet_sim.workload.trace import TraceWorkload

# ── Configuration ─────────────────────────────────────────────────────────────

SLO_MS = 500.0
TRACES = {
    "swe-agent": "examples/trace_samples/swe_agent_annotated.jsonl",
    "agentbench": "examples/trace_samples/agentbench_annotated.jsonl",
}

# Config A: Homogeneous fleet (60× A100-80GB, single pool)
HOMOGENEOUS_POOLS = [
    PoolConfig("default_pool", A100_80GB, 60, 8192),
]

# Config B: Phase-split fleet (40× A100 prefill + 20× H100 decode)
PHASE_SPLIT_POOLS = [
    PoolConfig("prefill_pool", A100_80GB, 40, 8192),
    PoolConfig("decode_pool", H100_80GB, 20, 65536),
]

FLEET_CONFIGS = {
    "homogeneous": HOMOGENEOUS_POOLS,
    "phase_split": PHASE_SPLIT_POOLS,
}

ARRIVAL_RATES = {
    "steady": 200,
    "burst_2x": 400,
    "burst_5x": 1000,
}


def build_policies(fleet_type: str) -> dict:
    """Build all 6 routing policies for the given fleet type.

    For homogeneous fleets, the phase-split policies fall back to the single
    pool (SemanticRouter default_pool). This is expected — the comparison
    shows whether phase-splitting + routing provides gains.
    """
    policies = {}

    if fleet_type == "homogeneous":
        # Baselines on single-pool fleet
        policies["1_random"] = FleetConfig(
            pools=list(HOMOGENEOUS_POOLS),
            router_type="RandomRouter",
        )
        policies["2_round_robin"] = FleetConfig(
            pools=list(HOMOGENEOUS_POOLS),
            router_type="RoundRobinRouter",
        )
        policies["3_least_loaded"] = FleetConfig(
            pools=list(HOMOGENEOUS_POOLS),
            router_type="LeastLoadedRouter",
        )
        # Experimental: on homogeneous fleet, semantic routing still goes to single pool
        # (demonstrates that stage-aware routing needs phase-split fleet to unlock gains)
        policies["4_stage_aware"] = FleetConfig(
            pools=list(HOMOGENEOUS_POOLS),
            router_type="SemanticRouter",
            router_kwargs={"classify_fn": lambda req: "default_pool"},
        )
        policies["5_stage_decode_vol"] = FleetConfig(
            pools=list(HOMOGENEOUS_POOLS),
            router_type="SemanticRouter",
            router_kwargs={"classify_fn": lambda req: "default_pool"},
        )
        policies["6_stage_interference"] = FleetConfig(
            pools=list(HOMOGENEOUS_POOLS),
            router_type="SemanticRouter",
            router_kwargs={"classify_fn": lambda req: "default_pool"},
        )
    else:
        # Phase-split fleet
        policies["1_random"] = FleetConfig(
            pools=list(PHASE_SPLIT_POOLS),
            router_type="RandomRouter",
        )
        policies["2_round_robin"] = FleetConfig(
            pools=list(PHASE_SPLIT_POOLS),
            router_type="RoundRobinRouter",
        )
        policies["3_least_loaded"] = FleetConfig(
            pools=list(PHASE_SPLIT_POOLS),
            router_type="LeastLoadedRouter",
        )
        policies["4_stage_aware"] = FleetConfig(
            pools=list(PHASE_SPLIT_POOLS),
            router_type="SemanticRouter",
            router_kwargs={"classify_fn": stage_classify},
        )
        policies["5_stage_decode_vol"] = FleetConfig(
            pools=list(PHASE_SPLIT_POOLS),
            router_type="SemanticRouter",
            router_kwargs={"classify_fn": stage_decode_volume_classify},
        )

        # Policy 6: interference avoidance needs live pool refs
        live_pools_ref = [None]

        def make_interference_policy():
            classify = stage_interference_classify_factory(live_pools_ref)
            cfg = FleetConfig(
                pools=list(PHASE_SPLIT_POOLS),
                router_type="SemanticRouter",
                router_kwargs={"classify_fn": classify},
            )
            return cfg, live_pools_ref

        cfg6, ref6 = make_interference_policy()
        policies["6_stage_interference"] = cfg6
        policies["_6_ref"] = ref6  # stash for wiring

    return policies


def run_single(
    policy_name: str,
    fleet_config: FleetConfig,
    arrivals: list,
    live_pools_ref=None,
) -> dict:
    """Run a single simulation and return metrics dict."""
    fleet = Fleet(fleet_config)
    fleet._build()

    # Wire live pool refs for policy 6 interference avoidance
    if live_pools_ref is not None:
        live_pools_ref[0] = fleet._pools

    result = fleet.run(arrivals)

    metrics = {
        "throughput_req_s": round(result.throughput(), 2),
        "token_throughput": round(result.token_throughput(), 0),
        "p50_ttft_ms": round(result.p50_ttft_ms(), 2),
        "p95_ttft_ms": round(result.p95_ttft_ms(), 2),
        "p99_ttft_ms": round(result.p99_ttft_ms(), 2),
        "p50_e2e_ms": round(result.p50_e2e_ms(), 2),
        "p99_e2e_ms": round(result.p99_e2e_ms(), 2),
        "p99_qwait_ms": round(result.p99_queue_wait_ms(), 2),
        "slo_compliance": round(result.slo_compliance(SLO_MS), 4),
        "mean_util": round(result.mean_utilisation(), 4),
        "util_variance": round(result.utilisation_variance(), 6),
        "interference_ratio": round(result.mean_interference_ratio(), 4),
        "total_completed": len(result.completed),
        "total_gpus": result.total_gpus(),
        "cost_per_hr": round(result.cost_per_hr(), 2),
    }

    risk = result.compute_prefix_risk()
    metrics.update({
        "migration_count": risk["session_migration_count"],
        "migration_rate": round(risk["migration_rate"], 4),
        "reprefill_amplification": round(risk["repeated_prefill_amplification"], 4),
        "worst_reprefill": risk["worst_case_reprefill"],
    })

    # Per-pool metrics
    for pool_id in result.pools:
        metrics[f"{pool_id}_util"] = round(result.mean_utilisation(pool_id), 4)
        metrics[f"{pool_id}_p99_ttft"] = round(result.p99_ttft_ms(pool_id), 2)
        metrics[f"{pool_id}_interference"] = round(
            result.mean_interference_ratio(pool_id), 4
        )

    return metrics


def format_table(results: list[dict], title: str) -> str:
    """Format results as an aligned text table."""
    if not results:
        return ""

    lines = [f"\n{'='*120}", f"  {title}", f"{'='*120}"]

    key_cols = [
        ("policy", 24),
        ("thru_req/s", 11),
        ("tok_thru", 10),
        ("p50_ttft", 10),
        ("p95_ttft", 10),
        ("p99_ttft", 10),
        ("p99_e2e", 10),
        ("SLO%", 8),
        ("util%", 7),
        ("interf%", 8),
        ("migr_rate", 10),
        ("reprefill%", 10),
    ]

    header = "  ".join(f"{name:>{width}}" for name, width in key_cols)
    lines.append(f"  {header}")
    lines.append(f"  {'-' * len(header)}")

    for r in results:
        vals = [
            f"{r['policy']:>24}",
            f"{r['throughput_req_s']:>11.2f}",
            f"{r['token_throughput']:>10.0f}",
            f"{r['p50_ttft_ms']:>10.2f}",
            f"{r['p95_ttft_ms']:>10.2f}",
            f"{r['p99_ttft_ms']:>10.2f}",
            f"{r['p99_e2e_ms']:>10.2f}",
            f"{r['slo_compliance']*100:>8.2f}",
            f"{r['mean_util']*100:>7.1f}",
            f"{r['interference_ratio']*100:>8.2f}",
            f"{r['migration_rate']:>10.4f}",
            f"{r['reprefill_amplification']*100:>10.2f}",
        ]
        lines.append(f"  {'  '.join(vals)}")

    lines.append("")
    return "\n".join(lines)


def run_experiment_set(
    dataset_name: str,
    trace_path: str,
    fleet_type: str,
    arrival_rate: int,
    rate_label: str,
) -> list[dict]:
    """Run all 6 policies for one dataset × fleet × arrival rate combo."""
    wl = TraceWorkload(
        path=trace_path,
        fmt="semantic_router",
        scale_lam=arrival_rate,
        field_map={
            "stage": "stage",
            "session_id": "session_id",
            "turn_index": "turn_index",
            "projected_decode_tokens": "projected_decode_tokens",
        },
    )
    arrivals = wl.generate()

    policies = build_policies(fleet_type)
    live_ref = policies.pop("_6_ref", None)

    results = []
    for policy_name, fc in sorted(policies.items()):
        t0 = time.time()
        ref = live_ref if "interference" in policy_name else None
        metrics = run_single(policy_name, fc, arrivals, live_pools_ref=ref)
        elapsed = time.time() - t0
        metrics["policy"] = policy_name
        metrics["dataset"] = dataset_name
        metrics["fleet"] = fleet_type
        metrics["rate_label"] = rate_label
        metrics["sim_time_s"] = round(elapsed, 2)
        results.append(metrics)
        print(f"    {policy_name:28s}  done in {elapsed:.1f}s  "
              f"(thru={metrics['throughput_req_s']:.1f} req/s, "
              f"p99={metrics['p99_ttft_ms']:.1f}ms, "
              f"SLO={metrics['slo_compliance']*100:.1f}%)")

    return results


def run_stress_tests(dataset_name: str, trace_path: str) -> list[dict]:
    """Section 9.3 stress tests: decode-skewed and synchronized bursts."""
    import json

    all_results = []

    # Stress 1: Decode-heavy only (filter to implement/test stages)
    print(f"\n  Stress test: decode-heavy skew ({dataset_name})")
    with open(trace_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    decode_records = [r for r in records if r.get("stage") in ("implement", "test")]
    if decode_records:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            for r in decode_records:
                tmp.write(json.dumps(r) + "\n")
            tmp_path = tmp.name

        results = run_experiment_set(
            dataset_name=f"{dataset_name}_decode_skew",
            trace_path=tmp_path,
            fleet_type="phase_split",
            arrival_rate=200,
            rate_label="decode_skew",
        )
        all_results.extend(results)
        Path(tmp_path).unlink()

    # Stress 2: Synchronized multi-agent burst (N sessions start at t=0)
    print(f"\n  Stress test: synchronized burst ({dataset_name})")
    sync_records = []
    for r in records[:200]:  # take first 200 requests
        r2 = dict(r)
        r2["timestamp"] = 0.0  # all arrive at t=0
        sync_records.append(r2)
    if sync_records:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            for r in sync_records:
                tmp.write(json.dumps(r) + "\n")
            tmp_path = tmp.name

        results = run_experiment_set(
            dataset_name=f"{dataset_name}_sync_burst",
            trace_path=tmp_path,
            fleet_type="phase_split",
            arrival_rate=None,  # use original timestamps (all at t=0)
            rate_label="sync_burst",
        )
        all_results.extend(results)
        Path(tmp_path).unlink()

    return all_results


def main():
    print("=" * 120)
    print("  SEMANTIC STAGE-AWARE ROUTING VALIDATION FOR CODING AGENTS")
    print("  Fleet Simulator Study")
    print("=" * 120)

    all_results = []
    overall_start = time.time()

    # ── Main experiment matrix ────────────────────────────────────────────────
    # For each dataset × fleet config: steady-state (200 req/s), burst (400, 1000 req/s)

    for dataset_name, trace_path in TRACES.items():
        for fleet_type, _pools in FLEET_CONFIGS.items():
            for rate_label, arrival_rate in ARRIVAL_RATES.items():
                title = f"{dataset_name} / {fleet_type} / {rate_label} ({arrival_rate} req/s)"
                print(f"\n{'─'*80}")
                print(f"  Running: {title}")
                print(f"{'─'*80}")

                results = run_experiment_set(
                    dataset_name=dataset_name,
                    trace_path=trace_path,
                    fleet_type=fleet_type,
                    arrival_rate=arrival_rate,
                    rate_label=rate_label,
                )
                all_results.extend(results)

                print(format_table(
                    results,
                    title=title,
                ))

    # ── Stress tests ──────────────────────────────────────────────────────────
    print(f"\n{'═'*120}")
    print("  STRESS TESTS")
    print(f"{'═'*120}")

    for dataset_name, trace_path in TRACES.items():
        stress_results = run_stress_tests(dataset_name, trace_path)
        all_results.extend(stress_results)
        for r in stress_results:
            pass  # already printed inline

    # ── Final comparative summary ─────────────────────────────────────────────
    print(f"\n{'═'*120}")
    print("  COMPARATIVE SUMMARY: Phase-Split Fleet, Steady State (200 req/s)")
    print(f"{'═'*120}")

    steady_phase = [
        r for r in all_results
        if r.get("fleet") == "phase_split" and r.get("rate_label") == "steady"
    ]
    for ds in ["swe-agent", "agentbench"]:
        ds_results = [r for r in steady_phase if r.get("dataset") == ds]
        if ds_results:
            print(format_table(ds_results, title=f"Phase-split / {ds} / steady 200 req/s"))

    # ── Homogeneous vs Phase-Split comparison ─────────────────────────────────
    print(f"\n{'═'*120}")
    print("  FLEET ARCHITECTURE COMPARISON: Homogeneous vs Phase-Split (Steady 200 req/s)")
    print(f"{'═'*120}")

    for ds in ["swe-agent", "agentbench"]:
        comparison = [
            r for r in all_results
            if r.get("dataset") == ds and r.get("rate_label") == "steady"
        ]
        if comparison:
            for r in comparison:
                r["policy"] = f"{r['fleet'][:5]}_{r['policy']}"
            print(format_table(comparison, title=f"{ds}: homogeneous vs phase-split"))

    # ── Risk analysis ─────────────────────────────────────────────────────────
    print(f"\n{'═'*120}")
    print("  PREFIX-CACHE RISK ANALYSIS")
    print(f"{'═'*120}")

    risk_results = [
        r for r in all_results
        if r.get("fleet") == "phase_split" and r.get("rate_label") == "steady"
    ]
    for r in risk_results:
        flag = " ⚠ EXCEEDS 10%" if r.get("reprefill_amplification", 0) > 0.10 else " ✓ OK"
        print(f"  {r.get('dataset', ''):12s} {r.get('policy', ''):28s}  "
              f"migrations={r.get('migration_count', 0):5d}  "
              f"rate={r.get('migration_rate', 0):.4f}  "
              f"reprefill_amp={r.get('reprefill_amplification', 0)*100:.2f}%{flag}")

    total_time = time.time() - overall_start
    print(f"\n{'═'*120}")
    print(f"  Total simulation time: {total_time:.1f}s")
    print(f"  Total experiment runs: {len(all_results)}")
    print(f"{'═'*120}")


if __name__ == "__main__":
    main()
