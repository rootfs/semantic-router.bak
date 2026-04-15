#!/usr/bin/env python3
"""High GPU utilization analysis for stage-aware routing.

Tests stage-aware routing under realistic production utilization (30-80%)
by using small fleets with moderate-to-high arrival rates. Compares
homogeneous vs phase-split fleets and baselines vs stage-aware routing
to show how interference and queuing behave under load.

Scenarios:
  1. Utilization sweep    — find fleet sizes that produce 30-80% util
  2. Routing comparison at high util — all policies on a tight fleet
  3. Saturation cliff     — push arrival rate until fleet saturates
  4. Interference under load — homogeneous vs phase-split at high util
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fleet_sim.core.fleet import Fleet, FleetConfig, PoolConfig
from fleet_sim.gpu_profiles.profiles import A100_80GB, H100_80GB
from fleet_sim.routing.stage_policies import stage_classify
from fleet_sim.workload.trace import TraceWorkload

TRACE = "examples/trace_samples/swe_agent_annotated.jsonl"
FMAP = {
    "stage": "stage",
    "session_id": "session_id",
    "turn_index": "turn_index",
    "projected_decode_tokens": "projected_decode_tokens",
}
SLO_MS = 500.0


def run_sim(fc, arrivals, slo=SLO_MS):
    fleet = Fleet(fc)
    result = fleet.run(arrivals)
    risk = result.compute_prefix_risk()
    return {
        "thru": round(result.throughput(), 1),
        "p50_ttft": round(result.p50_ttft_ms(), 1),
        "p99_ttft": round(result.p99_ttft_ms(), 1),
        "p99_e2e": round(result.p99_e2e_ms(), 1),
        "slo": round(result.slo_compliance(slo) * 100, 1),
        "util": round(result.mean_utilisation() * 100, 1),
        "cost": round(fc.total_cost_per_hr(), 2),
        "migr": round(risk["migration_rate"] * 100, 1),
        "reprfll": round(risk["repeated_prefill_amplification"] * 100, 1),
        "completed": len(result.completed),
        "total_gpus": fc.total_gpus(),
    }


def print_header(title):
    print(f"\n{'=' * 120}")
    print(f"  {title}")
    print(f"{'=' * 120}")


def print_row_header():
    hdr = (
        f"  {'Label':>36s}  {'thru':>6s}  {'p50':>7s}  {'p99':>7s}  "
        f"{'p99e2e':>8s}  {'SLO%':>6s}  {'util%':>6s}  {'$/hr':>7s}  "
        f"{'migr%':>6s}  {'reprfll%':>8s}"
    )
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")


def print_row(label, r):
    print(
        f"  {label:>36s}  {r['thru']:>6.1f}  {r['p50_ttft']:>7.1f}  "
        f"{r['p99_ttft']:>7.1f}  {r['p99_e2e']:>8.1f}  {r['slo']:>6.1f}  "
        f"{r['util']:>6.1f}  {r['cost']:>7.2f}  {r['migr']:>6.1f}  "
        f"{r['reprfll']:>8.1f}"
    )


def make_arrivals(rate):
    wl = TraceWorkload(
        path=TRACE, fmt="semantic_router", scale_lam=rate, field_map=FMAP
    )
    return wl.generate()


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 1: Utilization sweep — small fleets to drive high util
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_util_sweep():
    """Shrink fleet to drive utilization up, compare stage-aware vs baselines."""
    print_header(
        "SCENARIO 1: Utilization Sweep — shrink fleet to drive GPU util to 30-80%+"
    )
    print("  Fixed arrival rate: 200 req/s. Vary fleet size from 60 GPUs down to 3.")
    print("  Phase-split fleet with 2:1 A100:H100 ratio, stage-aware routing.\n")

    arrivals = make_arrivals(200)
    all_results = []

    configs = [
        (40, 20, "40A+20H (60 GPU, baseline)"),
        (20, 10, "20A+10H (30 GPU)"),
        (10, 5, "10A+5H (15 GPU)"),
        (6, 3, "6A+3H (9 GPU)"),
        (4, 2, "4A+2H (6 GPU)"),
        (3, 1, "3A+1H (4 GPU)"),
        (2, 1, "2A+1H (3 GPU)"),
    ]

    print_row_header()

    for na, nh, label in configs:
        pools = [
            PoolConfig("prefill_pool", A100_80GB, na, 8192),
            PoolConfig("decode_pool", H100_80GB, nh, 65536),
        ]
        fc = FleetConfig(
            pools=pools,
            router_type="SemanticRouter",
            router_kwargs={"classify_fn": stage_classify},
        )
        r = run_sim(fc, arrivals)
        print_row(label, r)
        r["label"] = label
        all_results.append(r)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 2: All policies at high utilization
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_policies_high_util():
    """Compare all routing policies on a tight fleet at high utilization."""
    print_header(
        "SCENARIO 2: All Policies at High Utilization"
    )
    print("  Tight fleets: 4A+2H phase-split and 6A homogeneous, 200 req/s.\n")

    arrivals = make_arrivals(200)
    all_results = []

    # Phase-split: 4A + 2H
    print("  --- Phase-split fleet: 4xA100 + 2xH100 (6 GPU) ---")
    print_row_header()

    phase_pools = [
        PoolConfig("prefill_pool", A100_80GB, 4, 8192),
        PoolConfig("decode_pool", H100_80GB, 2, 65536),
    ]

    policies = [
        ("Random", "RandomRouter", {}),
        ("Round-Robin", "RoundRobinRouter", {}),
        ("Least-Loaded", "LeastLoadedRouter", {}),
        ("Stage-Aware", "SemanticRouter", {"classify_fn": stage_classify}),
    ]

    for name, router_type, kwargs in policies:
        fc = FleetConfig(
            pools=list(phase_pools),
            router_type=router_type,
            router_kwargs=kwargs,
        )
        r = run_sim(fc, arrivals)
        label = f"Phase-split {name}"
        print_row(label, r)
        r["label"] = label
        r["fleet"] = "phase_split"
        all_results.append(r)

    # Homogeneous: 6A
    print(f"\n  --- Homogeneous fleet: 6xA100 (6 GPU) ---")
    print_row_header()

    homo_pools = [PoolConfig("default_pool", A100_80GB, 6, 8192)]

    for name, router_type, kwargs in policies[:3]:
        if router_type == "SemanticRouter":
            continue
        fc = FleetConfig(
            pools=list(homo_pools),
            router_type=router_type,
            router_kwargs=kwargs,
        )
        r = run_sim(fc, arrivals)
        label = f"Homogeneous {name}"
        print_row(label, r)
        r["label"] = label
        r["fleet"] = "homogeneous"
        all_results.append(r)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 3: Saturation cliff — push until fleet breaks
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_saturation():
    """Find the saturation point for different fleet sizes."""
    print_header(
        "SCENARIO 3: Saturation Cliff — push arrival rate until SLO breaks"
    )
    print("  Stage-aware routing on phase-split fleets of varying size.\n")

    all_results = []

    fleet_sizes = [
        (4, 2, "4A+2H (6 GPU)"),
        (6, 3, "6A+3H (9 GPU)"),
        (10, 5, "10A+5H (15 GPU)"),
        (20, 10, "20A+10H (30 GPU)"),
    ]

    rates = [50, 100, 200, 400, 800, 1500, 3000, 5000]

    for na, nh, fleet_label in fleet_sizes:
        print(f"  --- {fleet_label} ---")
        hdr = (
            f"  {'Rate':>8s}  {'thru':>6s}  {'p50':>7s}  {'p99':>7s}  "
            f"{'p99e2e':>8s}  {'SLO%':>6s}  {'util%':>6s}  {'migr%':>6s}"
        )
        print(hdr)
        print(f"  {'-' * (len(hdr) - 2)}")

        for rate in rates:
            arrivals = make_arrivals(rate)
            pools = [
                PoolConfig("prefill_pool", A100_80GB, na, 8192),
                PoolConfig("decode_pool", H100_80GB, nh, 65536),
            ]
            fc = FleetConfig(
                pools=pools,
                router_type="SemanticRouter",
                router_kwargs={"classify_fn": stage_classify},
            )
            r = run_sim(fc, arrivals)
            print(
                f"  {rate:>7d}  {r['thru']:>6.1f}  {r['p50_ttft']:>7.1f}  "
                f"{r['p99_ttft']:>7.1f}  {r['p99_e2e']:>8.1f}  {r['slo']:>6.1f}  "
                f"{r['util']:>6.1f}  {r['migr']:>6.1f}"
            )
            r["label"] = f"{fleet_label} @ {rate}r/s"
            r["fleet_label"] = fleet_label
            r["rate"] = rate
            all_results.append(r)

        print()

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 4: Homogeneous vs Phase-split at high utilization
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_homo_vs_split_high_util():
    """Compare homogeneous and phase-split at matching GPU counts under load."""
    print_header(
        "SCENARIO 4: Homogeneous vs Phase-Split at High Utilization"
    )
    print("  Same total GPU count, varying arrival rates to drive utilization.\n")

    all_results = []

    fleet_configs = [
        ("6A homo", [PoolConfig("default_pool", A100_80GB, 6, 8192)],
         "RandomRouter", {}, "homogeneous"),
        ("4A+2H stage", [
            PoolConfig("prefill_pool", A100_80GB, 4, 8192),
            PoolConfig("decode_pool", H100_80GB, 2, 65536),
        ], "SemanticRouter", {"classify_fn": stage_classify}, "phase_split"),
        ("9A homo", [PoolConfig("default_pool", A100_80GB, 9, 8192)],
         "RandomRouter", {}, "homogeneous"),
        ("6A+3H stage", [
            PoolConfig("prefill_pool", A100_80GB, 6, 8192),
            PoolConfig("decode_pool", H100_80GB, 3, 65536),
        ], "SemanticRouter", {"classify_fn": stage_classify}, "phase_split"),
    ]

    rates = [100, 200, 400, 800, 1500]

    for fleet_name, pools, router, kwargs, fleet_type in fleet_configs:
        print(f"  --- {fleet_name} ---")
        hdr = (
            f"  {'Rate':>8s}  {'thru':>6s}  {'p50':>7s}  {'p99':>7s}  "
            f"{'p99e2e':>8s}  {'SLO%':>6s}  {'util%':>6s}  {'$/hr':>7s}"
        )
        print(hdr)
        print(f"  {'-' * (len(hdr) - 2)}")

        for rate in rates:
            arrivals = make_arrivals(rate)
            fc = FleetConfig(
                pools=list(pools),
                router_type=router,
                router_kwargs=kwargs,
            )
            r = run_sim(fc, arrivals)
            print(
                f"  {rate:>7d}  {r['thru']:>6.1f}  {r['p50_ttft']:>7.1f}  "
                f"{r['p99_ttft']:>7.1f}  {r['p99_e2e']:>8.1f}  {r['slo']:>6.1f}  "
                f"{r['util']:>6.1f}  {r['cost']:>7.2f}"
            )
            r["label"] = f"{fleet_name} @ {rate}r/s"
            r["fleet_name"] = fleet_name
            r["fleet_type"] = fleet_type
            r["rate"] = rate
            all_results.append(r)

        print()

    return all_results


def main():
    print("=" * 120)
    print("  HIGH GPU UTILIZATION ANALYSIS")
    print("  Stage-Aware Routing Under Production Load Conditions")
    print("=" * 120)

    t0 = time.time()

    r1 = scenario_util_sweep()
    r2 = scenario_policies_high_util()
    r3 = scenario_saturation()
    r4 = scenario_homo_vs_split_high_util()

    elapsed = time.time() - t0
    print(f"\n{'=' * 120}")
    print(f"  Total high-utilization analysis time: {elapsed:.1f}s")
    print(f"{'=' * 120}")


if __name__ == "__main__":
    main()
