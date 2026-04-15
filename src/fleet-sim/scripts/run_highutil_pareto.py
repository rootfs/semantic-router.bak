#!/usr/bin/env python3
"""High-utilization Pareto frontier analysis.

Maps the cost vs latency trade-off space on tight fleets (6-30 GPUs)
at arrival rates that produce 15-60%+ utilization.  For each fleet
size and arrival rate, sweeps all A100:H100 ratios and compares
stage-aware routing against random baseline.

Scenarios:
  1. Pool ratio sweep at high util per fleet size (6, 9, 12, 15, 20 GPU)
  2. Arrival rate × pool ratio matrix on a 12-GPU fleet
  3. Stage-aware vs random Pareto comparison
  4. Minimum viable decode pool at each utilization level
"""
from __future__ import annotations

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


def make_arrivals(rate):
    wl = TraceWorkload(
        path=TRACE, fmt="semantic_router", scale_lam=rate, field_map=FMAP
    )
    return wl.generate()


def print_header(title):
    print(f"\n{'=' * 130}")
    print(f"  {title}")
    print(f"{'=' * 130}")


HDR = (
    f"  {'Label':>30s}  {'Router':>8s}  {'thru':>6s}  {'p50':>7s}  "
    f"{'p99':>7s}  {'p99e2e':>8s}  {'SLO%':>6s}  {'util%':>6s}  "
    f"{'$/hr':>7s}  {'migr%':>6s}  {'reprfll%':>8s}"
)


def print_row_header():
    print(HDR)
    print(f"  {'-' * (len(HDR) - 2)}")


def print_row(label, router, r):
    print(
        f"  {label:>30s}  {router:>8s}  {r['thru']:>6.1f}  "
        f"{r['p50_ttft']:>7.1f}  {r['p99_ttft']:>7.1f}  "
        f"{r['p99_e2e']:>8.1f}  {r['slo']:>6.1f}  {r['util']:>6.1f}  "
        f"{r['cost']:>7.2f}  {r['migr']:>6.1f}  {r['reprfll']:>8.1f}"
    )


def run_config(na, nh, arrivals, router="stage"):
    """Run one fleet config with either stage-aware or random routing."""
    if nh == 0:
        pools = [PoolConfig("prefill_pool", A100_80GB, na, 8192)]
        if router == "stage":
            fc = FleetConfig(
                pools=pools,
                router_type="SemanticRouter",
                router_kwargs={"classify_fn": lambda r: "prefill_pool"},
            )
        else:
            fc = FleetConfig(pools=pools, router_type="RandomRouter")
    else:
        pools = [
            PoolConfig("prefill_pool", A100_80GB, na, 8192),
            PoolConfig("decode_pool", H100_80GB, nh, 65536),
        ]
        if router == "stage":
            fc = FleetConfig(
                pools=pools,
                router_type="SemanticRouter",
                router_kwargs={"classify_fn": stage_classify},
            )
        else:
            fc = FleetConfig(pools=pools, router_type="RandomRouter")
    return run_sim(fc, arrivals)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 1: Pool ratio sweep per fleet size at high-util arrival rates
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_ratio_sweep():
    """Sweep A100:H100 ratio at fleet sizes that produce high utilization."""
    print_header(
        "SCENARIO 1: Pool Ratio Sweep at High Utilization"
    )

    all_results = []

    fleet_rate_pairs = [
        (6, 100, "6 GPU @ 100r/s"),
        (6, 200, "6 GPU @ 200r/s"),
        (9, 200, "9 GPU @ 200r/s"),
        (9, 400, "9 GPU @ 400r/s"),
        (12, 200, "12 GPU @ 200r/s"),
        (12, 400, "12 GPU @ 400r/s"),
        (15, 400, "15 GPU @ 400r/s"),
        (20, 400, "20 GPU @ 400r/s"),
        (20, 800, "20 GPU @ 800r/s"),
    ]

    for total, rate, section_label in fleet_rate_pairs:
        print(f"\n  --- {section_label} (stage-aware) ---")
        print_row_header()

        arrivals = make_arrivals(rate)

        for nh in range(0, total + 1):
            na = total - nh
            if na < 1:
                continue
            r = run_config(na, nh, arrivals, router="stage")
            label = f"{na}A+{nh}H"
            print_row(label, "stage", r)
            r["label"] = label
            r["router"] = "stage"
            r["total_gpus"] = total
            r["rate"] = rate
            r["n_a100"] = na
            r["n_h100"] = nh
            all_results.append(r)

        print()

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 2: Stage-aware vs Random head-to-head at each ratio
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_stage_vs_random():
    """Compare stage-aware vs random at each pool ratio on tight fleets."""
    print_header(
        "SCENARIO 2: Stage-Aware vs Random — Head-to-Head at High Utilization"
    )

    all_results = []

    configs = [
        (9, 200, "9 GPU @ 200r/s"),
        (12, 400, "12 GPU @ 400r/s"),
        (15, 400, "15 GPU @ 400r/s"),
    ]

    for total, rate, section_label in configs:
        print(f"\n  --- {section_label} ---")
        print_row_header()

        arrivals = make_arrivals(rate)

        for nh in range(0, total + 1, max(1, total // 6)):
            na = total - nh
            if na < 1:
                continue

            r_stage = run_config(na, nh, arrivals, router="stage")
            r_random = run_config(na, nh, arrivals, router="random")

            label = f"{na}A+{nh}H"
            print_row(label, "stage", r_stage)
            print_row(label, "random", r_random)

            for tag, r, rtype in [("stage", r_stage, "stage"), ("random", r_random, "random")]:
                r["label"] = label
                r["router"] = rtype
                r["total_gpus"] = total
                r["rate"] = rate
                r["n_a100"] = na
                r["n_h100"] = nh
                all_results.append(r)

        print()

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 3: Minimum viable decode pool at each utilization target
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_min_decode():
    """Find smallest decode pool that meets SLO at various util levels."""
    print_header(
        "SCENARIO 3: Minimum Viable Decode Pool at Each Utilization Level"
    )
    print("  Find the smallest number of H100s that achieves 99%+ SLO.\n")

    all_results = []

    test_cases = [
        (60, 200, "Low util (60 GPU, 200r/s)"),
        (30, 200, "Moderate (30 GPU, 200r/s)"),
        (15, 200, "High (15 GPU, 200r/s)"),
        (12, 200, "Higher (12 GPU, 200r/s)"),
        (9, 200, "Very high (9 GPU, 200r/s)"),
        (12, 400, "Stressed (12 GPU, 400r/s)"),
        (15, 400, "Stressed (15 GPU, 400r/s)"),
        (20, 800, "Stressed (20 GPU, 800r/s)"),
    ]

    print(
        f"  {'Scenario':>30s}  {'Min H100s':>9s}  {'Config':>12s}  "
        f"{'Util%':>6s}  {'P99 TTFT':>10s}  {'SLO%':>6s}  {'$/hr':>7s}"
    )
    print(f"  {'-' * 100}")

    for total, rate, scenario_label in test_cases:
        arrivals = make_arrivals(rate)

        best = None
        for nh in range(0, total):
            na = total - nh
            if na < 1:
                continue
            r = run_config(na, nh, arrivals, router="stage")
            if r["slo"] >= 99.0:
                best = (na, nh, r)
                break

        if best:
            na, nh, r = best
            label = f"{na}A+{nh}H"
            print(
                f"  {scenario_label:>30s}  {nh:>9d}  {label:>12s}  "
                f"{r['util']:>6.1f}  {r['p99_ttft']:>8.1f}ms  "
                f"{r['slo']:>6.1f}  {r['cost']:>7.2f}"
            )
            r["scenario"] = scenario_label
            r["min_h100"] = nh
            all_results.append(r)
        else:
            print(f"  {scenario_label:>30s}  {'NONE':>9s}  {'---':>12s}  "
                  f"{'---':>6s}  {'---':>10s}  {'<99%':>6s}  {'---':>7s}")
            all_results.append({
                "scenario": scenario_label, "min_h100": -1,
                "slo": 0, "total_gpus": total, "rate": rate,
            })

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 4: Cost vs P99 TTFT Pareto with utilization annotation
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_pareto_annotated():
    """Build the Pareto frontier with utilization explicitly annotated."""
    print_header(
        "SCENARIO 4: Cost vs P99 TTFT Pareto Frontier (utilization-annotated)"
    )
    print("  For each fleet×rate combo, find the Pareto-optimal configs.\n")

    all_results = []

    combos = [
        (9, 200),
        (12, 200),
        (12, 400),
        (15, 400),
        (20, 400),
        (20, 800),
    ]

    for total, rate in combos:
        print(f"  --- {total} GPUs, {rate} req/s ---")
        print(
            f"  {'Config':>12s}  {'Router':>8s}  {'P99 TTFT':>10s}  "
            f"{'SLO%':>6s}  {'Util%':>6s}  {'$/hr':>7s}  "
            f"{'P99 E2E':>10s}  {'Pareto?':>7s}"
        )
        print(f"  {'-' * 90}")

        arrivals = make_arrivals(rate)
        configs = []

        for nh in range(0, total + 1):
            na = total - nh
            if na < 1:
                continue
            for router in ["stage", "random"]:
                r = run_config(na, nh, arrivals, router=router)
                r["label"] = f"{na}A+{nh}H"
                r["router"] = router
                r["total_gpus"] = total
                r["rate"] = rate
                r["n_a100"] = na
                r["n_h100"] = nh
                configs.append(r)

        configs.sort(key=lambda x: x["cost"])

        best_p99 = float("inf")
        for c in configs:
            is_pareto = c["p99_ttft"] < best_p99 and c["slo"] >= 95.0
            if is_pareto:
                best_p99 = c["p99_ttft"]
            tag = "***" if is_pareto else ""
            print(
                f"  {c['label']:>12s}  {c['router']:>8s}  "
                f"{c['p99_ttft']:>8.1f}ms  {c['slo']:>6.1f}  "
                f"{c['util']:>6.1f}  {c['cost']:>7.2f}  "
                f"{c['p99_e2e']:>8.1f}ms  {tag:>7s}"
            )
            c["pareto"] = is_pareto
            all_results.append(c)

        print()

    return all_results


def main():
    print("=" * 130)
    print("  HIGH-UTILIZATION PARETO FRONTIER ANALYSIS")
    print("  Stage-Aware Routing Cost-Performance Trade-offs Under Load")
    print("=" * 130)

    t0 = time.time()

    r1 = scenario_ratio_sweep()
    r2 = scenario_stage_vs_random()
    r3 = scenario_min_decode()
    r4 = scenario_pareto_annotated()

    elapsed = time.time() - t0
    print(f"\n{'=' * 130}")
    print(f"  Total analysis time: {elapsed:.1f}s")
    print(f"{'=' * 130}")


if __name__ == "__main__":
    main()
