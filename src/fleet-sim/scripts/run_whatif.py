#!/usr/bin/env python3
"""What-if capacity planning analysis for stage-aware routing.

Runs six scenarios that demonstrate how white-boxing the agent workload
enables proactive fleet design:

  1. Workload mix shift     — shift stage distribution toward decode-heavy
  2. Model upgrade          — 2x longer decode output per Implement turn
  3. Hardware transition    — hypothetical H200 with 3x decode throughput
  4. SLA threshold sweep    — vary SLO target from 200ms to 1000ms
  5. Cost-performance Pareto — sweep pool ratios for Pareto frontier
  6. Peak/off-peak resizing — day (high traffic) vs night (low traffic)
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fleet_sim.core.fleet import Fleet, FleetConfig, PoolConfig
from fleet_sim.gpu_profiles.profiles import A100_80GB, H100_80GB, CUSTOM
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
    }


def load_raw_records():
    with open(TRACE) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_temp_trace(records):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for r in records:
        tmp.write(json.dumps(r) + "\n")
    tmp.close()
    return tmp.name


def make_phase_split(na, nh):
    pools = [
        PoolConfig("prefill_pool", A100_80GB, na, 8192),
        PoolConfig("decode_pool", H100_80GB, nh, 65536),
    ]
    return FleetConfig(
        pools=pools,
        router_type="SemanticRouter",
        router_kwargs={"classify_fn": stage_classify},
    )


def print_header(title):
    print(f"\n{'=' * 110}")
    print(f"  {title}")
    print(f"{'=' * 110}")


def print_row_header():
    hdr = (
        f"  {'Label':>32s}  {'thru':>6s}  {'p50':>7s}  {'p99':>7s}  "
        f"{'p99e2e':>8s}  {'SLO%':>6s}  {'util%':>6s}  {'$/hr':>7s}  "
        f"{'migr%':>6s}  {'reprfll%':>8s}"
    )
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")


def print_row(label, r):
    print(
        f"  {label:>32s}  {r['thru']:>6.1f}  {r['p50_ttft']:>7.1f}  "
        f"{r['p99_ttft']:>7.1f}  {r['p99_e2e']:>8.1f}  {r['slo']:>6.1f}  "
        f"{r['util']:>6.1f}  {r['cost']:>7.2f}  {r['migr']:>6.1f}  "
        f"{r['reprfll']:>8.1f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 1: Workload Mix Shift
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_workload_mix():
    """Shift stage distribution from current to more decode-heavy."""
    print_header("SCENARIO 1: Workload Mix Shift (what if traffic becomes more decode-heavy?)")
    print("  Baseline: Plan=19%, Explore=31%, Implement=31%, Test=19%")
    print("  We progressively shift Explore → Implement to model decode-heavy adoption.\n")

    records = load_raw_records()
    all_results = []

    shifts = [
        ("Baseline (19/31/31/19)", {}),
        ("Mild (19/25/37/19)", {"explore": "implement", "frac": 0.20}),
        ("Moderate (19/18/44/19)", {"explore": "implement", "frac": 0.40}),
        ("Aggressive (19/12/50/19)", {"explore": "implement", "frac": 0.60}),
        ("Extreme (19/5/57/19)", {"explore": "implement", "frac": 0.80}),
    ]

    print_row_header()

    for label, shift_cfg in shifts:
        if not shift_cfg:
            modified = records
        else:
            import random
            rng = random.Random(42)
            modified = []
            for rec in records:
                rec2 = dict(rec)
                if rec2.get("stage") == shift_cfg["explore"] == "explore" or (
                    rec2.get("stage") == "explore"
                ):
                    if rng.random() < shift_cfg["frac"]:
                        rec2["stage"] = "implement"
                        rec2["generated_tokens"] = int(
                            rec2.get("generated_tokens", 100) * 3
                        )
                modified.append(rec2)

        tmp_path = write_temp_trace(modified)
        wl = TraceWorkload(
            path=tmp_path, fmt="semantic_router", scale_lam=200, field_map=FMAP
        )
        arrivals = wl.generate()
        fc = make_phase_split(40, 20)
        r = run_sim(fc, arrivals)
        print_row(label, r)
        r["label"] = label
        all_results.append(r)
        Path(tmp_path).unlink()

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 2: Model Upgrade (longer decode)
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_model_upgrade():
    """What if a new model generates 2x or 3x more tokens per Implement turn?"""
    print_header(
        "SCENARIO 2: Model Upgrade (longer decode per Implement/Test turn)"
    )
    print("  Simulates models that produce more verbose code completions.\n")

    records = load_raw_records()
    all_results = []

    multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]

    print_row_header()

    for mult in multipliers:
        modified = []
        for rec in records:
            rec2 = dict(rec)
            if rec2.get("stage") in ("implement", "test"):
                rec2["generated_tokens"] = int(
                    rec2.get("generated_tokens", 100) * mult
                )
            modified.append(rec2)

        tmp_path = write_temp_trace(modified)
        wl = TraceWorkload(
            path=tmp_path, fmt="semantic_router", scale_lam=200, field_map=FMAP
        )
        arrivals = wl.generate()
        fc = make_phase_split(40, 20)
        r = run_sim(fc, arrivals)
        label = f"Decode {mult:.1f}x"
        print_row(label, r)
        r["label"] = label
        all_results.append(r)
        Path(tmp_path).unlink()

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 3: Hardware Transition (H200)
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_hardware_transition():
    """What if we replace H100s with hypothetical H200s (3x decode, higher cost)?"""
    print_header(
        "SCENARIO 3: Hardware Transition (H100 → hypothetical H200/B200)"
    )
    print("  H200: W=2.0ms (2x prefill), H=0.16ms (2x decode), chunk=2048,")
    print("         slots=384, cost=$5.50/hr")
    print("  B200: W=1.5ms (2.7x prefill), H=0.11ms (3x decode), chunk=2048,")
    print("         slots=512, cost=$7.00/hr\n")

    H200_80GB = CUSTOM(
        name="H200-141GB",
        W=0.0020,
        H=0.00016,
        chunk=2048,
        total_kv_blks=131072,
        max_slots=384,
        cost_per_hr=5.50,
        calibration_ctx=8192,
    )

    B200_192GB = CUSTOM(
        name="B200-192GB",
        W=0.0015,
        H=0.00011,
        chunk=2048,
        total_kv_blks=196608,
        max_slots=512,
        cost_per_hr=7.00,
        calibration_ctx=8192,
    )

    wl = TraceWorkload(
        path=TRACE, fmt="semantic_router", scale_lam=200, field_map=FMAP
    )
    arrivals = wl.generate()

    configs = [
        ("40xA100 + 20xH100 (baseline)", [
            PoolConfig("prefill_pool", A100_80GB, 40, 8192),
            PoolConfig("decode_pool", H100_80GB, 20, 65536),
        ]),
        ("40xA100 + 10xH200", [
            PoolConfig("prefill_pool", A100_80GB, 40, 8192),
            PoolConfig("decode_pool", H200_80GB, 10, 65536),
        ]),
        ("40xA100 + 5xH200", [
            PoolConfig("prefill_pool", A100_80GB, 40, 8192),
            PoolConfig("decode_pool", H200_80GB, 5, 65536),
        ]),
        ("40xA100 + 5xB200", [
            PoolConfig("prefill_pool", A100_80GB, 40, 8192),
            PoolConfig("decode_pool", B200_192GB, 5, 65536),
        ]),
        ("40xA100 + 3xB200", [
            PoolConfig("prefill_pool", A100_80GB, 40, 8192),
            PoolConfig("decode_pool", B200_192GB, 3, 65536),
        ]),
    ]

    print_row_header()
    all_results = []

    for label, pools in configs:
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
# Scenario 4: SLA Threshold Sweep
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_sla_sweep():
    """What SLO compliance do we achieve at various TTFT thresholds?"""
    print_header("SCENARIO 4: SLA Threshold Sweep (vary P99 TTFT target)")
    print("  Fixed fleet: 40xA100 + 20xH100, stage-aware, 200 req/s")
    print("  Sweep SLO target from 100ms to 1000ms.\n")

    wl = TraceWorkload(
        path=TRACE, fmt="semantic_router", scale_lam=200, field_map=FMAP
    )
    arrivals = wl.generate()
    fc = make_phase_split(40, 20)

    fleet = Fleet(fc)
    result = fleet.run(arrivals)

    slo_targets = [100, 150, 200, 250, 300, 400, 500, 750, 1000]

    hdr = f"  {'SLO Target':>12s}  {'Compliance':>12s}  {'Meets SLO?':>10s}"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    all_results = []
    for target in slo_targets:
        compliance = result.slo_compliance(target) * 100
        meets = "YES" if compliance >= 99.0 else ("MARGINAL" if compliance >= 95.0 else "NO")
        print(f"  {target:>10d}ms  {compliance:>11.1f}%  {meets:>10s}")
        all_results.append({
            "slo_target_ms": target,
            "compliance": round(compliance, 1),
            "meets": meets,
        })

    risk = result.compute_prefix_risk()
    p99 = result.p99_ttft_ms()
    p50 = result.p50_ttft_ms()
    print(f"\n  Actual P50 TTFT: {p50:.1f}ms,  P99 TTFT: {p99:.1f}ms")
    print(f"  Migration rate: {risk['migration_rate']*100:.1f}%")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 5: Cost-Performance Pareto Frontier
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_pareto():
    """Sweep pool ratios to map the cost vs latency Pareto frontier."""
    print_header("SCENARIO 5: Cost-Performance Pareto Frontier")
    print("  Fix 60 GPU budget, sweep A100:H100 ratio from 60:0 to 20:40.")
    print("  Also test smaller fleets (30 GPUs) with same ratios.\n")

    all_results = []

    for total, rate in [(60, 200), (30, 200), (60, 600)]:
        print(f"  --- {total} GPUs total, {rate} req/s ---")
        wl = TraceWorkload(
            path=TRACE, fmt="semantic_router", scale_lam=rate, field_map=FMAP
        )
        arrivals = wl.generate()

        print_row_header()

        ratios = []
        for nh in range(0, total, max(1, total // 12)):
            na = total - nh
            if na < 1:
                continue
            ratios.append((na, nh))
        if (total // 3, total - total // 3) not in ratios:
            ratios.append((total - total // 3, total // 3))
        ratios.sort(key=lambda x: x[1])

        for na, nh in ratios:
            if nh == 0:
                pools = [PoolConfig("prefill_pool", A100_80GB, na, 8192)]
                fc = FleetConfig(
                    pools=pools,
                    router_type="SemanticRouter",
                    router_kwargs={"classify_fn": lambda r: "prefill_pool"},
                )
            else:
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
            label = f"{na}A+{nh}H ({total}GPU,{rate}r/s)"
            print_row(label, r)
            r["label"] = label
            r["total_gpus"] = total
            r["n_a100"] = na
            r["n_h100"] = nh
            r["rate"] = rate
            all_results.append(r)

        print()

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 6: Peak/Off-Peak Dynamic Resizing
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_peak_offpeak():
    """Day/night traffic: high rate vs low rate with different pool configs."""
    print_header("SCENARIO 6: Peak/Off-Peak Dynamic Pool Resizing")
    print("  Peak (9am-6pm):  600 req/s — full fleet 40xA100 + 20xH100")
    print("  Off-peak (night): 80 req/s — can we shrink the decode pool?")
    print("  Reclaimed H100s could serve training or other workloads.\n")

    all_results = []

    scenarios = [
        ("Peak: 40A+20H @ 600r/s", 600, 40, 20),
        ("Off-peak: 40A+20H @ 80r/s", 80, 40, 20),
        ("Off-peak: 40A+10H @ 80r/s", 80, 40, 10),
        ("Off-peak: 40A+5H @ 80r/s", 80, 40, 5),
        ("Off-peak: 40A+2H @ 80r/s", 80, 40, 2),
        ("Off-peak: 40A+0H @ 80r/s", 80, 40, 0),
    ]

    print_row_header()

    for label, rate, na, nh in scenarios:
        wl = TraceWorkload(
            path=TRACE, fmt="semantic_router", scale_lam=rate, field_map=FMAP
        )
        arrivals = wl.generate()

        if nh == 0:
            pools = [PoolConfig("prefill_pool", A100_80GB, na, 8192)]
            fc = FleetConfig(
                pools=pools,
                router_type="SemanticRouter",
                router_kwargs={"classify_fn": lambda r: "prefill_pool"},
            )
        else:
            fc = make_phase_split(na, nh)

        r = run_sim(fc, arrivals)
        print_row(label, r)
        r["label"] = label
        r["rate"] = rate
        r["n_a100"] = na
        r["n_h100"] = nh
        all_results.append(r)

    # Compute savings
    peak_cost = all_results[0]["cost"]
    for r in all_results[1:]:
        savings = peak_cost - r["cost"]
        pct = savings / peak_cost * 100 if peak_cost > 0 else 0
        r["savings_vs_peak"] = round(savings, 2)
        r["savings_pct"] = round(pct, 1)

    print(f"\n  Peak fleet cost: ${peak_cost:.2f}/hr")
    for r in all_results[1:]:
        sv = r.get("savings_vs_peak", 0)
        sp = r.get("savings_pct", 0)
        slo_ok = "SLO OK" if r["slo"] >= 99.0 else "SLO BREACH"
        print(
            f"  {r['label']:>32s}  saves ${sv:.2f}/hr ({sp:.1f}%)  [{slo_ok}]"
        )

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 110)
    print("  WHAT-IF CAPACITY PLANNING ANALYSIS")
    print("  Stage-Aware Routing — Workload White-Boxing Perspective")
    print("=" * 110)

    t0 = time.time()

    r1 = scenario_workload_mix()
    r2 = scenario_model_upgrade()
    r3 = scenario_hardware_transition()
    r4 = scenario_sla_sweep()
    r5 = scenario_pareto()
    r6 = scenario_peak_offpeak()

    elapsed = time.time() - t0
    print(f"\n{'=' * 110}")
    print(f"  Total what-if analysis time: {elapsed:.1f}s")
    print(f"{'=' * 110}")


if __name__ == "__main__":
    main()
