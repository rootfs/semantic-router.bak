#!/usr/bin/env python3
"""Sensitivity analysis: vary pool ratio and fleet size for stage-aware routing."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fleet_sim.core.fleet import Fleet, FleetConfig, PoolConfig
from fleet_sim.gpu_profiles.profiles import A100_80GB, H100_80GB
from fleet_sim.routing.stage_policies import stage_classify
from fleet_sim.workload.trace import TraceWorkload

SLO_MS = 500.0
FMAP = {"stage": "stage", "session_id": "session_id",
        "turn_index": "turn_index", "projected_decode_tokens": "projected_decode_tokens"}

def run(fc, arrivals):
    fleet = Fleet(fc)
    result = fleet.run(arrivals)
    risk = result.compute_prefix_risk()
    return {
        "thru": round(result.throughput(), 1),
        "p50": round(result.p50_ttft_ms(), 1),
        "p99": round(result.p99_ttft_ms(), 1),
        "p99e2e": round(result.p99_e2e_ms(), 1),
        "slo": round(result.slo_compliance(SLO_MS)*100, 1),
        "util": round(result.mean_utilisation()*100, 1),
        "cost": round(fc.total_cost_per_hr(), 2),
        "migr": round(risk["migration_rate"]*100, 1),
        "reprfll": round(risk["repeated_prefill_amplification"]*100, 1),
        "completed": len(result.completed),
    }

def main():
    trace = "examples/trace_samples/swe_agent_annotated.jsonl"

    # Pool ratio sensitivity: vary prefill:decode split with 60 total GPUs
    print("=" * 100)
    print("  SENSITIVITY: Pool Ratio (60 total GPUs, stage-aware, SWE-agent, 200 req/s)")
    print("=" * 100)
    wl = TraceWorkload(path=trace, fmt="semantic_router", scale_lam=200, field_map=FMAP)
    arrivals = wl.generate()

    hdr = f"  {'Config':>22s}  {'thru':>6s}  {'p50':>7s}  {'p99':>7s}  {'p99e2e':>8s}  {'SLO%':>6s}  {'util%':>6s}  {'$/hr':>7s}  {'migr%':>6s}  {'reprfll%':>8s}"
    print(hdr)
    print(f"  {'-'*(len(hdr)-2)}")

    ratios = [(60,0), (55,5), (50,10), (45,15), (40,20), (35,25), (30,30), (20,40)]
    for na, nh in ratios:
        if nh == 0:
            pools = [PoolConfig("prefill_pool", A100_80GB, na, 8192)]
            fc = FleetConfig(pools=pools, router_type="SemanticRouter",
                             router_kwargs={"classify_fn": lambda r: "prefill_pool"})
        else:
            pools = [PoolConfig("prefill_pool", A100_80GB, na, 8192),
                     PoolConfig("decode_pool", H100_80GB, nh, 65536)]
            fc = FleetConfig(pools=pools, router_type="SemanticRouter",
                             router_kwargs={"classify_fn": stage_classify})
        r = run(fc, arrivals)
        label = f"{na}xA100+{nh}xH100"
        print(f"  {label:>22s}  {r['thru']:>6.1f}  {r['p50']:>7.1f}  {r['p99']:>7.1f}  "
              f"{r['p99e2e']:>8.1f}  {r['slo']:>6.1f}  {r['util']:>6.1f}  {r['cost']:>7.2f}  "
              f"{r['migr']:>6.1f}  {r['reprfll']:>8.1f}")

    # Fleet size sensitivity: fix 2:1 A100:H100 ratio, vary total
    print(f"\n{'='*100}")
    print("  SENSITIVITY: Fleet Size (2:1 A100:H100 ratio, stage-aware, SWE-agent, 200 req/s)")
    print("=" * 100)
    print(hdr)
    print(f"  {'-'*(len(hdr)-2)}")

    for total in [12, 24, 36, 48, 60, 90, 120]:
        na = total * 2 // 3
        nh = total - na
        pools = [PoolConfig("prefill_pool", A100_80GB, na, 8192),
                 PoolConfig("decode_pool", H100_80GB, nh, 65536)]
        fc = FleetConfig(pools=pools, router_type="SemanticRouter",
                         router_kwargs={"classify_fn": stage_classify})
        r = run(fc, arrivals)
        label = f"{na}xA100+{nh}xH100 ({total})"
        print(f"  {label:>22s}  {r['thru']:>6.1f}  {r['p50']:>7.1f}  {r['p99']:>7.1f}  "
              f"{r['p99e2e']:>8.1f}  {r['slo']:>6.1f}  {r['util']:>6.1f}  {r['cost']:>7.2f}  "
              f"{r['migr']:>6.1f}  {r['reprfll']:>8.1f}")

    # Arrival rate sensitivity: phase-split 40+20, vary rate
    print(f"\n{'='*100}")
    print("  SENSITIVITY: Arrival Rate (40xA100+20xH100, stage-aware, SWE-agent)")
    print("=" * 100)
    print(hdr)
    print(f"  {'-'*(len(hdr)-2)}")

    for rate in [50, 100, 200, 400, 600, 800, 1000, 1500, 2000]:
        wl2 = TraceWorkload(path=trace, fmt="semantic_router", scale_lam=rate, field_map=FMAP)
        arr2 = wl2.generate()
        pools = [PoolConfig("prefill_pool", A100_80GB, 40, 8192),
                 PoolConfig("decode_pool", H100_80GB, 20, 65536)]
        fc = FleetConfig(pools=pools, router_type="SemanticRouter",
                         router_kwargs={"classify_fn": stage_classify})
        r = run(fc, arr2)
        label = f"{rate} req/s"
        print(f"  {label:>22s}  {r['thru']:>6.1f}  {r['p50']:>7.1f}  {r['p99']:>7.1f}  "
              f"{r['p99e2e']:>8.1f}  {r['slo']:>6.1f}  {r['util']:>6.1f}  {r['cost']:>7.2f}  "
              f"{r['migr']:>6.1f}  {r['reprfll']:>8.1f}")

if __name__ == "__main__":
    main()
