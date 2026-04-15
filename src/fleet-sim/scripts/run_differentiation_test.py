#!/usr/bin/env python3
"""Differentiation test: stress the fleet enough to separate policies 4, 5, and 6.

The main simulation showed policies 4/5/6 identical because:
  - Policy 5's decode-volume threshold (400 tokens) rarely overrides stage routing
    since plan/explore have low projected decode (25-250 tokens)
  - Policy 6's interference avoidance triggers only at >85% load, but utilization
    was low (~2.8%)

This script runs:
  1. Varied decode-volume thresholds (100, 200, 400) for policy 5
  2. Higher arrival rates to trigger interference avoidance for policy 6
  3. A smaller fleet (fewer GPUs) to create genuine contention
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
    stage_interference_classify_factory,
    PREFILL_STAGES,
)
from fleet_sim.workload.trace import TraceWorkload

SLO_MS = 500.0

TIGHT_POOLS = [
    PoolConfig("prefill_pool", A100_80GB, 8, 8192),
    PoolConfig("decode_pool", H100_80GB, 4, 65536),
]


def make_volume_classify(threshold: int):
    """Policy 5 variant with configurable threshold."""
    def classify(req):
        if (
            req.projected_decode_tokens is not None
            and req.projected_decode_tokens > threshold
        ):
            return "decode_pool"
        if req.stage in PREFILL_STAGES:
            return "prefill_pool"
        return "decode_pool"
    return classify


def run_one(name: str, fc: FleetConfig, arrivals: list, live_ref=None) -> dict:
    fleet = Fleet(fc)
    fleet._build()
    if live_ref is not None:
        live_ref[0] = fleet._pools
    result = fleet.run(arrivals)
    risk = result.compute_prefix_risk()
    return {
        "policy": name,
        "throughput": round(result.throughput(), 2),
        "tok_thru": round(result.token_throughput(), 0),
        "p50_ttft": round(result.p50_ttft_ms(), 2),
        "p99_ttft": round(result.p99_ttft_ms(), 2),
        "p99_e2e": round(result.p99_e2e_ms(), 2),
        "slo": round(result.slo_compliance(SLO_MS) * 100, 2),
        "util": round(result.mean_utilisation() * 100, 2),
        "interf": round(result.mean_interference_ratio() * 100, 2),
        "migr_rate": round(risk["migration_rate"], 4),
        "reprefill": round(risk["repeated_prefill_amplification"] * 100, 2),
        "completed": len(result.completed),
    }


def print_results(results: list[dict], title: str):
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")
    hdr = (f"  {'policy':>30s}  {'thru':>8s}  {'tok_thru':>8s}  {'p50_ttft':>10s}  "
           f"{'p99_ttft':>10s}  {'p99_e2e':>10s}  {'SLO%':>7s}  {'util%':>6s}  "
           f"{'interf%':>7s}  {'migr%':>6s}  {'reprfll%':>8s}")
    print(hdr)
    print(f"  {'-'*(len(hdr)-2)}")
    for r in results:
        print(f"  {r['policy']:>30s}  {r['throughput']:>8.1f}  {r['tok_thru']:>8.0f}  "
              f"{r['p50_ttft']:>10.1f}  {r['p99_ttft']:>10.1f}  {r['p99_e2e']:>10.1f}  "
              f"{r['slo']:>7.1f}  {r['util']:>6.1f}  {r['interf']:>7.2f}  "
              f"{r['migr_rate']*100:>6.1f}  {r['reprefill']:>8.1f}")


def main():
    trace = "examples/trace_samples/swe_agent_annotated.jsonl"

    print("=" * 110)
    print("  DIFFERENTIATION TEST: Tight fleet to separate policies 4, 5, 6")
    print("=" * 110)

    # ── Test 1: Vary decode-volume thresholds on tight fleet ──────────────────
    for rate in [50, 100, 200]:
        wl = TraceWorkload(
            path=trace, fmt="semantic_router", scale_lam=rate,
            field_map={"stage": "stage", "session_id": "session_id",
                       "turn_index": "turn_index",
                       "projected_decode_tokens": "projected_decode_tokens"},
        )
        arrivals = wl.generate()
        results = []

        # Baselines
        for name, rtype in [("random", "RandomRouter"), ("least_loaded", "LeastLoadedRouter")]:
            fc = FleetConfig(pools=list(TIGHT_POOLS), router_type=rtype)
            results.append(run_one(name, fc, arrivals))

        # Policy 4: plain stage
        fc4 = FleetConfig(pools=list(TIGHT_POOLS), router_type="SemanticRouter",
                          router_kwargs={"classify_fn": stage_classify})
        results.append(run_one("P4_stage", fc4, arrivals))

        # Policy 5: vary threshold
        for thresh in [100, 200, 400, 600]:
            fc5 = FleetConfig(pools=list(TIGHT_POOLS), router_type="SemanticRouter",
                              router_kwargs={"classify_fn": make_volume_classify(thresh)})
            results.append(run_one(f"P5_vol_t{thresh}", fc5, arrivals))

        # Policy 6: interference avoidance
        live_ref = [None]
        classify6 = stage_interference_classify_factory(live_ref)
        fc6 = FleetConfig(pools=list(TIGHT_POOLS), router_type="SemanticRouter",
                          router_kwargs={"classify_fn": classify6})
        results.append(run_one("P6_interference", fc6, arrivals, live_ref))

        print_results(results, f"Tight fleet (8×A100+4×H100) @ {rate} req/s — SWE-agent")

    # ── Test 2: Same on agentbench ────────────────────────────────────────────
    trace2 = "examples/trace_samples/agentbench_annotated.jsonl"
    for rate in [50, 100, 200]:
        wl = TraceWorkload(
            path=trace2, fmt="semantic_router", scale_lam=rate,
            field_map={"stage": "stage", "session_id": "session_id",
                       "turn_index": "turn_index",
                       "projected_decode_tokens": "projected_decode_tokens"},
        )
        arrivals = wl.generate()
        results = []

        for name, rtype in [("random", "RandomRouter"), ("least_loaded", "LeastLoadedRouter")]:
            fc = FleetConfig(pools=list(TIGHT_POOLS), router_type=rtype)
            results.append(run_one(name, fc, arrivals))

        fc4 = FleetConfig(pools=list(TIGHT_POOLS), router_type="SemanticRouter",
                          router_kwargs={"classify_fn": stage_classify})
        results.append(run_one("P4_stage", fc4, arrivals))

        for thresh in [100, 200, 400, 600]:
            fc5 = FleetConfig(pools=list(TIGHT_POOLS), router_type="SemanticRouter",
                              router_kwargs={"classify_fn": make_volume_classify(thresh)})
            results.append(run_one(f"P5_vol_t{thresh}", fc5, arrivals))

        live_ref = [None]
        classify6 = stage_interference_classify_factory(live_ref)
        fc6 = FleetConfig(pools=list(TIGHT_POOLS), router_type="SemanticRouter",
                          router_kwargs={"classify_fn": classify6})
        results.append(run_one("P6_interference", fc6, arrivals, live_ref))

        print_results(results, f"Tight fleet (8×A100+4×H100) @ {rate} req/s — AgentBench")

    print(f"\n{'='*110}")
    print("  DONE")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()
