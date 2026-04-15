"""Experimental stage-aware routing policies for coding agents.

Three classify_fn implementations for SemanticRouter:

  Policy 4: stage_classify
    Routes by agent stage: Plan/Explore → prefill_pool, Implement/Test → decode_pool

  Policy 5: stage_decode_volume_classify
    Combines stage with projected decode volume. High projected decode → decode_pool
    regardless of stage.

  Policy 6: stage_interference_classify_factory
    Reads live pool telemetry to avoid sending decode-heavy stages to
    prefill-saturated engines and vice versa. Requires set_pools() injection.
"""

from __future__ import annotations

from ..core.request import Request

PREFILL_STAGES = {"plan", "explore"}
DECODE_STAGES = {"implement", "test"}
DECODE_VOLUME_THRESHOLD = 400


def stage_classify(req: Request) -> str:
    """Policy 4: Route by agent stage alone."""
    if req.stage in PREFILL_STAGES:
        return "prefill_pool"
    return "decode_pool"


def stage_decode_volume_classify(req: Request) -> str:
    """Policy 5: Stage + projected decode volume.

    If projected decode tokens exceed a threshold, always route to decode_pool
    even if the stage would normally go to prefill_pool.
    """
    if (
        req.projected_decode_tokens is not None
        and req.projected_decode_tokens > DECODE_VOLUME_THRESHOLD
    ):
        return "decode_pool"
    if req.stage in PREFILL_STAGES:
        return "prefill_pool"
    return "decode_pool"


def stage_interference_classify_factory(live_pools_ref: list):
    """Policy 6: Stage-aware with interference avoidance.

    Returns a classify_fn that reads live pool telemetry to avoid
    cross-phase interference.

    live_pools_ref is a mutable single-element list holding the live_pools
    dict. It's populated when SemanticRouter.set_pools() is called.
    The factory pattern lets us wire this up before pools exist.
    """
    def classify(req: Request) -> str:
        pools = live_pools_ref[0] if live_pools_ref else None

        base_pool = "prefill_pool" if req.stage in PREFILL_STAGES else "decode_pool"
        alt_pool = "decode_pool" if base_pool == "prefill_pool" else "prefill_pool"

        if pools is None:
            return base_pool

        prefill_pool = pools.get("prefill_pool")
        decode_pool = pools.get("decode_pool")
        if not prefill_pool or not decode_pool:
            return base_pool

        pp_prefill = sum(i._prefill_count for i in prefill_pool.instances)
        pp_decode = sum(i._decode_count for i in prefill_pool.instances)
        pp_cap = sum(i.n_slots for i in prefill_pool.instances)

        dp_prefill = sum(i._prefill_count for i in decode_pool.instances)
        dp_decode = sum(i._decode_count for i in decode_pool.instances)
        dp_cap = sum(i.n_slots for i in decode_pool.instances)

        pp_load = (pp_prefill + pp_decode) / max(1, pp_cap)
        dp_load = (dp_prefill + dp_decode) / max(1, dp_cap)

        if req.stage in DECODE_STAGES:
            # Decode-heavy request: avoid prefill-saturated pools
            pp_prefill_ratio = pp_prefill / max(1, pp_prefill + pp_decode) if (pp_prefill + pp_decode) > 0 else 0
            if dp_load < 0.85:
                return "decode_pool"
            # decode_pool is overloaded; if prefill_pool has low prefill contention, spill
            if pp_prefill_ratio < 0.3 and pp_load < 0.9:
                return "prefill_pool"
            return "decode_pool"
        else:
            # Prefill-heavy request: avoid decode-saturated pools
            dp_decode_ratio = dp_decode / max(1, dp_prefill + dp_decode) if (dp_prefill + dp_decode) > 0 else 0
            if pp_load < 0.85:
                return "prefill_pool"
            # prefill_pool is overloaded; if decode_pool has low decode contention, spill
            if dp_decode_ratio < 0.3 and dp_load < 0.9:
                return "decode_pool"
            return "prefill_pool"

    return classify
