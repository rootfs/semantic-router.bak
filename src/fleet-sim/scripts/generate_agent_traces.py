"""Generate synthetic but realistic pre-annotated agent traces.

Produces two datasets mimicking:
  1. SWE-agent trajectories (multi-turn coding agent with search/edit/test)
  2. AgentBench traces (deterministic benchmark tasks with clean stage boundaries)

Stage classification follows the heuristics from Appendix B:
  - Plan:      short prompt, short output (thinking/planning)
  - Explore:   large prompt (context loading), short output (search results)
  - Implement: medium prompt, large output (code generation)
  - Test:      medium prompt, variable output (test execution results)

Each session represents one agent task with 5-20 turns progressing through
stages in a realistic order: Plan -> Explore -> Implement -> Test, with
possible loops back to Explore or Implement on failures.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

STAGE_PROFILES = {
    "plan": {
        "l_in_range": (200, 1200),
        "l_out_range": (50, 300),
        "projected_decode_range": (40, 250),
    },
    "explore": {
        "l_in_range": (2000, 7000),
        "l_out_range": (30, 200),
        "projected_decode_range": (25, 180),
    },
    "implement": {
        "l_in_range": (1000, 4000),
        "l_out_range": (300, 1500),
        "projected_decode_range": (250, 1400),
    },
    "test": {
        "l_in_range": (800, 3000),
        "l_out_range": (50, 800),
        "projected_decode_range": (40, 700),
    },
}

# Markov transition probabilities for stage sequencing
SWE_AGENT_TRANSITIONS = {
    "plan":      {"explore": 0.7, "implement": 0.2, "plan": 0.1},
    "explore":   {"implement": 0.5, "explore": 0.3, "plan": 0.1, "test": 0.1},
    "implement": {"test": 0.5, "implement": 0.2, "explore": 0.2, "plan": 0.1},
    "test":      {"implement": 0.4, "explore": 0.3, "plan": 0.2, "test": 0.1},
}

AGENTBENCH_TRANSITIONS = {
    "plan":      {"explore": 0.8, "plan": 0.2},
    "explore":   {"implement": 0.6, "explore": 0.3, "plan": 0.1},
    "implement": {"test": 0.6, "implement": 0.3, "explore": 0.1},
    "test":      {"implement": 0.3, "explore": 0.2, "test": 0.3, "plan": 0.2},
}


def _next_stage(current: str, transitions: dict, rng: random.Random) -> str:
    targets = list(transitions[current].keys())
    weights = list(transitions[current].values())
    return rng.choices(targets, weights=weights, k=1)[0]


def generate_session(
    session_id: str,
    start_time: float,
    rng: random.Random,
    transitions: dict,
    n_turns_range: tuple[int, int] = (5, 20),
    inter_turn_mean: float = 0.3,
) -> list[dict]:
    """Generate a single agent session (one task) with realistic stage progression."""
    n_turns = rng.randint(*n_turns_range)
    stage = "plan"
    records = []
    t = start_time

    for turn_idx in range(n_turns):
        prof = STAGE_PROFILES[stage]
        l_in = rng.randint(*prof["l_in_range"])
        l_out = rng.randint(*prof["l_out_range"])
        proj_decode = rng.randint(*prof["projected_decode_range"])

        # Prompt growth: later turns accumulate context
        context_growth = int(turn_idx * rng.uniform(100, 400))
        l_in += context_growth

        records.append({
            "timestamp": round(t, 6),
            "prompt_tokens": l_in,
            "generated_tokens": l_out,
            "selected_model": "llama70b",
            "stage": stage,
            "session_id": session_id,
            "turn_index": turn_idx,
            "projected_decode_tokens": proj_decode,
            "category": "code",
            "complexity": "high" if stage in ("implement", "test") else "medium",
        })

        t += rng.expovariate(1.0 / inter_turn_mean)
        stage = _next_stage(stage, transitions, rng)

    return records


def generate_trace(
    output_path: str,
    n_sessions: int,
    arrival_rate: float,
    transitions: dict,
    n_turns_range: tuple[int, int],
    seed: int,
    dataset_name: str,
):
    rng = random.Random(seed)
    all_records = []
    t = 0.0

    for i in range(n_sessions):
        sid = f"{dataset_name}-session-{i:04d}"
        session_records = generate_session(
            session_id=sid,
            start_time=t,
            rng=rng,
            transitions=transitions,
            n_turns_range=n_turns_range,
        )
        all_records.extend(session_records)
        t += rng.expovariate(arrival_rate)

    all_records.sort(key=lambda r: r["timestamp"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    # Print summary statistics
    stages = {}
    total_prefill = 0
    total_decode = 0
    for r in all_records:
        s = r["stage"]
        stages[s] = stages.get(s, 0) + 1
        total_prefill += r["prompt_tokens"]
        total_decode += r["generated_tokens"]

    print(f"\n  {dataset_name} trace: {output_path}")
    print(f"    Sessions: {n_sessions}")
    print(f"    Total requests: {len(all_records)}")
    print(f"    Stage distribution:")
    for s in ["plan", "explore", "implement", "test"]:
        cnt = stages.get(s, 0)
        pct = 100 * cnt / len(all_records)
        print(f"      {s:12s}: {cnt:5d} ({pct:5.1f}%)")
    print(f"    Total prefill tokens: {total_prefill:,}")
    print(f"    Total decode tokens:  {total_decode:,}")
    ratio = total_decode / total_prefill if total_prefill else 0
    print(f"    Decode/prefill ratio: {ratio:.3f}")


def main():
    traces_dir = Path(__file__).parent.parent / "examples" / "trace_samples"

    print("=" * 70)
    print("Generating pre-annotated agent trace datasets")
    print("=" * 70)

    # SWE-agent traces: more sessions, more exploratory loops
    generate_trace(
        output_path=str(traces_dir / "swe_agent_annotated.jsonl"),
        n_sessions=500,
        arrival_rate=3.0,
        transitions=SWE_AGENT_TRANSITIONS,
        n_turns_range=(6, 20),
        seed=42,
        dataset_name="swe-agent",
    )

    # AgentBench traces: more structured, shorter sessions
    generate_trace(
        output_path=str(traces_dir / "agentbench_annotated.jsonl"),
        n_sessions=500,
        arrival_rate=4.0,
        transitions=AGENTBENCH_TRANSITIONS,
        n_turns_range=(5, 15),
        seed=123,
        dataset_name="agentbench",
    )

    print("\n" + "=" * 70)
    print("Trace generation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
