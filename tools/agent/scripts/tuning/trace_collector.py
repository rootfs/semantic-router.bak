"""Trace Collector — run probes and cache full eval traces.

Sends each probe through the running router with trace=true, caches the
complete response (signal_confidences, projection_scores, eval_trace) so
downstream analysis can reason about the network state without re-querying.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request


@dataclass
class ProbeTrace:
    """Cached trace data for a single probe."""

    probe_id: str
    query: str
    expected_decision: str
    actual_decision: str
    correct: bool
    confidence: float

    signal_confidences: dict[str, float] = field(default_factory=dict)
    projection_scores: dict[str, float] = field(default_factory=dict)
    matched_signals: dict[str, list[str]] = field(default_factory=dict)
    eval_traces: list[dict[str, Any]] = field(default_factory=list)

    raw_response: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class TraceCache:
    """Complete trace cache for all probes against a specific config."""

    config_hash: str
    timestamp: float
    probes: list[ProbeTrace] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.probes:
            return 0.0
        return sum(1 for p in self.probes if p.correct) / len(self.probes)

    @property
    def failures(self) -> list[ProbeTrace]:
        return [p for p in self.probes if not p.correct]

    def by_expected(self) -> dict[str, list[ProbeTrace]]:
        groups: dict[str, list[ProbeTrace]] = {}
        for p in self.probes:
            groups.setdefault(p.expected_decision, []).append(p)
        return groups

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "timestamp": self.timestamp,
            "accuracy": self.accuracy,
            "total": len(self.probes),
            "correct": sum(1 for p in self.probes if p.correct),
            "probes": [
                {
                    "id": p.probe_id,
                    "expected": p.expected_decision,
                    "actual": p.actual_decision,
                    "correct": p.correct,
                    "confidence": p.confidence,
                    "signal_confidences": p.signal_confidences,
                    "projection_scores": p.projection_scores,
                }
                for p in self.probes
            ],
        }


def _eval_probe(endpoint: str, query: str, timeout: int = 30) -> dict[str, Any]:
    url = f"{endpoint.rstrip('/')}/api/v1/eval?trace=true"
    body = json.dumps({"text": query}).encode()
    req = request.Request(url, data=body, method="POST",
                          headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _get_config_hash(endpoint: str) -> str:
    url = f"{endpoint.rstrip('/')}/config/hash"
    try:
        req = request.Request(url, method="GET")
        with request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("hash", "unknown")
    except Exception:
        return "unknown"


def _file_hash(path: str) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def collect_traces(
    endpoint: str,
    probes: list[dict[str, str]],
    config_path: str | None = None,
) -> TraceCache:
    """Run all probes with trace=true and build a TraceCache.

    Args:
        endpoint: Router API base URL (e.g. http://localhost:8080).
        probes: List of {"id": ..., "query": ..., "expected_decision": ...}.
        config_path: Optional local config path for file-based hashing.
    """
    config_hash = (
        _file_hash(config_path) if config_path and Path(config_path).exists()
        else _get_config_hash(endpoint)
    )

    cache = TraceCache(config_hash=config_hash, timestamp=time.time())

    for probe in probes:
        query = probe.get("query", "")
        expected = probe.get("expected_decision", "")
        probe_id = probe.get("id", "")

        try:
            resp = _eval_probe(endpoint, query)
        except Exception as exc:
            cache.probes.append(ProbeTrace(
                probe_id=probe_id,
                query=query,
                expected_decision=expected,
                actual_decision="ERROR",
                correct=False,
                confidence=0.0,
            ))
            continue

        dr = resp.get("decision_result", {})
        actual = dr.get("decision_name", "NONE")
        confidence = dr.get("confidence", 0.0)

        pt = ProbeTrace(
            probe_id=probe_id,
            query=query,
            expected_decision=expected,
            actual_decision=actual,
            correct=(actual == expected),
            confidence=confidence,
            signal_confidences=resp.get("signal_confidences", {}),
            projection_scores=resp.get("projection_scores", {}),
            matched_signals=dr.get("matched_signals", {}),
            eval_traces=dr.get("eval_trace", []),
            raw_response=resp,
        )
        cache.probes.append(pt)

    return cache


def save_cache(cache: TraceCache, path: str) -> None:
    Path(path).write_text(json.dumps(cache.to_dict(), indent=2))


def print_summary(cache: TraceCache) -> None:
    correct = sum(1 for p in cache.probes if p.correct)
    total = len(cache.probes)
    pct = 100 * correct / total if total else 0
    print(f"Accuracy: {correct}/{total} ({pct:.0f}%)")
    print(f"Config hash: {cache.config_hash[:16]}...")

    by_dec = cache.by_expected()
    for dec in sorted(by_dec):
        group = by_dec[dec]
        c = sum(1 for p in group if p.correct)
        t = len(group)
        print(f"  {dec:<35} {c}/{t} ({100*c/t:.0f}%)")

    if cache.failures:
        print(f"\nFailures ({len(cache.failures)}):")
        for f in cache.failures:
            print(f"  {f.probe_id}: expected={f.expected_decision}, "
                  f"got={f.actual_decision} (conf={f.confidence:.3f})")
