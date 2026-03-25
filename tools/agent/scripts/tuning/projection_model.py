"""Projection Model — local reconstruction of projection scores from config + traces.

Parses the projection score formulas and band boundaries from config.yaml,
then recomputes scores locally using signal_confidences from cached traces.
This enables analytical fix computation without querying the router.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ScoreInput:
    signal_type: str
    signal_name: str
    weight: float
    value_source: str = "binary"
    match: float = 1.0
    miss: float = 0.0


@dataclass
class ProjectionScore:
    name: str
    method: str
    inputs: list[ScoreInput] = field(default_factory=list)


@dataclass
class BandOutput:
    name: str
    lt: float | None = None
    lte: float | None = None
    gt: float | None = None
    gte: float | None = None

    def matches(self, score: float) -> bool:
        if self.gt is not None and not (score > self.gt):
            return False
        if self.gte is not None and not (score >= self.gte):
            return False
        if self.lt is not None and not (score < self.lt):
            return False
        if self.lte is not None and not (score <= self.lte):
            return False
        return True

    @property
    def lower_bound(self) -> float | None:
        if self.gte is not None:
            return self.gte
        if self.gt is not None:
            return self.gt
        return None


@dataclass
class ProjectionMapping:
    name: str
    source: str
    outputs: list[BandOutput] = field(default_factory=list)
    calibration_slope: float = 12.0


@dataclass
class ProjectionModel:
    """Local model of the projection computation graph."""

    scores: list[ProjectionScore] = field(default_factory=list)
    mappings: list[ProjectionMapping] = field(default_factory=list)

    # Lookup caches
    _score_by_name: dict[str, ProjectionScore] = field(default_factory=dict, repr=False)
    _mapping_by_source: dict[str, ProjectionMapping] = field(default_factory=dict, repr=False)
    _band_by_output: dict[str, tuple[ProjectionMapping, BandOutput]] = field(default_factory=dict, repr=False)

    def _build_indices(self) -> None:
        self._score_by_name = {s.name: s for s in self.scores}
        self._mapping_by_source = {m.source: m for m in self.mappings}
        self._band_by_output = {}
        for m in self.mappings:
            for b in m.outputs:
                self._band_by_output[b.name] = (m, b)

    def compute_score(
        self,
        score_name: str,
        signal_confidences: dict[str, float],
        matched_signals: dict[str, list[str]],
    ) -> float:
        """Recompute a projection score from signal data."""
        score_def = self._score_by_name.get(score_name)
        if not score_def:
            return 0.0

        total = 0.0
        for inp in score_def.inputs:
            total += inp.weight * self._input_value(inp, signal_confidences, matched_signals)
        return total

    def _input_value(
        self,
        inp: ScoreInput,
        signal_confidences: dict[str, float],
        matched_signals: dict[str, list[str]],
    ) -> float:
        matched = self._is_matched(inp.signal_type, inp.signal_name, matched_signals)

        if inp.value_source == "confidence":
            if not matched:
                return 0.0
            key = f"{inp.signal_type}:{inp.signal_name}"
            return signal_confidences.get(key, 1.0)
        else:
            return inp.match if matched else inp.miss

    @staticmethod
    def _is_matched(
        signal_type: str,
        signal_name: str,
        matched_signals: dict[str, list[str]],
    ) -> bool:
        rules = matched_signals.get(signal_type, [])
        return signal_name in rules

    def find_matching_band(self, score_name: str, score_value: float) -> BandOutput | None:
        mapping = self._mapping_by_source.get(score_name)
        if not mapping:
            return None
        for band in mapping.outputs:
            if band.matches(score_value):
                return band
        return None

    def get_band_for_output(self, output_name: str) -> tuple[ProjectionMapping, BandOutput] | None:
        return self._band_by_output.get(output_name)

    def get_score_for_output(self, output_name: str) -> ProjectionScore | None:
        """Given a projection output name, find the source score definition."""
        info = self._band_by_output.get(output_name)
        if not info:
            return None
        mapping, _ = info
        return self._score_by_name.get(mapping.source)


def load_projection_model(config_path: str) -> ProjectionModel:
    """Parse projection scores and mappings from a YAML config file."""
    import yaml

    raw = yaml.safe_load(Path(config_path).read_text())
    projections = raw.get("routing", {}).get("projections", {})

    scores = []
    for s in projections.get("scores", []):
        inputs = []
        for i in s.get("inputs", []):
            inputs.append(ScoreInput(
                signal_type=i.get("type", ""),
                signal_name=i.get("name", ""),
                weight=float(i.get("weight", 0)),
                value_source=i.get("value_source", "binary"),
                match=float(i.get("match", 1.0)),
                miss=float(i.get("miss", 0.0)),
            ))
        scores.append(ProjectionScore(
            name=s.get("name", ""),
            method=s.get("method", "weighted_sum"),
            inputs=inputs,
        ))

    mappings = []
    for m in projections.get("mappings", []):
        outputs = []
        for o in m.get("outputs", []):
            outputs.append(BandOutput(
                name=o.get("name", ""),
                lt=o.get("lt"),
                lte=o.get("lte"),
                gt=o.get("gt"),
                gte=o.get("gte"),
            ))
        cal = m.get("calibration", {})
        mappings.append(ProjectionMapping(
            name=m.get("name", ""),
            source=m.get("source", ""),
            outputs=outputs,
            calibration_slope=float(cal.get("slope", 12.0)) if cal else 12.0,
        ))

    model = ProjectionModel(scores=scores, mappings=mappings)
    model._build_indices()
    return model
