"""Privacy Routing Scenario — severity-weighted repair of routing policy.

Targets a four-lane policy (security, privacy, frontier reasoning, standard)
where misrouting a query to the wrong lane has differential severity impact.

Usage:
    from tuning import RouterClient, TuningLoop
    from tuning.scenarios import PrivacyScenario

    scenario = PrivacyScenario()
    loop = TuningLoop(
        scenario=scenario,
        router=RouterClient(),
        config_path=Path("config.yaml"),
        probes_path=Path("probes.yaml"),
    )
    output = loop.run()
"""
from __future__ import annotations

from ..scenario import Scenario
from ..engine import probe_severity


class PrivacyScenario(Scenario):

    @property
    def name(self) -> str:
        return "privacy_routing_causal_tracing"

    def severity(self, probe: dict) -> int:
        return probe_severity(probe)
