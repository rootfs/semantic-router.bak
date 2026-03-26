"""DSL Tuning Framework — extensible analytical optimization for semantic router configs.

Core modules:
  client     — RouterClient for eval/config/reload HTTP interactions
  probes     — probe loading and result persistence
  engine     — causal tracing engine (Algorithm 1, decomposition, fixes, regression)
  analyzer   — offline threshold optimization on cached data
  scenario   — Scenario ABC + TuningLoop for pluggable tuning pipelines

Scenario plugins live in tuning.scenarios.
"""

from .client import RouterClient
from .probes import load_probes, save_results
from .scenario import Scenario, TuningLoop
from .analyzer import OfflineAnalyzer

__all__ = [
    "RouterClient",
    "Scenario",
    "TuningLoop",
    "OfflineAnalyzer",
    "load_probes",
    "save_results",
]
