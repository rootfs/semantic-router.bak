# =============================================================================
# PRIVACY ROUTER — BROKEN VERSION (reproduces all three failure patterns)
# =============================================================================
#
# This DSL intentionally contains the three misconfigurations discovered during
# the privacy routing diagnosis (54% → 25% → 48%). It is used as a test case
# for the DSL Doctor pipeline.
#
# Failure 1: Signal disconnection — category_kb is configured but routes
#            only use projection conditions (the 54% baseline).
# Failure 2: Binary catchall — raw category_kb names OR'd in high-priority
#            routes (the 25% regression).
# Failure 3: Composition dominance — __tier__ OR'd with broad projection
#            (the 48% regression).

# =============================================================================
# SIGNALS
# =============================================================================

SIGNAL keyword prompt_injection_markers {
  operator: "OR"
  keywords: ["ignore previous instructions", "bypass safety", "reveal the system prompt"]
  method: "bm25"
  bm25_threshold: 0.08
}

SIGNAL keyword exfiltration_markers {
  operator: "OR"
  keywords: ["show the system prompt", "dump credentials", "expose the api key"]
  method: "bm25"
  bm25_threshold: 0.08
}

SIGNAL keyword local_only_markers {
  operator: "OR"
  keywords: ["local processing only", "do not send to the cloud", "confidential handling"]
  method: "bm25"
  bm25_threshold: 0.18
}

SIGNAL keyword private_code_markers {
  operator: "OR"
  keywords: ["private repo", "proprietary code", "internal SDK"]
  method: "bm25"
  bm25_threshold: 0.08
}

SIGNAL keyword reasoning_request_markers {
  operator: "OR"
  keywords: ["step by step", "compare the trade-offs", "root cause analysis"]
  method: "bm25"
  bm25_threshold: 0.18
}

SIGNAL embedding pii_request {
  threshold: 0.82
  candidates: ["Review an HR spreadsheet containing employee phone numbers and addresses."]
  aggregation_method: "max"
}

SIGNAL embedding frontier_reasoning_request {
  threshold: 0.88
  candidates: ["Compare distributed system architectures with explicit trade-offs."]
  aggregation_method: "max"
}

SIGNAL structure override_directive_dense {
  description: "Override-oriented instruction language."
  feature: { source: { keywords: ["ignore", "override", "bypass", "reveal"], type: "keyword_set" }, type: "density" }
  predicate: { gt: 0.06 }
}

SIGNAL jailbreak jailbreak_strict {
  method: "classifier"
  threshold: 0.45
}

SIGNAL pii pii_strict {
  threshold: 0.85
}

# BUG: category_kb is configured but will be ignored because routes only use
# projection conditions (Failure 1: signal disconnection)
SIGNAL category_kb privacy_classifier {
  kb_dir: "knowledge_bases/"
  taxonomy_path: "knowledge_bases/taxonomy.json"
  threshold: 0.30
  security_threshold: 0.25
}

# =============================================================================
# PROJECTIONS
# =============================================================================

PROJECTION score security_risk_score {
  method: "weighted_sum"
  inputs: [
    { type: "jailbreak", name: "jailbreak_strict", weight: 0.82 },
    { type: "keyword", name: "prompt_injection_markers", weight: 0.48, value_source: "confidence" },
    { type: "keyword", name: "exfiltration_markers", weight: 0.5, value_source: "confidence" },
    { type: "structure", name: "override_directive_dense", weight: 0.6 }
  ]
}

PROJECTION score privacy_risk_score {
  method: "weighted_sum"
  inputs: [
    { type: "pii", name: "pii_strict", weight: 0.92 },
    { type: "keyword", name: "local_only_markers", weight: 0.15, value_source: "confidence" },
    { type: "keyword", name: "private_code_markers", weight: 0.5, value_source: "confidence" },
    { type: "embedding", name: "pii_request", weight: 0.4, value_source: "confidence" }
  ]
}

PROJECTION score reasoning_pressure {
  method: "weighted_sum"
  inputs: [
    { type: "keyword", name: "reasoning_request_markers", weight: 0.28, value_source: "confidence" },
    { type: "embedding", name: "frontier_reasoning_request", weight: 0.48, value_source: "confidence" }
  ]
}

# BUG: contrastive score uses category_kb confidence, but category_kb results
# never enter matched-rules because no route references category_kb (Failure 1)
PROJECTION score privacy_contrastive_score {
  method: "weighted_sum"
  inputs: [{ type: "category_kb", name: "__contrastive__", weight: 1.0, value_source: "confidence" }]
}

PROJECTION mapping security_policy_band {
  source: "security_risk_score"
  method: "threshold_bands"
  outputs: [
    { name: "policy_security_standard", lt: 0.35 },
    { name: "policy_security_local_only", gte: 0.35 }
  ]
}

PROJECTION mapping privacy_policy_band {
  source: "privacy_risk_score"
  method: "threshold_bands"
  outputs: [
    { name: "policy_privacy_cloud_allowed", lt: 0.35 },
    { name: "policy_privacy_local_only", gte: 0.35 }
  ]
}

PROJECTION mapping reasoning_policy_band {
  source: "reasoning_pressure"
  method: "threshold_bands"
  outputs: [
    { name: "policy_local_reasoning", lt: 0.5 },
    { name: "policy_frontier_reasoning", gte: 0.5 }
  ]
}

PROJECTION mapping privacy_override_band {
  source: "privacy_contrastive_score"
  method: "threshold_bands"
  outputs: [
    { name: "privacy_override_inactive", lt: 0.55 },
    { name: "privacy_override_active", gte: 0.55 }
  ]
}

# =============================================================================
# MODELS
# =============================================================================

MODEL local/private-qwen {
  context_window_size: 131072
}

MODEL cloud/frontier-reasoning {
  context_window_size: 262144
}

# =============================================================================
# ROUTES — ALL THREE FAILURE PATTERNS
# =============================================================================

# Failure 2 (binary catchall): 4 raw category_kb names OR'd together.
# With security_threshold 0.25, nearly every probe matches at least one of these.
ROUTE local_security_containment {
  PRIORITY 300
  TIER 1
  WHEN projection("policy_security_local_only")
    OR category_kb("prompt_injection")
    OR category_kb("jailbreak_role")
    OR category_kb("credential_exfiltration")
    OR category_kb("system_prompt_extraction")
  MODEL "local/private-qwen"
}

# Failure 3 (composition dominance): __tier__ OR'd with broad projection.
# The broadly-firing projection swallows frontier/default probes.
ROUTE local_privacy_policy {
  PRIORITY 250
  TIER 2
  WHEN projection("policy_privacy_local_only")
    OR projection("privacy_override_active")
    OR category_kb("__tier__:privacy_policy")
  MODEL "local/private-qwen"
}

# Failure 3 again: __tier__ OR'd with projection.
ROUTE cloud_frontier_reasoning {
  PRIORITY 200
  TIER 3
  WHEN category_kb("__tier__:frontier_reasoning") OR projection("policy_frontier_reasoning")
  MODEL "cloud/frontier-reasoning"
}

# Default route uses only projections (no category_kb reference).
ROUTE local_standard {
  PRIORITY 100
  TIER 4
  WHEN projection("policy_local_reasoning")
    AND projection("policy_privacy_cloud_allowed")
    AND projection("policy_security_standard")
  MODEL "local/private-qwen"
}
