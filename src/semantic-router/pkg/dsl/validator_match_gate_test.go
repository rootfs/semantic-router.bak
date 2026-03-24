package dsl

import (
	"strings"
	"testing"
)

func TestCheckMatchGate_NoWarningWhenSignalReferenced(t *testing.T) {
	input := `
SIGNAL domain economics {
	mmlu_categories: ["economics"]
}

SIGNAL domain physics {
	mmlu_categories: ["physics"]
}

PROJECTION partition knowledge {
	semantics: softmax_exclusive
	temperature: 0.1
	members: [economics, physics]
	default: economics
}

PROJECTION score econ_score {
	method: weighted_sum
	inputs: [
		{
			type: domain
			name: "economics"
			weight: 1.0
			value_source: confidence
		}
	]
}

PROJECTION mapping econ_map {
	source: econ_score
	method: threshold_bands
	outputs: [
		{
			name: "econ_high"
			gte: 0.7
		}
	]
}

ROUTE econ_route {
	WHEN domain("economics")
	MODEL "gpt-4"
}
`
	diags, _ := Validate(input)
	for _, d := range diags {
		if strings.Contains(d.Message, "match-gate") || strings.Contains(d.Message, "matched-rules list") {
			t.Errorf("unexpected match-gate warning: %s", d.Message)
		}
	}
}

func TestCheckMatchGate_WarnsOnDisconnectedConfidenceInput(t *testing.T) {
	input := `
SIGNAL domain economics {
	mmlu_categories: ["economics"]
}

SIGNAL domain physics {
	mmlu_categories: ["physics"]
}

PROJECTION partition knowledge {
	semantics: softmax_exclusive
	temperature: 0.1
	members: [economics, physics]
	default: economics
}

PROJECTION score econ_score {
	method: weighted_sum
	inputs: [
		{
			type: domain
			name: "economics"
			weight: 1.0
			value_source: confidence
		}
	]
}

PROJECTION mapping econ_map {
	source: econ_score
	method: threshold_bands
	outputs: [
		{
			name: "econ_high"
			gte: 0.7
		}
	]
}

ROUTE some_route {
	WHEN projection("econ_high")
	MODEL "gpt-4"
}
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if strings.Contains(d.Message, "economics") &&
			strings.Contains(d.Message, "value_source: confidence") &&
			strings.Contains(d.Message, "matched-rules list") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected match-gate disconnection warning for domain economics confidence input")
	}
}

func TestCheckCategoryKBSyntheticNames_NoWarningForValidNames(t *testing.T) {
	input := `
SIGNAL category_kb privacy_kb {
	kb_dir: "/path/to/kb"
	threshold: 0.5
}

ROUTE tier_route {
	WHEN category_kb("__tier__:sensitive")
	MODEL "gpt-4"
}
`
	diags, _ := Validate(input)
	for _, d := range diags {
		if strings.Contains(d.Message, "unknown synthetic name") {
			t.Errorf("unexpected synthetic name warning: %s", d.Message)
		}
	}
}

func TestCheckCategoryKBSyntheticNames_WarnsOnUnknownPattern(t *testing.T) {
	input := `
SIGNAL category_kb privacy_kb {
	kb_dir: "/path/to/kb"
	threshold: 0.5
}

ROUTE unknown_route {
	WHEN category_kb("__invalid__:something")
	MODEL "gpt-4"
}
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if strings.Contains(d.Message, "__invalid__:something") && strings.Contains(d.Message, "unknown synthetic name") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected warning about unknown synthetic category_kb name")
	}
}
