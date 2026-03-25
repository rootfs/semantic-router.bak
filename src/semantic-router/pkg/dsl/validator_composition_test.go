package dsl

import (
	"strings"
	"testing"
)

func TestCheckORComposition_NoWarningWithoutTierORProjection(t *testing.T) {
	input := `
SIGNAL keyword code_help {
	keywords: ["code"]
}

SIGNAL keyword math_help {
	keywords: ["math"]
}

ROUTE mixed_route {
	WHEN keyword("code_help") OR keyword("math_help")
	MODEL "gpt-4"
}
`
	diags, _ := Validate(input)
	for _, d := range diags {
		if strings.Contains(d.Message, "projection") && strings.Contains(d.Message, "dominat") {
			t.Errorf("unexpected OR composition warning: %s", d.Message)
		}
	}
}

func TestCheckORComposition_WarnsOnTierORProjection(t *testing.T) {
	input := `
SIGNAL category_kb privacy_kb {
	kb_dir: "/path/to/kb"
	threshold: 0.5
}

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

PROJECTION score risk_score {
	method: weighted_sum
	inputs: [
		{
			type: domain
			name: "economics"
			weight: 1.0
		}
	]
}

PROJECTION mapping risk_map {
	source: risk_score
	method: threshold_bands
	outputs: [
		{
			name: "high_risk"
			gte: 0.7
		}
	]
}

ROUTE risky_route {
	WHEN category_kb("__tier__:sensitive") OR projection("high_risk")
	MODEL "gpt-4"
}
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if strings.Contains(d.Message, "__tier__:sensitive") &&
			strings.Contains(d.Message, "projection") &&
			strings.Contains(d.Message, "dominat") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected warning about category_kb tier OR'd with projection")
	}
}

func TestCheckORComposition_NoWarningForTierANDProjection(t *testing.T) {
	input := `
SIGNAL category_kb privacy_kb {
	kb_dir: "/path/to/kb"
	threshold: 0.5
}

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

PROJECTION score risk_score {
	method: weighted_sum
	inputs: [
		{
			type: domain
			name: "economics"
			weight: 1.0
		}
	]
}

PROJECTION mapping risk_map {
	source: risk_score
	method: threshold_bands
	outputs: [
		{
			name: "high_risk"
			gte: 0.7
		}
	]
}

ROUTE precise_route {
	WHEN category_kb("__tier__:sensitive") AND projection("high_risk")
	MODEL "gpt-4"
}
`
	diags, _ := Validate(input)
	for _, d := range diags {
		if strings.Contains(d.Message, "dominat") {
			t.Errorf("unexpected OR composition warning for AND expression: %s", d.Message)
		}
	}
}
