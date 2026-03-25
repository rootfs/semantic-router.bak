package dsl

import (
	"strings"
	"testing"
)

func TestCheckUnusedSignals_NoWarningWhenAllReferenced(t *testing.T) {
	input := `
SIGNAL keyword code_help {
	keywords: ["code", "debug"]
}

ROUTE code_route {
	WHEN keyword("code_help")
	MODEL "gpt-4"
}
`
	diags, _ := Validate(input)
	for _, d := range diags {
		if strings.Contains(d.Message, "never referenced") {
			t.Errorf("unexpected unused signal warning: %s", d.Message)
		}
	}
}

func TestCheckUnusedSignals_WarnsOnUnreferencedSignal(t *testing.T) {
	input := `
SIGNAL keyword code_help {
	keywords: ["code", "debug"]
}

SIGNAL keyword math_help {
	keywords: ["math", "calculus"]
}

ROUTE code_route {
	WHEN keyword("code_help")
	MODEL "gpt-4"
}
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if strings.Contains(d.Message, "math_help") && strings.Contains(d.Message, "never referenced") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected warning about unused signal math_help")
	}
}

func TestCheckUnusedSignals_ProjectionScoreCountsAsReference(t *testing.T) {
	input := `
SIGNAL domain economics {
	mmlu_categories: ["economics"]
}

SIGNAL domain physics {
	mmlu_categories: ["physics"]
}

PROJECTION partition knowledge_areas {
	semantics: softmax_exclusive
	temperature: 0.1
	members: [economics, physics]
	default: economics
}

PROJECTION score knowledge_score {
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

PROJECTION mapping knowledge_map {
	source: knowledge_score
	method: threshold_bands
	outputs: [
		{
			name: "econ_high"
			gte: 0.7
		}
	]
}

ROUTE econ_route {
	WHEN projection("econ_high")
	MODEL "gpt-4"
}
`
	diags, _ := Validate(input)
	for _, d := range diags {
		if strings.Contains(d.Message, "economics") && strings.Contains(d.Message, "never referenced") {
			t.Errorf("economics should be considered referenced via projection score input: %s", d.Message)
		}
	}
}

func TestCheckUnusedSignals_PartitionMemberCountsAsReference(t *testing.T) {
	input := `
SIGNAL domain math {
	mmlu_categories: ["math"]
}

SIGNAL domain science {
	mmlu_categories: ["physics"]
}

PROJECTION partition academic {
	semantics: softmax_exclusive
	temperature: 0.1
	members: [math, science]
	default: math
}

PROJECTION score academic_score {
	method: weighted_sum
	inputs: [
		{
			type: domain
			name: "math"
			weight: 1.0
		}
	]
}

PROJECTION mapping academic_map {
	source: academic_score
	method: threshold_bands
	outputs: [
		{
			name: "math_high"
			gte: 0.7
		}
	]
}

ROUTE math_route {
	WHEN projection("math_high")
	MODEL "gpt-4"
}
`
	diags, _ := Validate(input)
	for _, d := range diags {
		if strings.Contains(d.Message, "science") && strings.Contains(d.Message, "never referenced") {
			t.Errorf("science should be considered referenced via partition membership: %s", d.Message)
		}
	}
}
