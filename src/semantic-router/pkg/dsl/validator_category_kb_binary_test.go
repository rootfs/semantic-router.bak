package dsl

import (
	"strings"
	"testing"
)

func TestCheckCategoryKBBinary_NoWarningForFewORs(t *testing.T) {
	input := `
SIGNAL category_kb privacy_kb {
	kb_dir: "/path/to/kb"
	threshold: 0.5
}

ROUTE two_cats {
	WHEN category_kb("customer_data") OR category_kb("pii")
	MODEL "gpt-4"
}
`
	diags, _ := Validate(input)
	for _, d := range diags {
		if strings.Contains(d.Message, "raw category_kb names OR'd") {
			t.Errorf("unexpected binary vs best-match warning for only 2 categories: %s", d.Message)
		}
	}
}

func TestCheckCategoryKBBinary_WarnsOnThreeOrMoreRawORs(t *testing.T) {
	input := `
SIGNAL category_kb privacy_kb {
	kb_dir: "/path/to/kb"
	threshold: 0.5
}

ROUTE catchall_sensitive {
	WHEN category_kb("customer_data") OR category_kb("pii") OR category_kb("trade_secret")
	MODEL "gpt-4"
}
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if strings.Contains(d.Message, "raw category_kb names OR'd") &&
			strings.Contains(d.Message, "customer_data") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected binary vs best-match warning for 3+ OR'd raw category_kb names")
	}
}

func TestCheckCategoryKBBinary_NoWarningForTierPattern(t *testing.T) {
	input := `
SIGNAL category_kb privacy_kb {
	kb_dir: "/path/to/kb"
	threshold: 0.5
}

ROUTE tier_route {
	WHEN category_kb("__tier__:sensitive") OR category_kb("__tier__:public") OR category_kb("__tier__:internal")
	MODEL "gpt-4"
}
`
	diags, _ := Validate(input)
	for _, d := range diags {
		if strings.Contains(d.Message, "raw category_kb names OR'd") {
			t.Errorf("unexpected binary vs best-match warning for __tier__ pattern: %s", d.Message)
		}
	}
}

func TestCheckCategoryKBBinary_MixedRawAndSyntheticCounts(t *testing.T) {
	input := `
SIGNAL category_kb privacy_kb {
	kb_dir: "/path/to/kb"
	threshold: 0.5
}

ROUTE mixed {
	WHEN category_kb("customer_data") OR category_kb("__tier__:sensitive") OR category_kb("pii") OR category_kb("trade_secret")
	MODEL "gpt-4"
}
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if strings.Contains(d.Message, "3 raw category_kb names OR'd") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected warning for 3 raw names even with 1 synthetic name mixed in")
	}
}
