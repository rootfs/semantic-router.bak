package dsl

import (
	"fmt"
	"strings"
)

// checkCategoryKBBinaryVsBestMatch warns when a high-priority route uses
// multiple OR'd raw category_kb("real_name") conditions instead of the
// more precise __tier__/__best__ synthetic patterns.
//
// Using 3+ raw category names OR'd together creates a coarse binary
// match (any-of-N) that ignores the category KB classifier's relative
// ranking. The __tier__ pattern leverages the classifier's contrastive
// scoring to select the best-matching tier, producing tighter routing.
func (v *Validator) checkCategoryKBBinaryVsBestMatch() {
	for _, route := range v.prog.Routes {
		if route.When == nil {
			continue
		}
		v.checkCategoryKBBinaryPattern(route, route.When)
	}
}

func (v *Validator) checkCategoryKBBinaryPattern(route *RouteDecl, expr BoolExpr) {
	switch e := expr.(type) {
	case *BoolAnd:
		v.checkCategoryKBBinaryPattern(route, e.Left)
		v.checkCategoryKBBinaryPattern(route, e.Right)
	case *BoolOr:
		rawNames := collectRawCategoryKBFromOR(e)
		if len(rawNames) >= 3 {
			v.addDiag(DiagWarning, e.GetPos(),
				fmt.Sprintf(
					"ROUTE %q: %d raw category_kb names OR'd together (%s) — "+
						"this creates a coarse binary match ignoring the classifier's "+
						"relative ranking; consider using category_kb(\"__tier__:<tier>\") "+
						"for best-match semantics",
					route.Name, len(rawNames), strings.Join(quoteStrings(rawNames), ", "),
				),
				&QuickFix{
					Description: "Replace with category_kb(\"__tier__:<tier>\") pattern",
					NewText:     "category_kb(\"__tier__:<tier>\")",
				},
			)
		}
		v.checkCategoryKBBinaryPattern(route, e.Left)
		v.checkCategoryKBBinaryPattern(route, e.Right)
	case *BoolNot:
		v.checkCategoryKBBinaryPattern(route, e.Expr)
	}
}

// collectRawCategoryKBFromOR collects all raw (non-synthetic) category_kb
// signal names from an OR tree. Only counts names that are not __tier__,
// __best__, or __contrastive__ patterns.
func collectRawCategoryKBFromOR(expr BoolExpr) []string {
	var names []string
	collectRawCategoryKBNamesFromOR(expr, &names)
	return names
}

func collectRawCategoryKBNamesFromOR(expr BoolExpr, names *[]string) {
	switch e := expr.(type) {
	case *BoolOr:
		collectRawCategoryKBNamesFromOR(e.Left, names)
		collectRawCategoryKBNamesFromOR(e.Right, names)
	case *SignalRefExpr:
		if strings.ToLower(e.SignalType) == "category_kb" && isRawCategoryKBName(e.SignalName) {
			*names = append(*names, e.SignalName)
		}
	case *BoolAnd:
		// AND inside OR: check each side for category_kb
		andRefs := flattenSignalRefs(e)
		for _, ref := range andRefs {
			if strings.ToLower(ref.SignalType) == "category_kb" && isRawCategoryKBName(ref.SignalName) {
				*names = append(*names, ref.SignalName)
			}
		}
	}
}

// isRawCategoryKBName returns true for category_kb names that are NOT
// synthetic patterns (__tier__:*, __best__:*, __contrastive__).
func isRawCategoryKBName(name string) bool {
	if name == "__contrastive__" {
		return false
	}
	if strings.HasPrefix(name, "__tier__:") || strings.HasPrefix(name, "__best__:") {
		return false
	}
	return true
}

func quoteStrings(ss []string) []string {
	quoted := make([]string, len(ss))
	for i, s := range ss {
		quoted[i] = fmt.Sprintf("%q", s)
	}
	return quoted
}
