package dsl

import (
	"fmt"
	"strings"
)

// checkMatchGateConsistency verifies that projection score inputs with
// value_source: "confidence" reference signals that have a path to the
// matched-rules list. If a signal's confidence is used but its name never
// appears in matched rules, downstream projections will see zero input.
//
// It also warns on unknown synthetic category_kb names that are not part
// of the recognised set (__tier__:*, __best__:*, __contrastive__).
func (v *Validator) checkMatchGateConsistency() {
	routeReferencedSignals := v.collectRouteReferencedSignalsByType()

	for _, score := range v.prog.ProjectionScores {
		context := fmt.Sprintf("PROJECTION score %s", score.Name)
		for _, input := range score.Inputs {
			if input == nil {
				continue
			}
			v.checkScoreInputMatchGate(context, score.Pos, input, routeReferencedSignals)
		}
	}

	v.checkCategoryKBSyntheticNames()
}

// checkScoreInputMatchGate checks a single projection score input for
// match-gate disconnection.
func (v *Validator) checkScoreInputMatchGate(
	context string,
	pos Position,
	input *ProjectionScoreInputDecl,
	routeRefs map[string]map[string]bool,
) {
	if input.ValueSource != "confidence" {
		return
	}

	typeRefs := routeRefs[input.SignalType]
	if typeRefs != nil && typeRefs[input.SignalName] {
		return
	}

	v.addDiag(DiagWarning, pos,
		fmt.Sprintf(
			"%s: input %s(%q) uses value_source: confidence, but %s(%q) is not referenced "+
				"in any ROUTE WHEN clause — its confidence value will be stored but "+
				"the signal name won't appear in the matched-rules list, causing "+
				"this projection input to effectively contribute zero",
			context, input.SignalType, input.SignalName,
			input.SignalType, input.SignalName,
		),
		&QuickFix{
			Description: fmt.Sprintf(
				"Add %s(%q) to a ROUTE WHEN clause, or change value_source to binary",
				input.SignalType, input.SignalName,
			),
			NewText: "binary",
		},
	)
}

// collectRouteReferencedSignalsByType builds a map of signalType → {signalName → true}
// from all routes' WHEN clauses.
func (v *Validator) collectRouteReferencedSignalsByType() map[string]map[string]bool {
	refs := make(map[string]map[string]bool)
	for _, route := range v.prog.Routes {
		if route.When == nil {
			continue
		}
		collectSignalRefsByType(route.When, refs)
	}
	return refs
}

func collectSignalRefsByType(expr BoolExpr, refs map[string]map[string]bool) {
	switch e := expr.(type) {
	case *BoolAnd:
		collectSignalRefsByType(e.Left, refs)
		collectSignalRefsByType(e.Right, refs)
	case *BoolOr:
		collectSignalRefsByType(e.Left, refs)
		collectSignalRefsByType(e.Right, refs)
	case *BoolNot:
		collectSignalRefsByType(e.Expr, refs)
	case *SignalRefExpr:
		if refs[e.SignalType] == nil {
			refs[e.SignalType] = make(map[string]bool)
		}
		refs[e.SignalType][e.SignalName] = true
	}
}

// checkCategoryKBSyntheticNames warns on category_kb signal references that
// use unknown synthetic name patterns.
func (v *Validator) checkCategoryKBSyntheticNames() {
	for _, route := range v.prog.Routes {
		if route.When == nil {
			continue
		}
		checkCategoryKBNamesInExpr(route.When, v)
	}
}

func checkCategoryKBNamesInExpr(expr BoolExpr, v *Validator) {
	switch e := expr.(type) {
	case *BoolAnd:
		checkCategoryKBNamesInExpr(e.Left, v)
		checkCategoryKBNamesInExpr(e.Right, v)
	case *BoolOr:
		checkCategoryKBNamesInExpr(e.Left, v)
		checkCategoryKBNamesInExpr(e.Right, v)
	case *BoolNot:
		checkCategoryKBNamesInExpr(e.Expr, v)
	case *SignalRefExpr:
		if e.SignalType != "category_kb" {
			return
		}
		if !isKnownCategoryKBName(e.SignalName, v) {
			v.addDiag(DiagWarning, e.Pos,
				fmt.Sprintf(
					"category_kb(%q) uses an unknown synthetic name — "+
						"recognised patterns: __tier__:<tier>, __best__:<category>, __contrastive__, "+
						"or a declared category_kb signal name",
					e.SignalName,
				),
				nil,
			)
		}
	}
}

// isKnownCategoryKBName returns true for declared category_kb signal names
// and the recognised synthetic patterns.
func isKnownCategoryKBName(name string, v *Validator) bool {
	if v.isSignalDefined("category_kb", name) {
		return true
	}
	if name == "__contrastive__" {
		return true
	}
	if strings.HasPrefix(name, "__tier__:") || strings.HasPrefix(name, "__best__:") {
		return true
	}
	return false
}
