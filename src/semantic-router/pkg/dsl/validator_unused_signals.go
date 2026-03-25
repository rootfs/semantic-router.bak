package dsl

import (
	"fmt"
	"strings"
)

// checkUnusedSignals warns when configured signals are never referenced by any
// decision's WHEN clause or projection score input. An unreferenced signal is
// evaluated at runtime but cannot influence routing, indicating either a
// configuration oversight or dead signal that should be removed.
func (v *Validator) checkUnusedSignals() {
	referencedSignals := v.collectAllReferencedSignals()

	for _, s := range v.prog.Signals {
		key := signalKey(s.SignalType, s.Name)
		if referencedSignals[key] {
			continue
		}
		v.addDiag(DiagWarning, s.Pos,
			fmt.Sprintf(
				"SIGNAL %s %q is configured but never referenced by any ROUTE WHEN clause or PROJECTION score input — "+
					"it will be evaluated at runtime without influencing routing",
				s.SignalType, s.Name,
			),
			&QuickFix{
				Description: fmt.Sprintf("Remove unused SIGNAL %s %q", s.SignalType, s.Name),
				NewText:     "",
			},
		)
	}

	v.checkCategoryKBCoverageGap()
}

// collectAllReferencedSignals walks all routes' WHEN clauses and projection
// score inputs to build the set of {type:name} pairs that are actually used.
func (v *Validator) collectAllReferencedSignals() map[string]bool {
	refs := make(map[string]bool)

	for _, route := range v.prog.Routes {
		if route.When != nil {
			collectSignalKeysFromExpr(route.When, refs)
		}
	}

	for _, score := range v.prog.ProjectionScores {
		for _, input := range score.Inputs {
			refs[signalKey(input.SignalType, input.SignalName)] = true
		}
	}

	for _, partition := range v.prog.ProjectionPartitions {
		for _, member := range partition.Members {
			sig := v.findSignalByName(member)
			if sig != nil {
				refs[signalKey(sig.SignalType, sig.Name)] = true
			}
		}
	}

	return refs
}

// collectSignalKeysFromExpr walks a boolean expression tree and adds all
// signal references to the provided set.
func collectSignalKeysFromExpr(expr BoolExpr, refs map[string]bool) {
	switch e := expr.(type) {
	case *BoolAnd:
		collectSignalKeysFromExpr(e.Left, refs)
		collectSignalKeysFromExpr(e.Right, refs)
	case *BoolOr:
		collectSignalKeysFromExpr(e.Left, refs)
		collectSignalKeysFromExpr(e.Right, refs)
	case *BoolNot:
		collectSignalKeysFromExpr(e.Expr, refs)
	case *SignalRefExpr:
		refs[signalKey(e.SignalType, e.SignalName)] = true
	}
}

// checkCategoryKBCoverageGap warns when category_kb signals are configured
// but routes use only projection(*) conditions without referencing category_kb
// directly. This pattern causes a match-gate disconnection where the KB
// classifier runs but its results never enter the matched-rules list.
func (v *Validator) checkCategoryKBCoverageGap() {
	hasCategoryKB := false
	for _, s := range v.prog.Signals {
		if s.SignalType == "category_kb" {
			hasCategoryKB = true
			break
		}
	}
	if !hasCategoryKB {
		return
	}

	routeRefsCategoryKB := false
	routeRefsProjection := false
	for _, route := range v.prog.Routes {
		if route.When == nil {
			continue
		}
		types := collectSignalTypesFromExpr(route.When)
		if types["category_kb"] {
			routeRefsCategoryKB = true
		}
		if types["projection"] {
			routeRefsProjection = true
		}
	}

	if routeRefsProjection && !routeRefsCategoryKB {
		v.addDiag(DiagWarning, Position{},
			"category_kb signals are configured but routes only reference projection(*) conditions — "+
				"category_kb match results won't enter the matched-rules gate, "+
				"causing projection scores that depend on category_kb confidence to see zero input; "+
				"add category_kb conditions to route WHEN clauses or use projection score inputs with value_source: confidence",
			nil,
		)
	}
}

// collectSignalTypesFromExpr extracts all distinct signal types from an expression.
func collectSignalTypesFromExpr(expr BoolExpr) map[string]bool {
	types := make(map[string]bool)
	collectSignalTypesFromExprInto(expr, types)
	return types
}

func collectSignalTypesFromExprInto(expr BoolExpr, types map[string]bool) {
	switch e := expr.(type) {
	case *BoolAnd:
		collectSignalTypesFromExprInto(e.Left, types)
		collectSignalTypesFromExprInto(e.Right, types)
	case *BoolOr:
		collectSignalTypesFromExprInto(e.Left, types)
		collectSignalTypesFromExprInto(e.Right, types)
	case *BoolNot:
		collectSignalTypesFromExprInto(e.Expr, types)
	case *SignalRefExpr:
		types[strings.ToLower(e.SignalType)] = true
	}
}

func signalKey(signalType, signalName string) string {
	return signalType + ":" + signalName
}
