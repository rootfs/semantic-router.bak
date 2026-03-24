package dsl

import (
	"fmt"
	"strings"
)

// checkORCompositionRisks detects problematic OR compositions in route WHEN
// clauses. The primary pattern flagged is category_kb("__tier__:*") OR'd with
// projection(*), where the broad projection condition dominates and makes the
// precise tier-based signal redundant.
func (v *Validator) checkORCompositionRisks() {
	for _, route := range v.prog.Routes {
		if route.When == nil {
			continue
		}
		v.checkORCompositionInExpr(route, route.When)
	}
}

func (v *Validator) checkORCompositionInExpr(route *RouteDecl, expr BoolExpr) {
	switch e := expr.(type) {
	case *BoolAnd:
		v.checkORCompositionInExpr(route, e.Left)
		v.checkORCompositionInExpr(route, e.Right)
	case *BoolOr:
		v.checkORBranches(route, e)
		v.checkORCompositionInExpr(route, e.Left)
		v.checkORCompositionInExpr(route, e.Right)
	case *BoolNot:
		v.checkORCompositionInExpr(route, e.Expr)
	}
}

// checkORBranches examines the two sides of an OR node for problematic
// combinations of precise and broad signals.
func (v *Validator) checkORBranches(route *RouteDecl, orNode *BoolOr) {
	leftSignals := flattenSignalRefs(orNode.Left)
	rightSignals := flattenSignalRefs(orNode.Right)

	v.checkTierORProjection(route, orNode, leftSignals, rightSignals)
	v.checkTierORProjection(route, orNode, rightSignals, leftSignals)
}

func (v *Validator) checkTierORProjection(
	route *RouteDecl,
	orNode *BoolOr,
	sideA []SignalRefExpr,
	sideB []SignalRefExpr,
) {
	var tierRef *SignalRefExpr
	for i := range sideA {
		if isCategoryKBTier(sideA[i]) {
			tierRef = &sideA[i]
			break
		}
	}
	if tierRef == nil {
		return
	}

	for _, ref := range sideB {
		if strings.ToLower(ref.SignalType) == "projection" {
			v.addDiag(DiagWarning, orNode.GetPos(),
				fmt.Sprintf(
					"ROUTE %q: category_kb(%q) OR'd with projection(%q) — "+
						"the broad projection condition will dominate, making the "+
						"precise tier-based category_kb signal redundant; "+
						"consider removing the projection branch and using __tier__ alone",
					route.Name, tierRef.SignalName, ref.SignalName,
				),
				&QuickFix{
					Description: fmt.Sprintf(
						"Remove projection(%q) branch from OR; use category_kb(%q) only",
						ref.SignalName, tierRef.SignalName,
					),
					NewText: fmt.Sprintf("category_kb(\"%s\")", tierRef.SignalName),
				},
			)
			return
		}
	}
}

func isCategoryKBTier(ref SignalRefExpr) bool {
	return strings.ToLower(ref.SignalType) == "category_kb" &&
		strings.HasPrefix(ref.SignalName, "__tier__:")
}

// flattenSignalRefs collects all SignalRefExpr nodes from an expression,
// without descending into nested OR nodes (we only want the immediate branches).
func flattenSignalRefs(expr BoolExpr) []SignalRefExpr {
	var refs []SignalRefExpr
	flattenSignalRefsInto(expr, &refs)
	return refs
}

func flattenSignalRefsInto(expr BoolExpr, refs *[]SignalRefExpr) {
	switch e := expr.(type) {
	case *SignalRefExpr:
		*refs = append(*refs, *e)
	case *BoolAnd:
		flattenSignalRefsInto(e.Left, refs)
		flattenSignalRefsInto(e.Right, refs)
	case *BoolOr:
		flattenSignalRefsInto(e.Left, refs)
		flattenSignalRefsInto(e.Right, refs)
	case *BoolNot:
		flattenSignalRefsInto(e.Expr, refs)
	}
}
