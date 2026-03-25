package dsl

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

// DoctorDiagnostic represents a single finding from the DSL Doctor.
type DoctorDiagnostic struct {
	Level   string    `json:"level"`
	Message string    `json:"message"`
	Fix     *QuickFix `json:"fix,omitempty"`
}

// DoctorReport is the structured output of a doctor run.
type DoctorReport struct {
	InputPath   string             `json:"input_path"`
	Diagnostics []DoctorDiagnostic `json:"diagnostics"`
	Summary     DoctorSummary      `json:"summary"`
}

// DoctorSummary counts issues by severity.
type DoctorSummary struct {
	Errors      int `json:"errors"`
	Warnings    int `json:"warnings"`
	Constraints int `json:"constraints"`
}

// CLIDoctor runs static DSL analysis and outputs a diagnosis report.
// When jsonOutput is true, prints structured JSON instead of human-readable text.
// Returns the number of errors found.
func CLIDoctor(inputPath string, w io.Writer, jsonOutput bool) int {
	data, err := os.ReadFile(inputPath)
	if err != nil {
		_, _ = fmt.Fprintf(w, "failed to read input file: %s\n", err)
		return 1
	}

	diags, _ := Validate(string(data))

	report := buildDoctorReport(inputPath, diags)

	if jsonOutput {
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		_ = enc.Encode(report)
	} else {
		writeDoctorHumanReport(w, report)
	}

	return report.Summary.Errors
}

func buildDoctorReport(inputPath string, diags []Diagnostic) DoctorReport {
	report := DoctorReport{
		InputPath: inputPath,
	}

	for _, d := range diags {
		dd := DoctorDiagnostic{
			Level:   d.Level.String(),
			Message: d.Message,
			Fix:     d.Fix,
		}
		report.Diagnostics = append(report.Diagnostics, dd)

		switch d.Level {
		case DiagError:
			report.Summary.Errors++
		case DiagWarning:
			report.Summary.Warnings++
		case DiagConstraint:
			report.Summary.Constraints++
		}
	}

	return report
}

func writeDoctorHumanReport(w io.Writer, report DoctorReport) {
	_, _ = fmt.Fprintf(w, "DSL Doctor Report: %s\n", report.InputPath)
	_, _ = fmt.Fprintf(w, "%s\n\n", strings.Repeat("=", 40+len(report.InputPath)))

	if len(report.Diagnostics) == 0 {
		_, _ = fmt.Fprintln(w, "No issues found. DSL configuration is healthy.")
		return
	}

	// Group by category for the doctor-specific checks
	unusedSignals := filterDiagnostics(report.Diagnostics, "never referenced")
	matchGate := filterDiagnostics(report.Diagnostics, "matched-rules list")
	composition := filterDiagnostics(report.Diagnostics, "dominat")
	binaryCatchall := filterDiagnostics(report.Diagnostics, "raw category_kb names OR'd")
	syntheticNames := filterDiagnostics(report.Diagnostics, "unknown synthetic name")

	if len(unusedSignals) > 0 {
		_, _ = fmt.Fprintf(w, "## Signal Disconnection (%d)\n\n", len(unusedSignals))
		for _, d := range unusedSignals {
			_, _ = fmt.Fprintf(w, "  %s\n", d.Message)
			if d.Fix != nil {
				_, _ = fmt.Fprintf(w, "    Fix: %s\n", d.Fix.Description)
			}
		}
		_, _ = fmt.Fprintln(w)
	}

	if len(matchGate) > 0 {
		_, _ = fmt.Fprintf(w, "## Match-Gate Disconnection (%d)\n\n", len(matchGate))
		for _, d := range matchGate {
			_, _ = fmt.Fprintf(w, "  %s\n", d.Message)
			if d.Fix != nil {
				_, _ = fmt.Fprintf(w, "    Fix: %s\n", d.Fix.Description)
			}
		}
		_, _ = fmt.Fprintln(w)
	}

	if len(composition) > 0 {
		_, _ = fmt.Fprintf(w, "## Composition Pathology (%d)\n\n", len(composition))
		for _, d := range composition {
			_, _ = fmt.Fprintf(w, "  %s\n", d.Message)
			if d.Fix != nil {
				_, _ = fmt.Fprintf(w, "    Fix: %s\n", d.Fix.Description)
			}
		}
		_, _ = fmt.Fprintln(w)
	}

	if len(binaryCatchall) > 0 {
		_, _ = fmt.Fprintf(w, "## Binary vs Best-Match (%d)\n\n", len(binaryCatchall))
		for _, d := range binaryCatchall {
			_, _ = fmt.Fprintf(w, "  %s\n", d.Message)
			if d.Fix != nil {
				_, _ = fmt.Fprintf(w, "    Fix: %s\n", d.Fix.Description)
			}
		}
		_, _ = fmt.Fprintln(w)
	}

	if len(syntheticNames) > 0 {
		_, _ = fmt.Fprintf(w, "## Unknown Synthetic Names (%d)\n\n", len(syntheticNames))
		for _, d := range syntheticNames {
			_, _ = fmt.Fprintf(w, "  %s\n", d.Message)
		}
		_, _ = fmt.Fprintln(w)
	}

	// Remaining diagnostics
	categorized := len(unusedSignals) + len(matchGate) + len(composition) + len(binaryCatchall) + len(syntheticNames)
	remaining := len(report.Diagnostics) - categorized
	if remaining > 0 {
		_, _ = fmt.Fprintf(w, "## Other Issues (%d)\n\n", remaining)
		for _, d := range report.Diagnostics {
			msg := d.Message
			if containsAny(msg, "never referenced", "matched-rules list", "dominat", "raw category_kb names OR'd", "unknown synthetic name") {
				continue
			}
			_, _ = fmt.Fprintf(w, "  [%s] %s\n", d.Level, d.Message)
			if d.Fix != nil {
				_, _ = fmt.Fprintf(w, "    Fix: %s\n", d.Fix.Description)
			}
		}
		_, _ = fmt.Fprintln(w)
	}

	_, _ = fmt.Fprintf(w, "\nSummary: %d error(s), %d warning(s), %d constraint(s)\n",
		report.Summary.Errors, report.Summary.Warnings, report.Summary.Constraints)
}

func filterDiagnostics(diags []DoctorDiagnostic, substr string) []DoctorDiagnostic {
	var filtered []DoctorDiagnostic
	for _, d := range diags {
		if strings.Contains(d.Message, substr) {
			filtered = append(filtered, d)
		}
	}
	return filtered
}

func containsAny(s string, substrs ...string) bool {
	for _, sub := range substrs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}
