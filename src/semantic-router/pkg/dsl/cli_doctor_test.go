package dsl

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

func TestCLIDoctor_HumanOutput(t *testing.T) {
	dslContent := `
SIGNAL keyword code_help {
	keywords: ["code", "debug"]
}

SIGNAL keyword unused_signal {
	keywords: ["unused"]
}

ROUTE code_route {
	WHEN keyword("code_help")
	MODEL "gpt-4"
}
`
	tmpFile, err := os.CreateTemp("", "test-doctor-*.dsl")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(dslContent); err != nil {
		t.Fatal(err)
	}
	tmpFile.Close()

	var buf bytes.Buffer
	errCount := CLIDoctor(tmpFile.Name(), &buf, false)

	output := buf.String()
	if !strings.Contains(output, "Signal Disconnection") {
		t.Error("expected Signal Disconnection section in human output")
	}
	if !strings.Contains(output, "unused_signal") {
		t.Error("expected mention of unused_signal")
	}
	_ = errCount
}

func TestCLIDoctor_JSONOutput(t *testing.T) {
	dslContent := `
SIGNAL keyword code_help {
	keywords: ["code"]
}

ROUTE code_route {
	WHEN keyword("code_help")
	MODEL "gpt-4"
}
`
	tmpFile, err := os.CreateTemp("", "test-doctor-*.dsl")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(dslContent); err != nil {
		t.Fatal(err)
	}
	tmpFile.Close()

	var buf bytes.Buffer
	errCount := CLIDoctor(tmpFile.Name(), &buf, true)

	if errCount != 0 {
		t.Errorf("expected 0 errors for valid DSL, got %d", errCount)
	}

	output := buf.String()
	if !strings.Contains(output, "input_path") {
		t.Error("expected JSON output with input_path field")
	}
	if !strings.Contains(output, "diagnostics") {
		t.Error("expected JSON output with diagnostics field")
	}
}

func TestCLIDoctor_CategoryKBBinaryWarning(t *testing.T) {
	dslContent := `
SIGNAL category_kb privacy_kb {
	kb_dir: "/path/to/kb"
	threshold: 0.5
}

ROUTE catchall {
	WHEN category_kb("customer_data") OR category_kb("pii") OR category_kb("trade_secret")
	MODEL "gpt-4"
}
`
	tmpFile, err := os.CreateTemp("", "test-doctor-*.dsl")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(dslContent); err != nil {
		t.Fatal(err)
	}
	tmpFile.Close()

	var buf bytes.Buffer
	CLIDoctor(tmpFile.Name(), &buf, false)

	output := buf.String()
	if !strings.Contains(output, "Binary vs Best-Match") {
		t.Error("expected Binary vs Best-Match section")
	}
}
