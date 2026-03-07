package promptcompression

import (
	"math/rand"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"
)

func TestSpecificityScoresBasic(t *testing.T) {
	rules := DefaultRules()

	sentences := []string{
		"John Smith visited Tokyo on January 15, 2024.",
		"the thing is really nice and good and great.",
		"Please review document DR-20240115-A from the compliance department.",
	}
	sentTokens := make([][]string, len(sentences))
	for i, s := range sentences {
		sentTokens[i] = TokenizeWords(s)
	}
	tfidfScorer := NewTFIDFScorer(sentTokens)

	scores := SpecificityScores(sentences, sentTokens, tfidfScorer, rules)

	if len(scores) != len(sentences) {
		t.Fatalf("expected %d scores, got %d", len(sentences), len(scores))
	}

	// Sentence with proper nouns + date should score highest.
	if scores[0] <= scores[1] {
		t.Errorf("specific sentence (%.3f) should outscore generic sentence (%.3f)",
			scores[0], scores[1])
	}

	// Sentence with document ID should also score well.
	if scores[2] <= scores[1] {
		t.Errorf("sentence with document ID (%.3f) should outscore generic (%.3f)",
			scores[2], scores[1])
	}
}

func TestSpecificityScoresEmpty(t *testing.T) {
	rules := DefaultRules()
	scores := SpecificityScores(nil, nil, nil, rules)
	if scores != nil {
		t.Errorf("expected nil for empty input, got %v", scores)
	}
}

func TestCountProperNouns(t *testing.T) {
	tests := []struct {
		name string
		sent string
		want int
	}{
		{"no proper nouns", "the quick brown fox", 0},
		{"two proper nouns", "the quick Brown Fox", 2},
		{"multiple proper", "John Smith met Mary Johnson in Paris", 4},
		{"first word not counted", "Hello world", 0},
		{"all caps not proper", "go to THE store", 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := countProperNouns(tt.sent)
			if got != tt.want {
				t.Errorf("countProperNouns(%q) = %d, want %d", tt.sent, got, tt.want)
			}
		})
	}
}

func TestHasDigit(t *testing.T) {
	tests := []struct {
		input string
		want  bool
	}{
		{"hello", false},
		{"hello123", true},
		{"2024", true},
		{"", false},
		{"日本語", false},
	}
	for _, tt := range tests {
		got := hasDigit(tt.input)
		if got != tt.want {
			t.Errorf("hasDigit(%q) = %v, want %v", tt.input, got, tt.want)
		}
	}
}

func TestCountQuotedSegments(t *testing.T) {
	tests := []struct {
		name string
		sent string
		want int
	}{
		{"double quotes", `he said "hello" and "goodbye"`, 2},
		{"single quotes", "it's called 'the thing'", 1},
		{"CJK brackets", "「こんにちは」と言った", 1},
		{"no quotes", "hello world", 0},
		{"smart quotes", "\u201CHello\u201D said John", 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := countQuotedSegments(tt.sent)
			if got != tt.want {
				t.Errorf("countQuotedSegments(%q) = %d, want %d", tt.sent, got, tt.want)
			}
		})
	}
}

// ===========================================================================
// Parity tests: optimized functions vs original allocating versions
// ===========================================================================

func TestCountProperNounsParity(t *testing.T) {
	cases := []string{
		"the quick brown fox",
		"the quick Brown Fox",
		"John Smith met Mary Johnson in Paris",
		"Hello world",
		"go to THE store",
		"",
		"A",
		"a",
		"AB",
		"Ab",
		"HELLO WORLD FOO BAR",
		"John",
		"john Smith",
		"  John   Smith  ",
		"John-Smith visited Tokyo",
		"Dr. Martin Luther King Jr.",
		"I went to McDonald's",
		"  Leading spaces John Smith trailing  ",
		"CamelCase and PascalCase words Test",
		"αΒγΔ εΖηΘ",
		"Ünïcödé Tëst",
		"日本語のテスト",
		"Mixed English and 日本語 text",
		"São Paulo and Zürich are cities",
		"ALL-CAPS-WORD and Title-Case word",
		"X Y Z",
		"OneWord",
		"twoWords Here",
	}

	for _, sent := range cases {
		got := countProperNouns(sent)
		want := countProperNounsOld(sent)
		if got != want {
			t.Errorf("countProperNouns(%q) = %d, old = %d", sent, got, want)
		}
	}
}

func TestCountProperNounsEdgeCases(t *testing.T) {
	tests := []struct {
		name string
		sent string
		want int
	}{
		{"empty", "", 0},
		{"single char", "A", 0},
		{"single lowercase", "a", 0},
		{"only spaces", "   ", 0},
		{"only punctuation", "...!!??", 0},
		{"tab separated", "Hello\tWorld\tFoo", 2},
		{"newline separated", "Hello\nWorld\nFoo", 2},
		{"consecutive spaces", "Hello   World   Foo", 2},
		{"hyphenated proper", "Jean-Pierre visited", 1},
		{"apostrophe in name", "O'Brien came home", 1},
		{"unicode uppercase", "Über cool Größe test", 1},
		{"single upper+lower", "I met Jo", 1},
		{"first word proper but skipped", "Paris is beautiful", 0},
		{"all single char", "A B C D", 0},
		{"mixed case ALL+title", "I love THE New York", 2},
		{"trailing punct", "John Smith.", 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := countProperNouns(tt.sent)
			if got != tt.want {
				t.Errorf("countProperNouns(%q) = %d, want %d", tt.sent, got, tt.want)
			}
			// Also check parity with old version.
			old := countProperNounsOld(tt.sent)
			if got != old {
				t.Errorf("countProperNouns(%q) = %d, old = %d (parity mismatch)", tt.sent, got, old)
			}
		})
	}
}

// Fuzz-like randomized parity test for countProperNouns.
func TestCountProperNounsFuzz(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	charset := []rune("abcdefABCDEF 日本αΒ.,!?'\"-")

	for iter := 0; iter < 500; iter++ {
		length := rng.Intn(100) + 1
		runes := make([]rune, length)
		for i := range runes {
			runes[i] = charset[rng.Intn(len(charset))]
		}
		sent := string(runes)

		got := countProperNouns(sent)
		want := countProperNounsOld(sent)
		if got != want {
			t.Fatalf("fuzz parity fail: countProperNouns(%q) = %d, old = %d", sent, got, want)
		}
	}
}

func TestCountQuotedSegmentsEdgeCases(t *testing.T) {
	tests := []struct {
		name string
		sent string
		want int
	}{
		{"empty", "", 0},
		{"unmatched open", `"hello`, 0},
		{"unmatched close", `hello"`, 0},
		{"adjacent quotes", `"a""b"`, 2},
		{"empty quoted", `""`, 1},
		{"nested same type greedy", `"outer "inner" end"`, 2},
		{"mixed quote types", `"hello" and 「world」`, 2},
		{"guillemets", "«bonjour» said Pierre", 1},
		{"german quotes", "\u201Ehallo\u201D sagte Hans", 1},
		{"CJK double bracket", "『日本語』テスト", 1},
		{"only opener no closer", "\u300C", 0},
		{"multibyte throughout", "「一」「二」「三」", 3},
		{"single char between", `"x"`, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := countQuotedSegments(tt.sent)
			if got != tt.want {
				t.Errorf("countQuotedSegments(%q) = %d, want %d", tt.sent, got, tt.want)
			}
		})
	}
}

// ===========================================================================
// Concurrency safety: the optimized functions are pure (no shared mutable
// state), but verify they produce consistent results under concurrent access.
// ===========================================================================

func TestCountProperNounsConcurrency(t *testing.T) {
	sentences := []string{
		"John Smith met Mary Johnson in Paris last December.",
		"THE quick Brown Fox jumped over THE lazy Dog.",
		"São Paulo and Zürich are beautiful cities to visit.",
		"Dr. Martin Luther King Jr. gave a speech in Washington.",
		"I went to McDonald's in New York City yesterday.",
	}

	expected := make([]int, len(sentences))
	for i, sent := range sentences {
		expected[i] = countProperNouns(sent)
	}

	var wg sync.WaitGroup
	const goroutines = 50
	const iterations = 100

	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for iter := 0; iter < iterations; iter++ {
				for i, sent := range sentences {
					got := countProperNouns(sent)
					if got != expected[i] {
						t.Errorf("concurrent countProperNouns(%q) = %d, want %d", sent, got, expected[i])
						return
					}
				}
			}
		}()
	}
	wg.Wait()
}

func TestCountQuotedSegmentsConcurrency(t *testing.T) {
	sentences := []string{
		`he said "hello" and "goodbye" to everyone`,
		"「こんにちは」と「さようなら」",
		"«bonjour» et «au revoir»",
		`no quotes here at all`,
		"\u201CHello\u201D \u201CWorld\u201D",
	}

	expected := make([]int, len(sentences))
	for i, sent := range sentences {
		expected[i] = countQuotedSegments(sent)
	}

	var wg sync.WaitGroup
	const goroutines = 50
	const iterations = 100

	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for iter := 0; iter < iterations; iter++ {
				for i, sent := range sentences {
					got := countQuotedSegments(sent)
					if got != expected[i] {
						t.Errorf("concurrent countQuotedSegments(%q) = %d, want %d", sent, got, expected[i])
						return
					}
				}
			}
		}()
	}
	wg.Wait()
}

// ===========================================================================
// Allocation benchmarks for the optimized functions
// ===========================================================================

func BenchmarkCountProperNouns(b *testing.B) {
	sent := "Dr. John Smith and Mary Johnson visited the New York City Museum of Art with Professor Albert Einstein last December."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		countProperNouns(sent)
	}
}

func BenchmarkCountProperNounsOld(b *testing.B) {
	sent := "Dr. John Smith and Mary Johnson visited the New York City Museum of Art with Professor Albert Einstein last December."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		countProperNounsOld(sent)
	}
}

func BenchmarkCountQuotedSegments(b *testing.B) {
	sent := `he said "hello" and she replied "goodbye" and they both said "the end" and "farewell"`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		countQuotedSegments(sent)
	}
}

func BenchmarkCountProperNounsLongSentence(b *testing.B) {
	var sb strings.Builder
	for i := 0; i < 50; i++ {
		sb.WriteString("John Smith ")
	}
	sent := sb.String()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		countProperNouns(sent)
	}
}

func BenchmarkCountProperNounsOldLongSentence(b *testing.B) {
	var sb strings.Builder
	for i := 0; i < 50; i++ {
		sb.WriteString("John Smith ")
	}
	sent := sb.String()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		countProperNounsOld(sent)
	}
}

// ===========================================================================
// Verify countQuotedSegments handles invalid UTF-8 gracefully
// ===========================================================================

func TestCountQuotedSegmentsInvalidUTF8(t *testing.T) {
	invalid := "hello\xff\xfe\"world\"\x80"
	got := countQuotedSegments(invalid)
	if !utf8.ValidString(invalid) {
		t.Logf("input has invalid UTF-8 (expected), countQuotedSegments=%d", got)
	}
}

func TestCountProperNounsInvalidUTF8(t *testing.T) {
	invalid := "Hello\xff World\xfe Test"
	got := countProperNouns(invalid)
	old := countProperNounsOld(invalid)
	if got != old {
		t.Errorf("invalid UTF-8 parity: new=%d, old=%d", got, old)
	}
}
