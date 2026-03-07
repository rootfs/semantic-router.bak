package promptcompression

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestIsInterrogative(t *testing.T) {
	rules := DefaultRules()

	tests := []struct {
		name string
		sent string
		want bool
	}{
		{"question mark", "What is your name?", true},
		{"lowercase wh no qmark", "who is that", true},
		{"statement", "The weather is nice today.", false},
		{"empty", "", false},
		{"trailing spaces qmark", "What do you mean?  ", true},
		{"indirect wondering default rules", "I was wondering about your plans.", false},
		{"indirect curious has wh", "I am curious what happened.", false},
		{"no signal", "Hello there friend.", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsInterrogative(tt.sent, rules)
			if got != tt.want {
				t.Errorf("IsInterrogative(%q) = %v, want %v", tt.sent, got, tt.want)
			}
		})
	}
}

func TestInterrogativeScoresNormalized(t *testing.T) {
	rules := DefaultRules()
	sentences := []string{
		"Hello, how are you doing today?",
		"I went to the store yesterday.",
		"What time does the meeting start?",
		"The project deadline is next Friday.",
	}

	scores := InterrogativeScores(sentences, rules)
	if len(scores) != len(sentences) {
		t.Fatalf("expected %d scores, got %d", len(sentences), len(scores))
	}

	// Questions should score higher than non-questions.
	if scores[0] <= scores[1] {
		t.Errorf("question %q (%.3f) should score higher than statement %q (%.3f)",
			sentences[0], scores[0], sentences[1], scores[1])
	}
	if scores[2] <= scores[3] {
		t.Errorf("question %q (%.3f) should score higher than statement %q (%.3f)",
			sentences[2], scores[2], sentences[3], scores[3])
	}

	// At least one score should be 1.0 (normalized).
	maxScore := 0.0
	for _, s := range scores {
		if s > maxScore {
			maxScore = s
		}
	}
	if maxScore != 1.0 {
		t.Errorf("max score should be 1.0 after normalization, got %.3f", maxScore)
	}
}

func TestInterrogativeScoresEmpty(t *testing.T) {
	rules := DefaultRules()
	scores := InterrogativeScores(nil, rules)
	if scores != nil {
		t.Errorf("expected nil for empty input, got %v", scores)
	}
}

func TestLoadRulesAndInterrogative(t *testing.T) {
	// Find the rules file relative to this test file.
	rulesPath := filepath.Join("..", "..", "..", "..", "config", "prompt-compression", "compression-rules.yaml")
	if _, err := os.Stat(rulesPath); os.IsNotExist(err) {
		t.Skipf("rules file not found at %s, skipping", rulesPath)
	}

	rules, err := LoadRules(rulesPath)
	if err != nil {
		t.Fatalf("LoadRules: %v", err)
	}

	// CJK question mark.
	if !rules.IsQuestionMark('\uFF1F') {
		t.Error("expected CJK fullwidth question mark to be recognized")
	}

	// Chinese wh-word.
	if !rules.IsWhWord("什么") {
		t.Error("expected 什么 to be a wh-word")
	}

	// Japanese question particle.
	if !rules.IsQuestionParticle("ですか") {
		t.Error("expected ですか to be a question particle")
	}

	// Full interrogative check with loaded rules.
	sentences := []string{
		"What is your name?",
		"今天天气怎么样？",
		"Good morning.",
	}
	scores := InterrogativeScores(sentences, rules)
	if scores[0] <= scores[2] {
		t.Errorf("English question should outscore English statement: %.3f vs %.3f", scores[0], scores[2])
	}
	if scores[1] <= scores[2] {
		t.Errorf("Chinese question should outscore English statement: %.3f vs %.3f", scores[1], scores[2])
	}
}

func TestEndsWithParticle(t *testing.T) {
	rules := DefaultRules()
	rules.questionParticles = map[string]bool{
		"か":  true,
		"ですか": true,
		"吗":  true,
	}

	tests := []struct {
		name string
		sent string
		want bool
	}{
		{"Japanese ka", "これは何か", true},
		{"Japanese desuka", "これは何ですか", true},
		{"Chinese ma", "你好吗", true},
		{"No particle", "Hello world", false},
		{"Japanese with period after", "これは何か。", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := endsWithParticle(tt.sent, rules)
			if got != tt.want {
				t.Errorf("endsWithParticle(%q) = %v, want %v", tt.sent, got, tt.want)
			}
		})
	}
}

// ===========================================================================
// Edge-case tests for optimized interrogative functions
// ===========================================================================

func TestEndsWithParticleEdgeCases(t *testing.T) {
	rules := DefaultRules()
	rules.questionParticles = map[string]bool{
		"か":    true,
		"ですか":  true,
		"でしょうか": true,
		"吗":    true,
	}

	tests := []struct {
		name string
		sent string
		want bool
	}{
		{"empty", "", false},
		{"only spaces", "   ", false},
		{"only punctuation", "...!!??", false},
		{"particle at very end", "テストか", true},
		{"long particle", "テストでしょうか", true},
		{"particle with trailing space", "テストか ", true},
		{"particle with trailing newline", "テストか\n", true},
		{"particle with multiple trailing punct", "テストか。。。", true},
		{"partial match not particle", "テストで", false},
		{"very short string", "か", true},
		{"single char non-particle", "a", false},
		{"emoji after particle", "テストか😀", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := endsWithParticle(tt.sent, rules)
			if got != tt.want {
				t.Errorf("endsWithParticle(%q) = %v, want %v", tt.sent, got, tt.want)
			}
		})
	}
}

func TestEndsWithParticleEmptyRules(t *testing.T) {
	rules := DefaultRules()
	rules.questionParticles = nil
	got := endsWithParticle("テストか", rules)
	if got != false {
		t.Error("expected false when questionParticles is nil")
	}

	rules.questionParticles = map[string]bool{}
	got = endsWithParticle("テストか", rules)
	if got != false {
		t.Error("expected false when questionParticles is empty")
	}
}

func TestScoreSentenceInterrogativeFastPath(t *testing.T) {
	rules := DefaultRules()

	// The fast path fires when both hasQMark and hasWhWord are true.
	fastPathSents := []string{
		"What is this?",
		"Who are you?",
		"Where did you go?",
		"  How does it work?  ",
	}
	for _, sent := range fastPathSents {
		score := scoreSentenceInterrogative(sent, rules)
		expected := max64(rules.ScoreQuestionMark, rules.ScoreWhWithQMark)
		if score != expected {
			t.Errorf("fast path: scoreSentenceInterrogative(%q) = %.3f, want %.3f", sent, score, expected)
		}
	}

	// Non-fast-path cases should still work.
	nonFast := []struct {
		sent string
		min  float64
	}{
		{"who is that", rules.ScoreWhWithoutQMark},
		{"Is this a test?", rules.ScoreQuestionMark},
		{"The weather is nice.", 0},
	}
	for _, tt := range nonFast {
		score := scoreSentenceInterrogative(tt.sent, rules)
		if score < tt.min {
			t.Errorf("scoreSentenceInterrogative(%q) = %.3f, want >= %.3f", tt.sent, score, tt.min)
		}
	}
}

func TestScoreSentenceInterrogativeLazyLower(t *testing.T) {
	// With DefaultRules (no requestPatterns/indirectMarkers),
	// strings.ToLower should NOT be called. Verify the result is correct.
	rules := DefaultRules()
	if len(rules.requestPatterns) != 0 || len(rules.indirectMarkers) != 0 {
		t.Skip("DefaultRules has patterns/markers — test assumptions invalid")
	}

	// A sentence with indirect markers should NOT match with DefaultRules.
	score := scoreSentenceInterrogative("I was wondering about things.", rules)
	if score != 0 {
		t.Errorf("expected 0 for indirect marker with DefaultRules (no patterns loaded), got %.3f", score)
	}

	// With loaded rules that have patterns, it should match.
	rulesPath := filepath.Join("..", "..", "..", "..", "config", "prompt-compression", "compression-rules.yaml")
	if _, err := os.Stat(rulesPath); os.IsNotExist(err) {
		t.Skipf("rules file not found at %s", rulesPath)
	}
	fullRules, err := LoadRules(rulesPath)
	if err != nil {
		t.Fatalf("LoadRules: %v", err)
	}
	score = scoreSentenceInterrogative("tell me about the weather", fullRules)
	if score <= 0 {
		t.Errorf("expected > 0 for request pattern with full rules, got %.3f", score)
	}
}

// ===========================================================================
// Concurrency safety for interrogative scoring
// ===========================================================================

func TestInterrogativeScoresConcurrency(t *testing.T) {
	rules := DefaultRules()
	rules.questionParticles = map[string]bool{
		"か":  true,
		"ですか": true,
		"吗":  true,
	}

	sentences := []string{
		"What is your name?",
		"The weather is nice.",
		"Who are you?",
		"これは何ですか。",
		"I went to the store.",
		"你好吗",
		"where is that",
		"",
	}

	expected := make([]float64, len(sentences))
	for i, sent := range sentences {
		expected[i] = scoreSentenceInterrogative(sent, rules)
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
					got := scoreSentenceInterrogative(sent, rules)
					if got != expected[i] {
						t.Errorf("concurrent scoreSentenceInterrogative(%q) = %.3f, want %.3f",
							sent, got, expected[i])
						return
					}
				}
			}
		}()
	}
	wg.Wait()
}

// ===========================================================================
// Allocation benchmarks for interrogative scoring
// ===========================================================================

func BenchmarkScoreSentenceInterrogative(b *testing.B) {
	rules := DefaultRules()
	sent := "What is the meaning of life?"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scoreSentenceInterrogative(sent, rules)
	}
}

func BenchmarkScoreSentenceInterrogativeNoMatch(b *testing.B) {
	rules := DefaultRules()
	sent := "The weather is sunny and warm today."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scoreSentenceInterrogative(sent, rules)
	}
}

func BenchmarkEndsWithParticle(b *testing.B) {
	rules := DefaultRules()
	rules.questionParticles = map[string]bool{
		"か":    true,
		"ですか":  true,
		"でしょうか": true,
		"吗":    true,
		"嗎":    true,
	}
	sent := "これは何でしょうか。"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		endsWithParticle(sent, rules)
	}
}

func BenchmarkInterrogativeScores(b *testing.B) {
	rules := DefaultRules()
	sentences := make([]string, 40)
	for i := range sentences {
		if i%3 == 0 {
			sentences[i] = "What is the meaning of this sentence?"
		} else {
			sentences[i] = "The quick brown fox jumped over the lazy dog."
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		InterrogativeScores(sentences, rules)
	}
}
