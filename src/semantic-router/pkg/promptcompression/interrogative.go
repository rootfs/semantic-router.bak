package promptcompression

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// InterrogativeScores computes a per-sentence interrogative score in [0, 1]
// driven by the patterns in rules. The score reflects how likely the sentence
// is to be a user's actual question or information request.
//
// Scoring tiers (highest match wins — not additive):
//
//	1.0  question mark at end of sentence
//	1.0  wh-word at start + question mark
//	0.8  sentence-final question particle (CJK)
//	0.7  wh-word at start without question mark
//	0.7  starts with a request pattern ("tell me", "do you remember")
//	0.3  contains an indirect question marker ("wondering", "curious")
//
// All tier scores are configurable via the rules file.
func InterrogativeScores(sentences []string, rules *CompressionRules) []float64 {
	if rules == nil {
		rules = DefaultRules()
	}
	n := len(sentences)
	if n == 0 {
		return nil
	}

	scores := make([]float64, n)
	for i, sent := range sentences {
		scores[i] = scoreSentenceInterrogative(sent, rules)
	}

	// Normalize to [0, 1] by dividing by max.
	maxScore := 0.0
	for _, s := range scores {
		if s > maxScore {
			maxScore = s
		}
	}
	if maxScore > 0 {
		for i := range scores {
			scores[i] /= maxScore
		}
	}

	return scores
}

// IsInterrogative returns true if the sentence scores above zero,
// i.e. it matches any interrogative pattern. Useful for PreserveInterrogative.
func IsInterrogative(sent string, rules *CompressionRules) bool {
	if rules == nil {
		rules = DefaultRules()
	}
	return scoreSentenceInterrogative(sent, rules) > 0
}

func scoreSentenceInterrogative(sent string, rules *CompressionRules) float64 {
	sent = strings.TrimSpace(sent)
	if sent == "" {
		return 0
	}

	hasQMark := endsWithQuestionMark(sent, rules)
	hasWhWord := startsWithWhWord(sent, rules)

	// Fast path: question-mark + wh-word is the maximum possible score.
	// No need to check particles, patterns, or markers.
	if hasQMark && hasWhWord {
		return max64(rules.ScoreQuestionMark, rules.ScoreWhWithQMark)
	}

	best := 0.0
	if hasQMark {
		best = max64(best, rules.ScoreQuestionMark)
	}
	if hasWhWord {
		best = max64(best, rules.ScoreWhWithoutQMark)
	}

	hasParticle := endsWithParticle(sent, rules)
	if hasParticle {
		best = max64(best, rules.ScoreParticle)
	}

	// Defer lowerSent allocation until we actually need it.
	needLower := len(rules.requestPatterns) > 0 || len(rules.indirectMarkers) > 0
	if needLower {
		lowerSent := strings.ToLower(sent)
		if rules.MatchesRequestPattern(lowerSent) {
			best = max64(best, rules.ScoreRequestPattern)
		}
		if rules.MatchesIndirectMarker(lowerSent) {
			best = max64(best, rules.ScoreIndirectMarker)
		}
	}

	return best
}

// endsWithQuestionMark checks whether the last non-whitespace rune is a
// configured question mark. Works across scripts (Latin ?, CJK ？, Arabic ؟).
func endsWithQuestionMark(sent string, rules *CompressionRules) bool {
	for i := len(sent); i > 0; {
		rn, sz := utf8.DecodeLastRuneInString(sent[:i])
		if rn == utf8.RuneError {
			return false
		}
		if !unicode.IsSpace(rn) {
			return rules.IsQuestionMark(rn)
		}
		i -= sz
	}
	return false
}

// startsWithWhWord checks if the first token of the sentence is a wh-word.
func startsWithWhWord(sent string, rules *CompressionRules) bool {
	// Extract first token: run of non-space runes.
	start := 0
	for start < len(sent) {
		rn, sz := utf8.DecodeRuneInString(sent[start:])
		if !unicode.IsSpace(rn) {
			break
		}
		start += sz
	}
	end := start
	for end < len(sent) {
		rn, sz := utf8.DecodeRuneInString(sent[end:])
		if unicode.IsSpace(rn) || unicode.IsPunct(rn) {
			break
		}
		end += sz
	}
	if end <= start {
		return false
	}
	token := strings.ToLower(sent[start:end])
	return rules.IsWhWord(token)
}

// endsWithParticle checks if the last token (before punctuation) is a
// sentence-final question particle.
//
// Zero-allocation: uses byte-index arithmetic on the original string instead
// of converting to []rune.
func endsWithParticle(sent string, rules *CompressionRules) bool {
	if len(rules.questionParticles) == 0 {
		return false
	}

	// Strip trailing punctuation and whitespace to find the content end.
	end := len(sent)
	for end > 0 {
		rn, sz := utf8.DecodeLastRuneInString(sent[:end])
		if unicode.IsSpace(rn) || unicode.IsPunct(rn) || isSentenceTerminator(rn) {
			end -= sz
		} else {
			break
		}
	}
	if end == 0 {
		return false
	}

	// Walk backwards up to 5 runes collecting byte offsets, then check
	// longest-first candidates against the particle set.
	const maxParticleRunes = 5
	var offsets [maxParticleRunes + 1]int // offsets[0]=end, offsets[k]=start of k-th rune from end
	offsets[0] = end
	nRunes := 0
	pos := end
	for nRunes < maxParticleRunes && pos > 0 {
		_, sz := utf8.DecodeLastRuneInString(sent[:pos])
		pos -= sz
		nRunes++
		offsets[nRunes] = pos
	}

	for pLen := nRunes; pLen >= 1; pLen-- {
		candidate := sent[offsets[pLen]:end]
		if rules.IsQuestionParticle(candidate) {
			return true
		}
	}
	return false
}

func max64(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
