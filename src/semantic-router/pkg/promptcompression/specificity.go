package promptcompression

import (
	"unicode"
	"unicode/utf8"
)

// SpecificityScores computes a per-sentence specificity score in [0, 1].
// Sentences with more specific, retrievable content (proper nouns, numbers,
// dates, quoted text, long unique terms) score higher.
//
// This is language-agnostic: capitalization works for Latin-script languages,
// numeric tokens work universally, and quoted content is detected by
// standard quotation marks across scripts.
func SpecificityScores(
	sentences []string,
	sentTokens [][]string,
	tfidfScorer *TFIDFScorer,
	rules *CompressionRules,
) []float64 {
	if rules == nil {
		rules = DefaultRules()
	}
	n := len(sentences)
	if n == 0 {
		return nil
	}

	scores := make([]float64, n)
	numDocs := float64(n)

	for i, sent := range sentences {
		tokens := sentTokens[i]
		if len(tokens) == 0 {
			continue
		}

		var weighted float64
		total := float64(len(tokens))

		// Proper nouns: capitalized words not at sentence start.
		properCount := countProperNouns(sent)
		weighted += rules.SpecProperNounWeight * float64(properCount)

		// Numeric tokens.
		numericCount := 0
		for _, tok := range tokens {
			if hasDigit(tok) {
				numericCount++
			}
		}
		weighted += rules.SpecNumericWeight * float64(numericCount)

		// Quoted content.
		quotedCount := countQuotedSegments(sent)
		weighted += rules.SpecQuotedWeight * float64(quotedCount)

		// Long unique tokens (high specificity, low document frequency).
		if tfidfScorer != nil {
			longUniqueCount := 0
			for _, tok := range tokens {
				if len(tok) >= rules.LongTokenMinChars {
					df, ok := tfidfScorer.docFreq[tok]
					if ok && numDocs > 0 {
						ratio := float64(df) / numDocs
						if ratio < rules.LongTokenMaxDFRatio {
							longUniqueCount++
						}
					} else if !ok {
						// Token not seen in any other sentence — unique.
						longUniqueCount++
					}
				}
			}
			weighted += rules.SpecLongUniqueWeight * float64(longUniqueCount)
		}

		scores[i] = weighted / total
	}

	// Normalize to [0, 1].
	normalizeSlice(scores)
	return scores
}

// countProperNouns counts title-case words (first letter upper, at least one
// lower) that are NOT the first word of the sentence. This excludes ALL-CAPS
// words (likely emphasis or acronyms) and single-character words.
//
// Zero-allocation: scans runes in-place without building a []string slice.
// Word boundaries are detected by space/punct transitions; title-case is
// checked inline as each word's runes are visited.
func countProperNouns(sent string) int {
	count := 0
	wordIdx := 0
	inWord := false
	runePos := 0 // rune index within current word
	firstUpper := false
	hasLower := false

	// evaluateWord checks the just-finished word and increments wordIdx.
	evaluateWord := func() {
		if wordIdx > 0 && runePos >= 2 && firstUpper && hasLower {
			count++
		}
		wordIdx++
		inWord = false
	}

	for i := 0; i < len(sent); {
		rn, sz := utf8.DecodeRuneInString(sent[i:])
		i += sz

		if unicode.IsSpace(rn) || unicode.IsPunct(rn) {
			if inWord {
				evaluateWord()
			}
			continue
		}

		if !inWord {
			inWord = true
			runePos = 0
			firstUpper = false
			hasLower = false
		}

		if runePos == 0 {
			firstUpper = unicode.IsUpper(rn)
		} else if unicode.IsLower(rn) {
			hasLower = true
		}
		runePos++
	}

	// Flush the last word if we ended inside one.
	if inWord {
		evaluateWord()
	}

	return count
}

// countProperNounsOld is the original allocating version, kept for testing
// parity. Not called in production.
func countProperNounsOld(sent string) int {
	words := splitWords(sent)
	count := 0
	for i, w := range words {
		if i == 0 || len(w) < 2 {
			continue
		}
		if isTitleCase(w) {
			count++
		}
	}
	return count
}

func splitWords(s string) []string {
	var words []string
	start := -1
	for i := 0; i < len(s); {
		rn, sz := utf8.DecodeRuneInString(s[i:])
		if unicode.IsSpace(rn) || unicode.IsPunct(rn) {
			if start >= 0 {
				words = append(words, s[start:i])
				start = -1
			}
		} else if start < 0 {
			start = i
		}
		i += sz
	}
	if start >= 0 {
		words = append(words, s[start:])
	}
	return words
}

// isTitleCase returns true if the first rune is uppercase and at least one
// subsequent rune is lowercase. "John" → true, "THE" → false, "a" → false.
func isTitleCase(w string) bool {
	firstUpper := false
	hasLower := false
	first := true
	for _, rn := range w {
		if first {
			if !unicode.IsUpper(rn) {
				return false
			}
			firstUpper = true
			first = false
			continue
		}
		if unicode.IsLower(rn) {
			hasLower = true
			break
		}
	}
	return firstUpper && hasLower
}

// hasDigit returns true if the string contains any digit character.
func hasDigit(s string) bool {
	for _, rn := range s {
		if unicode.IsDigit(rn) {
			return true
		}
	}
	return false
}

// countQuotedSegments counts the number of quoted substrings in the sentence.
// Recognizes: "...", '...', «...», „...", 「...」, 『...』
//
// Zero-allocation: iterates with utf8.DecodeRuneInString instead of []rune.
func countQuotedSegments(sent string) int {
	count := 0
	i := 0
	for i < len(sent) {
		opener, sz := utf8.DecodeRuneInString(sent[i:])
		closer := matchingQuote(opener)
		if closer != 0 {
			j := i + sz
			for j < len(sent) {
				rn, rsz := utf8.DecodeRuneInString(sent[j:])
				if rn == closer {
					count++
					j += rsz
					i = j
					goto next
				}
				j += rsz
			}
		}
		i += sz
		continue
	next:
	}
	return count
}

func matchingQuote(r rune) rune {
	switch r {
	case '"':
		return '"'
	case '\u201C': // "
		return '\u201D' // "
	case '\u00AB': // «
		return '\u00BB' // »
	case '\u201E': // „
		return '\u201D' // "
	case '\u300C': // 「
		return '\u300D' // 」
	case '\u300E': // 『
		return '\u300F' // 』
	case '\'':
		return '\''
	case '\u2018': // '
		return '\u2019' // '
	}
	return 0
}
