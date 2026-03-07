package promptcompression

import (
	"fmt"
	"os"
	"strings"
	"unicode/utf8"

	"gopkg.in/yaml.v3"
)

// CompressionRules holds the parsed, indexed pattern sets loaded from the
// external compression-rules YAML file. All pattern matching at scoring
// time uses the pre-built lookup structures (maps / slices) rather than
// re-parsing the YAML on every call.
type CompressionRules struct {
	// questionMarks is a set of runes that mark interrogative sentences.
	questionMarks map[rune]bool

	// whWords is a flattened, lowercased set of interrogative words
	// across all configured languages.
	whWords map[string]bool

	// questionParticles is a flattened set of sentence-final particles.
	questionParticles map[string]bool

	// requestPatterns is a list of lowercased prefix strings.
	// Matched against the lowercased start of each sentence.
	requestPatterns []string

	// indirectMarkers is a list of lowercased substrings.
	// Matched anywhere within the lowercased sentence.
	indirectMarkers []string

	// Interrogative score tiers.
	ScoreQuestionMark   float64
	ScoreWhWithQMark    float64
	ScoreWhWithoutQMark float64
	ScoreParticle       float64
	ScoreRequestPattern float64
	ScoreIndirectMarker float64

	// Specificity weights.
	SpecProperNounWeight float64
	SpecNumericWeight    float64
	SpecQuotedWeight     float64
	SpecLongUniqueWeight float64
	LongTokenMinChars    int
	LongTokenMaxDFRatio  float64
}

// rulesYAML mirrors the YAML structure for unmarshalling.
type rulesYAML struct {
	Interrogative struct {
		QuestionMarks     []string              `yaml:"question_marks"`
		WhWords           map[string][]string    `yaml:"wh_words"`
		QuestionParticles map[string][]string    `yaml:"question_particles"`
		RequestPatterns   map[string][]string    `yaml:"request_patterns"`
		IndirectMarkers   map[string][]string    `yaml:"indirect_markers"`
		Scores            struct {
			QuestionMark   float64 `yaml:"question_mark"`
			WhWithQMark    float64 `yaml:"wh_with_qmark"`
			WhWithoutQMark float64 `yaml:"wh_without_qmark"`
			Particle       float64 `yaml:"particle"`
			RequestPattern float64 `yaml:"request_pattern"`
			IndirectMarker float64 `yaml:"indirect_marker"`
		} `yaml:"scores"`
	} `yaml:"interrogative"`

	Specificity struct {
		Weights struct {
			ProperNoun float64 `yaml:"proper_noun"`
			Numeric    float64 `yaml:"numeric"`
			Quoted     float64 `yaml:"quoted"`
			LongUnique float64 `yaml:"long_unique"`
		} `yaml:"weights"`
		LongTokenMinChars   int     `yaml:"long_token_min_chars"`
		LongTokenMaxDFRatio float64 `yaml:"long_token_max_df_ratio"`
	} `yaml:"specificity"`
}

// LoadRules reads and parses a compression-rules YAML file, building
// optimized lookup structures for scoring. Returns an error if the file
// cannot be read or parsed.
func LoadRules(path string) (*CompressionRules, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("promptcompression: read rules file %q: %w", path, err)
	}

	var raw rulesYAML
	if err := yaml.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("promptcompression: parse rules file %q: %w", path, err)
	}

	return buildRules(&raw), nil
}

// DefaultRules returns a minimal built-in ruleset that provides basic
// interrogative detection (question marks only) without requiring an
// external file. Use LoadRules for full multilingual support.
func DefaultRules() *CompressionRules {
	return &CompressionRules{
		questionMarks: map[rune]bool{
			'?':      true,
			'\uFF1F': true, // ？
			'\u061F': true, // ؟
		},
		whWords: map[string]bool{
			"who": true, "what": true, "where": true, "when": true,
			"why": true, "how": true, "which": true, "whose": true, "whom": true,
		},
		questionParticles: map[string]bool{},
		requestPatterns:   nil,
		indirectMarkers:   nil,

		ScoreQuestionMark:   1.0,
		ScoreWhWithQMark:    1.0,
		ScoreWhWithoutQMark: 0.7,
		ScoreParticle:       0.8,
		ScoreRequestPattern: 0.7,
		ScoreIndirectMarker: 0.3,

		SpecProperNounWeight: 1.0,
		SpecNumericWeight:    0.8,
		SpecQuotedWeight:     0.6,
		SpecLongUniqueWeight: 0.4,
		LongTokenMinChars:    8,
		LongTokenMaxDFRatio:  0.3,
	}
}

func buildRules(raw *rulesYAML) *CompressionRules {
	r := &CompressionRules{
		questionMarks:     make(map[rune]bool, len(raw.Interrogative.QuestionMarks)),
		whWords:           make(map[string]bool, 64),
		questionParticles: make(map[string]bool, 32),
	}

	for _, s := range raw.Interrogative.QuestionMarks {
		rn, _ := utf8.DecodeRuneInString(s)
		if rn != utf8.RuneError {
			r.questionMarks[rn] = true
		}
	}

	for _, words := range raw.Interrogative.WhWords {
		for _, w := range words {
			r.whWords[strings.ToLower(w)] = true
		}
	}

	for _, particles := range raw.Interrogative.QuestionParticles {
		for _, p := range particles {
			r.questionParticles[p] = true
		}
	}

	for _, patterns := range raw.Interrogative.RequestPatterns {
		for _, p := range patterns {
			r.requestPatterns = append(r.requestPatterns, strings.ToLower(p))
		}
	}

	for _, markers := range raw.Interrogative.IndirectMarkers {
		for _, m := range markers {
			r.indirectMarkers = append(r.indirectMarkers, strings.ToLower(m))
		}
	}

	// Scores — use parsed values, falling back to sensible defaults.
	r.ScoreQuestionMark = withDefault(raw.Interrogative.Scores.QuestionMark, 1.0)
	r.ScoreWhWithQMark = withDefault(raw.Interrogative.Scores.WhWithQMark, 1.0)
	r.ScoreWhWithoutQMark = withDefault(raw.Interrogative.Scores.WhWithoutQMark, 0.7)
	r.ScoreParticle = withDefault(raw.Interrogative.Scores.Particle, 0.8)
	r.ScoreRequestPattern = withDefault(raw.Interrogative.Scores.RequestPattern, 0.7)
	r.ScoreIndirectMarker = withDefault(raw.Interrogative.Scores.IndirectMarker, 0.3)

	// Specificity.
	r.SpecProperNounWeight = withDefault(raw.Specificity.Weights.ProperNoun, 1.0)
	r.SpecNumericWeight = withDefault(raw.Specificity.Weights.Numeric, 0.8)
	r.SpecQuotedWeight = withDefault(raw.Specificity.Weights.Quoted, 0.6)
	r.SpecLongUniqueWeight = withDefault(raw.Specificity.Weights.LongUnique, 0.4)
	r.LongTokenMinChars = raw.Specificity.LongTokenMinChars
	if r.LongTokenMinChars <= 0 {
		r.LongTokenMinChars = 8
	}
	r.LongTokenMaxDFRatio = raw.Specificity.LongTokenMaxDFRatio
	if r.LongTokenMaxDFRatio <= 0 {
		r.LongTokenMaxDFRatio = 0.3
	}

	return r
}

func withDefault(v, def float64) float64 {
	if v > 0 {
		return v
	}
	return def
}

// IsQuestionMark returns true if the rune is a configured question mark.
func (r *CompressionRules) IsQuestionMark(rn rune) bool {
	return r.questionMarks[rn]
}

// IsWhWord returns true if the lowercased token is a configured wh-word.
func (r *CompressionRules) IsWhWord(token string) bool {
	return r.whWords[strings.ToLower(token)]
}

// IsQuestionParticle returns true if the token is a configured question particle.
func (r *CompressionRules) IsQuestionParticle(token string) bool {
	return r.questionParticles[token]
}

// MatchesRequestPattern returns true if the lowercased sentence starts with
// any configured request pattern.
func (r *CompressionRules) MatchesRequestPattern(lowerSentence string) bool {
	for _, p := range r.requestPatterns {
		if strings.HasPrefix(lowerSentence, p) {
			return true
		}
	}
	return false
}

// MatchesIndirectMarker returns true if the lowercased sentence contains
// any configured indirect question marker.
func (r *CompressionRules) MatchesIndirectMarker(lowerSentence string) bool {
	for _, m := range r.indirectMarkers {
		if strings.Contains(lowerSentence, m) {
			return true
		}
	}
	return false
}
