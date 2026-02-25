package memory

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// =============================================================================
// Memory Extractor
// =============================================================================

// MemoryExtractor extracts facts from conversation history using an LLM.
// It analyzes conversation messages and identifies important information
// to store in long-term memory (facts, preferences, procedural knowledge).
//
// The LLM endpoint is configured via external_models with model_role="memory_extraction".
//
// It supports two modes:
//  1. Extraction only: Use ExtractFacts() to extract facts without storing them
//  2. Extraction + Storage with deduplication: Use ProcessResponse() to extract and store with deduplication
//
// Usage:
//
//	// Extraction + Storage with deduplication:
//	extractorWithStore := NewMemoryExtractorWithStore(routerCfg, batchSize, store)
//	err := extractorWithStore.ProcessResponse(ctx, sessionID, userID, history)
type MemoryExtractor struct {
	endpoint    string       // Resolved LLM endpoint
	model       string       // Resolved model name
	accessKey   string       // Optional: Bearer token for LLM endpoint auth
	client      *http.Client // Reused for connection pooling
	store       Store        // Optional: for ProcessResponse with deduplication
	turnCounts  map[string]int
	mu          sync.Mutex
	dedupConfig DeduplicationConfig
	// LLM generation parameters from external_models config
	maxTokens   int
	temperature float64
	batchSize   int
}

// NewMemoryExtractorWithStore creates a new MemoryExtractor with router config for external model resolution and store.
// This enables ProcessResponse which handles extraction + storage with deduplication.
// The LLM endpoint is resolved from external_models using model_role="memory_extraction".
// batchSize controls how often extraction runs (every N turns). Use 0 for default (10).
func NewMemoryExtractorWithStore(routerCfg *config.RouterConfig, batchSize int, store Store) *MemoryExtractor {
	// Resolve LLM endpoint and params from external_models
	resolved := resolveExtractionConfig(routerCfg)
	if resolved == nil {
		return nil
	}

	if batchSize <= 0 {
		batchSize = 10 // default
	}

	return &MemoryExtractor{
		endpoint:    resolved.endpoint,
		model:       resolved.model,
		accessKey:   resolved.accessKey,
		client:      &http.Client{Timeout: resolved.timeout},
		store:       store,
		turnCounts:  make(map[string]int),
		dedupConfig: DefaultDeduplicationConfig(),
		maxTokens:   resolved.maxTokens,
		temperature: resolved.temperature,
		batchSize:   batchSize,
	}
}

// resolvedExtractionConfig holds resolved extraction LLM configuration
type resolvedExtractionConfig struct {
	endpoint    string
	model       string
	accessKey   string
	timeout     time.Duration
	maxTokens   int
	temperature float64
}

// resolveExtractionConfig resolves the LLM endpoint and params from external_models.
func resolveExtractionConfig(routerCfg *config.RouterConfig) *resolvedExtractionConfig {
	if routerCfg == nil {
		return nil
	}

	externalCfg := routerCfg.FindExternalModelByRole(config.ModelRoleMemoryExtraction)
	if externalCfg == nil || externalCfg.ModelEndpoint.Address == "" {
		return nil
	}

	timeout := 30 * time.Second
	if externalCfg.TimeoutSeconds > 0 {
		timeout = time.Duration(externalCfg.TimeoutSeconds) * time.Second
	}

	maxTokens := externalCfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 2048 // default — 500 was too small for long conversations
	}

	temperature := externalCfg.Temperature
	if temperature <= 0 {
		temperature = 0.1 // default
	}

	return &resolvedExtractionConfig{
		endpoint:    fmt.Sprintf("http://%s:%d", externalCfg.ModelEndpoint.Address, externalCfg.ModelEndpoint.Port),
		model:       externalCfg.ModelName,
		accessKey:   externalCfg.AccessKey,
		timeout:     timeout,
		maxTokens:   maxTokens,
		temperature: temperature,
	}
}

// SetDeduplicationConfig sets the deduplication configuration.
func (e *MemoryExtractor) SetDeduplicationConfig(config DeduplicationConfig) {
	e.dedupConfig = config
}

// =============================================================================
// LLM-Based Fact Extraction
// =============================================================================

// extractionSystemPrompt is the system prompt for fact extraction
const extractionSystemPrompt = `You are a memory extraction system. Extract important USER information from conversations.

CRITICAL RULES:
1. Extract ONLY facts stated by or about the USER
2. DO NOT extract assistant suggestions, recommendations, or general knowledge
3. ALWAYS include context - never extract isolated values
4. Use self-contained phrases that make sense without the conversation
5. Return ONLY a valid JSON array - no explanations, no markdown, no thinking
6. ALWAYS phrase facts as STATEMENTS, never as questions
7. Include CONSTRAINTS and LIMITATIONS explicitly (cannot, must not, excluded, etc.)
8. The "type" field MUST be exactly one of: "semantic", "procedural", or "episodic" — no other values

MEMORY TYPES (use ONLY these three values for "type"):

"semantic" — facts, preferences, identity, constraints, knowledge about the user:
  Examples: name, job, allergies, preferences, tech stack, budget, limitations

"procedural" — user's personal workflows or routines they explicitly describe:
  Examples: morning routine, deployment process, debugging approach

"episodic" — specific events or experiences the user describes:
  Examples: "User visited Paris in June 2025", "User attended AWS re:Invent 2024"

OUTPUT FORMAT — return a JSON array of objects with exactly two fields:
  [{"type": "semantic", "content": "..."}, ...]

Return [] if nothing worth remembering about the USER.`

// ExtractFacts extracts memorable facts from a conversation using an LLM.
// This is a pure extraction function - it does NOT store the facts.
// Use ProcessResponse if you want extraction + storage with deduplication.
//
// Error handling:
//   - Returns empty slice on any error (graceful degradation)
//   - Logs warnings for debugging but doesn't fail the response
//
// Example:
//
//	messages := []Message{
//	    {Role: "user", Content: "My budget for Hawaii is $10,000"},
//	    {Role: "assistant", Content: "Great! That's a good budget for Hawaii."},
//	}
//	facts, err := extractor.ExtractFacts(ctx, messages)
//	// facts = [{Type: "semantic", Content: "User's budget for Hawaii vacation is $10,000"}]
func (e *MemoryExtractor) ExtractFacts(ctx context.Context, messages []Message) ([]ExtractedFact, error) {
	if e == nil || e.endpoint == "" {
		logging.Debugf("Memory: Fact extraction not configured")
		return nil, nil
	}

	if len(messages) == 0 {
		return nil, nil
	}

	// Format messages for the prompt
	conversationText := formatMessagesForExtraction(messages)

	// Build user prompt
	userPrompt := fmt.Sprintf("Extract important information from this conversation:\n\n%s\n\nReturn JSON array:", conversationText)

	// TODO: Remove debug logs after POC demo
	logging.Infof("╔══════════════════════════════════════════════════════════════════╗")
	logging.Infof("║                    MEMORY FACT EXTRACTION                        ║")
	logging.Infof("╠══════════════════════════════════════════════════════════════════╣")
	logging.Infof("║ MESSAGES TO EXTRACT FROM (%d messages):                          ║", len(messages))
	for _, msg := range messages {
		logging.Infof("║   [%s]: %s", msg.Role, truncateForLog(msg.Content, 50))
	}
	logging.Infof("╚══════════════════════════════════════════════════════════════════╝")

	// Call LLM for extraction
	facts, err := e.callLLMForExtraction(ctx, userPrompt)
	if err != nil {
		logging.Warnf("Memory: Fact extraction failed: %v", err)
		return nil, nil // Graceful degradation
	}

	// TODO: Remove debug logs after POC demo
	logging.Infof("╔══════════════════════════════════════════════════════════════════╗")
	logging.Infof("║ EXTRACTED FACTS (%d):                                            ║", len(facts))
	for i, fact := range facts {
		logging.Infof("║   %d. [%s] %s", i+1, fact.Type, fact.Content) // Full content for demo
	}
	logging.Infof("╚══════════════════════════════════════════════════════════════════╝")

	return facts, nil
}

// callLLMForExtraction calls the configured LLM endpoint for fact extraction.
// Uses response_format: json_object to enforce valid JSON output and prevent
// reasoning model artifacts like <think> tags.
func (e *MemoryExtractor) callLLMForExtraction(ctx context.Context, userPrompt string) ([]ExtractedFact, error) {
	jsonFormat := shared.NewResponseFormatJSONObjectParam()
	reqBody := openai.ChatCompletionNewParams{
		Model: e.model,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(extractionSystemPrompt),
			openai.UserMessage(userPrompt),
		},
		MaxTokens:   openai.Int(int64(e.maxTokens)),
		Temperature: openai.Float(e.temperature),
		ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &jsonFormat,
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1/chat/completions", strings.TrimSuffix(e.endpoint, "/"))
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if e.accessKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+e.accessKey)
	}

	resp, err := e.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("LLM request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("LLM returned status %d", resp.StatusCode)
	}

	var llmResp openai.ChatCompletion
	if err := json.NewDecoder(resp.Body).Decode(&llmResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if len(llmResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in LLM response")
	}

	content := llmResp.Choices[0].Message.Content
	logging.Debugf("Memory extraction raw LLM content (%d chars): %s", len(content), truncateForLog(content, 200))
	return parseExtractedFacts(content)
}

// =============================================================================
// Extraction + Storage with Deduplication
// =============================================================================

// ProcessResponse extracts and stores memories from conversation history.
// It runs extraction every N turns (batchSize) to avoid excessive LLM calls.
// This method combines extraction (using ExtractFacts) + storage with deduplication.
func (e *MemoryExtractor) ProcessResponse(
	ctx context.Context,
	sessionID string,
	userID string,
	history []Message,
) error {
	// TODO: Remove demo logging after POC
	logging.Infof("╔══════════════════════════════════════════════════════════════════╗")
	logging.Infof("║              MEMORY EXTRACTION: ProcessResponse                  ║")
	logging.Infof("╠══════════════════════════════════════════════════════════════════╣")
	logging.Infof("║ sessionID: %s", sessionID)
	logging.Infof("║ userID: %s", userID)
	logging.Infof("║ historyLen: %d", len(history))
	logging.Infof("╚══════════════════════════════════════════════════════════════════╝")

	if e.store == nil || !e.store.IsEnabled() {
		logging.Infof("Memory extraction: SKIPPED - store not enabled (store=%v)", e.store != nil)
		return nil // Store not enabled, skip extraction
	}

	if e == nil || e.endpoint == "" {
		logging.Infof("Memory extraction: SKIPPED - extraction not configured")
		return nil // Extraction not enabled
	}

	// Track turn count for this session
	e.mu.Lock()
	e.turnCounts[sessionID]++
	turnCount := e.turnCounts[sessionID]
	e.mu.Unlock()

	// Use batch size from struct (set at construction)
	batchSize := e.batchSize

	// TODO: Remove demo logging after POC
	logging.Infof("Memory extraction: turnCount=%d, batchSize=%d, shouldExtract=%v",
		turnCount, batchSize, turnCount%batchSize == 0)

	// Only extract every N turns
	if turnCount%batchSize != 0 {
		logging.Infof("Memory extraction: SKIPPED - not batch turn (turn %d, batch every %d)", turnCount, batchSize)
		return nil
	}

	// Get recent batch (last N+5 messages for context)
	batchStart := 0
	if len(history) > batchSize+5 {
		batchStart = len(history) - batchSize - 5
	}
	batch := history[batchStart:]

	// Use ExtractFacts for extraction
	extracted, err := e.ExtractFacts(ctx, batch)
	if err != nil {
		logging.Warnf("Memory extraction failed: %v", err)
		return err // Return error but don't block response
	}

	if len(extracted) == 0 {
		logging.Debugf("Memory extraction: no facts extracted from batch")
		return nil
	}

	// Store with deduplication — strip think tags from fact content
	for _, fact := range extracted {
		fact.Content = stripThinkTags(fact.Content)
		if fact.Content == "" {
			continue
		}
		if err := e.storeWithDeduplication(ctx, userID, fact); err != nil {
			logging.Warnf("Failed to store memory with deduplication: %v", err)
		}
	}

	return nil
}

// storeWithDeduplication stores a fact with deduplication logic.
// It checks for similar existing memories and either updates or creates new ones.
func (e *MemoryExtractor) storeWithDeduplication(
	ctx context.Context,
	userID string,
	fact ExtractedFact,
) error {
	// TODO: Remove demo logging after POC
	logging.Infof("╔══════════════════════════════════════════════════════════════════╗")
	logging.Infof("║              DEDUPLICATION CHECK                                 ║")
	logging.Infof("╠══════════════════════════════════════════════════════════════════╣")
	logging.Infof("║ userID: %s", userID)
	logging.Infof("║ fact.Type: %s", fact.Type)
	logging.Infof("║ fact.Content: %s", fact.Content) // Full content for demo
	logging.Infof("╚══════════════════════════════════════════════════════════════════╝")

	// Check for similar memories using deduplication logic
	result := CheckDeduplication(ctx, e.store, userID, fact.Content, fact.Type, e.dedupConfig)

	// TODO: Remove demo logging after POC
	logging.Infof("Deduplication result: action=%s, similarity=%.3f, existingID=%v",
		result.Action, result.Similarity, result.ExistingMemory != nil)

	switch result.Action {
	case "update":
		// Very similar → UPDATE existing memory
		if result.ExistingMemory == nil {
			// Should not happen, but handle gracefully
			logging.Warnf("Memory deduplication: update action but no existing memory")
			return e.createNewMemory(ctx, userID, fact)
		}

		// Update existing memory with new content
		result.ExistingMemory.Content = fact.Content // Use newer content
		result.ExistingMemory.UpdatedAt = time.Now()

		if err := e.store.Update(ctx, result.ExistingMemory.ID, result.ExistingMemory); err != nil {
			return fmt.Errorf("failed to update memory: %w", err)
		}

		logging.Infof("Memory deduplication: UPDATED memory id=%s (similarity=%.3f)",
			result.ExistingMemory.ID, result.Similarity)
		return nil

	case "create":
		// Create new memory (either no similar found, or in gray zone)
		return e.createNewMemory(ctx, userID, fact)

	default:
		// Unknown action - default to create
		logging.Warnf("Memory deduplication: unknown action '%s', defaulting to create", result.Action)
		return e.createNewMemory(ctx, userID, fact)
	}
}

// createNewMemory creates a new memory from an extracted fact.
func (e *MemoryExtractor) createNewMemory(
	ctx context.Context,
	userID string,
	fact ExtractedFact,
) error {
	mem := &Memory{
		ID:         generateMemoryID(),
		Type:       fact.Type,
		Content:    fact.Content,
		UserID:     userID,
		Source:     "conversation",
		CreatedAt:  time.Now(),
		Importance: 0.5, // Default importance
	}

	if err := e.store.Store(ctx, mem); err != nil {
		return fmt.Errorf("failed to store memory: %w", err)
	}

	logging.Infof("Memory deduplication: CREATED new memory id=%s, type=%s", mem.ID, mem.Type)
	return nil
}

// generateMemoryID generates a unique memory ID
func generateMemoryID() string {
	return fmt.Sprintf("mem_%d", time.Now().UnixNano())
}

// =============================================================================
// Response Parsing
// =============================================================================

// parseExtractedFacts parses the LLM response into ExtractedFact structs.
// Handles both bare JSON arrays and wrapped objects (e.g. {"facts": [...]})
// since response_format: json_object may cause the model to wrap the array.
// Also recovers partial results from truncated JSON arrays (token limit hit).
func parseExtractedFacts(content string) ([]ExtractedFact, error) {
	content = strings.TrimSpace(content)
	content = cleanJSONResponse(content)

	if content == "" || content == "[]" {
		return nil, nil
	}

	// Try parsing as a JSON array first
	var facts []ExtractedFact
	if err := json.Unmarshal([]byte(content), &facts); err != nil {
		// Try as a single object (model sometimes omits the array wrapper)
		var single ExtractedFact
		if sErr := json.Unmarshal([]byte(content), &single); sErr == nil && single.Content != "" {
			facts = []ExtractedFact{single}
		}
		// Try as a wrapper object like {"facts": [...]} or {"memories": [...]}
		if len(facts) == 0 {
			var wrapper map[string]json.RawMessage
			if wErr := json.Unmarshal([]byte(content), &wrapper); wErr == nil {
				for _, v := range wrapper {
					if jErr := json.Unmarshal(v, &facts); jErr == nil && len(facts) > 0 {
						break
					}
					// Also try unwrapping single objects inside wrapper values
					var singleInner ExtractedFact
					if jErr := json.Unmarshal(v, &singleInner); jErr == nil && singleInner.Content != "" {
						facts = append(facts, singleInner)
					}
				}
			}
		}
		// Try recovering from truncated JSON array
		if len(facts) == 0 {
			if recovered := recoverTruncatedJSON(content); len(recovered) > 0 {
				logging.Warnf("Memory: Recovered %d facts from truncated JSON", len(recovered))
				facts = recovered
			}
		}
		if len(facts) == 0 {
			return nil, fmt.Errorf("failed to parse facts JSON: %w (content: %s)", err, truncateForLog(content, 100))
		}
	}

	// Validate and filter facts
	validFacts := make([]ExtractedFact, 0, len(facts))
	for _, fact := range facts {
		// Skip empty content
		if strings.TrimSpace(fact.Content) == "" {
			continue
		}

		// Normalize type (empty content was already filtered above)
		normalizedType := normalizeMemoryType(string(fact.Type))
		if normalizedType == "" {
			continue
		}

		validFacts = append(validFacts, ExtractedFact{
			Type:    normalizedType,
			Content: strings.TrimSpace(fact.Content),
		})
	}

	return validFacts, nil
}

// recoverTruncatedJSON attempts to salvage complete JSON objects from a
// truncated array. When the model hits the token limit mid-output, the JSON
// array is cut off (e.g. `[{"type":"semantic","content":"A"},{"type":"sem`).
// We find the last complete object boundary and re-close the array.
func recoverTruncatedJSON(content string) []ExtractedFact {
	content = strings.TrimSpace(content)
	if !strings.HasPrefix(content, "[") {
		return nil
	}

	// Find the last complete object: look for "}," or "}" followed by truncation
	lastComplete := strings.LastIndex(content, "}")
	if lastComplete < 0 {
		return nil
	}

	// Slice up to and including the last "}", then close the array
	recovered := content[:lastComplete+1]
	if !strings.HasSuffix(recovered, "]") {
		recovered = strings.TrimRight(recovered, ", \t\n") + "]"
	}

	var facts []ExtractedFact
	if err := json.Unmarshal([]byte(recovered), &facts); err != nil {
		return nil
	}
	return facts
}

// cleanJSONResponse removes markdown code blocks, <think> tags, and other
// formatting artifacts from LLM output before JSON parsing.
//
// Even with response_format: json_object, some reasoning models (e.g. MiniMax-M2.1
// via vLLM) still emit <think>...</think> blocks. The JSON may appear after, before,
// or even INSIDE the think tags (when </think> is missing).
func cleanJSONResponse(content string) string {
	// Strip closed <think>...</think> blocks.
	thinkPattern := regexp.MustCompile(`(?s)<think>.*?</think>`)
	content = thinkPattern.ReplaceAllString(content, "")
	content = strings.TrimSpace(content)

	// If nothing remains or content still starts with <think> (unclosed),
	// extract JSON by finding the first [ or { in the original/remaining text.
	if content == "" || strings.HasPrefix(content, "<") {
		content = extractJSONFromContent(content)
	}

	// Remove markdown code blocks
	codeBlockPattern := regexp.MustCompile("(?s)```(?:json)?\\s*(.+?)\\s*```")
	if matches := codeBlockPattern.FindStringSubmatch(content); len(matches) > 1 {
		content = matches[1]
	}

	return strings.TrimSpace(content)
}

// extractJSONFromContent finds the first JSON array or object in a string,
// regardless of surrounding non-JSON text (think tags, prose, etc.).
func extractJSONFromContent(content string) string {
	// Prefer arrays first (our expected format)
	if idx := strings.Index(content, "["); idx >= 0 {
		return content[idx:]
	}
	// Fall back to object (wrapper like {"facts": [...]})
	if idx := strings.Index(content, "{"); idx >= 0 {
		return content[idx:]
	}
	return ""
}

// normalizeMemoryType converts string to MemoryType.
// Unrecognized types default to semantic because most facts are semantic and
// some models invent their own taxonomy (identity, preference, health, etc.).
func normalizeMemoryType(typeStr string) MemoryType {
	switch strings.ToLower(strings.TrimSpace(typeStr)) {
	case "semantic":
		return MemoryTypeSemantic
	case "procedural":
		return MemoryTypeProcedural
	case "episodic":
		return MemoryTypeEpisodic
	default:
		if typeStr == "" {
			return ""
		}
		logging.Debugf("Memory: Mapping unknown type %q to semantic", typeStr)
		return MemoryTypeSemantic
	}
}

// =============================================================================
// Helper Functions
// =============================================================================

// stripThinkTags removes <think>...</think> blocks and trailing unclosed
// <think> from text. Used to clean LLM responses before they enter the
// memory pipeline (conversation history, extracted facts, rewritten queries).
var thinkClosedPattern = regexp.MustCompile(`(?s)<think>.*?</think>\s*`)
var thinkUnclosedPattern = regexp.MustCompile(`(?s)<think>.*`)

func stripThinkTags(s string) string {
	s = thinkClosedPattern.ReplaceAllString(s, "")
	s = thinkUnclosedPattern.ReplaceAllString(s, "")
	return strings.TrimSpace(s)
}

// formatMessagesForExtraction formats messages for the LLM extraction prompt.
// Strips <think> tags from assistant responses so reasoning artifacts don't
// contaminate extraction or get stored as memory content.
func formatMessagesForExtraction(messages []Message) string {
	var lines []string
	for _, msg := range messages {
		content := msg.Content
		if msg.Role == "assistant" {
			content = stripThinkTags(content)
		}
		lines = append(lines, fmt.Sprintf("[%s]: %s", msg.Role, content))
	}
	return strings.Join(lines, "\n")
}

// truncateForLog truncates a string for logging purposes
func truncateForLog(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// LLM request and response types are provided by github.com/openai/openai-go:
//   Request:  openai.ChatCompletionNewParams
//   Response: openai.ChatCompletion

// =============================================================================
// Utility Functions
// =============================================================================

// ExtractFactsFromReader extracts facts from a reader (e.g., for testing)
func (e *MemoryExtractor) ExtractFactsFromReader(reader io.Reader) ([]ExtractedFact, error) {
	content, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read content: %w", err)
	}

	return parseExtractedFacts(string(content))
}
