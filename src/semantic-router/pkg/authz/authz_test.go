package authz

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

const testConfig = `
tokens:
  - access_token: "test-token-abc"
    description: "Test User 1"
    api_keys:
      openai: "sk-openai-test-1"
      anthropic: "sk-ant-test-1"
  - access_token: "test-token-def"
    description: "Test User 2"
    api_keys:
      openai: "sk-openai-test-2"
`

func setupTestConfig(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "auth_tokens.yaml")
	if err := os.WriteFile(path, []byte(testConfig), 0644); err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}
	return path
}

func TestLoadConfig(t *testing.T) {
	path := setupTestConfig(t)
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig failed: %v", err)
	}
	if len(cfg.Tokens) != 2 {
		t.Errorf("Expected 2 tokens, got %d", len(cfg.Tokens))
	}
	if cfg.Tokens[0].AccessToken != "test-token-abc" {
		t.Errorf("Expected token 'test-token-abc', got '%s'", cfg.Tokens[0].AccessToken)
	}
}

func TestTokenStore_Lookup(t *testing.T) {
	path := setupTestConfig(t)
	cfg, _ := LoadConfig(path)
	store := NewTokenStore(cfg)

	// Valid lookup
	entry := store.Lookup("test-token-abc")
	if entry == nil {
		t.Fatal("Expected to find token")
	}
	if entry.Description != "Test User 1" {
		t.Errorf("Expected 'Test User 1', got '%s'", entry.Description)
	}

	// Invalid lookup
	if store.Lookup("nonexistent") != nil {
		t.Error("Expected nil for nonexistent token")
	}
}

func TestTokenStore_GetAPIKey(t *testing.T) {
	path := setupTestConfig(t)
	cfg, _ := LoadConfig(path)
	store := NewTokenStore(cfg)

	tests := []struct {
		token    string
		provider string
		expected string
	}{
		{"test-token-abc", "openai", "sk-openai-test-1"},
		{"test-token-abc", "anthropic", "sk-ant-test-1"},
		{"test-token-def", "openai", "sk-openai-test-2"},
		{"test-token-def", "anthropic", ""},     // not configured
		{"nonexistent", "openai", ""},            // unknown token
		{"test-token-abc", "nonexistent", ""},    // unknown provider
	}

	for _, tt := range tests {
		result := store.GetAPIKey(tt.token, tt.provider)
		if result != tt.expected {
			t.Errorf("GetAPIKey(%q, %q) = %q, want %q", tt.token, tt.provider, result, tt.expected)
		}
	}
}

func TestServer_HandleCheck_ValidToken(t *testing.T) {
	path := setupTestConfig(t)
	cfg, _ := LoadConfig(path)
	store := NewTokenStore(cfg)
	server := NewServer(store, ":0")

	req := httptest.NewRequest("GET", "/", nil)
	req.Header.Set("Authorization", "Bearer test-token-abc")
	w := httptest.NewRecorder()

	server.handleCheck(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d", w.Code)
	}
	if w.Header().Get(HeaderUserOpenAIKey) != "sk-openai-test-1" {
		t.Errorf("Expected OpenAI key in response header, got %q", w.Header().Get(HeaderUserOpenAIKey))
	}
	if w.Header().Get(HeaderUserAnthropicKey) != "sk-ant-test-1" {
		t.Errorf("Expected Anthropic key in response header, got %q", w.Header().Get(HeaderUserAnthropicKey))
	}
}

func TestServer_HandleCheck_InvalidToken(t *testing.T) {
	path := setupTestConfig(t)
	cfg, _ := LoadConfig(path)
	store := NewTokenStore(cfg)
	server := NewServer(store, ":0")

	req := httptest.NewRequest("GET", "/", nil)
	req.Header.Set("Authorization", "Bearer wrong-token")
	w := httptest.NewRecorder()

	server.handleCheck(w, req)

	if w.Code != http.StatusForbidden {
		t.Errorf("Expected 403, got %d", w.Code)
	}

	var errResp authErrorResponse
	if err := json.NewDecoder(w.Body).Decode(&errResp); err != nil {
		t.Fatalf("Failed to decode error response: %v", err)
	}
	if errResp.Message != "Invalid access token" {
		t.Errorf("Expected 'Invalid access token', got %q", errResp.Message)
	}
}

func TestServer_HandleCheck_MissingAuth(t *testing.T) {
	path := setupTestConfig(t)
	cfg, _ := LoadConfig(path)
	store := NewTokenStore(cfg)
	server := NewServer(store, ":0")

	req := httptest.NewRequest("GET", "/", nil)
	w := httptest.NewRecorder()

	server.handleCheck(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("Expected 401, got %d", w.Code)
	}
}

func TestServer_HandleCheck_BadAuthFormat(t *testing.T) {
	path := setupTestConfig(t)
	cfg, _ := LoadConfig(path)
	store := NewTokenStore(cfg)
	server := NewServer(store, ":0")

	req := httptest.NewRequest("GET", "/", nil)
	req.Header.Set("Authorization", "Basic dXNlcjpwYXNz")
	w := httptest.NewRecorder()

	server.handleCheck(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("Expected 401, got %d", w.Code)
	}
}

func TestServer_HandleHealth(t *testing.T) {
	store := NewTokenStore(&AuthConfig{})
	server := NewServer(store, ":0")

	req := httptest.NewRequest("GET", "/healthz", nil)
	w := httptest.NewRecorder()

	server.handleHealth(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d", w.Code)
	}
}
