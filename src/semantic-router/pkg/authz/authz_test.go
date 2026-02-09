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
	server := NewServer(store, nil, ":0")

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
	server := NewServer(store, nil, ":0")

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
	server := NewServer(store, nil, ":0")

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
	server := NewServer(store, nil, ":0")

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
	server := NewServer(store, nil, ":0")

	req := httptest.NewRequest("GET", "/healthz", nil)
	w := httptest.NewRecorder()

	server.handleHealth(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d", w.Code)
	}
}

// ── OIDC tests ──────────────────────────────────────────────────────

// mockOIDCServer returns an httptest server that simulates a GitHub-like
// userinfo endpoint.  It validates the bearer token and returns user claims.
func mockOIDCServer(validToken, login string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		if auth != "Bearer "+validToken {
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"message":"Bad credentials"}`)) //nolint:errcheck
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{ //nolint:errcheck
			"login": login,
			"id":    42,
			"name":  "Test User",
		})
	}))
}

func TestOIDCValidator_ValidToken_ExactMatch(t *testing.T) {
	mock := mockOIDCServer("ghp_valid123", "octocat")
	defer mock.Close()

	providers := []OIDCProvider{{
		Name:             "github",
		UserinfoEndpoint: mock.URL,
		UserIDClaim:      "login",
		TokenCacheTTL:    "1m",
		UserMappings: []OIDCUserMapping{
			{UserID: "octocat", Description: "Octocat", APIKeys: map[string]string{"openai": "sk-octocat-oai"}},
		},
	}}
	v := NewOIDCValidator(providers)

	entry := v.Validate("ghp_valid123")
	if entry == nil {
		t.Fatal("Expected OIDC validation to succeed")
	}
	if entry.APIKeys["openai"] != "sk-octocat-oai" {
		t.Errorf("Expected openai key 'sk-octocat-oai', got %q", entry.APIKeys["openai"])
	}
	if entry.Description == "" || entry.Description == "octocat" {
		t.Errorf("Expected enriched description, got %q", entry.Description)
	}
}

func TestOIDCValidator_ValidToken_WildcardMatch(t *testing.T) {
	mock := mockOIDCServer("ghp_wildcard", "randomuser")
	defer mock.Close()

	providers := []OIDCProvider{{
		Name:             "github",
		UserinfoEndpoint: mock.URL,
		UserIDClaim:      "login",
		UserMappings: []OIDCUserMapping{
			{UserID: "specific-user", APIKeys: map[string]string{"openai": "sk-specific"}},
			{UserID: "*", Description: "Any GitHub user", APIKeys: map[string]string{"openai": "sk-shared-github"}},
		},
	}}
	v := NewOIDCValidator(providers)

	entry := v.Validate("ghp_wildcard")
	if entry == nil {
		t.Fatal("Expected OIDC wildcard match to succeed")
	}
	if entry.APIKeys["openai"] != "sk-shared-github" {
		t.Errorf("Expected shared key, got %q", entry.APIKeys["openai"])
	}
}

func TestOIDCValidator_InvalidToken(t *testing.T) {
	mock := mockOIDCServer("ghp_valid123", "octocat")
	defer mock.Close()

	providers := []OIDCProvider{{
		Name:             "github",
		UserinfoEndpoint: mock.URL,
		UserIDClaim:      "login",
		UserMappings: []OIDCUserMapping{
			{UserID: "*", APIKeys: map[string]string{"openai": "sk-any"}},
		},
	}}
	v := NewOIDCValidator(providers)

	entry := v.Validate("ghp_wrong_token")
	if entry != nil {
		t.Error("Expected nil for invalid OIDC token")
	}
}

func TestOIDCValidator_CachesResult(t *testing.T) {
	callCount := 0
	mock := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"login": "cached-user"}) //nolint:errcheck
	}))
	defer mock.Close()

	providers := []OIDCProvider{{
		Name:             "test",
		UserinfoEndpoint: mock.URL,
		UserIDClaim:      "login",
		TokenCacheTTL:    "10m",
		UserMappings:     []OIDCUserMapping{{UserID: "*", APIKeys: map[string]string{"openai": "sk-x"}}},
	}}
	v := NewOIDCValidator(providers)

	// First call hits the server.
	v.Validate("token-cache-test")
	// Second call should be served from cache.
	v.Validate("token-cache-test")

	if callCount != 1 {
		t.Errorf("Expected 1 userinfo call (cached), got %d", callCount)
	}
}

func TestOIDCValidator_UserNotInMappings(t *testing.T) {
	mock := mockOIDCServer("ghp_unmapped", "stranger")
	defer mock.Close()

	providers := []OIDCProvider{{
		Name:             "github",
		UserinfoEndpoint: mock.URL,
		UserIDClaim:      "login",
		UserMappings: []OIDCUserMapping{
			{UserID: "octocat", APIKeys: map[string]string{"openai": "sk-octocat"}},
			// No wildcard — stranger is denied.
		},
	}}
	v := NewOIDCValidator(providers)

	entry := v.Validate("ghp_unmapped")
	if entry != nil {
		t.Error("Expected nil — user not in mappings and no wildcard")
	}
}

func TestOIDCValidator_NilWhenNoProviders(t *testing.T) {
	v := NewOIDCValidator(nil)
	if v != nil {
		t.Error("Expected nil validator when no providers configured")
	}
	v = NewOIDCValidator([]OIDCProvider{})
	if v != nil {
		t.Error("Expected nil validator when empty providers slice")
	}
}

func TestServer_HandleCheck_OIDCFallback(t *testing.T) {
	mock := mockOIDCServer("ghp_oidc_test", "ghuser")
	defer mock.Close()

	// Empty static store — only OIDC.
	store := NewTokenStore(&AuthConfig{})
	oidc := NewOIDCValidator([]OIDCProvider{{
		Name:             "github",
		UserinfoEndpoint: mock.URL,
		UserIDClaim:      "login",
		UserMappings: []OIDCUserMapping{
			{UserID: "ghuser", Description: "GitHub User", APIKeys: map[string]string{
				"openai":    "sk-gh-openai",
				"anthropic": "sk-ant-gh-anthropic",
			}},
		},
	}})

	server := NewServer(store, oidc, ":0")

	req := httptest.NewRequest("GET", "/", nil)
	req.Header.Set("Authorization", "Bearer ghp_oidc_test")
	w := httptest.NewRecorder()

	server.handleCheck(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d", w.Code)
	}
	if w.Header().Get(HeaderUserOpenAIKey) != "sk-gh-openai" {
		t.Errorf("Expected OIDC OpenAI key, got %q", w.Header().Get(HeaderUserOpenAIKey))
	}
	if w.Header().Get(HeaderUserAnthropicKey) != "sk-ant-gh-anthropic" {
		t.Errorf("Expected OIDC Anthropic key, got %q", w.Header().Get(HeaderUserAnthropicKey))
	}
}

func TestLoadConfig_WithOIDCProviders(t *testing.T) {
	cfg := `
tokens:
  - access_token: "static-1"
    api_keys:
      openai: "sk-1"
oidc_providers:
  - name: "github"
    userinfo_endpoint: "https://api.github.com/user"
    user_id_claim: "login"
    token_cache_ttl: "10m"
    user_mappings:
      - user_id: "octocat"
        description: "Octocat"
        api_keys:
          openai: "sk-octocat"
      - user_id: "*"
        api_keys:
          openai: "sk-default"
`
	dir := t.TempDir()
	path := filepath.Join(dir, "oidc_config.yaml")
	os.WriteFile(path, []byte(cfg), 0644) //nolint:errcheck

	config, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig failed: %v", err)
	}
	if len(config.Tokens) != 1 {
		t.Errorf("Expected 1 static token, got %d", len(config.Tokens))
	}
	if len(config.OIDCProviders) != 1 {
		t.Fatalf("Expected 1 OIDC provider, got %d", len(config.OIDCProviders))
	}
	p := config.OIDCProviders[0]
	if p.Name != "github" {
		t.Errorf("Expected provider name 'github', got %q", p.Name)
	}
	if len(p.UserMappings) != 2 {
		t.Errorf("Expected 2 user_mappings, got %d", len(p.UserMappings))
	}
	if p.TokenCacheTTL != "10m" {
		t.Errorf("Expected token_cache_ttl '10m', got %q", p.TokenCacheTTL)
	}
}
