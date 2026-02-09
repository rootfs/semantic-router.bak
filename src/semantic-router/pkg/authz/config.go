// Package authz provides an Envoy ext_authz HTTP service for user token
// validation and provider API key injection.
package authz

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// AuthConfig represents the top-level auth tokens configuration file.
type AuthConfig struct {
	Tokens []TokenEntry `yaml:"tokens"`
}

// TokenEntry represents a single user's token mapping.
type TokenEntry struct {
	// AccessToken is the bearer token the user sends in the Authorization header.
	AccessToken string `yaml:"access_token"`
	// Description is a human-readable label for this token (e.g., "User 1 - Dev").
	Description string `yaml:"description,omitempty"`
	// APIKeys maps provider names to their API keys for this user.
	APIKeys map[string]string `yaml:"api_keys"`
}

// TokenStore provides fast lookup of user tokens.
type TokenStore struct {
	// tokenMap maps access_token -> TokenEntry
	tokenMap map[string]*TokenEntry
}

// LoadConfig reads and parses the auth tokens YAML configuration file.
func LoadConfig(path string) (*AuthConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read auth config %s: %w", path, err)
	}

	var cfg AuthConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse auth config %s: %w", path, err)
	}

	return &cfg, nil
}

// NewTokenStore creates a TokenStore from the given config.
func NewTokenStore(cfg *AuthConfig) *TokenStore {
	store := &TokenStore{
		tokenMap: make(map[string]*TokenEntry, len(cfg.Tokens)),
	}
	for i := range cfg.Tokens {
		entry := &cfg.Tokens[i]
		store.tokenMap[entry.AccessToken] = entry
	}
	return store
}

// Lookup returns the TokenEntry for the given access token, or nil if not found.
func (s *TokenStore) Lookup(accessToken string) *TokenEntry {
	return s.tokenMap[accessToken]
}

// GetAPIKey returns the API key for a specific provider (e.g., "openai", "anthropic").
// Returns empty string if the token or provider is not found.
func (s *TokenStore) GetAPIKey(accessToken, provider string) string {
	entry := s.Lookup(accessToken)
	if entry == nil {
		return ""
	}
	return entry.APIKeys[provider]
}

// TokenCount returns the number of configured tokens.
func (s *TokenStore) TokenCount() int {
	return len(s.tokenMap)
}
