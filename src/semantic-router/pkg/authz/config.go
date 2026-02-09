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
	Tokens        []TokenEntry   `yaml:"tokens"`
	OIDCProviders []OIDCProvider `yaml:"oidc_providers,omitempty"`
}

// OIDCProvider configures an external OIDC / OAuth2 identity provider.
// Tokens not found in the static store are validated by calling the
// provider's userinfo endpoint with the bearer token.
type OIDCProvider struct {
	// Name identifies this provider (e.g. "github", "google").
	Name string `yaml:"name"`
	// UserinfoEndpoint is called with the bearer token to retrieve user claims.
	// For GitHub: https://api.github.com/user
	UserinfoEndpoint string `yaml:"userinfo_endpoint"`
	// UserIDClaim is the JSON field in the userinfo response used as the user identity.
	// For GitHub: "login" (username). For Google: "email".
	UserIDClaim string `yaml:"user_id_claim"`
	// TokenCacheTTL controls how long a validated token is cached (e.g. "5m", "1h").
	// Default: 5m.
	TokenCacheTTL string `yaml:"token_cache_ttl,omitempty"`
	// UserMappings maps OIDC user identities to provider API keys.
	// A mapping with user_id "*" acts as a wildcard for any authenticated user.
	UserMappings []OIDCUserMapping `yaml:"user_mappings"`
}

// OIDCUserMapping maps an OIDC user identity to provider API keys.
type OIDCUserMapping struct {
	// UserID is the value of the UserIDClaim that this mapping matches.
	// Use "*" as a wildcard to match any authenticated user.
	UserID string `yaml:"user_id"`
	// Description is a human-readable label for this mapping.
	Description string `yaml:"description,omitempty"`
	// APIKeys maps provider names to their API keys (same as TokenEntry).
	APIKeys map[string]string `yaml:"api_keys"`
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
