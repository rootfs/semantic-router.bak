package authz

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
	"time"
)

// Default cache TTL when not specified in config.
const defaultCacheTTL = 5 * time.Minute

// oidcCacheEntry stores a validated OIDC token result with an expiry.
type oidcCacheEntry struct {
	entry     *TokenEntry
	expiresAt time.Time
}

// OIDCValidator validates bearer tokens against configured OIDC / OAuth2
// providers by calling their userinfo endpoints. Results are cached.
type OIDCValidator struct {
	providers []OIDCProvider
	cache     map[string]*oidcCacheEntry
	mu        sync.RWMutex
	client    *http.Client
}

// NewOIDCValidator creates a validator for the given OIDC providers.
// Returns nil if providers is empty (no-op).
func NewOIDCValidator(providers []OIDCProvider) *OIDCValidator {
	if len(providers) == 0 {
		return nil
	}
	return &OIDCValidator{
		providers: providers,
		cache:     make(map[string]*oidcCacheEntry),
		client:    &http.Client{Timeout: 10 * time.Second},
	}
}

// ProviderCount returns the number of configured OIDC providers.
func (v *OIDCValidator) ProviderCount() int {
	return len(v.providers)
}

// Validate tries each configured provider until one accepts the token.
// Returns a synthetic TokenEntry on success, or nil if all providers reject it.
func (v *OIDCValidator) Validate(token string) *TokenEntry {
	// Fast path: check cache under read lock.
	v.mu.RLock()
	if cached, ok := v.cache[token]; ok && time.Now().Before(cached.expiresAt) {
		v.mu.RUnlock()
		return cached.entry
	}
	v.mu.RUnlock()

	// Slow path: call each provider's userinfo endpoint.
	for i := range v.providers {
		entry := v.validateWithProvider(token, &v.providers[i])
		if entry != nil {
			ttl := parseTTL(v.providers[i].TokenCacheTTL)
			v.mu.Lock()
			v.cache[token] = &oidcCacheEntry{
				entry:     entry,
				expiresAt: time.Now().Add(ttl),
			}
			v.mu.Unlock()
			return entry
		}
	}
	return nil
}

// validateWithProvider calls the provider's userinfo endpoint and maps the
// returned identity to a TokenEntry via user_mappings.
func (v *OIDCValidator) validateWithProvider(token string, provider *OIDCProvider) *TokenEntry {
	req, err := http.NewRequest("GET", provider.UserinfoEndpoint, nil)
	if err != nil {
		log.Printf("[ext_authz/oidc] Failed to build request for %s: %v", provider.Name, err)
		return nil
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "semantic-router-ext-authz")

	resp, err := v.client.Do(req)
	if err != nil {
		log.Printf("[ext_authz/oidc] %s userinfo request failed: %v", provider.Name, err)
		return nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("[ext_authz/oidc] %s userinfo returned %d", provider.Name, resp.StatusCode)
		return nil
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20)) // 1 MB limit
	if err != nil {
		log.Printf("[ext_authz/oidc] %s failed to read userinfo body: %v", provider.Name, err)
		return nil
	}

	var claims map[string]interface{}
	if err := json.Unmarshal(body, &claims); err != nil {
		log.Printf("[ext_authz/oidc] %s failed to parse userinfo JSON: %v", provider.Name, err)
		return nil
	}

	// Extract user identity from the configured claim.
	userID := extractClaim(claims, provider.UserIDClaim)
	if userID == "" {
		log.Printf("[ext_authz/oidc] %s userinfo missing claim %q", provider.Name, provider.UserIDClaim)
		return nil
	}

	// Match against user_mappings.
	var wildcardMapping *OIDCUserMapping
	for i := range provider.UserMappings {
		m := &provider.UserMappings[i]
		if m.UserID == userID {
			desc := m.Description
			if desc == "" {
				desc = userID
			}
			return &TokenEntry{
				AccessToken: token, // synthetic — the original bearer token
				Description: fmt.Sprintf("%s (OIDC/%s: %s)", desc, provider.Name, userID),
				APIKeys:     m.APIKeys,
			}
		}
		if m.UserID == "*" {
			wildcardMapping = m
		}
	}

	// Fall back to wildcard mapping.
	if wildcardMapping != nil {
		desc := wildcardMapping.Description
		if desc == "" {
			desc = "authenticated user"
		}
		return &TokenEntry{
			AccessToken: token,
			Description: fmt.Sprintf("%s (OIDC/%s: %s)", desc, provider.Name, userID),
			APIKeys:     wildcardMapping.APIKeys,
		}
	}

	log.Printf("[ext_authz/oidc] %s user %q not in user_mappings (no wildcard)", provider.Name, userID)
	return nil
}

// extractClaim pulls a string value from a JSON claims map.
// Handles both string and numeric types (GitHub returns "id" as a number).
func extractClaim(claims map[string]interface{}, key string) string {
	v, ok := claims[key]
	if !ok {
		return ""
	}
	switch val := v.(type) {
	case string:
		return val
	default:
		return fmt.Sprintf("%v", val)
	}
}

// parseTTL parses a duration string, returning defaultCacheTTL on error or empty input.
func parseTTL(s string) time.Duration {
	if s == "" {
		return defaultCacheTTL
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		log.Printf("[ext_authz/oidc] invalid token_cache_ttl %q, using default %v", s, defaultCacheTTL)
		return defaultCacheTTL
	}
	return d
}
