package authz

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
)

// Header names injected by ext_authz into upstream requests.
const (
	// HeaderUserOpenAIKey is the header carrying the user's OpenAI API key.
	HeaderUserOpenAIKey = "x-user-openai-key"
	// HeaderUserAnthropicKey is the header carrying the user's Anthropic API key.
	HeaderUserAnthropicKey = "x-user-anthropic-key"
)

// Server implements the Envoy ext_authz HTTP service.
// It validates user bearer tokens and injects provider API keys as headers.
type Server struct {
	store *TokenStore
	oidc  *OIDCValidator // optional — nil disables OIDC
	addr  string
}

// NewServer creates a new ext_authz HTTP server.
// Pass nil for oidc to disable OIDC validation.
func NewServer(store *TokenStore, oidc *OIDCValidator, addr string) *Server {
	return &Server{
		store: store,
		oidc:  oidc,
		addr:  addr,
	}
}

// authErrorResponse is the JSON error body returned on auth failure.
type authErrorResponse struct {
	Error   string `json:"error"`
	Message string `json:"message"`
}

// Start starts the HTTP ext_authz server. This blocks until the server is stopped.
func (s *Server) Start() error {
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleCheck)
	mux.HandleFunc("/healthz", s.handleHealth)

	oidcInfo := ""
	if s.oidc != nil {
		oidcInfo = fmt.Sprintf(" + %d OIDC provider(s)", s.oidc.ProviderCount())
	}
	log.Printf("[ext_authz] Starting on %s with %d static token(s)%s", s.addr, s.store.TokenCount(), oidcInfo)
	return http.ListenAndServe(s.addr, mux)
}

// handleHealth returns 200 OK for health checks.
func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	fmt.Fprintln(w, "ok")
}

// handleCheck implements the ext_authz check logic.
//
// Envoy sends the original client request (including headers) to this endpoint.
// The response determines whether the request is allowed or denied:
//   - 200: Request is allowed. Response headers matching allowed_upstream_headers
//     in the Envoy config are injected into the upstream request.
//   - 401/403: Request is denied. The response body is returned to the client.
func (s *Server) handleCheck(w http.ResponseWriter, r *http.Request) {
	// Extract bearer token from Authorization header
	authHeader := r.Header.Get("Authorization")
	if authHeader == "" {
		s.denyRequest(w, http.StatusUnauthorized, "Missing Authorization header. Use: Authorization: Bearer <token>")
		return
	}

	token := extractBearerToken(authHeader)
	if token == "" {
		s.denyRequest(w, http.StatusUnauthorized, "Invalid Authorization header format. Expected: Bearer <token>")
		return
	}

	// Look up token — try static store first, then OIDC providers.
	entry := s.store.Lookup(token)
	if entry == nil && s.oidc != nil {
		entry = s.oidc.Validate(token)
	}
	if entry == nil {
		log.Printf("[ext_authz] DENIED: unknown token (prefix: %s...)", safePrefix(token, 8))
		s.denyRequest(w, http.StatusForbidden, "Invalid access token")
		return
	}

	log.Printf("[ext_authz] ALLOWED: %s", entry.Description)

	// Return 200 with provider API keys as response headers.
	// Envoy will inject these into the upstream request based on
	// allowed_upstream_headers configuration.
	if key, ok := entry.APIKeys["openai"]; ok && key != "" {
		w.Header().Set(HeaderUserOpenAIKey, key)
	}
	if key, ok := entry.APIKeys["anthropic"]; ok && key != "" {
		w.Header().Set(HeaderUserAnthropicKey, key)
	}

	w.WriteHeader(http.StatusOK)
	fmt.Fprintln(w, "ok")
}

// denyRequest sends an auth denial response with the given status and message.
func (s *Server) denyRequest(w http.ResponseWriter, statusCode int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	resp := authErrorResponse{
		Error:   http.StatusText(statusCode),
		Message: message,
	}
	json.NewEncoder(w).Encode(resp) //nolint:errcheck
}

// extractBearerToken extracts the token from "Bearer <token>".
func extractBearerToken(authHeader string) string {
	const prefix = "Bearer "
	if !strings.HasPrefix(authHeader, prefix) {
		// Also accept lowercase
		if !strings.HasPrefix(authHeader, "bearer ") {
			return ""
		}
		return strings.TrimSpace(authHeader[len("bearer "):])
	}
	return strings.TrimSpace(authHeader[len(prefix):])
}

// safePrefix returns the first n characters of s, or s if it's shorter.
func safePrefix(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
