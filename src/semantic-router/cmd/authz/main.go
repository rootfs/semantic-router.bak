// Command authz runs the Envoy ext_authz HTTP service for user token
// validation and provider API key injection.
//
// Usage:
//
//	authz [-config path/to/auth_tokens.yaml] [-addr :9001]
package main

import (
	"flag"
	"log"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
)

func main() {
	configPath := flag.String("config", "config/auth_tokens.yaml", "Path to auth tokens YAML config")
	addr := flag.String("addr", ":9001", "Listen address for ext_authz HTTP service")
	flag.Parse()

	// Load configuration
	cfg, err := authz.LoadConfig(*configPath)
	if err != nil {
		log.Fatalf("Failed to load auth config: %v", err)
	}

	store := authz.NewTokenStore(cfg)
	log.Printf("Loaded %d static token(s) from %s", store.TokenCount(), *configPath)

	// Initialise OIDC validator if providers are configured.
	oidc := authz.NewOIDCValidator(cfg.OIDCProviders)
	if oidc != nil {
		log.Printf("Loaded %d OIDC provider(s) from %s", oidc.ProviderCount(), *configPath)
	}

	// Start server
	server := authz.NewServer(store, oidc, *addr)
	if err := server.Start(); err != nil {
		log.Fatalf("ext_authz server failed: %v", err)
	}
}
