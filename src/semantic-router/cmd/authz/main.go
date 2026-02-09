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
	log.Printf("Loaded %d user tokens from %s", store.TokenCount(), *configPath)

	// Start server
	server := authz.NewServer(store, *addr)
	if err := server.Start(); err != nil {
		log.Fatalf("ext_authz server failed: %v", err)
	}
}
