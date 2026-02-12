# Authz Demos

Complete demos for each auth backend — configs, scripts, tests, and recordings in one place.

## Providers

| Directory | Auth Backend | Identity Headers | K8s Required |
|---|---|---|---|
| `authorino/` | Authorino ext_authz (K8s Secrets) | `x-authz-user-id`, `x-authz-user-groups` | Yes |
| `envoy-jwt/` | Envoy jwt_authn (RSA256, local JWKS) | `x-jwt-sub`, `x-jwt-groups` | No |

## Directory Layout

```
demo/authz/
├── README.md                           # this file
├── authorino/
│   ├── README.md                       # Authorino-specific setup guide
│   ├── config.yaml                     # router config — default identity headers, RBAC bindings
│   ├── envoy.yaml                      # Envoy — ext_authz (Authorino) + ext_proc (Router)
│   ├── profile.yaml                    # authz credential chain profile
│   ├── envoy-authorino-test.yaml       # Envoy config for testing
│   ├── authz-profile-*.yaml           # alternative authz profiles (static, gemini, local-vllm)
│   ├── k8s/                            # Kubernetes manifests
│   │   ├── authconfig.yaml
│   │   ├── k8s-deploy.yaml
│   │   ├── secrets-byot.yaml
│   │   ├── secrets-per-user.yaml
│   │   └── secrets-shared.yaml
│   ├── demo-authz.sh                   # Authorino ext_authz demo (fail-closed)
│   ├── demo-integration.sh             # end-to-end Authorino integration demo
│   ├── demo-oidc.sh                    # OIDC/GitHub auth demo
│   ├── demo-rbac-asciinema.sh          # RBAC asciinema recording script
│   ├── setup-rbac-sample.sh            # RBAC sample infrastructure setup
│   └── test.sh                         # live integration test (8 tests)
└── envoy-jwt/
    ├── README.md                       # JWT-specific setup guide
    ├── config.yaml                     # router config — custom identity headers (x-jwt-*)
    ├── config-rbac-demo.yaml           # simplified RBAC-only router config (no keywords)
    ├── envoy.yaml                      # Envoy — jwt_authn + ext_proc + ORIGINAL_DST
    ├── generate-jwt-keys.py            # RSA key pair + JWKS generation
    ├── generate-jwt-tokens.py          # JWT minting per test user
    ├── jwt-artifacts/                  # generated keys, tokens
    │   ├── jwks.json
    │   ├── private-key.pem
    │   └── tokens.env
    ├── setup.sh                        # generate keys/tokens, start Envoy container
    ├── test.sh                         # live integration test (8 tests, real JWTs)
    ├── demo-asciinema.sh               # terminal demo recording script
    ├── demo-rbac.sh                    # RBAC JWT routing demo
    ├── demo-rbac.cast                  # asciinema recording of RBAC demo
    └── demo.cast                       # asciinema recording of JWT demo
```

## Quick Start

### Envoy JWT (no Kubernetes required)

```bash
cd demo/authz/envoy-jwt

# 1. Generate keys, tokens, start Envoy
bash setup.sh

# 2. Start the router
go run cmd/main.go --config demo/authz/envoy-jwt/config.yaml --port 50053

# 3. Run tests
bash test.sh

# 4. Run RBAC demo
bash demo-rbac.sh
```

### Authorino (requires Kind cluster)

```bash
cd demo/authz/authorino

# 1. Set up K8s resources
kubectl apply -f k8s/

# 2. Start the router
go run cmd/main.go --config demo/authz/authorino/config.yaml --port 50051

# 3. Run tests
bash test.sh
```

## Adding a New Provider

1. Create `demo/authz/<provider-name>/`
2. Add `config.yaml` with the appropriate `authz.identity` headers
3. Add `envoy.yaml` for the Envoy/gateway configuration
4. Add `test.sh` for integration testing
5. Update this README
