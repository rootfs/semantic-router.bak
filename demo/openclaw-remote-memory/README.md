# OpenClaw + Semantic Router Demo

End-to-end demonstration of [OpenClaw](https://github.com/anthropics/openclaw) using
**semantic-router** as both the **LLM inference gateway** and the **remote vector store**
for OpenClaw's memory backend.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  OpenClaw Agent                                                  │
│  (Node.js container, port 18788)                                 │
│                                                                  │
│  ┌──── LLM inference ─────┐    ┌──── Memory (vector store) ───┐ │
│  │ POST /v1/chat/          │    │ POST /v1/vector_stores/…/    │ │
│  │      completions        │    │      search                  │ │
│  └──────────┬──────────────┘    └──────────┬──────────────────┘ │
└─────────────┼──────────────────────────────┼────────────────────┘
              │                              │
              ▼                              ▼
      ┌───────────────┐             ┌──────────────────┐
      │ Envoy  :8801  │             │ Semantic Router  │
      │ (ext_proc     │───gRPC────▶ │ API  :8080       │
      │  filter)      │  :50051     │ gRPC :50051      │
      └───────┬───────┘             └──────────────────┘
              │ x-vsr-destination-endpoint
              ▼
      ┌───────────────┐
      │ vLLM  :8100   │
      │ Qwen2.5-14B   │
      │ (tool calling)│
      └───────────────┘
```

**Traffic flow:**
1. OpenClaw sends chat completions to **Envoy** (`:8801`)
2. Envoy forwards headers/body to **semantic-router** `ext_proc` (`:50051`) for routing decisions
3. Semantic-router selects the best vLLM endpoint and returns a routing header
4. Envoy forwards the request to **vLLM** (`:8100`)
5. OpenClaw sends memory operations (file upload, search) directly to **semantic-router** API (`:8080`)

## Directory Structure

```
demo/
├── README.md                           # This file
├── demo.cast                           # asciinema recording of the full demo
├── openclaw/
│   └── openclaw.json                   # OpenClaw configuration
├── semantic-router/
│   ├── config.yaml                     # Python CLI config (generates Go config)
│   └── router-config-generated.yaml    # Go router config (with vector_store)
├── envoy/
│   └── envoy.yaml                      # Envoy proxy config (ext_proc + dynamic cluster)
├── workspace/
│   └── memory/
│       ├── project-notes.md            # Sample memory doc (team contacts, architecture)
│       └── runbook.md                  # Sample memory doc (incident runbook)
├── scripts/
│   ├── start-all.sh                    # Start all 4 containers
│   ├── stop-all.sh                     # Stop all containers
│   └── record-demo.sh                  # asciinema recording script
└── tests/
    ├── test-all.sh                     # Full integration test suite (8 tests)
    ├── test-vector-store.sh            # Vector store CRUD + search tests
    └── test-llm-inference.sh           # LLM inference chain tests
```

## Prerequisites

- Docker
- ROCm GPU (adapt `--device` flags for NVIDIA)
- Pre-built images:
  - `vllm/vllm-openai-rocm:v0.15.0` (or equivalent CUDA image)
  - `ghcr.io/vllm-project/semantic-router/vllm-sr:latest`
  - `envoyproxy/envoy:v1.31-latest`
  - `openclaw:local` — built from the OpenClaw repo: `docker build -t openclaw:local .`
- `semantic-router` Go binary built from source (for vector store support):
  ```bash
  cd /path/to/semantic-router/src/semantic-router
  CGO_ENABLED=1 go build -o demo/semantic-router/router ./cmd/main.go
  ```
- Rust shared libraries (`*.so`) copied to `demo/semantic-router/libs/`

## Quick Start

```bash
# 1. Start everything
bash demo/scripts/start-all.sh

# 2. Wait ~90s for vLLM to load the model
watch -n5 'curl -sf localhost:8100/v1/models | jq .data[0].id'

# 3. Run integration tests
bash demo/tests/test-all.sh

# 4. Run individual test suites
bash demo/tests/test-vector-store.sh
bash demo/tests/test-llm-inference.sh

# 5. Stop everything
bash demo/scripts/stop-all.sh
```

## Configuration Details

### OpenClaw (`openclaw/openclaw.json`)

| Setting | Value | Purpose |
|---------|-------|---------|
| `models.providers.semantic-router.baseUrl` | `http://127.0.0.1:8801/v1` | LLM via Envoy |
| `memory.backend` | `remote` | Use semantic-router vector store |
| `memory.remote.baseUrl` | `http://127.0.0.1:8080` | Semantic-router API |
| `memory.remote.vectorStoreName` | `openclaw-demo` | Named vector store |
| Model context window | `32768` tokens | Qwen2.5-14B max |
| Model max tokens | `4096` | Output limit |

### Semantic Router (`semantic-router/router-config-generated.yaml`)

Key sections that differ from the auto-generated default:

- **`vector_store`** — Enables the OpenAI-compatible vector store API (`embedding_model: mmbert`, `embedding_dimension: 768`)
- **`model_config`** — Maps `Qwen/Qwen2.5-14B-Instruct` to `preferred_endpoints: ["Qwen_Qwen2.5-14B-Instruct_local_vllm"]`
- **`vllm_endpoints`** — Points to `127.0.0.1:8100` (the vLLM container)

### Envoy (`envoy/envoy.yaml`)

- Listens on `:8801`, forwards to `ext_proc` at `:50051`
- Dynamic cluster uses `x-vsr-destination-endpoint` header for routing
- 300s timeouts for long generation requests

### vLLM

Started with `--enable-auto-tool-choice --tool-call-parser hermes` for OpenClaw's
tool-calling requirements, and `--max-model-len 32768` for the extended context window.

## Ports Summary

| Port | Service | Protocol |
|------|---------|----------|
| 8100 | vLLM (Qwen2.5-14B) | HTTP |
| 8080 | Semantic Router API | HTTP |
| 50051 | Semantic Router gRPC | gRPC (ext_proc) |
| 8801 | Envoy proxy | HTTP |
| 18788 | OpenClaw gateway | HTTP |

## Workspace Memory

The `workspace/memory/` directory contains sample documents that OpenClaw syncs
to the semantic-router vector store:

- **`project-notes.md`** — Team contacts, architecture notes, sprint priorities
- **`runbook.md`** — Incident response procedures (DB pool exhaustion, API latency spikes)

These files are automatically uploaded and chunked by OpenClaw's remote memory
backend when the agent starts.

## Troubleshooting

**Vector store returns 404:**
The auto-generated config may strip the `vector_store` section. Use
`router-config-generated.yaml` directly with the Go binary.

**Embedding errors:**
Ensure `embedding_model: mmbert` and `embedding_dimension: 768` — the semantic cache
pre-loads `mmbert`, and using `bert` (384-dim) separately will fail.

**Envoy 503 (no healthy upstream):**
Check that `model_config` in the router config has `preferred_endpoints` mapping
the model name to the correct `vllm_endpoints` entry name.

**vLLM tool-call errors (400):**
Restart vLLM with `--enable-auto-tool-choice --tool-call-parser hermes`.
