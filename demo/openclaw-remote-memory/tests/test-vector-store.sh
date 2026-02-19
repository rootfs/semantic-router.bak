#!/usr/bin/env bash
# Test vector store CRUD and search operations against semantic-router.
set -euo pipefail

API="http://localhost:8080"
PASS=0; FAIL=0
pass() { echo -e "  \033[1;32m✓\033[0m $1"; PASS=$((PASS+1)); }
fail() { echo -e "  \033[1;31m✗\033[0m $1"; FAIL=$((FAIL+1)); }

echo "Vector Store API Tests ($API)"
echo ""

# Create
echo "[1] Create vector store"
VS=$(curl -sf -X POST "$API/v1/vector_stores" \
  -H 'Content-Type: application/json' \
  -d '{"name":"test-vs-crud"}')
VS_ID=$(echo "$VS" | jq -r '.id')
[[ "$VS_ID" == vs_* ]] && pass "Created: $VS_ID" || fail "Create failed"

# Get
echo "[2] Get vector store"
GOT=$(curl -sf "$API/v1/vector_stores/$VS_ID")
GOT_NAME=$(echo "$GOT" | jq -r '.name')
[[ "$GOT_NAME" == "test-vs-crud" ]] && pass "Name matches" || fail "Name mismatch: $GOT_NAME"

# Upload file
echo "[3] Upload file"
FILE=$(curl -sf -X POST "$API/v1/files" \
  -F 'purpose=assistants' \
  -F 'file=@/dev/stdin;filename=test-doc.txt' <<< "The quick brown fox jumps over the lazy dog. This is a test document about animals and jumping.")
FILE_ID=$(echo "$FILE" | jq -r '.id')
[[ "$FILE_ID" == file_* ]] && pass "Uploaded: $FILE_ID" || fail "Upload failed"

# Attach file
echo "[4] Attach file to vector store"
ATTACH=$(curl -sf -X POST "$API/v1/vector_stores/$VS_ID/files" \
  -H 'Content-Type: application/json' \
  -d "{\"file_id\":\"$FILE_ID\"}")
VSF_ID=$(echo "$ATTACH" | jq -r '.id')
[[ "$VSF_ID" == vsf_* ]] && pass "Attached: $VSF_ID" || fail "Attach failed"

# Wait for ingestion
echo "[5] Wait for ingestion..."
sleep 5
FILES=$(curl -sf "$API/v1/vector_stores/$VS_ID/files")
STATUS=$(echo "$FILES" | jq -r '.data[0].status')
[[ "$STATUS" == "completed" ]] && pass "Ingestion completed" || fail "Status: $STATUS"

# Search
echo "[6] Semantic search"
SEARCH=$(curl -sf -X POST "$API/v1/vector_stores/$VS_ID/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"animals jumping","max_num_results":3}')
HITS=$(echo "$SEARCH" | jq '.data | length')
[[ "$HITS" -gt 0 ]] && pass "Search returned $HITS hit(s)" || fail "No search results"

# Cleanup — delete vector store
echo "[7] Delete vector store"
DEL=$(curl -sf -X DELETE "$API/v1/vector_stores/$VS_ID")
DELETED=$(echo "$DEL" | jq -r '.deleted')
[[ "$DELETED" == "true" ]] && pass "Deleted" || fail "Delete failed"

echo ""
echo "Results: $PASS passed, $FAIL failed"
exit $FAIL
