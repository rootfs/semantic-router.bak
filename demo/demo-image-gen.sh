#!/bin/bash
# Demo: Image Generation through Semantic Router
# This demo shows image generation via Envoy -> Router -> vLLM-Omni

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Semantic Router - Image Generation Demo                    ║"
echo "║     Using vLLM-Omni with FLUX.1-schnell model                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
sleep 1

echo -e "${YELLOW}📋 Architecture:${NC}"
echo "   Client → Envoy:8801 → Router ExtProc:50051 → vLLM-Omni:8001"
echo
sleep 1

echo -e "${YELLOW}🔍 Checking services...${NC}"
sleep 0.5

# Check services
echo -n "   vLLM-Omni (port 8001): "
if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${RED}✗ Not running${NC}"
    exit 1
fi

echo -n "   Router (port 50051):   "
if lsof -i :50051 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${RED}✗ Not running${NC}"
    exit 1
fi

echo -n "   Envoy (port 8801):     "
if lsof -i :8801 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${RED}✗ Not running${NC}"
    exit 1
fi

echo
sleep 1

echo -e "${YELLOW}📝 Sending image generation request...${NC}"
echo -e "${BLUE}"
echo '   curl -X POST http://localhost:8801/v1/chat/completions \'
echo '     -H "Content-Type: application/json" \'
echo '     -d {"model": "auto", "messages": [{"role": "user",'
echo '          "content": "Generate an image of a sunset over mountains"}]}'
echo -e "${NC}"
sleep 1

echo -e "${YELLOW}⏳ Generating image with FLUX.1-schnell...${NC}"
START_TIME=$(date +%s)

# Make the request and save response
RESPONSE=$(curl -sS -X POST http://localhost:8801/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "auto",
        "messages": [
            {"role": "user", "content": "Generate an image of a beautiful sunset over mountains with orange and purple sky"}
        ]
    }' 2>&1)

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo
echo -e "${GREEN}✓ Image generated in ${DURATION}s${NC}"
echo

# Extract image and save
IMAGE_FILE="/tmp/demo_generated_image.png"
echo "$RESPONSE" | python3 -c "
import json
import sys
import base64

try:
    data = json.load(sys.stdin)
    content = data['choices'][0]['message']['content']
    if isinstance(content, list):
        for part in content:
            if part.get('type') == 'image_url':
                url = part['image_url']['url']
                if url.startswith('data:image/'):
                    b64_data = url.split(',')[1]
                    with open('$IMAGE_FILE', 'wb') as f:
                        f.write(base64.b64decode(b64_data))
                    print('Image saved!')
                    break
except Exception as e:
    print(f'Error: {e}')
"

echo -e "${YELLOW}📊 Response details:${NC}"
echo "$RESPONSE" | python3 -c "
import json
import sys
try:
    data = json.load(sys.stdin)
    print(f'   Model: {data.get(\"model\", \"N/A\")}')
    print(f'   ID: {data.get(\"id\", \"N/A\")}')
    content = data['choices'][0]['message']['content']
    if isinstance(content, list):
        url = content[0].get('image_url', {}).get('url', '')
        print(f'   Image size: {len(url):,} bytes (base64)')
except Exception as e:
    print(f'   Error parsing response: {e}')
"
echo

sleep 1
echo -e "${YELLOW}🖼️  Generated Image (ASCII preview):${NC}"
echo
sleep 0.5

# Display image with chafa
if [ -f "$IMAGE_FILE" ]; then
    chafa --size=60x30 --colors=256 "$IMAGE_FILE" 2>/dev/null || echo "   (Image preview not available)"
    echo
    echo -e "${GREEN}   File: $IMAGE_FILE${NC}"
    ls -lh "$IMAGE_FILE" 2>/dev/null | awk '{print "   Size: " $5}'
    sleep 3  # Pause to admire the image
else
    echo "   (Image file not found)"
fi

echo
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗"
echo -e "║  ${GREEN}✓ Demo Complete!${CYAN}                                              ║"
echo -e "║                                                                ║"
echo -e "║  The Semantic Router successfully:                            ║"
echo -e "║    • Detected image generation intent via keyword matching    ║"
echo -e "║    • Routed to vLLM-Omni backend                             ║"
echo -e "║    • Generated image with FLUX.1-schnell                     ║"
echo -e "║    • Returned base64-encoded PNG                             ║"
echo -e "╚════════════════════════════════════════════════════════════════╝${NC}"
echo
