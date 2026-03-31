#!/bin/bash
set -e
MODEL=${MODEL:-/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf}
WIKI=$(find /home/erol/ai/turboquant -name "wiki.test.raw" 2>/dev/null | head -1)

echo "=== TurboQuant Quality Gate ==="
echo "Model: $MODEL"
echo "Wiki:  $WIKI"
echo ""

# 1. PPL (turbo3 ctx=512 < 6.89)
echo "--- Step 1: PPL (turbo3 ctx=512, threshold < 6.89) ---"
PPL=$(./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99 --no-mmap 2>&1 | grep "Final estimate" | awk '{print $5}')
echo "PPL: $PPL"
if (( $(echo "$PPL > 6.89" | bc -l) )); then
    echo "FAIL: PPL $PPL > 6.89"
    exit 1
fi
echo "PASS: PPL $PPL <= 6.89"
echo ""

# 2. Speed (warmup + measure, turbo3 short > 55 tok/s)
echo "--- Step 2: Speed (turbo3 short, threshold > 55 tok/s) ---"
echo "  Warmup run..."
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d 0 -ngl 99 -t 1 -r 1 -p 0 -n 128 -mmp 0 > /dev/null 2>&1
echo "  Measuring..."
SPEED=$(./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d 0 -ngl 99 -t 1 -r 3 -p 0 -n 128 -mmp 0 2>&1 | grep "tg128" | grep -oP 'tg128\s*\|\s*\K[0-9.]+')
echo "Speed: $SPEED tok/s"
if (( $(echo "$SPEED < 55.0" | bc -l) )); then
    echo "FAIL: speed $SPEED < 55.0"
    exit 1
fi
echo "PASS: speed $SPEED >= 55.0 tok/s"
echo ""

# 3. Generation (Qwen 9B D=256 turbo3 — must produce non-empty content)
echo "--- Step 3: Generation (turbo3 D=256, must produce content) ---"
GEN_MODEL=/home/erol/ai/turboquant/models/Qwen3.5-9B-Q8_0.gguf
if [ ! -f "$GEN_MODEL" ]; then
    echo "SKIP: Generation model not found at $GEN_MODEL"
else
    echo "  Starting server..."
    ./build/bin/llama-server -m $GEN_MODEL \
        -ctk turbo3 -ctv turbo3 -fa on -ngl 99 -c 4096 --port 8091 --no-mmap --log-disable &
    SERVER_PID=$!
    sleep 30

    CONTENT=$(curl -s http://localhost:8091/v1/chat/completions -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":200,"temperature":0}' \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:50])" 2>/dev/null || echo "")

    kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; sleep 3
    pkill -f "llama-server.*8091" 2>/dev/null || true; sleep 2

    echo "Generation: '$CONTENT'"
    if [ -z "$CONTENT" ]; then
        echo "FAIL: empty generation"
        exit 1
    fi
    echo "PASS: generation produced content"
fi
echo ""

echo "=== ALL CHECKS PASSED ==="
